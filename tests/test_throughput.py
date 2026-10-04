"""Throughput: the progress line, the parser's worker processes and load options, and
streaming pattern totals."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import types
from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile import measure, runtime, status
from styleprofile.cli import main
from styleprofile.core import Phase, Progress
from styleprofile.corpus.reading import load_chunks
from styleprofile.corpus.windows import window
from styleprofile.measure import Measurer, workers_can_start
from styleprofile.reference import build_reference
from styleprofile.reports import dumps_report
from styleprofile.status import StatusLine

ROOT = Path(__file__).resolve().parent.parent
WRITER, CONTRAST, DRAFT = (
    ROOT / "examples/writer",
    ROOT / "examples/llm-drafts",
    ROOT / "examples/draft.md",
)
HAS_SPACY = importlib.util.find_spec("spacy") is not None


class Terminal(io.StringIO):
    """A stderr that says it is a terminal."""

    def isatty(self) -> bool:
        return True


def _parser() -> Any:
    pytest.importorskip("spacy")
    from styleprofile.syntax import SyntaxUnavailableError, load_parser

    try:
        return load_parser()
    except SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")


# The progress line.


def _build(tmp_path: Path, stderr: io.StringIO, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(sys, "stderr", stderr)
    output = tmp_path / "writer.json"
    args = ["build", str(WRITER), "--contrast", str(CONTRAST), "--no-syntax", "-o", str(output)]
    assert main(args) == 0
    return stderr.getvalue()


def test_the_cli_shows_progress_only_on_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TERM", raising=False)
    shown = _build(tmp_path, Terminal(), monkeypatch)
    on_terminal = capsys.readouterr().out
    assert "\rmeasuring" in shown and "chunks" in shown
    assert "\rcalibrating" in shown and "\rlearning the contrast" in shown
    # The line is erased before anything else is written (the thin-reference warnings).
    progress, erased, after = shown.rpartition("\r\x1b[K")
    assert erased and "\r" not in after and "Thin reference" in after
    assert "Thin reference" not in progress
    piped = _build(tmp_path, io.StringIO(), monkeypatch)
    assert "\r" not in piped and "measuring" not in piped
    # What goes to stdout is the same either way.
    assert capsys.readouterr().out == on_terminal


def test_a_dumb_terminal_gets_no_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    assert "\r" not in _build(tmp_path, Terminal(), monkeypatch)


def test_json_and_quiet_scores_show_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TERM", raising=False)
    reference = tmp_path / "writer.json"
    assert main(["build", str(WRITER), "--no-syntax", "-o", str(reference)]) == 0
    for flag in ("--json", "-q"):
        stderr = Terminal()
        monkeypatch.setattr(sys, "stderr", stderr)
        assert main(["score", flag, str(DRAFT), str(reference)]) == 0
        assert "\r" not in stderr.getvalue()
    stderr = Terminal()
    monkeypatch.setattr(sys, "stderr", stderr)
    assert main(["score", str(DRAFT), str(reference)]) == 0
    assert "\rmeasuring" in stderr.getvalue()
    capsys.readouterr()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_line_shows_rate_and_time_left_once_it_settles() -> None:
    clock, stream = Clock(), io.StringIO()
    line = StatusLine(stream, clock=clock, columns=200)
    line(Progress(Phase.MEASURE, 0, 100, 0))
    assert stream.getvalue().endswith("\rmeasuring 0/100 chunks\x1b[K")
    clock.now = 0.5
    line(Progress(Phase.MEASURE, 10, 100, 5_000))
    assert "words/s" not in stream.getvalue()  # too early to tell
    clock.now = 2.0
    line(Progress(Phase.MEASURE, 40, 100, 20_000))
    assert stream.getvalue().endswith(
        "\rmeasuring 40/100 chunks, 10,000 words/s, about 3s left\x1b[K"
    )
    clock.now = 2.05
    line(Progress(Phase.MEASURE, 41, 100, 20_500))  # redrawn at most every 0.1s
    assert stream.getvalue().endswith("about 3s left\x1b[K")
    clock.now = 5.0
    line(Progress(Phase.MEASURE, 100, 100, 50_000))
    assert stream.getvalue().endswith("\rmeasuring 100/100 chunks, 10,000 words/s\x1b[K")
    line(Progress(Phase.DONE))
    assert stream.getvalue().endswith("\r\x1b[K")
    assert status.TIP not in stream.getvalue()


def test_a_slow_parse_suggests_leaving_out_syntax_once() -> None:
    clock, stream = Clock(), io.StringIO()
    line = StatusLine(stream, clock=clock, columns=200)
    line(Progress(Phase.MEASURE, 0, 1_000, 0, parsing=True))
    for done in range(1, 30):
        clock.now = done * 0.5  # 2 chunks a second: 500 s in all
        line(Progress(Phase.MEASURE, done, 1_000, done * 500, parsing=True))
    assert stream.getvalue().count(status.TIP) == 1
    assert f"\r\x1b[K{status.TIP}\n" in stream.getvalue()
    # Without the parser there is nothing to suggest.
    clock, stream = Clock(), io.StringIO()
    line = StatusLine(stream, clock=clock, columns=200)
    line(Progress(Phase.MEASURE, 0, 1_000, 0))
    clock.now = 10.0
    line(Progress(Phase.MEASURE, 1, 1_000, 500))
    assert status.TIP not in stream.getvalue()


def test_start_up_time_does_not_count_toward_the_estimate() -> None:
    # Workers take 2 s to start, then measure 20 chunks a second: 394 chunks take about 22 s.
    clock, stream = Clock(), io.StringIO()
    line = StatusLine(stream, clock=clock, columns=200)
    line(Progress(Phase.MEASURE, 0, 394, 0, parsing=True))
    for done in range(1, 395):
        clock.now = 2.0 + done / 20
        line(Progress(Phase.MEASURE, done, 394, done * 500, parsing=True))
    assert status.TIP not in stream.getvalue()
    assert "about 1m" not in stream.getvalue()


def test_the_line_fits_the_terminal() -> None:
    stream = io.StringIO()
    line = StatusLine(stream, clock=Clock(), columns=20)
    line(Progress(Phase.MEASURE_CONTRAST, 0, 123_456, 0))
    assert stream.getvalue() == "\r" + "measuring the contra"[:19] + "\x1b[K"


def test_durations() -> None:
    assert status.duration(8.4) == "8s"
    assert status.duration(65) == "1m 05s"
    assert status.duration(3_720) == "1h 02m"


def test_measuring_reports_every_chunk() -> None:
    events: list[Progress] = []
    chunks = window(load_chunks([str(WRITER)]), 200)
    with Measurer(progress=events.append) as measurer:
        report = build_reference(chunks, measurer=measurer)
    measured = [event for event in events if event.phase is Phase.MEASURE]
    assert [event.done for event in measured] == list(range(report["chunk_count"] + 1))
    assert measured[-1].words == report["word_count"]
    assert Progress(Phase.CALIBRATE) in events


# Worker processes.


def test_jobs_must_be_a_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for jobs in (-1, 1.5, "2", True):
        with pytest.raises(sp.StyleProfileError, match="jobs") as error:
            sp.build(WRITER, jobs=jobs)  # pyright: ignore[reportArgumentType]
        assert error.value.code == "invalid_setting"
        assert error.value.setting == "jobs"
    output = str(tmp_path / "x.json")
    assert main(["build", str(WRITER), "--jobs", "-1", "-o", output]) == 1
    assert "--jobs must be 0" in capsys.readouterr().err


def test_memory_jobs_follow_physical_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    gigabyte = 1024**3
    for memory, expected in ((2 * gigabyte, 1), (4 * gigabyte, 3), (8 * gigabyte, 6)):
        monkeypatch.setattr(
            measure.os,
            "sysconf",
            lambda name, memory=memory: 4096 if name == "SC_PAGE_SIZE" else memory // 4096,
            raising=False,
        )
        assert measure.memory_jobs() == expected

    def unavailable(name: str) -> int:
        raise ValueError(name)

    monkeypatch.setattr(measure.os, "sysconf", unavailable)
    assert measure.memory_jobs() is None


def test_automatic_jobs_wait_for_a_large_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(measure, "workers_can_start", lambda: True)
    monkeypatch.setattr(measure, "cpus", lambda: 16)
    monkeypatch.setattr(measure, "memory_jobs", lambda: None)
    automatic = Measurer(jobs=0)
    assert automatic.workers(measure.PARALLEL_WORDS - 1) == 1
    assert automatic.workers(measure.PARALLEL_WORDS) == measure.AUTO_JOBS
    monkeypatch.setattr(measure, "cpus", lambda: 2)
    assert automatic.workers(10**7) == 2
    # Memory caps them too: a quarter of an 8 GB machine holds 6 workers, of 2 GB one.
    monkeypatch.setattr(measure, "cpus", lambda: 16)
    monkeypatch.setattr(measure, "memory_jobs", lambda: 3)
    assert automatic.workers(10**7) == 3
    assert Measurer(jobs=4).workers(10) == 3  # explicit jobs respect memory too
    assert Measurer(jobs=1).workers(10**7) == 1
    assert Measurer(jobs=3).workers(10) == 3  # asked for, so used however little to parse
    # A script whose workers would run it again gets one process, even when it asks for more.
    monkeypatch.setattr(measure, "workers_can_start", lambda: False)
    assert automatic.workers(10**7) == 1
    assert Measurer(jobs=3).workers(10**7) == 1


def _can_start(tmp_path: Path, source: str) -> bool:
    """What ``workers_can_start`` says when called from a script with ``source``."""
    script = tmp_path / "script.py"
    script.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True, cwd=tmp_path
    )
    return done.stdout.strip() == "True"


CHECK = "from styleprofile.measure import workers_can_start\n"


def test_workers_start_only_from_a_guarded_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A call under the guard, directly or through a function the guard calls.
    assert _can_start(
        tmp_path, CHECK + "if __name__ == '__main__':\n    print(workers_can_start())\n"
    )
    assert _can_start(
        tmp_path,
        CHECK
        + "def main():\n    print(workers_can_start())\n\nif '__main__' == __name__:\n    main()\n",
    )
    # A script with a guard that calls at the top level anyway, or has none.
    assert not _can_start(
        tmp_path,
        CHECK + "print(workers_can_start())\nif __name__ == '__main__':\n    pass\n",
    )
    assert not _can_start(tmp_path, CHECK + "print(workers_can_start())\n")
    # The else of the guard is not under it.
    assert not _can_start(
        tmp_path,
        CHECK + "if __name__ != '__main__':\n    pass\nelse:\n    print(workers_can_start())\n",
    )
    # An interactive session or notebook has no script.
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    assert workers_can_start()


@pytest.mark.skipif(not HAS_SPACY, reason="needs spaCy")
def test_worker_processes_give_the_same_profile() -> None:
    parser = _parser()
    chunks = window(load_chunks([str(WRITER)]), 200)
    contrast = window(load_chunks([str(CONTRAST)]), 200)
    reports = []
    for jobs in (1, 2):
        with Measurer(jobs=jobs) as measurer:
            reports.append(
                build_reference(chunks, parser=parser, contrast=contrast, measurer=measurer)
            )
            assert (measurer._pool is not None) is (jobs == 2)  # pyright: ignore[reportPrivateUsage]
    assert json.dumps(reports[0]) == json.dumps(reports[1])


class _BrokenPool:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def submit(self, *args: Any, **kwargs: Any) -> Any:
        raise BrokenProcessPool("a worker died while starting")

    def map(self, *args: Any, **kwargs: Any) -> Any:
        raise BrokenProcessPool("a worker died while starting")

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.mark.skipif(not HAS_SPACY, reason="needs spaCy")
def test_workers_that_fail_leave_the_parse_to_this_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _parser()
    chunks = window(load_chunks([str(WRITER)]), 200)
    expected = build_reference(chunks, parser=parser)
    monkeypatch.setattr(measure, "start_pool", _BrokenPool)
    with Measurer(jobs=2) as measurer:
        assert json.dumps(build_reference(chunks, parser=parser, measurer=measurer)) == (
            json.dumps(expected)
        )
        assert measurer.workers(10**7) == 1  # and no more attempts this run


@pytest.mark.skipif(not HAS_SPACY, reason="needs spaCy")
def test_the_parser_loads_only_what_the_metrics_read() -> None:
    parser = _parser()
    names = set(parser.nlp.component_names)
    assert {"tok2vec", "tagger", "parser", "attribute_ruler"} <= names
    assert not names & {"ner", "lemmatizer", "senter"}


# Streaming totals and stable pieces.


def test_a_reference_sums_pattern_counts_without_keeping_each_chunks() -> None:
    chunks = window(load_chunks([str(WRITER)]), 200)
    kept = measure.measure(chunks, None, 1)
    streamed = measure.measure(chunks, None, 1, keep_distributions=False)
    assert streamed.distributions == [] and len(kept.distributions) == len(chunks)
    totals: dict[str, Counter[str]] = {}
    for counts in kept.distributions:
        for name, entry in counts.items():
            totals.setdefault(name, Counter()).update(entry)
    assert streamed.totals == kept.totals == totals
    # Same keys in the same order, so ties among the most common break the same way.
    assert {name: list(entry) for name, entry in streamed.totals.items()} == {
        name: list(entry) for name, entry in totals.items()
    }


def test_the_api_reports_the_same_numbers_with_or_without_jobs_and_cache() -> None:
    settings = sp.Settings(syntax=False)
    first = sp.build(WRITER, settings, contrast=CONTRAST, jobs=1, cache=False)
    second = sp.build(WRITER, settings, contrast=CONTRAST, jobs=0, cache=True)
    assert dumps_report(first.report) == dumps_report(second.report)


@pytest.mark.skipif(not HAS_SPACY, reason="needs spaCy")
def test_a_lazy_parser_loads_its_model_only_to_parse() -> None:
    eager = _parser()
    from styleprofile.syntax import load_parser

    lazy = load_parser(lazy=True)
    assert not lazy.loaded
    assert lazy.used() == eager.used()  # the same versions, read without loading
    [(metrics, _)] = lazy.parse(["A sentence to parse, which loads the model."])
    assert lazy.loaded and metrics


@pytest.mark.skipif(not HAS_SPACY, reason="needs spaCy")
def test_a_piece_whose_prose_repeats_is_found_as_main_found_it() -> None:
    # Two pieces with the same prose in one window: the second is searched for after the
    # first, so it gets the second occurrence's span (and its own parse context).
    parser = _parser()
    sentence = "The dog that the cat chased ran away quickly across the wide field today."
    other = "Nothing else here reads like it at all, whatever the weather does next week."
    text = " ".join([sentence, other, sentence, other])
    part_length = 10
    offsets = measure.piece_offsets(text, [(part_length, sentence), (part_length, sentence)])
    assert offsets == [0, text.index(sentence, 1)]
    # A piece of another length starts its own search.
    assert measure.piece_offsets(text, [(10, sentence), (20, sentence)]) == [0, 0]
    [(_, _, pieces, _)] = measure.parse(
        parser,
        [(text, [(offset, offset + len(sentence)) for offset in offsets if offset is not None])],
    )
    assert all(piece is not None for piece in pieces)


def test_a_model_that_fails_to_load_leaves_syntax_out_under_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from styleprofile import syntax

    def broken(model: str) -> Any:
        raise syntax.SyntaxUnavailableError(f"{model} is broken")

    monkeypatch.setattr(syntax, "_load", broken)
    runtime._default_parser.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        profile = sp.build(WRITER, jobs=1, cache=False)
        assert profile.report["settings"]["syntax_used"] is None
        assert sp.NoteCode.NO_SYNTAX in [note.code for note in profile.notes]
        with pytest.raises(sp.SyntaxUnavailableError):
            sp.build(WRITER, sp.Settings(syntax=True), jobs=1, cache=False)
    finally:
        runtime._default_parser.cache_clear()  # pyright: ignore[reportPrivateUsage]
