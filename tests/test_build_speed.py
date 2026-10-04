"""Build options, worker limits and deterministic surface workers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import styleprofile as sp
from styleprofile import measure
from styleprofile.cli import main
from styleprofile.core import NoteCode
from styleprofile.corpus.duplicates import drop_duplicates
from styleprofile.corpus.reading import load_chunks
from styleprofile.corpus.windows import window
from styleprofile.measure import Measurer
from styleprofile.reference import build_reference
from styleprofile.reports import dumps_report

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples" / "writer"
CONTRAST = ROOT / "examples" / "llm-drafts"
DRAFT = ROOT / "examples" / "draft.md"


def test_paragraph_calibration_is_opt_in_and_other_profile_bytes_stay_identical() -> None:
    plain = sp.build(WRITER, syntax=False, contrast=CONTRAST)
    asked = sp.build(WRITER, syntax=False, contrast=CONTRAST, passages=True)
    assert plain.report["version"] == asked.report["version"] == 8
    assert "drift" not in plain.report["calibration"]
    assert asked.report["calibration"].pop("drift")["documents"] == 7
    assert json.dumps(plain.report) == json.dumps(asked.report)
    strings = [path.read_text() for path in sorted(WRITER.glob("*.md"))]
    assert "drift" in sp.build_texts(strings, syntax=False, passages=True).report["calibration"]


@pytest.mark.parametrize("json_output", [False, True])
def test_score_without_paragraph_calibration_names_rebuild_and_scores_normally(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    path = tmp_path / "writer.json"
    sp.build(WRITER, syntax=False, contrast=CONTRAST).save(path)
    args = ["score", str(DRAFT), str(path), "--by-paragraph"]
    if json_output:
        args.append("--json")
    assert main(args) == 0
    output = capsys.readouterr()
    assert "Paragraph checks need" in output.err
    assert "styleprofile build" in output.err and "--by-paragraph" in output.err
    assert "--contrast" in output.err
    if json_output:
        report = json.loads(output.out)
        assert report["reference"]["delta_mean"] is not None
        assert not report.get("passages")
    else:
        assert "Delta" in output.out and "Where it drifts" not in output.out


def test_surface_workers_preserve_profile_bytes_and_stop() -> None:
    chunks = window(load_chunks([str(WRITER)]), 200)
    contrast = window(load_chunks([str(CONTRAST)]), 200)
    expected = build_reference(chunks, contrast=contrast, passages=True)
    with Measurer(jobs=2) as measurer:
        actual = build_reference(chunks, contrast=contrast, passages=True, measurer=measurer)
        assert measurer._pool is not None  # pyright: ignore[reportPrivateUsage]
        assert json.dumps(actual) == json.dumps(expected)
    assert measurer._pool is None  # pyright: ignore[reportPrivateUsage]


def test_surface_worker_failure_falls_back_without_changing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures.process import BrokenProcessPool

    def broken(*args: object) -> object:
        raise BrokenProcessPool("cannot start")

    chunks = window(load_chunks([str(WRITER)]), 200)
    expected = build_reference(chunks)
    monkeypatch.setattr(measure, "start_pool", broken)
    with Measurer(jobs=2) as measurer:
        assert json.dumps(build_reference(chunks, measurer=measurer)) == json.dumps(expected)
        assert measurer.workers(10**7, surface=True) == 1


@pytest.mark.parametrize(
    "quota,expected",
    [("max 100000", 8), ("150000 100000", 2), ("100000 100000", 1), ("bad", 8), ("100 0", 8)],
)
def test_cgroup_cpu_quota_on_older_python(
    quota: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(measure.sys, "version_info", (3, 12))
    monkeypatch.setattr(measure.os, "cpu_count", lambda: 8)
    monkeypatch.delattr(measure.os, "process_cpu_count", raising=False)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: quota)
    assert measure.cpus() == expected


def test_explicit_workers_are_capped_and_noted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(measure, "memory_jobs", lambda: 1)
    profile = sp.build(WRITER, syntax=False, jobs=8)
    assert any(
        note.code == NoteCode.WORKERS_LIMITED and "8 to 1" in note.message for note in profile.notes
    )


def test_duplicate_keys_are_compact_normalized_digests() -> None:
    chunks = load_chunks([str(WRITER)])
    seen: dict[bytes, str] = {}
    kept, _ = drop_duplicates(chunks, seen)
    assert kept == chunks and all(len(key) == 64 for key in seen)
    repeated, note = drop_duplicates(chunks, seen)
    assert repeated == [] and note is not None


def test_group_fallback_reads_jsonl_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = tmp_path / "writer.jsonl"
    contrast = tmp_path / "contrast.jsonl"
    writer.write_text(
        "\n".join(
            json.dumps({"id": i, "thread": i, "text": path.read_text()})
            for i, path in enumerate(sorted(WRITER.glob("*.md")))
        ),
        encoding="utf-8",
    )
    contrast.write_text(
        "\n".join(
            json.dumps({"id": i, "text": path.read_text()})
            for i, path in enumerate(sorted(CONTRAST.glob("*.md")))
        ),
        encoding="utf-8",
    )
    counts: dict[Path, int] = {}
    read = Path.read_bytes

    def counted(path: Path) -> bytes:
        counts[path] = counts.get(path, 0) + 1
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    profile = sp.build(writer, contrast=contrast, group_field="thread", syntax=False)
    assert profile.report.get("contrast")
    assert counts[writer] == counts[contrast] == 1
    assert any(note.code == NoteCode.MISSING_GROUP for note in profile.notes)


def test_surface_failure_after_completed_batch_keeps_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    from typing import Any

    class Pool:
        calls = 0

        def submit(self, function: Any, items: Any) -> Future[Any]:
            self.calls += 1
            future: Future[Any] = Future()
            if self.calls == 2:
                future.set_exception(BrokenProcessPool("second batch failed"))
            else:
                future.set_result(function(items))
            return future

        def shutdown(self, **kwargs: object) -> None:
            pass

    chunks = window(load_chunks([str(WRITER)]), 75)
    assert len(chunks) > measure.TASK_TEXTS
    expected = build_reference(chunks)
    monkeypatch.setattr(measure, "start_pool", lambda *args: Pool())
    with Measurer(jobs=2) as measurer:
        actual = build_reference(chunks, measurer=measurer)
    assert json.dumps(actual) == json.dumps(expected)


def test_surface_failure_cancels_queued_syntax_and_retries_remaining_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import Future
    from typing import Any

    class Pool:
        calls = 0

        def __init__(self) -> None:
            self.pending: Future[Any] = Future()

        def submit(self, function: Any, items: Any) -> Future[Any]:
            self.calls += 1
            future: Future[Any] = Future()
            if self.calls == 2:
                future.set_exception(OSError("surface batch failed"))
            else:
                future.set_result(function(items))
            return future

        def map(self, function: Any, tasks: Any) -> Any:
            yield [({}, {}, [], None)] * measure.TASK_TEXTS
            yield self.pending.result()

        def shutdown(self, **kwargs: object) -> None:
            self.pending.cancel()

    pool = Pool()
    monkeypatch.setattr(measure, "start_pool", lambda *args: pool)
    monkeypatch.setattr(measure, "workers_can_start", lambda: True)
    monkeypatch.setattr(measure, "memory_jobs", lambda: 4)
    retried: list[str] = []

    def parse(parser: Any, items: Any, keep: Any) -> Any:
        for text, _ in items:
            retried.append(text)
            yield ({}, {}, [], None)

    monkeypatch.setattr(measure, "parse", parse)
    from types import SimpleNamespace

    parser = SimpleNamespace(model="unused")
    texts = [f"Sentence number {index}." for index in range(65)]
    items = [(text, measure.prose(text), []) for text in texts]
    with Measurer(jobs=2) as measurer:
        surface = measurer.surface(items, parser)  # type: ignore[arg-type]
        syntax = measurer.parse(parser, [(text, []) for text in texts])  # type: ignore[arg-type]
        for _ in texts:
            next(surface)
            next(syntax)
    assert retried == texts[measure.TASK_TEXTS :]


def test_one_shot_paragraph_scoring_builds_the_requested_calibration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = sp.build(WRITER, contrast=CONTRAST, syntax=False, passages=True).score(
        DRAFT, passages=True
    )
    assert (
        main(
            [
                "score",
                str(DRAFT),
                "--against",
                str(WRITER),
                "--contrast",
                str(CONTRAST),
                "--by-paragraph",
                "--no-syntax",
                "--no-cache",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["passages"]
    encoded_expected = json.loads(dumps_report(expected.report))
    assert report["reference"]["delta_mean"] == encoded_expected["reference"]["delta_mean"]
    assert "Paragraph checks need" not in output.err
    assert "To reuse:" in output.err and "--by-paragraph" in output.err
