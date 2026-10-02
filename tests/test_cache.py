"""The measurement cache (``styleprofile.cache``): what its keys depend on, that a report built
from it is byte-for-byte the one measuring gives, that only new text is measured, and that it
never fails a run."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile import cache as caching
from styleprofile import measure, metrics
from styleprofile.cache import MeasurementCache, fingerprint
from styleprofile.cli import main
from styleprofile.measure import Measurer
from styleprofile.profile import build_reference, dumps_report, load_chunks, score, window

ROOT = Path(__file__).resolve().parent.parent
WRITER, CONTRAST, DRAFT = (
    ROOT / "examples/writer",
    ROOT / "examples/llm-drafts",
    ROOT / "examples/draft.md",
)
HAS_SPACY = importlib.util.find_spec("spacy") is not None


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache of this test's own, empty at the start."""
    home = tmp_path / "cache-home"
    monkeypatch.setenv("XDG_CACHE_HOME", str(home))
    return home / "styleprofile" / caching.FILENAME


class Counting:
    """Counts the chunks measured rather than read from the cache."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.texts: list[str] = []
        original = measure._distributions  # pyright: ignore[reportPrivateUsage]

        def counting(text: str, *rest: Any) -> Any:
            self.texts.append(text)
            return original(text, *rest)

        monkeypatch.setattr(measure, "_distributions", counting)


def _parser() -> Any:
    pytest.importorskip("spacy")
    from styleprofile.syntax import SyntaxUnavailableError, load_parser

    try:
        return load_parser()
    except SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")


# The key.


def test_changing_a_metric_definition_changes_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = fingerprint(None)
    assert fingerprint(None) == before  # deterministic
    first, *rest = metrics.METRICS
    monkeypatch.setattr(metrics, "METRICS", (dataclasses.replace(first, label="Other"), *rest))
    assert fingerprint(None) != before


def test_the_fingerprint_covers_the_code_versions_and_parser() -> None:
    base = fingerprint(None)
    # The source of the measuring modules: another set of modules is other code.
    assert fingerprint(None, modules=("styleprofile.metrics",)) != base
    assert fingerprint(None, version="9.9.9") != base
    syntax = {"model": "en_core_web_sm", "model_version": "3.8.0", "spacy_version": "3.8.7"}
    with_syntax = fingerprint(syntax)
    assert with_syntax != base
    assert fingerprint({**syntax, "model_version": "3.8.1"}) != with_syntax
    assert fingerprint({**syntax, "spacy_version": "3.9.0"}) != with_syntax


def test_keys_separate_texts_pieces_and_fingerprints() -> None:
    one, two = fingerprint(None), fingerprint(None, version="other")
    assert caching.key(one, "text") == caching.key(one, "text")
    assert caching.key(one, "text") != caching.key(two, "text")
    assert caching.key(one, "text") != caching.key(one, "text", "")
    assert caching.key(one, "ab", "c") != caching.key(one, "a", "bc")


# Hits.


def test_a_cache_hit_gives_a_byte_identical_report(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = sp.Settings(syntax=False)
    cold = sp.build(WRITER, settings, contrast=CONTRAST)
    assert cache_home.exists()
    counting = Counting(monkeypatch)
    spans: list[int] = []
    measure_spans = measure.measure_spans

    def counting_spans(parts: Any, whole: Any) -> Any:
        spans.append(len(parts))
        return measure_spans(parts, whole)

    monkeypatch.setattr(measure, "measure_spans", counting_spans)
    warm = sp.build(WRITER, settings, contrast=CONTRAST)
    assert counting.texts == []  # every chunk and piece came from the cache
    assert spans == []  # and every document read in parts for drift calibration
    assert "drift" in (warm.report.get("calibration") or {})
    cold.save(tmp_path / "cold.json")
    warm.save(tmp_path / "warm.json")
    assert (tmp_path / "cold.json").read_bytes() == (tmp_path / "warm.json").read_bytes()
    uncached = sp.build(WRITER, settings, contrast=CONTRAST, cache=False)
    assert dumps_report(uncached.report) == dumps_report(warm.report)
    # Scores too, whose chunks keep their own pattern counts.
    first = warm.score(DRAFT, cache=True)
    measured = len(counting.texts)
    assert measured
    second = warm.score(DRAFT, cache=True)
    assert len(counting.texts) == measured
    assert dumps_report(first.report) == dumps_report(second.report)


def test_the_cache_reproduces_every_float_exactly(tmp_path: Path) -> None:
    # Below the report's rounding: the low-level functions, unrounded, from a cold and a warm
    # cache and from none.
    chunks = window(load_chunks([str(WRITER)]), 200)
    contrast = window(load_chunks([str(CONTRAST)]), 200)
    path = tmp_path / "cache.sqlite3"
    reports = []
    for store in (None, MeasurementCache(path), MeasurementCache(path)):
        with Measurer(cache=store) as measurer:
            reports.append(build_reference(chunks, contrast=contrast, measurer=measurer))
    assert json.dumps(reports[0]) == json.dumps(reports[1]) == json.dumps(reports[2])
    scored = []
    for store in (None, MeasurementCache(path), MeasurementCache(path)):
        with Measurer(cache=store) as measurer:
            scored.append(
                score(window(load_chunks([str(DRAFT)]), 200), reports[0], measurer=measurer)
            )
    assert json.dumps(scored[0]) == json.dumps(scored[1]) == json.dumps(scored[2])


@pytest.mark.skipif(not HAS_SPACY, reason="needs spaCy")
def test_a_cache_hit_with_syntax_is_identical_and_parses_nothing(tmp_path: Path) -> None:
    parser = _parser()
    chunks = window(load_chunks([str(WRITER)]), 200)
    path = tmp_path / "cache.sqlite3"
    parsed: list[str] = []
    pipe = parser.nlp.pipe

    def counting(texts: Any, **kwargs: Any) -> Any:
        texts = list(texts)
        parsed.extend(texts)
        return pipe(texts, **kwargs)

    reports = []
    parser.nlp.pipe = counting
    try:
        for store in (None, MeasurementCache(path), MeasurementCache(path)):
            parsed.clear()
            with Measurer(cache=store) as measurer:
                reports.append(build_reference(chunks, parser=parser, measurer=measurer))
    finally:
        parser.nlp.pipe = pipe
    assert parsed == []  # the last build read every chunk and piece from the cache
    assert json.dumps(reports[0]) == json.dumps(reports[1]) == json.dumps(reports[2])


def test_adding_a_document_measures_only_the_new_one(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = tmp_path / "writer"
    shutil.copytree(WRITER, corpus)
    settings = sp.Settings(syntax=False)
    sp.build(corpus, settings)
    new = corpus / "zz-new.md"
    new.write_text((ROOT / "examples/draft.md").read_text(encoding="utf-8"), encoding="utf-8")
    counting = Counting(monkeypatch)
    grown = sp.build(corpus, settings)
    new_windows = window(load_chunks([str(new)]), settings.window_words)
    assert len(counting.texts) == len(new_windows)
    # And the profile is the one a build without the cache gives.
    cold = sp.build(corpus, settings, cache=False)
    assert dumps_report(grown.report) == dumps_report(cold.report)


def test_no_cache_neither_reads_nor_writes(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = sp.Settings(syntax=False)
    sp.build(WRITER, settings, cache=False)
    assert not cache_home.exists()
    sp.build(WRITER, settings)
    size = cache_home.stat().st_size
    counting = Counting(monkeypatch)
    sp.build(WRITER, settings, cache=False)
    assert counting.texts  # measured again
    assert cache_home.stat().st_size == size


# Never failing a run.


def test_an_unusable_cache_is_left_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = sp.Settings(syntax=False)
    expected = dumps_report(sp.build(WRITER, settings, cache=False).report)
    # The cache directory's place is taken by a file.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocked))
    assert dumps_report(sp.build(WRITER, settings).report) == expected
    # The cache file is not a database.
    home = tmp_path / "garbage"
    (home / "styleprofile").mkdir(parents=True)
    (home / "styleprofile" / caching.FILENAME).write_bytes(b"\x00garbage" * 1000)
    monkeypatch.setenv("XDG_CACHE_HOME", str(home))
    assert dumps_report(sp.build(WRITER, settings).report) == expected


def test_an_unreadable_entry_is_measured_again(tmp_path: Path) -> None:
    chunks = window(load_chunks([str(WRITER)]), 200)
    path = tmp_path / "cache.sqlite3"
    with Measurer(cache=MeasurementCache(path)) as measurer:
        expected = build_reference(chunks, measurer=measurer)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE entries SET value = x'00'")
    db.close()
    with Measurer(cache=MeasurementCache(path)) as measurer:
        assert json.dumps(build_reference(chunks, measurer=measurer)) == json.dumps(expected)


def test_a_relative_xdg_cache_home_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caching.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    assert caching.cache_dir() == Path.home() / ".cache" / "styleprofile"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert caching.cache_dir() == Path.home() / ".cache" / "styleprofile"


def test_the_cache_file_is_private(cache_home: Path) -> None:
    sp.build(WRITER, sp.Settings(syntax=False))
    assert cache_home.stat().st_mode & 0o077 == 0
    assert cache_home.parent.stat().st_mode & 0o077 == 0


# Size and sharing.


def test_the_cache_drops_the_least_recently_used_entries(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"

    def value(number: int) -> dict[str, Any]:
        # About 10 KB once compressed: random text does not compress.
        return {"n": number, "blob": secrets.token_hex(10_000)}

    store = MeasurementCache(path, clock=lambda: 3_600.0)
    for number in range(40):
        store.put(f"old{number}".encode(), 1, value(number))
    store.close()
    store = MeasurementCache(path, clock=lambda: 3 * 3_600.0)
    assert store.fetch(b"old39") is not None  # used again, so now recent
    store.close()
    small = MeasurementCache(path, max_bytes=200_000, clock=lambda: 5 * 3_600.0)
    for number in range(10):
        small.put(f"new{number}".encode(), 1, value(number))
    small.close()
    store = MeasurementCache(path)
    old = [number for number in range(39) if store.fetch(f"old{number}".encode())]
    assert store.fetch(b"old39") is not None
    assert all(store.fetch(f"new{number}".encode()) for number in range(10))
    store.close()
    assert len(old) < 20  # the least recently used went first
    # The pages it freed went back to the file system: none are left free in the file.
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA freelist_count").fetchall() == [(0,)]
    db.close()
    assert path.stat().st_size < 1_000_000


WRITER_SCRIPT = """
import sys
from pathlib import Path
from styleprofile.cache import MeasurementCache
store = MeasurementCache(Path(sys.argv[1]))
for number in range(300):
    store.put(f"{sys.argv[2]}{number}".encode(), number, {"n": number})
store.close()
"""


def test_processes_can_share_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    processes = [
        subprocess.Popen([sys.executable, "-c", WRITER_SCRIPT, str(path), name])
        for name in ("a", "b", "c")
    ]
    assert [process.wait(timeout=60) for process in processes] == [0, 0, 0]
    store = MeasurementCache(path)
    for name in ("a", "b", "c"):
        found = store.fetch(f"{name}299".encode())
        assert found is not None and caching.decode(found[1]) == {"n": 299}
    store.close()
    assert caching.describe(path)[2] == 900


def test_the_cache_command_shows_and_clears_it(
    cache_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cache"]) == 0
    assert "empty" in capsys.readouterr().out
    sp.build(WRITER, sp.Settings(syntax=False))
    assert main(["cache"]) == 0
    out = capsys.readouterr().out
    assert str(cache_home) in out and "entries" in out
    assert main(["cache", "--clear"]) == 0
    assert "deleted the measurement cache" in capsys.readouterr().out
    assert not cache_home.exists()
    assert main(["cache", "--clear"]) == 0
    assert "already empty" in capsys.readouterr().out


def test_the_cli_can_skip_the_cache(
    cache_home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "writer.json"
    assert main(["build", str(WRITER), "--no-syntax", "--no-cache", "-o", str(output)]) == 0
    assert not cache_home.exists()
    assert main(["build", str(WRITER), "--no-syntax", "-o", str(output)]) == 0
    assert cache_home.exists()
    capsys.readouterr()


# Failures the run survives, and says so.


def _notes(profile: sp.Profile) -> list[sp.Note]:
    return [note for note in profile.notes if note.code == sp.NoteCode.CACHE_UNAVAILABLE]


def test_a_damaged_cache_in_a_read_only_folder_is_left_out(
    cache_home: Path, tmp_path: Path
) -> None:
    settings = sp.Settings(syntax=False)
    expected = dumps_report(sp.build(WRITER, settings, cache=False).report)
    cache_home.parent.mkdir(parents=True)
    cache_home.write_bytes(b"\x00garbage" * 1000)  # not a database, and cannot be replaced
    cache_home.parent.chmod(0o500)
    try:
        profile = sp.build(WRITER, settings)
    finally:
        cache_home.parent.chmod(0o700)
    assert dumps_report(profile.report) == expected
    [note] = _notes(profile)
    assert str(cache_home) in note.message


def test_a_writer_that_fails_never_holds_up_the_run(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = sp.Settings(syntax=False)
    expected = dumps_report(sp.build(WRITER, settings, contrast=CONTRAST, cache=False).report)
    opened = MeasurementCache._open  # pyright: ignore[reportPrivateUsage]

    def failing(self: MeasurementCache) -> Any:
        if threading.current_thread() is not threading.main_thread():
            raise PermissionError(13, "Permission denied", str(self.path))
        return opened(self)

    monkeypatch.setattr(MeasurementCache, "_open", failing)
    monkeypatch.setattr(caching, "WRITE_BATCH", 1)  # a batch per entry: the queue fills up
    profile = sp.build(WRITER, settings, contrast=CONTRAST)
    assert dumps_report(profile.report) == expected
    [note] = _notes(profile)
    assert "permission denied" in note.message


def test_a_writer_that_stops_never_holds_up_the_run(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A writer thread that ends without taking any batch: handing it one must not wait.
    monkeypatch.setattr(MeasurementCache, "_write", lambda self: None)
    monkeypatch.setattr(caching, "WRITE_BATCH", 1)
    done: list[sp.Profile] = []
    runner = threading.Thread(
        target=lambda: done.append(sp.build(WRITER, sp.Settings(syntax=False)))
    )
    runner.start()
    runner.join(60)
    assert done, "the build waited on a stopped writer"
    [note] = _notes(done[0])
    assert "writer stopped" in note.message


def test_the_environment_can_turn_the_cache_off(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("STYLEPROFILE_NO_CACHE", "1")
    sp.build(WRITER, sp.Settings(syntax=False))
    assert not cache_home.exists()
    assert main(["cache"]) == 0
    assert "off: STYLEPROFILE_NO_CACHE is set" in capsys.readouterr().out
    monkeypatch.setenv("STYLEPROFILE_NO_CACHE", "0")
    sp.build(WRITER, sp.Settings(syntax=False))
    assert cache_home.exists()


def test_scoring_leaves_nothing_in_the_cache_unless_asked(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    entries = caching.describe(cache_home)[2]
    profile.score(DRAFT)
    assert caching.describe(cache_home)[2] == entries
    profile.score(DRAFT, cache=True)
    assert caching.describe(cache_home)[2] > entries


def test_the_cache_command_says_when_the_cache_is_unavailable(
    cache_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_home.parent.mkdir(parents=True)
    cache_home.parent.chmod(0o500)
    try:
        assert main(["cache"]) == 0
    finally:
        cache_home.parent.chmod(0o700)
    out = capsys.readouterr().out
    assert "unavailable:" in out and "not writable" in out


def test_a_stuck_writer_is_given_up_on_within_seconds(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A writer that takes a batch and then never finishes it: the run waits only as long as
    # the writer makes no progress (HAND_DEADLINE_S), then goes on without the cache.
    monkeypatch.setattr(caching, "HAND_DEADLINE_S", 0.5)
    monkeypatch.setattr(caching, "HAND_TIMEOUT_S", 0.1)
    written = MeasurementCache._write  # pyright: ignore[reportPrivateUsage]
    release = threading.Event()

    def stuck(self: MeasurementCache) -> None:
        get = self._writes.get  # pyright: ignore[reportPrivateUsage]

        def slow_get(*args: Any, **kwargs: Any) -> Any:
            item = get(*args, **kwargs)
            if item is not None:
                release.wait(60)
            return item

        self._writes.get = slow_get  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
        written(self)

    monkeypatch.setattr(MeasurementCache, "_write", stuck)
    settings = sp.Settings(syntax=False)
    expected = dumps_report(sp.build(WRITER, settings, contrast=CONTRAST, cache=False).report)
    started = time.monotonic()
    try:
        profile = sp.build(WRITER, settings, contrast=CONTRAST)
    finally:
        release.set()
    assert time.monotonic() - started < 20
    assert dumps_report(profile.report) == expected
    [note] = _notes(profile)
    assert "stopped responding" in note.message
