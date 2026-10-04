"""Snapshot tests: the CLI's exact output on the sample corpus in examples/.

Every command's stdout and stderr, with color off and a fixed terminal width, is compared with
a file in tests/snapshots/, so any change to what users see shows up as a reviewable diff.
After an intended change, regenerate the files with ``make snapshots`` (or run this module with
``UPDATE_SNAPSHOTS=1``) and commit them with the change.

There are two modes. ``surface`` builds with ``--no-syntax`` and runs everywhere; ``syntax``
uses the spaCy parser and is skipped when spaCy is not installed. Updating fails instead of
skipping without spaCy, so the syntax snapshots can't silently go stale, and refuses to run
under CI (``CI`` set), where snapshots are only ever checked.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from styleprofile.cache import MeasurementCache
from styleprofile.cli import main
from styleprofile.profile import MINOR_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
gen: Any = importlib.import_module("gen")

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"
UPDATE = os.environ.get("UPDATE_SNAPSHOTS", "") not in ("", "0")
HAS_SPACY = importlib.util.find_spec("spacy") is not None
TMP = "<tmp>"  # stands in for the per-run temporary directory

WRITER, CONTRAST, DRAFT = "examples/writer", "examples/llm-drafts", "examples/draft.md"
REFERENCE, REPORT, EVALUATION = f"{TMP}/writer.json", f"{TMP}/draft.json", f"{TMP}/evaluation.json"
MIXED = f"{TMP}/mixed.json"
# The demo's larger reference in the same voice (``bench/gen.py --corpus demo``, as `make demo`
# builds it), against which draft.md's two planted paragraphs are found.
DEMO, DEMO_REFERENCE = f"{TMP}/demo", f"{TMP}/demo-writer.json"
DEMO_REPORT = f"{TMP}/demo-draft.json"

# (name, arguments) in the order they run; later commands read what earlier ones wrote.
# `build` gains --no-syntax in surface mode, and `score` and `show` inherit it from the profile.
COMMANDS: list[tuple[str, list[str]]] = [
    ("build-default", ["build", f"{TMP}/writer"]),
    ("score-against", ["score", DRAFT, "--against", WRITER, "--contrast", CONTRAST]),
    ("demo", ["demo", "--dir", f"{TMP}/bundled-demo"]),
    ("build", ["build", WRITER, "--contrast", CONTRAST, "-o", REFERENCE]),
    ("show-newer-minor", ["show", f"{TMP}/newer-minor.json"]),
    ("score", ["score", DRAFT, REFERENCE, "-o", REPORT]),
    ("score-quiet", ["score", "-q", DRAFT, REFERENCE]),
    ("score-all", ["score", "--all", DRAFT, REFERENCE]),
    (
        "score-two-files",
        ["score", f"{WRITER}/sharpening.md", f"{WRITER}/old-maps.md", REFERENCE],
    ),
    # The writer's own essay and an LLM draft: one close, one very different.
    (
        "score-writer-and-draft",
        ["score", f"{WRITER}/sharpening.md", f"{CONTRAST}/old-maps.md", REFERENCE, "-o", MIXED],
    ),
    (
        "score-writer-and-draft-quiet",
        [
            "score",
            "-q",
            "--fail-above",
            "clearly",
            f"{WRITER}/sharpening.md",
            f"{CONTRAST}/old-maps.md",
            REFERENCE,
        ],
    ),
    ("show-score-writer-and-draft", ["show", MIXED]),
    ("score-by-paragraph", ["score", "--by-paragraph", DRAFT, REFERENCE]),
    (
        "build-demo",
        [
            "build",
            "--by-paragraph",
            f"{DEMO}/writer",
            "--contrast",
            f"{DEMO}/contrast",
            "-o",
            DEMO_REFERENCE,
        ],
    ),
    # Paragraph drift is experimental, shown only with --by-paragraph.
    ("score-demo", ["score", DRAFT, DEMO_REFERENCE]),
    ("score-demo-quiet", ["score", "-q", DRAFT, DEMO_REFERENCE]),
    (
        "score-demo-by-paragraph",
        ["score", "--by-paragraph", DRAFT, DEMO_REFERENCE, "-o", DEMO_REPORT],
    ),
    ("score-demo-by-paragraph-quiet", ["score", "-q", "--by-paragraph", DRAFT, DEMO_REFERENCE]),
    ("show-demo-by-paragraph", ["show", "--by-paragraph", DEMO_REPORT]),
    (
        "score-demo-two-files-by-paragraph",
        ["score", "--by-paragraph", DRAFT, f"{WRITER}/old-maps.md", DEMO_REFERENCE],
    ),
    (
        "score-demo-two-files-by-paragraph-quiet",
        ["score", "-q", "--by-paragraph", DRAFT, f"{WRITER}/old-maps.md", DEMO_REFERENCE],
    ),
    ("show-reference", ["show", REFERENCE]),
    ("show-reference-all", ["show", "--all", REFERENCE]),
    ("show-score", ["show", REPORT]),
]
# Commands whose output doesn't depend on spaCy run in surface mode only.
SURFACE_ONLY: list[tuple[str, list[str]]] = [
    ("metrics", ["metrics"]),
    (
        "evaluate",
        [
            "evaluate",
            WRITER,
            "--contrast",
            CONTRAST,
            "--edited",
            f"plain={TMP}/plain",
            "--no-syntax",
            "-o",
            EVALUATION,
        ],
    ),
    ("show-evaluation", ["show", EVALUATION]),
]
MODES = {
    "surface": [*COMMANDS, *SURFACE_ONLY],
    "syntax": COMMANDS,
}


def _plain(text: str) -> str:
    """A deterministic stand-in for a light human edit: drop Markdown headings, lists and bold."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^#+\s+", "", line)
        line = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line)
        lines.append(line.replace("**", ""))
    return "\n".join(lines) + "\n"


def _run(argv: list[str], tmp: Path) -> str:
    """Run one command; return its exit code, stdout and stderr with local paths normalized."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        pytest.MonkeyPatch.context() as patch,
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        if argv[0] == "demo":
            # Snapshot the verdict, independent of the disk cache's availability. The
            # cache-enabled demo's warning and identical verdict are tested in test_demo.
            patch.setenv("STYLEPROFILE_NO_CACHE", "1")
        code = main([arg.replace(TMP, str(tmp)) for arg in argv])
    text = f"$ styleprofile {shlex.join(argv)}\n[exit {code}]\n"
    text += f"--- stdout\n{stdout.getvalue()}"
    if stderr.getvalue():
        text += f"--- stderr\n{stderr.getvalue()}"
    text = re.sub(r"in [0-9.]+ s, not saved", "in <time> s, not saved", text)
    # Longest first: the temporary directory may sit inside the repository or vice versa.
    for path, name in sorted(
        [
            (str(tmp), TMP),
            (str(tmp.resolve()), TMP),
            (tmp.as_posix(), TMP),
            (tmp.resolve().as_posix(), TMP),
            (str(ROOT), "<repo>"),
            (ROOT.as_posix(), "<repo>"),
        ],
        key=lambda item: -len(item[0]),
    ):
        text = text.replace(path, name)
    return text


@pytest.fixture(scope="module", params=list(MODES))
def outputs(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[str, dict[str, str]]]:
    """Run every command of one mode, in order, from the repository root."""
    mode: str = request.param
    if UPDATE and os.environ.get("CI"):
        pytest.fail("UPDATE_SNAPSHOTS is set under CI; snapshots are only updated locally")
    if mode == "syntax" and not HAS_SPACY:
        if UPDATE:
            pytest.fail("updating snapshots needs spaCy for the syntax mode; run `make snapshots`")
        pytest.skip("spaCy is not installed")
    tmp = tmp_path_factory.mktemp(mode)
    gen.generate("demo", out=tmp)
    shutil.copytree(ROOT / WRITER, tmp / "writer")
    plain = tmp / "plain"
    plain.mkdir()
    for draft in sorted((ROOT / CONTRAST).glob("*.md")):
        (plain / draft.name).write_text(_plain(draft.read_text(encoding="utf-8")), encoding="utf-8")
    results: dict[str, str] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)
        patch.setenv("NO_COLOR", "1")
        patch.delenv("FORCE_COLOR", raising=False)
        patch.setenv("COLUMNS", "100")
        for name, args in MODES[mode]:
            if name == "show-newer-minor":
                newer = json.loads((tmp / "writer.json").read_text(encoding="utf-8"))
                newer["minor_version"] = MINOR_VERSION + 1
                (tmp / "newer-minor.json").write_text(json.dumps(newer), encoding="utf-8")
            if mode == "surface" and (args[0] in ("build", "demo") or "--against" in args):
                args = [*args, "--no-syntax"]
            if name == "build-default":
                with pytest.MonkeyPatch.context() as cwd:
                    cwd.chdir(tmp)
                    results[name] = _run(args, tmp)
            else:
                results[name] = _run(args, tmp)
    yield mode, results


def _snapshot(mode: str, name: str) -> Path:
    return SNAPSHOTS / mode / f"{name}.txt"


def test_demo_snapshot_is_independent_of_disk_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(self: MeasurementCache) -> sqlite3.Connection:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("STYLEPROFILE_NO_CACHE", raising=False)
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setattr(MeasurementCache, "_open", unavailable)
    actual = _run(["demo", "--dir", f"{TMP}/bundled-demo", "--no-syntax"], tmp_path)
    assert actual == _snapshot("surface", "demo").read_text(encoding="utf-8")
    assert "STYLEPROFILE_NO_CACHE" not in os.environ


@pytest.mark.parametrize("name", sorted({name for cases in MODES.values() for name, _ in cases}))
def test_snapshot(outputs: tuple[str, dict[str, str]], name: str) -> None:
    mode, results = outputs
    if name not in results:
        pytest.skip(f"{name} runs in surface mode only")
    path = _snapshot(mode, name)
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(results[name], encoding="utf-8")
        return
    relative = path.relative_to(ROOT)
    if not path.exists():
        pytest.fail(f"{relative} is missing; run `make snapshots`", pytrace=False)
    expected = path.read_text(encoding="utf-8")
    if results[name] != expected:
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            results[name].splitlines(keepends=True),
            fromfile=f"{relative} (snapshot)",
            tofile=f"{relative} (actual)",
        )
        pytest.fail(
            f"output differs from {relative}; if the change is intended, run `make snapshots` "
            "and commit the result\n\n" + "".join(diff),
            pytrace=False,
        )


def test_no_stale_snapshots() -> None:
    """Every snapshot file belongs to a command; `make snapshots` deletes those that don't."""
    known = {_snapshot(mode, name) for mode, cases in MODES.items() for name, _ in cases}
    stale = sorted(path for path in SNAPSHOTS.rglob("*.txt") if path not in known)
    if UPDATE and not os.environ.get("CI"):
        for path in stale:
            path.unlink()
        for folder in SNAPSHOTS.iterdir():
            if folder.is_dir() and not any(folder.iterdir()):
                shutil.rmtree(folder)
        return
    assert not stale, f"snapshots with no command: {[str(p.relative_to(ROOT)) for p in stale]}"
