"""Human notes are concise; verbose and serialized reports keep the full explanation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from styleprofile import api
from styleprofile.cli import main
from styleprofile.core import Note, NoteCode, warning_text

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples/writer"
DRAFTS = ROOT / "examples/llm-drafts"
DRAFT = ROOT / "examples/draft.md"


def test_unknown_saved_warning_is_preserved() -> None:
    future = "A future report's warning must stay visible, even when its meaning is unknown."
    assert warning_text(future) == future
    assert warning_text(future, verbose=True) == future


def test_every_note_has_bounded_short_and_original_long_forms() -> None:
    for code in NoteCode:
        assert code.forms
        for name, form in code.forms.items():
            # Exercise arbitrary context lengths, numeric formats already resolved by callers.
            values = ["x" * 300 for _ in range(12)]
            message = form.long.format(*values, missing="spaCy is missing", fix="install syntax")
            note = Note(message, code)
            assert note.text(verbose=True) == message
            assert "\n" not in note.text(), (code, name)
            assert len(note.text()) <= 100, (code, name)


def test_build_combines_thin_notes_and_verbose_restores_them(tmp_path, capsys) -> None:
    output = tmp_path / "writer.json"
    command = ["build", str(WRITER), "--no-syntax", "--no-cache", "-o", str(output)]
    assert main(command) == 0
    short = capsys.readouterr()
    short_report = output.read_bytes()
    assert short.err.count("Thin reference:") == 1
    assert "7 chunks, 4,496 words (aim for 15+ chunks and 20,000+ words)" in short.err
    assert len(short.err.strip()) <= 120
    assert main([*command, "--verbose"]) == 0
    verbose = capsys.readouterr()
    assert verbose.err.count("Thin reference:") == 2
    assert "aim for 15 or more by adding documents" in verbose.err
    assert "aim for 20,000 or more of the writer's text, in one genre" in verbose.err
    assert output.read_bytes() == short_report


@pytest.mark.parametrize("command", ["score", "evaluate"])
def test_short_verbose_and_json_keep_identical_reports(command, tmp_path, capsys) -> None:
    # Stand-ins give a saved, long calibration warning; partial edits add evaluation warnings.
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "\n\n".join(path.read_text().replace("# ", "") for path in sorted(WRITER.glob("*.md"))),
        encoding="utf-8",
    )
    if command == "score":
        profile = api.build(manuscript, syntax=False, split_on="none", cache=False)
        reference = tmp_path / "reference.json"
        profile.save(reference)
        arguments = [str(DRAFT), str(reference)]
    else:
        edited = tmp_path / "edited"
        edited.mkdir()
        first = next(DRAFTS.glob("*.md"))
        (edited / first.name).write_text(first.read_text().lower(), encoding="utf-8")
        arguments = [str(manuscript), "--contrast", str(DRAFTS), "--edited", f"plain={edited}"]
    output = tmp_path / "result.json"
    base = [command, *arguments, "--no-syntax", "--no-cache", "-o", str(output)]
    assert main(base) == 0
    short = capsys.readouterr()
    saved = output.read_bytes()
    report = json.loads(saved)
    assert report["warnings"]
    for warning in report["warnings"]:
        assert warning_text(warning) in short.out
    assert main([*base, "--verbose"]) == 0
    verbose = capsys.readouterr()
    assert output.read_bytes() == saved
    for warning in report["warnings"]:
        assert warning in verbose.out
    assert main([*base, "--json"]) == 0
    serialized = capsys.readouterr()
    assert json.loads(serialized.out) == report


def test_default_snapshots_fit_120_columns() -> None:
    paths = list((ROOT / "tests/snapshots").rglob("*.txt"))
    assert paths
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # The '$' header is the test's command, not CLI output. Quoted commands are
            # reproducible literal text; aligned verdict rows deliberately retain columns.
            command = line.startswith("$ ")
            quoted = line.lstrip().startswith(("`", "To reuse: `")) or bool(
                re.match(r'^\s*[“"].*[”"]\s*$', line)
            )
            verdict_table = bool(re.search(r"^  .*\S {3,}\S.*(?:▲|▼|—)", line))
            if not (command or quoted or verdict_table):
                assert len(line) <= 120, (path.name, number, len(line), line)
