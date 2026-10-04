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


def test_dynamic_note_context_stays_on_one_line() -> None:
    name = "an\nEnglish\tfile.md"
    note = NoteCode.NON_ENGLISH.note("0", name)
    assert "an English file.md" in note.text()
    assert "\n" not in note.text() and "\t" not in note.text()
    assert name in note.text(verbose=True)


def test_cli_flag_translation_does_not_change_dynamic_context(capsys) -> None:
    from styleprofile.cli import FLAGS, _notes

    note = NoteCode.OVERSIZE_CHUNK.note("0", "pool.md", "2,001", "500", setting="window_words")
    _notes([note])
    text = capsys.readouterr().err
    assert "pool.md:" in text and "--pool.md" not in text
    assert "twice --window-words" in text
    assert len(text.strip()) <= 106
    assert note.text(settings=FLAGS) == text.removeprefix("note: ").strip()


def test_cli_warning_flags_preserve_labels_and_original_reports(tmp_path, capsys) -> None:
    # Both score and show must translate advice, without rewriting dynamic labels.
    profile = api.build(WRITER, syntax=False, min_words=75, cache=False)
    reference = tmp_path / "reference.json"
    profile.save(reference)
    tiny = tmp_path / "tiny.md"
    tiny.write_text("A few words that cannot be scored.", encoding="utf-8")
    output = tmp_path / "score.json"
    command = ["score", str(DRAFT), str(tiny), str(reference), "--no-cache", "-o", str(output)]
    assert main(command) == 0
    short = capsys.readouterr().out
    assert "lower --min-words" in short
    saved = output.read_bytes()
    assert "lower --min-words" not in json.loads(saved)["warnings"][0]
    assert main([*command, "--verbose"]) == 0
    verbose = capsys.readouterr().out
    assert "chunk(s) with fewer than 75 prose words" in verbose
    assert output.read_bytes() == saved
    assert main(["show", str(output)]) == 0
    assert "lower --min-words" in capsys.readouterr().out

    message = NoteCode.SHORT_EDITS.message("0", "min_words", "1", "75", "'draft'")
    assert warning_text(message, settings={"min_words": "--min-words"}).startswith("min_words:")
    message = NoteCode.LEFT_OUT.message("details", "draft", "")
    assert "use --pool" in warning_text(message, settings={"pool": "--pool"})


def test_unknown_warning_wraps_unbroken_context_without_losing_text() -> None:
    from styleprofile.display import _human_text, _strip

    warning = "Note: Future warning " + "x" * 300
    short = _human_text([warning], verbose=False)
    assert all(len(line) <= 120 for line in short.splitlines())
    assert "".join(short.split()) == "".join(warning.split())
    assert _human_text([warning], verbose=True) == warning
    for size in (226, 227, 228, 300):
        colored = "\033[33mNote: " + "x" * size + "\033[0m"
        rendered = _human_text([colored], verbose=False)
        assert "\033" not in _strip(rendered)
        assert all(len(line) <= 120 for line in _strip(rendered).splitlines())
        assert "".join(_strip(rendered).split()) == "".join(_strip(colored).split())
        assert _human_text([colored], verbose=True) == colored


def test_combined_thin_reference_keeps_structural_remedies(capsys) -> None:
    from styleprofile.cli import _thin_warnings

    profile = api.build(WRITER, syntax=False, cache=False)
    report = profile.report
    grouped = NoteCode.THIN_REFERENCE.note("0", "1 document", "thread", setting="group_field")
    dominant = NoteCode.THIN_REFERENCE.note(
        "dominant", "40", "42 chunks", "2 chunks", "20", "3", ""
    )
    single = NoteCode.THIN_REFERENCE.note("3", "1 document")
    for note, remedy in [
        (grouped, "finer --group-field"),
        (dominant, "similar-sized documents"),
        (single, "no held-out calibration or contrast"),
    ]:
        candidate = report.copy()
        candidate["document_count"] = 1 if note != dominant else 3
        thin = api.Profile(candidate, notes=[note])
        _thin_warnings(thin)
        short = capsys.readouterr().err
        assert len(short.splitlines()) == 1 and len(short.strip()) <= 120
        assert remedy in short
        _thin_warnings(thin, verbose=True)
        verbose = capsys.readouterr().err
        assert "Thin reference: " in verbose
        assert note.message.replace("group_field", "--group-field") in verbose


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


def test_requested_split_without_boundaries_says_text_was_kept_whole() -> None:
    profile = api.build(WRITER, syntax=False, cache=False)
    result = profile.score_text(DRAFT.read_text(encoding="utf-8"), split_on="heading", cache=False)
    [note] = [note for note in result.notes if note.code == NoteCode.SPLIT]
    assert "kept whole" in note.text()
    assert "split into documents" not in note.text()
    assert "no headings" in note.text(verbose=True)


@pytest.mark.parametrize("model_missing", [False, True])
def test_default_syntax_note_names_the_missing_dependency(
    model_missing, tmp_path, capsys, monkeypatch
) -> None:
    from styleprofile import SyntaxUnavailableError

    def unavailable():
        raise SyntaxUnavailableError("dependency unavailable", model_missing=model_missing)

    monkeypatch.setattr(api, "_default_parser", unavailable)
    assert main(["build", str(WRITER), "--no-cache", "-o", str(tmp_path / "reference.json")]) == 0
    lines = capsys.readouterr().err.splitlines()
    [note] = [line for line in lines if line.startswith("note:")]
    assert len(note) <= 106
    if model_missing:
        assert "English model missing" in note
        assert "run `styleprofile setup`" in note
    else:
        assert "spaCy is not installed" in note
        assert "Install the syntax extra" in note
