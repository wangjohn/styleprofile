"""The library pipeline (``styleprofile.api``): the same numbers as the CLI, inputs as paths,
``Text`` or chunks, inherited settings, and notes instead of printing."""

from __future__ import annotations

import doctest
import json
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile import api
from styleprofile.cli import main
from styleprofile.display import format_evaluation
from styleprofile.profile import base_id, document_of, dumps_report

ROOT = Path(__file__).resolve().parent.parent
WRITER, CONTRAST, DRAFT = "examples/writer", "examples/llm-drafts", "examples/draft.md"
POSTS = [
    "I don't know what I expected. We tried it anyway, and it mostly worked for a while.",
    "You can't plan everything. We shipped it on a Tuesday, and I think that was right.",
    "It's a small thing. So I wrote it down, because I'll forget it otherwise, like always.",
]


@pytest.fixture
def examples(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from the repository root, so paths are recorded as the CLI records them."""
    monkeypatch.chdir(ROOT)
    return ROOT


def _saved(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_saved(report: dict[str, Any]) -> dict[str, Any]:
    """A report as it is written to disk, with floats rounded."""
    return json.loads(dumps_report(report))


def test_api_and_cli_give_identical_reports(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference, scored = tmp_path / "writer.json", tmp_path / "draft.json"
    assert main(["build", WRITER, "--contrast", CONTRAST, "-o", str(reference)]) == 0
    assert main(["score", DRAFT, str(reference), "-o", str(scored)]) == 0
    capsys.readouterr()

    profile = sp.build([WRITER], contrast=[CONTRAST])
    library_reference = tmp_path / "library.json"
    profile.save(library_reference)
    assert library_reference.read_bytes() == reference.read_bytes()

    # Scored in memory, the report differs only in where the reference was saved.
    cli_report = _saved(scored)
    library_report = _as_saved(profile.score([DRAFT]).report)
    assert cli_report["reference"].pop("path") == str(reference)
    assert library_report["reference"].pop("path") == str(library_reference)
    assert library_report == cli_report
    # Loaded from the CLI's file, it is identical, path included.
    assert _as_saved(sp.Profile.load(reference).score([DRAFT]).report) == _saved(scored)
    assert capsys.readouterr() == ("", ""), "the library never prints"


def test_api_and_cli_evaluate_identically(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edited = tmp_path / "plain"
    edited.mkdir()
    for path in (ROOT / CONTRAST).glob("*.md"):
        text = path.read_text(encoding="utf-8").replace(" — ", ", ")
        (edited / path.name).write_text(text, encoding="utf-8")
    output = tmp_path / "evaluation.json"
    command = ["evaluate", WRITER, "--contrast", CONTRAST, "--edited", f"plain={edited}"]
    assert main([*command, "--no-syntax", "-o", str(output)]) == 0
    capsys.readouterr()

    result = sp.evaluate([WRITER], [CONTRAST], {"plain": str(edited)}, sp.Settings(syntax=False))
    assert _as_saved(result.report) == _saved(output)
    assert result.to_text() == format_evaluation(_saved(output))


def test_library_example_runs(examples: Path) -> None:
    """docs/library.md, run as a doctest."""
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    failures, tried = doctest.testfile(
        str(ROOT / "docs" / "library.md"), module_relative=False, optionflags=flags
    )
    assert tried > 0 and failures == 0


def test_lower_level_functions_need_no_parser() -> None:
    chunks = [sp.Chunk(f"post{index}", f"post{index}.md", text) for index, text in enumerate(POSTS)]
    reference = sp.build_reference(chunks)
    assert reference["settings"]["syntax"] is None
    report = sp.score([sp.Chunk("draft", "draft.md", POSTS[0])], reference)
    assert report["reference"]["delta_mean"] is not None


def test_texts_are_scored_like_the_same_file(examples: Path) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    text = (ROOT / DRAFT).read_text(encoding="utf-8")
    from_text = profile.score(sp.Text(text))
    from_file = profile.score(DRAFT)
    assert from_text.delta == from_file.delta
    assert from_text.verdict == from_file.verdict == "close"
    assert from_text.sources == ("<text>",)
    assert [row["id"] for row in from_text.report["chunks"]] == ["text1#w1"]
    assert from_text.report["settings"]["inputs"] == "<text>"
    named = profile.score([sp.Text(text, name="draft"), sp.Text(POSTS[0])])
    assert [row["id"] for row in named.report["chunks"]] == ["draft#w1", "text2#w1"]


def test_inputs_can_be_chunks_paths_or_an_iterator(examples: Path) -> None:
    profile = sp.build(Path(WRITER), sp.Settings(syntax=False, window_words=0))
    assert profile.report["settings"]["inputs"] == WRITER
    chunk = sp.Chunk("mine", "mine.md", POSTS[1] * 20)
    result = profile.score(iter([chunk, Path(DRAFT)]))
    assert result.report["settings"]["inputs"] == ["mine", DRAFT]
    assert [row["id"] for row in result.report["chunks"]] == ["mine", "draft.md"]
    with pytest.raises(TypeError, match="expected a path"):
        profile.score([b"bytes"])  # type: ignore[list-item]


def test_a_string_is_a_path_never_text(examples: Path) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    with pytest.raises(sp.StyleProfileError, match=r"^nope\.md not found$") as missing:
        profile.score("nope.md")
    assert missing.value.code == "input_not_found"
    with pytest.raises(sp.StyleProfileError, match=r"pass raw text as Text\(\.\.\.\)"):
        profile.score("A first line.\nA second line.")


def test_missing_input_error_on_the_command_line(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["build", "nope/", "-o", str(tmp_path / "x.json")]) == 1
    assert capsys.readouterr().err == "error: nope/ not found\n"


def test_score_inherits_the_profile_settings(examples: Path) -> None:
    settings = sp.Settings(window_words=200, min_words=3, syntax=False, top_k=50)
    profile = sp.build(WRITER, settings)
    assert profile.settings == settings
    result = profile.score(DRAFT)
    recorded = result.report["settings"]
    assert (recorded["window_words"], recorded["min_words"], recorded["top_k"]) == (200, 3, 50)
    assert result.notes == ()
    assert not any("window sizes differ" in warning for warning in result.warnings)

    overridden = profile.score(DRAFT, window_words=0, min_words=5)
    assert overridden.report["settings"]["window_words"] is None
    assert any("(off vs 200)" in warning for warning in overridden.warnings)
    assert overridden.notes == (sp.Note("min_words 5 overrides the reference's 3", "min_words"),)


def test_settings_are_checked_without_naming_flags() -> None:
    for options, code in [
        ({"window_words": -1}, "window_words"),
        ({"min_words": -1}, "min_words"),
        ({"top_k": 0}, "top_k"),
        ({"syntax": "yes"}, "syntax"),
        ({"input_format": "docx"}, "input_format"),
    ]:
        with pytest.raises(sp.StyleProfileError) as error:
            sp.Settings(**options)
        assert error.value.code == code
        assert str(error.value).startswith(code) and "--" not in str(error.value)


def test_setting_errors_name_the_flag_on_the_command_line(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = str(tmp_path / "x.json")
    assert main(["build", WRITER, "-o", output, "--window-words", "-5"]) == 1
    assert capsys.readouterr().err == "error: --window-words must be 0 (no windowing) or positive\n"


def test_auto_syntax_without_spacy_is_a_note_not_output(
    examples: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable() -> None:
        raise sp.SyntaxUnavailableError("no spaCy")

    monkeypatch.setattr(api, "_default_parser", unavailable)
    profile = sp.build(WRITER)
    assert [note.code for note in profile.notes] == ["no_syntax"]
    assert "surface metrics only" in profile.notes[0].message
    assert profile.report["settings"]["syntax"] is None
    with pytest.raises(sp.SyntaxUnavailableError):
        sp.build(WRITER, sp.Settings(syntax=True))
    assert capsys.readouterr() == ("", "")


def test_repeated_inputs_are_read_once_with_a_note(examples: Path) -> None:
    profile = sp.build([WRITER, f"{WRITER}/old-maps.md"], sp.Settings(syntax=False))
    once = sp.build(WRITER, sp.Settings(syntax=False))
    assert profile.report["summary"] == once.report["summary"]
    assert profile.notes == (
        sp.Note(f"{WRITER}/old-maps.md was already given; using it once", "repeated_input"),
    )
    with pytest.raises(sp.StyleProfileError, match="only once"):
        sp.build(["-", "-"])


def test_progress_reports_coarse_phases(examples: Path) -> None:
    phases: list[str] = []
    profile = sp.build(
        WRITER, sp.Settings(syntax=False), progress=lambda event: phases.append(event.phase)
    )
    assert phases == [api.READ, api.BUILD, api.DONE]
    phases.clear()
    profile.score(DRAFT, progress=lambda event: phases.append(event.phase))
    assert phases == [api.READ, api.SCORE, api.DONE]


def test_profiles_save_load_and_refuse_other_reports(examples: Path, tmp_path: Path) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    assert profile.path is None
    path = tmp_path / "writer.json"
    profile.save(path)
    assert profile.path == path.resolve()
    loaded = sp.Profile.load(path)
    assert loaded.report == _saved(path) and loaded.settings == profile.settings

    result = loaded.score(DRAFT)
    assert result.report["reference"]["path"] == str(path.resolve())
    report_path = tmp_path / "draft.json"
    result.save(report_path)
    with pytest.raises(sp.StyleProfileError, match="score report") as error:
        sp.Profile.load(report_path)
    assert error.value.code == "score_as_reference"
    # A saved score renders the same without its reference.
    assert sp.ScoreResult(_saved(report_path)).to_text() == result.to_text()


def test_verdicts_match_the_command_line_headline(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = tmp_path / "writer.json"
    profile = sp.build(WRITER, contrast=CONTRAST, settings=sp.Settings(syntax=False))
    profile.save(reference)
    result = profile.score(DRAFT)
    assert main(["score", DRAFT, str(reference), "-q"]) == 0
    headline = capsys.readouterr().out
    assert result.likeness is not None and result.delta is not None
    assert headline == (
        f"{DRAFT}: {result.verdict} (Delta {result.delta:.2f}); "
        f"LLM-likeness {result.likeness_verdict} ({result.likeness:.2f})\n"
    )
    assert result.contrast_label == "LLM"
    assert repr(result) == f"<ScoreResult: {result.verdict} (Delta {result.delta:.2f})>"


def test_notes_reach_the_caller_when_a_run_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post = tmp_path / "post.md"
    post.write_text(POSTS[0], encoding="utf-8")
    repeated = sp.Note(f"{post} was already given; using it once", "repeated_input")
    # One document cannot learn a contrast; the note about the repeat comes with the error.
    with pytest.raises(sp.StyleProfileError) as error:
        sp.build([post, post], sp.Settings(syntax=False), contrast=sp.Text(POSTS[1]))
    assert error.value.code == "contrast_needs_documents"
    assert error.value.notes == [repeated]

    draft = tmp_path / "draft.md"
    draft.write_text(POSTS[1], encoding="utf-8")
    command = ["build", str(post), str(post), "--contrast", str(draft), "--no-syntax"]
    assert main([*command, "-o", str(tmp_path / "x.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"note: {repeated.message}\nerror: ")


def test_rewindowing_chunks_keeps_their_documents() -> None:
    chunks = [sp.Chunk("post", "post.md", "\n\n".join(POSTS * 40))]
    twice = sp.window(sp.window(chunks, 100), 100)
    assert twice[0].id == "post#w1#w1"
    assert {base_id(chunk.id) for chunk in twice} == {"post"}
    assert {document_of(chunk.source, chunk.id) for chunk in twice} == {
        document_of("post.md", "post")
    }
