"""The library pipeline (``styleprofile.api``): the same numbers as the CLI, inputs as paths,
``Text`` or chunks, inherited settings, and notes instead of printing."""

from __future__ import annotations

import dataclasses
import doctest
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile import api
from styleprofile.cli import _flagged, main
from styleprofile.display import format_evaluation
from styleprofile.profile import (
    EVALUATION_VERSION,
    VERSION,
    Chunk,
    base_id,
    build_reference,
    document_of,
    dumps_report,
    load_chunks,
    load_report,
    score,
    window,
)
from styleprofile.syntax import SyntaxUnavailableError

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


def _saved(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_saved(report: Mapping[str, Any]) -> Any:
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
    # Reports save only the profile's file name, never where it lives.
    assert cli_report["reference"].pop("path") == reference.name
    assert library_report["reference"].pop("path") == library_reference.name
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
        str(ROOT / "docs" / "library.md"),
        module_relative=False,
        optionflags=flags,
        extraglobs={
            "essays": [p.read_text(encoding="utf-8") for p in sorted((ROOT / WRITER).glob("*.md"))],
            "drafts": [
                p.read_text(encoding="utf-8") for p in sorted((ROOT / CONTRAST).glob("*.md"))
            ],
            "draft": (ROOT / DRAFT).read_text(encoding="utf-8"),
        },
    )
    assert tried > 0 and failures == 0


def test_the_top_level_is_the_high_level_api() -> None:
    expected = {"build", "evaluate", "Profile", "ScoreResult", "Settings", "Text", "Note"}
    expected |= {"NoteCode", "Progress", "Phase", "Verdict", "StyleProfileError"}
    assert expected <= set(sp.__all__)
    for lower_level in ("score", "build_reference", "load_chunks", "window", "write_report"):
        assert lower_level not in sp.__all__
        assert not hasattr(sp, lower_level)


def test_lower_level_functions_need_no_parser() -> None:
    chunks = [Chunk(f"post{index}", f"post{index}.md", text) for index, text in enumerate(POSTS)]
    reference = build_reference(chunks)
    assert reference["settings"]["syntax_used"] is None
    report = score([Chunk("draft", "draft.md", POSTS[0])], reference)
    assert report["reference"]["delta_mean"] is not None


def test_texts_are_scored_like_the_same_file(examples: Path) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    text = (ROOT / DRAFT).read_text(encoding="utf-8")
    from_text = profile.score(sp.Text(text))
    from_file = profile.score(DRAFT)
    assert from_text.delta == from_file.delta
    assert from_text.verdict == from_file.verdict == sp.Verdict.CLOSE
    assert from_text.sources == ("<text>",)
    assert [row["id"] for row in from_text.report["chunks"]] == ["text1#w1"]
    assert from_text.report["settings"]["inputs"] == ["<text>"]


def test_unnamed_texts_skip_taken_names_and_duplicates_are_refused() -> None:
    texts = [sp.Text(POSTS[0], name="text2"), sp.Text(POSTS[1]), sp.Text(POSTS[2])]
    profile = sp.build(texts, sp.Settings(syntax=False, window_words=0), keep_chunks=True)
    assert [row["id"] for row in profile.report["chunks"]] == ["text2", "text1", "text3"]

    twice = [sp.Text(POSTS[0], name="a"), sp.Text(POSTS[1], name="a")]
    with pytest.raises(sp.StyleProfileError, match="two texts are named 'a'") as error:
        sp.build(twice, sp.Settings(syntax=False))
    assert error.value.code == "duplicate_names"


def test_evaluate_pairs_edited_texts_with_their_originals(examples: Path) -> None:
    drafts = [path.read_text(encoding="utf-8") for path in sorted((ROOT / CONTRAST).glob("*.md"))]
    plain = [draft.replace(" — ", ", ") for draft in drafts]
    settings = sp.Settings(syntax=False)
    by_position = sp.evaluate(
        WRITER, [sp.Text(d) for d in drafts], {"plain": [sp.Text(p) for p in plain]}, settings
    )
    named = sp.evaluate(
        WRITER,
        [sp.Text(d, name=f"d{i}") for i, d in enumerate(drafts)],
        {"plain": [sp.Text(p, name=f"d{i}") for i, p in reversed(list(enumerate(plain)))]},
        settings,
    )
    for result in (by_position, named):
        assert result.report["sets"]["plain"]["missing"] == []
    assert by_position.report["sets"]["plain"]["auc"] == named.report["sets"]["plain"]["auc"]
    with pytest.raises(sp.StyleProfileError, match="no original") as error:
        sp.evaluate(WRITER, [sp.Text(d) for d in drafts], {"x": sp.Text(plain[0], "z")}, settings)
    assert error.value.code == "unmatched_edits"


def test_inputs_can_be_chunks_paths_or_an_iterator(examples: Path) -> None:
    profile = sp.build(Path(WRITER), sp.Settings(syntax=False, window_words=0))
    # Paths are recorded by the name their sources were saved under, never as paths.
    assert profile.report["settings"]["inputs"] == [Path(WRITER).name]
    chunk = Chunk("mine", "mine.md", POSTS[1] * 20)
    result = profile.score(iter([chunk, Path(DRAFT)]))
    assert result.report["settings"]["inputs"] == ["mine", Path(DRAFT).name]
    assert [row["id"] for row in result.report["chunks"]] == ["mine", "draft.md"]
    with pytest.raises(TypeError, match="expected a path"):
        profile.score([3])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="bytes are not an input"):
        profile.score(b"draft.md")  # type: ignore[arg-type]


def test_a_string_is_a_path_never_text(examples: Path) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    with pytest.raises(sp.StyleProfileError, match=r"^nope\.md not found$") as missing:
        profile.score("nope.md")
    assert missing.value.code == "input_not_found"
    for text in ("Some text I wrote.", "I wrote this/that today", "Sometext"):
        with pytest.raises(
            sp.StyleProfileError, match=r"use build_texts\(\.\.\.\) or Profile.score_text"
        ):
            profile.score(text)


def test_missing_inputs_on_the_command_line_are_named_as_typed(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for missing in ("nope/", "posts", "my posts"):
        assert main(["build", missing, "-o", str(tmp_path / "x.json")]) == 1
        assert capsys.readouterr().err == f"error: {missing} not found\n"


def _accepted(name: str) -> list[Any]:
    """The values a setting accepts from a fixed pool of plausible ones."""
    pool = [0, 1, 7, 150, 500, None, True, False, "auto", "text", "body", "html", "", -1, 2.5]
    accepted = []
    for value in pool:
        try:
            sp.Settings(**{name: value})
        except sp.StyleProfileError:
            continue
        accepted.append(value)
    return accepted


def test_settings_round_trip_through_a_report() -> None:
    """Every field is recorded verbatim and read back, so a new setting needs no other code
    to be saved, inherited or overridden."""
    samples = {field.name: _accepted(field.name) for field in dataclasses.fields(sp.Settings)}
    assert all(samples.values()), f"a setting accepts nothing from the pool: {samples}"
    rng = random.Random(3)
    for _ in range(200):
        chosen = {name: rng.choice(values) for name, values in samples.items()}
        settings = sp.Settings(**chosen)
        recorded = json.loads(json.dumps(settings.to_report()))
        assert recorded == chosen
        assert sp.Settings.from_report(recorded) == settings
        assert sp.Settings.from_report({**recorded, "inputs": [], "syntax_used": None}) == settings
    # A report made without settings (by the lower-level functions) was not windowed.
    assert sp.Settings.from_report({}) == sp.Settings(window_words=0)
    # Overrides are typed with the same fields.
    assert set(api.SettingsOverrides.__annotations__) == set(samples)


def test_profiles_record_settings_verbatim(examples: Path) -> None:
    settings = sp.Settings(window_words=0, min_words=0, syntax="auto")
    profile = sp.build(WRITER, settings)
    assert profile.settings == settings
    recorded = profile.report["settings"]
    assert (recorded["window_words"], recorded["min_words"], recorded["syntax"]) == (0, 0, "auto")
    assert "syntax_used" in recorded
    # Building again with a profile's settings asks for nothing more than the first build.
    assert sp.build(WRITER, profile.settings).settings == settings


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
    assert overridden.report["settings"]["window_words"] == 0
    assert any("(off vs 200)" in warning for warning in overridden.warnings)
    assert overridden.notes == (
        sp.Note(
            "min_words 5 overrides the reference's 3",
            sp.NoteCode.SETTING_OVERRIDDEN,
            setting="min_words",
        ),
    )
    # Whole settings replace the inherited ones, and overrides apply on top.
    with pytest.warns(DeprecationWarning, match="keyword overrides"):
        replaced = profile.score(
            DRAFT, dataclasses.replace(settings, window_words=100), min_words=3
        )
    assert replaced.report["settings"]["window_words"] == 100 and replaced.notes == ()
    with pytest.raises(TypeError, match="unknown setting"):
        profile.score(DRAFT, window=100)  # pyright: ignore[reportCallIssue]


def test_settings_are_checked_strictly_without_naming_flags() -> None:
    for options in [
        {"window_words": -1},
        {"window_words": "500"},
        {"window_words": True},
        {"min_words": -1},
        {"top_k": 0},
        {"top_k": 2.5},
        {"syntax": "yes"},
        {"syntax": 0},
        {"syntax": 1.0},
        {"text_field": ""},
        {"input_format": "docx"},
    ]:
        [name] = options
        with pytest.raises(sp.StyleProfileError) as error:
            sp.Settings(**options)
        assert (error.value.code, error.value.setting) == ("invalid_setting", name)
        assert str(error.value).startswith(name) and "--" not in str(error.value)


def test_setting_errors_name_the_flag_on_the_command_line(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = str(tmp_path / "x.json")
    assert main(["build", WRITER, "-o", output, "--window-words", "-5"]) == 1
    assert capsys.readouterr().err == "error: --window-words must be 0 (no windowing) or positive\n"
    # Only the setting itself, as a whole word, becomes a flag.
    assert _flagged("min_words_extra and min_words 5", "min_words") == (
        "min_words_extra and --min-words 5"
    )


def test_thin_references_are_noted_by_the_library(examples: Path) -> None:
    essays = [f"{WRITER}/old-maps.md", f"{WRITER}/sharpening.md"]
    profile = sp.build(essays, sp.Settings(syntax=False))
    thin = [note for note in profile.notes if note.code == sp.NoteCode.THIN_REFERENCE]
    assert [note.message.split(";")[0] for note in thin] == [
        "it has 2 chunks",
        "it has 1,277 words",
    ]
    assert thin[0].setting == "window_words"
    assert "--" not in thin[0].message and "a smaller window_words" in thin[0].message


def test_auto_syntax_without_spacy_is_a_note_not_output(
    examples: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable() -> None:
        raise SyntaxUnavailableError("no spaCy")

    monkeypatch.setattr(api, "_default_parser", unavailable)
    profile = sp.build(WRITER)
    assert profile.notes[0].code == sp.NoteCode.NO_SYNTAX
    assert "surface metrics only" in profile.notes[0].message
    assert profile.report["settings"]["syntax"] == "auto"
    assert profile.report["settings"]["syntax_used"] is None
    with pytest.raises(sp.StyleProfileError) as error:
        sp.build(WRITER, sp.Settings(syntax=True))
    assert isinstance(error.value, sp.SyntaxUnavailableError)
    assert error.value.code == "syntax_unavailable"
    assert capsys.readouterr() == ("", "")


def test_repeated_inputs_are_read_once_with_a_note(examples: Path) -> None:
    profile = sp.build([WRITER, f"{WRITER}/old-maps.md"], sp.Settings(syntax=False))
    once = sp.build(WRITER, sp.Settings(syntax=False))
    assert profile.report["summary"] == once.report["summary"]
    assert profile.notes[0] == sp.Note(
        f"{WRITER}/old-maps.md was already given; using it once", sp.NoteCode.REPEATED_INPUT
    )
    with pytest.raises(sp.StyleProfileError, match="only once") as error:
        sp.build(["-", "-"])
    assert error.value.code == "stdin_twice"


def _phases(events: list[sp.Progress]) -> list[sp.Phase]:
    """The phases in the order they started (each chunk measured repeats its phase)."""
    return [
        event.phase
        for index, event in enumerate(events)
        if not index or event.phase != events[index - 1].phase
    ]


def test_every_run_reports_phases_in_one_order(examples: Path) -> None:
    events: list[sp.Progress] = []
    profile = sp.build(WRITER, sp.Settings(syntax=False), contrast=CONTRAST, progress=events.append)
    P = sp.Phase
    assert _phases(events) == [
        P.READ,
        P.BUILD,
        P.MEASURE,
        P.CALIBRATE,
        P.MEASURE_CONTRAST,
        P.CONTRAST,
        P.DONE,
    ]
    # Measuring reports each chunk, with the words done so far.
    measured = [event for event in events if event.phase == P.MEASURE]
    assert measured[0].done == 0 and measured[-1].done == measured[-1].total
    assert measured[-1].total == profile.report["chunk_count"]
    assert measured[-1].words == profile.report["word_count"]
    assert not any(event.parsing for event in events)
    events.clear()
    profile.score(DRAFT, progress=events.append)
    assert _phases(events) == [P.READ, P.SCORE, P.MEASURE, P.DONE]
    events.clear()
    profile.score(DRAFT, syntax="auto", progress=events.append)
    assert _phases(events)[:2] == [P.READ, P.LOAD_PARSER]
    assert _phases(events)[2:] == [P.SCORE, P.MEASURE, P.DONE]


def test_profiles_save_load_and_refuse_other_reports(examples: Path, tmp_path: Path) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False))
    assert profile.path is None
    path = tmp_path / "writer.json"
    profile.save(path)
    assert profile.path == path.resolve()
    loaded = sp.Profile.load(path)
    assert loaded.report == _saved(path) and loaded.settings == profile.settings

    result = loaded.score(DRAFT)
    assert result.report["reference"]["path"] == path.name
    assert isinstance(result.warnings, tuple) and isinstance(result.notes, tuple)
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
    assert isinstance(result.verdict, sp.Verdict)
    assert result.likeness_verdict is sp.LikenessVerdict.LIKE_REFERENCE
    assert headline == (
        f"{DRAFT}: {result.verdict} (Delta {result.delta:.2f}); "
        f"LLM-likeness {result.likeness_verdict.words('LLM')} ({result.likeness:.2f})\n"
    )
    assert result.contrast_label == "LLM"
    assert repr(result) == f"<ScoreResult: {result.verdict} (Delta {result.delta:.2f})>"
    assert sp.Verdict.TOO_SHORT == "too short to judge"
    assert sp.LikenessVerdict.LEANS.words("LLM") == "leans LLM"


def test_a_result_with_nothing_compared_is_not_comparable() -> None:
    baseline = {"chunk_count": 3, "summary": {}, "calibration": None, "contrast": None}
    report: Any = {
        "chunk_count": 1,
        "warnings": [],
        "reference": {"delta_mean": None, "likeness_mean": None, "baseline": baseline},
    }
    result = sp.ScoreResult(report)
    assert result.verdict is sp.Verdict.NOT_COMPARABLE and result.likeness_verdict is None


def test_notes_reach_the_caller_when_a_run_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    post = tmp_path / "post.md"
    post.write_text(POSTS[0], encoding="utf-8")
    repeated = sp.Note(f"{post} was already given; using it once", sp.NoteCode.REPEATED_INPUT)
    # One document cannot learn a contrast; the note about the repeat comes with the error.
    with pytest.raises(sp.StyleProfileError) as error:
        sp.build([post, post], sp.Settings(syntax=False), contrast=sp.Text(POSTS[1]))
    assert error.value.code == "contrast_needs_documents"
    assert error.value.notes == (repeated,)

    draft = tmp_path / "draft.md"
    draft.write_text(POSTS[1], encoding="utf-8")
    command = ["build", str(post), str(post), "--contrast", str(draft), "--no-syntax"]
    assert main([*command, "-o", str(tmp_path / "x.json")]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"note: {repeated.message}\nerror: ")


def test_os_errors_keep_the_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    post, locked = tmp_path / "post.md", tmp_path / "locked.md"
    post.write_text(POSTS[0], encoding="utf-8")
    locked.write_text(POSTS[1], encoding="utf-8")

    def unreadable(inputs: Any, text_field: Any = None, **options: Any) -> Any:
        if str(inputs[0]) == str(locked):
            raise PermissionError(13, "Permission denied", str(locked))
        return load_chunks(inputs, text_field, **options)

    monkeypatch.setattr(api, "load_chunks", unreadable)
    with pytest.raises(sp.StyleProfileError, match="Permission denied") as error:
        sp.build([post, post, locked], sp.Settings(syntax=False))
    assert error.value.code == "unreadable"
    assert isinstance(error.value.__cause__, PermissionError)
    assert [note.code for note in error.value.notes] == [sp.NoteCode.REPEATED_INPUT]


def test_rewindowing_chunks_keeps_their_documents() -> None:
    chunks = [Chunk("post", "post.md", "\n\n".join(POSTS * 40))]
    twice = window(window(chunks, 100), 100)
    assert twice[0].id == "post#w1#w1"
    assert {base_id(chunk.id) for chunk in twice} == {"post"}
    assert {document_of(chunk.source, chunk.id) for chunk in twice} == {
        document_of("post.md", "post")
    }


def _as_main_saved_it(report: Mapping[str, Any]) -> Any:
    """A report shaped as main (report version 5) saved it: syntax is the parser that ran,
    no windowing is null, and one input is recorded bare."""
    settings = {
        key: value
        for key, value in report["settings"].items()
        if key not in ("syntax", "syntax_used", "input_format")
    }
    settings["syntax"] = {"model": "en_core_web_sm", "model_version": "3.8.0"}
    settings["window_words"] = settings["window_words"] or None
    return {**report, "version": 5, "settings": settings}


def test_reports_from_older_versions_are_refused_with_rebuild_it(
    examples: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = sp.build(WRITER, sp.Settings(syntax=False, window_words=0))
    old_profile, old_score = tmp_path / "old.json", tmp_path / "old-score.json"
    old_profile.write_text(json.dumps(_as_main_saved_it(profile.report)), encoding="utf-8")
    score_report = profile.score(DRAFT).report
    old_score.write_text(json.dumps({**score_report, "version": 5}), encoding="utf-8")

    with pytest.raises(sp.StyleProfileError, match="rebuild it with `styleprofile build`") as error:
        sp.Profile.load(old_profile)
    assert error.value.code == "outdated"
    assert f"report version 5; this one reads {VERSION}" in str(error.value)
    with pytest.raises(sp.StyleProfileError, match="an older styleprofile") as error:
        sp.Profile(_as_main_saved_it(profile.report))
    with pytest.raises(sp.StyleProfileError, match="score it again") as error:
        load_report(old_score)
    assert error.value.code == "outdated"

    # The command line says the same for both, naming the file as typed; the message says how
    # to fix it, so no hint follows, and it never blames a flag.
    for command, again in (
        (["score", DRAFT, str(old_profile)], "rebuild it with `styleprofile build`"),
        (["show", str(old_score)], "score it again with `styleprofile score`"),
    ):
        assert main(command) == 1
        err = capsys.readouterr().err
        assert err == (
            f"error: {command[-1]} was made by an older styleprofile (report version 5; this "
            f"one reads {VERSION}); {again}\n"
        )


def test_evaluation_reports_have_their_own_version(examples: Path, tmp_path: Path) -> None:
    drafts = [path.read_text(encoding="utf-8") for path in sorted((ROOT / CONTRAST).glob("*.md"))]
    result = sp.evaluate(
        WRITER,
        [sp.Text(d) for d in drafts],
        {"plain": [sp.Text(d.replace(" — ", ", ")) for d in drafts]},
        sp.Settings(syntax=False),
    )
    assert result.report["version"] == EVALUATION_VERSION
    assert "top_k" not in result.report["settings"], "evaluate never uses top_k"
    path = tmp_path / "evaluation.json"
    result.save(path)
    assert load_report(path)["kind"] == "evaluation"
    path.write_text(json.dumps({**result.report, "version": 1}), encoding="utf-8")
    with pytest.raises(sp.StyleProfileError, match="run `styleprofile evaluate` again"):
        load_report(path)


def test_lower_level_profiles_are_read_as_not_windowed() -> None:
    chunks = [Chunk(f"post{index}", f"post{index}.md", text) for index, text in enumerate(POSTS)]
    profile = sp.Profile(build_reference(chunks))
    assert profile.report["settings"]["window_words"] == 0
    assert profile.settings.window_words == 0
    result = profile.score(sp.Text(POSTS[0] * 30))
    assert not any("window sizes differ" in warning for warning in result.warnings)


def test_build_refuses_one_chunk_and_suggests_working_window(examples: Path) -> None:
    with pytest.raises(sp.StyleProfileError, match="at least 2 chunks") as error:
        sp.build(f"{WRITER}/old-maps.md", sp.Settings(syntax=False), cache=False)
    assert error.value.code == "reference_needs_chunks"
    assert "--window-words (320)" in str(error.value)
    profile = sp.build(
        f"{WRITER}/old-maps.md", sp.Settings(syntax=False, window_words=320), cache=False
    )
    assert profile.report["chunk_count"] >= 2


def test_incomparable_old_reference_is_not_judged(examples: Path) -> None:
    report = build_reference(load_chunks([f"{WRITER}/old-maps.md"]), parser=None)
    result = sp.Profile(report).score(DRAFT, syntax=False, cache=False)
    assert result.verdict is sp.Verdict.NOT_COMPARABLE
    assert not result.judged and result.reason
    assert all(not doc.judged and doc.reason for doc in result.documents)
    assert result.report["reference"]["verdict"]["judged"] is False


def test_one_chunk_suggestion_respects_minimum_words(examples: Path) -> None:
    with pytest.raises(sp.StyleProfileError) as error:
        sp.build(f"{WRITER}/old-maps.md", sp.Settings(syntax=False, min_words=400), cache=False)
    assert error.value.code == "reference_needs_chunks"
    assert "add documents with at least 400 prose words each" in str(error.value)
    assert "--window-words" not in str(error.value)
