"""Input formatting and language warnings do not become false style evidence."""

from pathlib import Path

import pytest

import styleprofile as sp
from styleprofile.profile import Chunk, window
from styleprofile.surface import (
    PARAGRAPH_METRICS,
    paragraph_metrics_missing,
    prose,
    surface_metrics,
    unlikely_english,
    words,
)
from styleprofile.weighting import _values

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "I went to the old house and found that it was still standing there."
LANGUAGES = [
    "これは日本語の文章です。静かな町に住んでいます。毎日新しい本を読みます。",
    "Das kleine Haus steht neben einem Garten. Jeden Morgen gehe ich durch die Stadt "
    "und sehe Menschen auf ihrem Weg zur Arbeit.",
    "La casa pequeña está junto al jardín. Cada mañana camino por la ciudad y veo "
    "personas que van hacia sus trabajos.",
]


@pytest.mark.parametrize("length,missing", [(299, False), (300, True), (500, True)])
def test_single_paragraph_metrics_are_missing_at_300(length: int, missing: bool) -> None:
    text = " ".join(["word"] * length)
    parsed = prose(text)
    metrics = surface_metrics(text)
    assert paragraph_metrics_missing(parsed, length) is missing
    for name in PARAGRAPH_METRICS:
        assert (metrics["sentence_shape"][name] is None) is missing
        assert (("sentence_shape", name) not in _values(metrics)) is missing
    # Two actual paragraphs still carry paragraph evidence.
    two = text + "\n\nAnother sentence."
    assert surface_metrics(two)["sentence_shape"]["paragraph_words_mean"] is not None


def test_long_plain_block_is_windowed_without_creating_paragraphs() -> None:
    text = " ".join([SENTENCE] * 300)  # 4,200 words
    chunks = window([Chunk("long", "long", text)], 500)
    assert len(chunks) >= 8
    assert " ".join(chunk.text for chunk in chunks) == text
    assert all(250 <= len(words(prose(chunk.text).text)) <= 750 for chunk in chunks)
    assert all(len(prose(chunk.text).paragraphs) == 1 for chunk in chunks)


def test_flat_essays_are_close_and_notes_reach_build_and_score() -> None:
    profile = sp.build(ROOT / "examples/writer", sp.Settings(syntax=False), cache=False)
    flat = [
        sp.Text("\n".join(line for line in path.read_text().splitlines() if line), name=path.name)
        for path in sorted((ROOT / "examples/writer").glob("*.md"))
    ]
    result = profile.score(flat, cache=False)
    assert len(result.report["documents"]) == 7
    assert all(document.verdict == sp.Verdict.CLOSE for document in result.documents)
    assert sum(note.code == sp.NoteCode.NO_PARAGRAPH_BREAKS for note in result.notes) == 7
    built = sp.build(flat, sp.Settings(syntax=False), cache=False, keep_chunks=True)
    assert sum(note.code == sp.NoteCode.NO_PARAGRAPH_BREAKS for note in built.notes) == 7
    for chunk in built.report["chunks"]:
        assert all(chunk["metrics"]["sentence_shape"][name] is None for name in PARAGRAPH_METRICS)
    for name in PARAGRAPH_METRICS:
        assert built.report["summary"]["sentence_shape"][name]["n"] == 0


@pytest.mark.parametrize("text", LANGUAGES)
def test_other_languages_warn_for_build_and_score(text: str) -> None:
    text *= 10
    assert unlikely_english(prose(text))
    settings = sp.Settings(syntax=False, min_words=1, window_words=0)
    built = sp.build([sp.Text(text), sp.Text(text + " 1")], settings, cache=False)
    assert any(note.code == sp.NoteCode.NON_ENGLISH for note in built.notes)
    profile = sp.build(ROOT / "examples/writer", settings, cache=False)
    result = profile.score(sp.Text(text), cache=False)
    assert any(note.code == sp.NoteCode.NON_ENGLISH for note in result.notes)


def test_english_samples_and_short_ascii_texts_do_not_warn() -> None:
    for path in (ROOT / "examples").rglob("*.md"):
        assert not unlikely_english(prose(path.read_text())), path
    assert not unlikely_english(prose("Specialist quartz crystallography."))


def test_oversize_sentence_warns_and_stays_whole() -> None:
    text = " ".join(["word"] * 1100)
    settings = sp.Settings(syntax=False, window_words=500)
    built = sp.build([sp.Text(text), sp.Text(text + " 1")], settings, cache=False)
    assert built.report["chunk_count"] == 2
    assert any(note.code == sp.NoteCode.OVERSIZE_CHUNK for note in built.notes)
    result = built.score(sp.Text(text), cache=False)
    assert any(note.code == sp.NoteCode.OVERSIZE_CHUNK for note in result.notes)


@pytest.mark.parametrize(
    "text,cause",
    [
        ("# Heading", "only headings"),
        ("```\nprint(1)\n```", "only code"),
        ("", "empty after reading or conversion"),
        ("https://example.com", "no readable prose"),
    ],
)
def test_no_chunks_explains_cause_and_action(text: str, cause: str) -> None:
    with pytest.raises(sp.StyleProfileError, match=cause) as error:
        sp.build(sp.Text(text), sp.Settings(syntax=False), cache=False)
    assert error.value.code == "no_chunks"
    assert "Add readable prose" in str(error.value)


def test_long_draft_is_scored_in_window_sized_chunks() -> None:
    profile = sp.build(ROOT / "examples/writer", sp.Settings(syntax=False), cache=False)
    result = profile.score(sp.Text(" ".join([SENTENCE] * 300)), cache=False)
    assert result.report["chunk_count"] >= 8
    for chunk in result.report["chunks"]:
        size = chunk["metrics"]["size"]["words"]
        assert size is not None and 250 <= size <= 750
    assert any(note.code == sp.NoteCode.NO_PARAGRAPH_BREAKS for note in result.notes)
    assert not any(note.code == sp.NoteCode.OVERSIZE_CHUNK for note in result.notes)


def test_language_notes_are_printed_by_build_and_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from styleprofile.cli import main

    sample = tmp_path / "spanish.md"
    sample.write_text(LANGUAGES[2] * 10)
    reference = tmp_path / "profile.json"
    assert (
        main(["build", str(sample), "--no-syntax", "--window-words", "100", "-o", str(reference)])
        == 0
    )
    assert "may not be English" in capsys.readouterr().err
    assert main(["score", str(sample), str(reference)]) == 0
    assert "may not be English" in capsys.readouterr().err
