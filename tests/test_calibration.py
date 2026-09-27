"""Length-aware verdicts: calibration for shorter texts, and "too short to judge".

The acceptance tests measure, on text the reference never saw, how often the writer's own
pieces of about 75, 150 and 300 words read "clearly different" or worse (at most 5%), and
how often LLM drafts cut to about 150 words are still flagged (at least 80%). They run on
the sample corpus in examples/ (leaving one document out at a time) and on a synthetic
corpus from bench/gen.py (held-out documents), without spaCy so they run everywhere.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from styleprofile.calibration import (
    CALIBRATION_LENGTHS,
    MIN_CALIBRATION_PIECES,
    MIN_JUDGED_WORDS,
    Lengths,
    chunk_level,
    chunk_likeness_level,
    cut,
    plan_pieces,
)
from styleprofile.cli import main
from styleprofile.profile import (
    VERSION,
    Chunk,
    build_reference,
    load_chunks,
    score,
    window,
    write_report,
)
from styleprofile.surface import prose, words
from styleprofile.weighting import (
    DISTANCE_WORDS,
    TOO_SHORT,
    mean_ceiling,
    pooled_ceiling,
)

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples" / "writer"
DRAFTS = ROOT / "examples" / "llm-drafts"
DRAFT = ROOT / "examples" / "draft.md"
WINDOW = 500
# The paragraph from the repo review: the writer's own voice, which scored "very different,
# Delta 2.76, a few LLM traits" against 500-word windows.
FENCE = (
    "I have walked this fence line every spring for thirty years. The stones shift; the posts "
    "do not. My father said a good fence was a promise you kept with your neighbor, and he "
    "kept his."
)
# "clearly different" or worse, and "leans LLM" or worse.
CLEARLY = 2
LEANS = 2
MAX_FALSE_POSITIVES = 0.05
MIN_DETECTION = 0.8


def _words(text: str) -> int:
    return len(words(prose(text).text))


@cache
def _examples_reference() -> dict[str, Any]:
    return build_reference(
        window(load_chunks([str(WRITER)]), WINDOW),
        parser=None,
        contrast=window(load_chunks([str(DRAFTS)]), WINDOW),
        settings={"window_words": WINDOW},
    )


def _reference() -> dict[str, Any]:
    return copy.deepcopy(_examples_reference())


def _score(texts: Sequence[str], reference: dict[str, Any]) -> dict[str, Any]:
    chunks = [Chunk(f"t{index}", f"t{index}", text) for index, text in enumerate(texts)]
    return score(chunks, reference, parser=None, settings={"window_words": WINDOW})


# Cutting windows into pieces.


def test_pieces_are_near_their_length_and_lose_no_text() -> None:
    text = (WRITER / "fence-lines.md").read_text(encoding="utf-8")
    total = _words(text)
    for length in CALIBRATION_LENGTHS:
        pieces = cut(text, length)
        assert len(pieces) == round(total / length)
        sizes = [_words(piece) for piece in pieces]
        assert sum(sizes) == total
        assert all(0.5 * length <= size <= 1.6 * length for size in sizes), (length, sizes)
    # A chunk shorter than one and a half pieces is not cut.
    assert cut(FENCE, 75) == []


def test_pieces_cut_at_sentences_and_keep_blocks_whole() -> None:
    paragraph = " ".join(f"Sentence number {index} is here." for index in range(40))
    listing = "\n".join(f"- item {index} of the list" for index in range(12))
    code = "```python\nx = 1\ny = 2\n```"
    text = f"# Title\n\n{paragraph}\n\n{listing}\n\n{code}\n\n{paragraph}"
    pieces = cut(text, 75)
    assert len(pieces) > 2
    for piece in pieces:
        # Every piece starts at a sentence or a block, never mid-sentence.
        assert piece.startswith(("Sentence number", "- item", "```", "# Title"))
    joined = "\n\n".join(pieces)
    assert listing in joined and code in joined


def test_pieces_prefer_paragraph_breaks() -> None:
    first = " ".join(["Alpha beta gamma delta."] * 18)  # 72 words
    second = " ".join(["Epsilon zeta eta theta."] * 20)  # 80 words
    pieces = cut(f"{first}\n\n{second}", 75)
    # The exact halfway point falls inside the second paragraph; the break is taken instead.
    assert pieces == [first, second]


def test_calibration_cuts_a_bounded_share_of_a_large_corpus() -> None:
    texts = [" ".join(["Word after word goes here."] * 100)] * 100  # 100 chunks of 500 words
    sizes = [500] * 100
    planned = plan_pieces(texts, sizes, CALIBRATION_LENGTHS, 10_000)
    assert {index for index, _, _ in planned} == set(range(0, 100, 5))
    # Only lengths the median chunk holds one and a half times are cut.
    assert {length for _, length, _ in plan_pieces(texts, [200] * 100, (75, 150, 300), 10**9)} == {
        75
    }


# What a reference stores.


def test_reference_stores_calibration_by_length() -> None:
    reference = _reference()
    assert reference["version"] == VERSION == 6
    calibration = reference["calibration"]
    assert calibration["chunk_words"] == pytest.approx(640, abs=20)
    lengths = calibration["by_length"]
    assert set(lengths) == {"75", "150", "300"}
    for length, entry in lengths.items():
        assert entry["documents"] == 7
        assert entry["words"] == pytest.approx(int(length), rel=0.15)
        if entry["pieces"] >= MIN_CALIBRATION_PIECES:
            assert set(entry) >= {"reliability", "delta", "likeness", "contrast_pieces"}
            # Shorter texts vary more by chance than whole windows do.
            assert entry["delta"]["p95"] > calibration["delta"]["p95"]
        else:
            assert "delta" not in entry and "reliability" not in entry
    # 300-word pieces: 2 per document, too few to calibrate on 7 documents.
    assert lengths["300"]["pieces"] < MIN_CALIBRATION_PIECES
    assert len(json.dumps(lengths)) < 20_000


def test_calibration_interpolates_between_lengths() -> None:
    lengths = Lengths(_reference())
    window_p95 = _reference()["calibration"]["delta"]["p95"]
    at = {words: lengths.at(words) for words in (75, 100, 162, 300, 640, 2000)}
    assert all(item.judged for item in at.values())
    p95 = [(item.delta or {})["p95"] for item in at.values()]
    assert p95 == sorted(p95, reverse=True), "shorter texts get wider ranges"
    # At and above the windows' own length, the windows' range and rms apply unchanged.
    assert p95[-2] == pytest.approx(window_p95)
    assert p95[-1] == pytest.approx(window_p95)
    assert at[640].scale == {}
    # Shorter texts swing more, so their arrows are scaled down.
    assert at[100].scale and all(factor > 1 for factor in at[100].scale.values())


def test_pooled_ceiling_is_mean_ceiling_for_equal_ranges() -> None:
    stats = {"median": 0.7, "p95": 1.1}
    for count in (1, 4, 9):
        assert pooled_ceiling([stats] * count) == pytest.approx(mean_ceiling(stats, count))
    short, long = {"median": 1.1, "p95": 2.1}, {"median": 0.7, "p95": 0.95}
    mixed = pooled_ceiling([short, long])
    assert mixed == pytest.approx(0.9 + (1.0**2 + 0.25**2) ** 0.5 / 2)


# Too short to judge.


def test_the_review_paragraph_abstains(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _words(FENCE) == 36
    report = _score([FENCE], _reference())
    verdict = report["reference"]["verdict"]
    assert verdict["judged"] is False
    assert verdict["verdict"] == TOO_SHORT
    assert verdict["delta"]["level"] is None
    assert verdict["likeness"]["verdict"] == TOO_SHORT
    assert all(area["level"] is None for area in verdict["by_group"].values())
    assert "under 75 words" in verdict["reason"]

    reference = tmp_path / "writer.json"
    write_report(_reference(), reference)
    sample = tmp_path / "fence.md"
    sample.write_text(FENCE, encoding="utf-8")
    assert main(["score", "-q", str(sample), str(reference)]) == 0
    assert capsys.readouterr().out == f"{sample}: too short to judge (36 words)\n"

    assert main(["score", str(sample), str(reference), "--no-syntax"]) == 0
    text = capsys.readouterr().out
    assert "Overall: too short to judge (36 words)" in text
    assert "indicative only" in text
    area_block = text.split("By area", 1)[1].split("\n\n", 1)[0]
    assert not any(word in area_block for word in DISTANCE_WORDS)
    assert "Indicative LLM signals" in text or "Strongest" not in text

    assert main(["score", "--json", str(sample), str(reference)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["reference"]["verdict"]["verdict"] == TOO_SHORT


def test_a_length_with_too_few_pieces_abstains_and_says_why() -> None:
    # Three 400-word documents: 15 pieces of 75 words, too few to calibrate that length.
    texts = [
        " ".join(f"Document {doc} sentence {index} runs about eight words." for index in range(50))
        for doc in range(3)
    ]
    chunks = [Chunk(f"d{doc}", f"d{doc}.md", text) for doc, text in enumerate(texts)]
    reference = build_reference(chunks, parser=None)
    entry = reference["calibration"]["by_length"]["75"]
    assert entry["pieces"] < MIN_CALIBRATION_PIECES
    at = Lengths(reference).at(100)
    assert not at.judged
    assert at.reason is not None
    assert f"{entry['pieces']} pieces of about 75 words" in at.reason
    assert "add more of the writer's text" in at.reason


def test_each_chunk_is_judged_at_its_own_length() -> None:
    essay = (WRITER / "old-maps.md").read_text(encoding="utf-8")
    report = _score([essay, FENCE], _reference())
    long, short = report["chunks"]
    assert long["reference"]["calibration"]["judged"]
    assert not short["reference"]["calibration"]["judged"]
    assert chunk_level(short) is None and chunk_likeness_level(short) is None
    verdict = report["reference"]["verdict"]
    # The short chunk is left out of the verdict and the means, with a warning.
    assert verdict["judged"] and verdict["chunks_judged"] == 1
    assert report["reference"]["delta_mean"] == pytest.approx(long["reference"]["delta"])
    assert any("1 chunk(s) are too short to judge" in warning for warning in report["warnings"])


def test_without_shorter_lengths_only_texts_near_a_window_are_judged() -> None:
    # Windows too short to cut into pieces (or a profile from before length calibration).
    reference = _reference()
    reference["calibration"]["by_length"] = {}
    lengths = Lengths(reference)
    assert lengths.at(330).judged, "half a window or more reads the windows' own range"
    at = lengths.at(200)
    assert not at.judged
    assert at.reason is not None and "rebuild it, with windows of at least 113 words" in at.reason


def test_the_demo_draft_gets_a_sensible_verdict() -> None:
    report = _score([DRAFT.read_text(encoding="utf-8")], _reference())
    verdict = report["reference"]["verdict"]
    assert verdict["judged"]
    # Mostly the writer's voice with two paragraphs that slip into the LLM register.
    assert verdict["delta"]["level"] <= 1
    assert verdict["likeness"]["level"] <= 1


# Acceptance: false positives and detection at each length, on held-out text.


def _rates(reference: dict[str, Any], texts: Sequence[str], length: int) -> tuple[int, int, int]:
    """(pieces judged, clearly different or worse, flagged) for ``texts`` cut to ``length``."""
    pieces = [piece for text in texts for piece in cut(text, length)]
    if not pieces:
        return 0, 0, 0
    report = _score(pieces, reference)
    judged = clearly = flagged = 0
    for row in report["chunks"]:
        level = chunk_level(row)
        if level is None:
            continue
        judged += 1
        likeness = chunk_likeness_level(row) or 0
        clearly += level >= CLEARLY
        flagged += level >= CLEARLY or likeness >= LEANS
    return judged, clearly, flagged


def _accumulate(totals: dict[int, list[int]], length: int, counts: tuple[int, int, int]) -> None:
    for index, count in enumerate(counts):
        totals.setdefault(length, [0, 0, 0])[index] += count


Rates = tuple[dict[int, list[int]], dict[int, list[int]]]


@pytest.fixture(scope="module")
def examples_rates() -> Rates:
    """Leave one document out: each essay scored against the other six, and each LLM draft
    against a contrast of the other four."""
    writer = load_chunks([str(WRITER)])
    drafts = load_chunks([str(DRAFTS)])
    own: dict[int, list[int]] = {}
    llm: dict[int, list[int]] = {}
    for held in writer:
        rest = [chunk for chunk in writer if chunk is not held]
        reference = build_reference(
            window(rest, WINDOW), parser=None, contrast=window(drafts, WINDOW)
        )
        for length in CALIBRATION_LENGTHS:
            _accumulate(own, length, _rates(reference, [held.text], length))
    for held in drafts:
        rest = [chunk for chunk in drafts if chunk is not held]
        reference = build_reference(
            window(writer, WINDOW), parser=None, contrast=window(rest, WINDOW)
        )
        _accumulate(llm, 150, _rates(reference, [held.text], 150))
    return own, llm


def _generate(out: Path) -> Path:
    """The benchmark's medium corpus (bench/gen.py), written to ``out``."""
    spec = importlib.util.spec_from_file_location("bench_gen", ROOT / "bench" / "gen.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module.generate("medium", out)


@pytest.fixture(scope="module")
def synthetic_rates(tmp_path_factory: pytest.TempPathFactory) -> Rates:
    """A synthetic writer of 80 documents with 20 contrast drafts, scored on 40 held-out
    documents and 10 held-out drafts."""
    folder = _generate(tmp_path_factory.mktemp("corpus"))
    writer = sorted((folder / "writer").glob("*.md"))
    drafts = sorted((folder / "contrast").glob("*.md"))
    reference = build_reference(
        window(load_chunks([str(path) for path in writer[:80]]), WINDOW),
        parser=None,
        contrast=window(load_chunks([str(path) for path in drafts[:20]]), WINDOW),
    )
    held_writer = [path.read_text(encoding="utf-8") for path in writer[80:120]]
    held_drafts = [path.read_text(encoding="utf-8") for path in drafts[20:30]]
    own: dict[int, list[int]] = {}
    llm: dict[int, list[int]] = {}
    for length in CALIBRATION_LENGTHS:
        _accumulate(own, length, _rates(reference, held_writer, length))
    _accumulate(llm, 150, _rates(reference, held_drafts, 150))
    return own, llm


@pytest.mark.parametrize("corpus", ["examples_rates", "synthetic_rates"])
@pytest.mark.parametrize("length", CALIBRATION_LENGTHS)
def test_held_out_writer_text_is_rarely_clearly_different(
    corpus: str, length: int, request: pytest.FixtureRequest
) -> None:
    judged, clearly, _ = request.getfixturevalue(corpus)[0][length]
    assert judged >= 10, "too few judged pieces to measure a rate"
    assert clearly / judged <= MAX_FALSE_POSITIVES, f"{clearly} of {judged} pieces"


@pytest.mark.parametrize("corpus", ["examples_rates", "synthetic_rates"])
def test_llm_drafts_cut_to_150_words_are_still_flagged(
    corpus: str, request: pytest.FixtureRequest
) -> None:
    judged, _, flagged = request.getfixturevalue(corpus)[1][150]
    assert judged >= 10
    assert flagged / judged >= MIN_DETECTION, f"{flagged} of {judged} pieces"


def test_min_judged_words_is_the_shortest_length() -> None:
    assert min(CALIBRATION_LENGTHS) == MIN_JUDGED_WORDS


def test_pieces_are_parsed_once_with_their_windows() -> None:
    pytest.importorskip("spacy")
    from styleprofile.syntax import SyntaxUnavailableError, load_parser

    try:
        parser = load_parser()
    except SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")
    parsed: list[str] = []
    pipe = parser.nlp.pipe

    def counting(texts: Any, **kwargs: Any) -> Any:
        texts = list(texts)
        parsed.extend(texts)
        return pipe(texts, **kwargs)

    parser.nlp.pipe = counting
    try:
        writer = window(load_chunks([str(WRITER)]), WINDOW)
        reference = build_reference(writer, parser=parser)
    finally:
        parser.nlp.pipe = pipe
    assert len(parsed) == len(writer)
    lengths = reference["calibration"]["by_length"]
    assert "syntax" in lengths["75"]["reliability"]
