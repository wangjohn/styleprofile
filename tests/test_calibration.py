"""Length-aware verdicts: calibration for shorter texts, and "too short to judge".

The acceptance tests measure, on text the reference never saw, how often the writer's own
pieces of about 75, 150 and 300 words read "clearly different" or worse (at most 5%), how
often they land above the stored 95% bound, and how often LLM drafts cut to about 150 words
are still flagged (at least 80%). The pieces are cut by the test's own cutters, not by the
one calibration uses: whole paragraphs, and greedy runs of sentences that split paragraphs.
They run on the sample corpus in examples/ (leaving one document out at a time) and on a
synthetic corpus from bench/gen.py (held-out documents), without spaCy so they run
everywhere.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile.calibration import (
    CALIBRATION_LENGTHS,
    FLOOR_DOCUMENTS,
    MIN_CALIBRATION_DOCUMENTS,
    MIN_CALIBRATION_PIECES,
    MIN_JUDGED_WORDS,
    SIMILARITY_FLOOR,
    Lengths,
    chunk_level,
    chunk_likeness_level,
    cut,
    enough,
    plan_pieces,
    similarity,
    verdict,
)
from styleprofile.cli import main
from styleprofile.core import StyleProfileError
from styleprofile.profile import (
    VERSION,
    Chunk,
    build_reference,
    load_chunks,
    load_reference,
    score,
    window,
    write_report,
)
from styleprofile.schema import ReferenceReport, ScoredChunk, ScoreReport
from styleprofile.surface import prose, words
from styleprofile.weighting import (
    DISTANCE_WORDS,
    MIN_CEILING,
    RUN_SIMILARITY,
    TOO_SHORT,
    centre,
    delta_level,
    effective_count,
    intraclass_correlation,
    mean_ceiling,
    pooled_ceiling,
    quantile,
    upper_mean,
    upper_quantile,
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
# A 110-word passage in the same voice, from the review.
BEES = (
    "My neighbor keeps bees. He has kept them since before I moved here, which is a long time "
    "now, and every August he leaves a jar of honey on my porch without a note. I have never "
    "asked him to. I don't think he'd know what to say if I thanked him properly, so I don't. "
    "I leave a loaf of bread on his porch in December instead.\n\n"
    "The bees don't care about any of this. They go where the clover is, and in a dry year "
    "they go farther, and some mornings I find one drowned in the dog's water dish and I fish "
    "it out with a leaf."
)
# "somewhat different" or worse, "clearly different" or worse, and "leans LLM" or worse.
SOMEWHAT = 1
CLEARLY = 2
LEANS = 2
MAX_FALSE_POSITIVES = 0.05
# The share of the writer's own pieces above the stored 95% bound. The bound is an upper
# confidence bound, so it is usually met; this allows for the sampling noise of ~150 pieces.
MAX_ABOVE_BOUND = 0.08
MIN_DETECTION = 0.8


def _words(text: str) -> int:
    return len(words(prose(text).text))


def _body(path: Path) -> str:
    """A generated document without its title line."""
    return path.read_text(encoding="utf-8").split("\n", 1)[1]


@cache
def _examples_reference() -> ReferenceReport:
    return build_reference(
        window(load_chunks([str(WRITER)]), WINDOW),
        parser=None,
        contrast=window(load_chunks([str(DRAFTS)]), WINDOW),
        settings={"window_words": WINDOW},
    )


def _reference() -> ReferenceReport:
    return copy.deepcopy(_examples_reference())


def _score(texts: Sequence[str], reference: ReferenceReport) -> ScoreReport:
    chunks = [Chunk(f"t{index}", f"t{index}", text) for index, text in enumerate(texts)]
    return score(chunks, reference, parser=None, settings={"window_words": WINDOW})


@cache
def _bench() -> Any:
    """The benchmark's corpus generator, bench/gen.py."""
    spec = importlib.util.spec_from_file_location("bench_gen", ROOT / "bench" / "gen.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def _generate(out: Path) -> Path:
    """The benchmark's medium corpus (bench/gen.py), written to ``out``."""
    return _bench().generate("medium", out)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _generate(tmp_path_factory.mktemp("corpus"))


# The test's own cutters, independent of calibration's ``cut``.

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _paragraphs(text: str) -> list[list[str]]:
    """Each prose paragraph as its sentences; lists, quotes and tables whole; no headings."""
    found = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        if block.startswith(("-", "*", ">", "|", "1.")):
            found.append([block])
        else:
            found.append([part for part in _SENTENCE.split(" ".join(block.split())) if part])
    return found


def cut_sentences(text: str, length: int) -> list[str]:
    """Greedy runs of sentences, splitting paragraphs wherever ``length`` falls."""
    pieces: list[str] = []
    current: list[list[str]] = [[]]
    count = 0
    for paragraph in _paragraphs(text):
        for sentence in paragraph:
            size = len(sentence.split())
            if count >= 0.6 * length and abs(count + size - length) > abs(count - length):
                pieces.append("\n\n".join(" ".join(part) for part in current if part))
                current, count = [[]], 0
            current[-1].append(sentence)
            count += size
        current.append([])
    if count >= 0.6 * length:
        pieces.append("\n\n".join(" ".join(part) for part in current if part))
    return pieces


def cut_paragraphs(text: str, length: int) -> list[str]:
    """Whole paragraphs, as many as come closest to ``length``."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in _paragraphs(text):
        joined = " ".join(paragraph)
        size = len(joined.split())
        if current and abs(count + size - length) > abs(count - length):
            pieces.append("\n\n".join(current))
            current, count = [], 0
        current.append(joined)
        count += size
    if current:
        pieces.append("\n\n".join(current))
    return [piece for piece in pieces if 0.6 * length <= len(piece.split()) <= 1.6 * length]


CUTTERS: dict[str, Callable[[str, int], list[str]]] = {
    "sentences": cut_sentences,
    "paragraphs": cut_paragraphs,
}


# Cutting windows into pieces for calibration.


def test_pieces_are_near_their_length_and_lose_no_text() -> None:
    text = (WRITER / "fence-lines.md").read_text(encoding="utf-8")
    total = _words(text)
    for length in CALIBRATION_LENGTHS:
        for excerpts in (False, True):
            pieces = cut(text, length, excerpts=excerpts)
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


def test_pieces_prefer_paragraph_breaks_unless_cut_as_excerpts() -> None:
    first = " ".join(["Alpha beta gamma delta."] * 18)  # 72 words
    second = " ".join(["Epsilon zeta eta theta."] * 20)  # 80 words
    # The exact halfway point falls inside the second paragraph; the break is taken instead.
    assert cut(f"{first}\n\n{second}", 75) == [first, second]
    excerpt, _ = cut(f"{first}\n\n{second}", 75, excerpts=True)
    assert excerpt.startswith(first) and len(excerpt) > len(first)


def test_calibration_samples_pieces_from_every_chunk() -> None:
    texts = [" ".join(["Word after word goes here."] * 100)] * 100  # 100 chunks of 500 words
    planned = plan_pieces(texts, [500] * 100, CALIBRATION_LENGTHS, 10_000)
    for length in CALIBRATION_LENGTHS:
        chosen = [(index, text) for index, size, text in planned if size == length]
        # Every k-th piece, not every k-th chunk, so the pieces span the corpus.
        indexes = {index for index, _ in chosen}
        assert len(indexes) >= 30 and max(indexes) >= 90
        assert sum(len(text.split()) for _, text in chosen) <= 1.1 * 10_000
    # Only lengths the median chunk holds one and a half times are cut.
    assert {length for _, length, _ in plan_pieces(texts, [200] * 100, (75, 150, 300), 10**9)} == {
        75
    }


def test_a_few_large_documents_without_windows_are_calibrated(corpus: Path) -> None:
    files = sorted((corpus / "writer").glob("*.md"))
    books = [
        sp.Text(
            "\n\n".join(path.read_text(encoding="utf-8") for path in files[i::6][:8]), f"book{i}"
        )
        for i in range(6)
    ]
    profile = sp.build(books, sp.Settings(window_words=0, syntax=False))
    lengths = profile.report["calibration"]["by_length"]
    # Every book contributes pieces; six are enough even counted as alike within a book.
    assert all(entry["documents"] == 6 and enough(entry) for entry in lengths.values())
    draft = "\n\n".join(_body(path) for path in files[100:105])
    assert 4000 < _words(draft) < 6500
    result = profile.score(sp.Text(draft), syntax=False)
    assert result.judged, result.reason


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
        if enough(entry):
            assert set(entry) >= {"reliability", "delta", "likeness", "effective"}
            delta = entry["delta"]
            assert delta["median"] <= min(delta["p95"], delta["p99"])
            assert "max" not in delta and "p99" not in delta["by_group"]["voice"]
            # Shorter texts vary more by chance than whole windows do.
            assert entry["delta"]["p95"] > calibration["delta"]["p95"]
        else:
            assert "delta" not in entry and "reliability" not in entry
    # 300-word pieces: 4 per document, worth fewer than 20 independent ones on 7 documents.
    assert not enough(lengths["300"]) and lengths["300"]["effective"] < MIN_CALIBRATION_PIECES
    assert len(json.dumps(lengths)) < 20_000


def test_the_bound_counts_documents_not_pieces() -> None:
    # Ten documents of ten pieces each, alike within a document: worth about ten values.
    alike = [float(doc) for doc in range(10) for _ in range(10)]
    groups = [str(doc) for doc in range(10) for _ in range(10)]
    assert effective_count(alike, groups) == pytest.approx(10, rel=0.05)
    # The same values in a hundred documents are worth a hundred.
    assert effective_count(alike, [str(index) for index in range(100)]) == 100
    spread = [float(index % 37) for index in range(100)]
    assert upper_quantile(spread, groups, 0.95) >= quantile(spread, 0.95)
    assert upper_quantile(alike, groups, 0.95) == max(alike)


def test_similarity_is_pooled_over_lengths_and_floored_for_few_documents() -> None:
    # Pieces alike within their document at one length, unrelated at another: the pooled
    # estimate lies between, weighted by pieces.
    alike = (
        [float(doc) for doc in range(12) for _ in range(4)],
        [str(doc) for doc in range(12) for _ in range(4)],
    )
    mixed = ([float(index % 5) for index in range(48)], alike[1])
    pooled = similarity([alike, mixed], 12)
    assert 0 < pooled < 1
    assert pooled == pytest.approx(
        ((intraclass_correlation(*alike) or 0) + (intraclass_correlation(*mixed) or 0)) / 2
    )
    # With fewer than FLOOR_DOCUMENTS documents it is never below the floor.
    assert similarity([mixed], FLOOR_DOCUMENTS - 1) >= SIMILARITY_FLOOR
    assert similarity([mixed], FLOOR_DOCUMENTS) < SIMILARITY_FLOOR


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
    stats = {"median": 0.7, "mean": 0.75, "p95": 1.15}
    for count in (1, 4, 9):
        assert pooled_ceiling([stats] * count) == pytest.approx(mean_ceiling(stats, count))
        assert mean_ceiling(stats, count) == pytest.approx(
            0.75 + 0.4 * (RUN_SIMILARITY + (1 - RUN_SIMILARITY) / count) ** 0.5
        )
    short, long = (
        {"median": 1.1, "mean": 1.2, "p95": 2.2},
        {"median": 0.7, "mean": 0.7, "p95": 0.95},
    )
    mixed = pooled_ceiling([short, long])
    squares, total = 1.0**2 + 0.25**2, (1.0 + 0.25) ** 2
    spread = ((1 - RUN_SIMILARITY) * squares + RUN_SIMILARITY * total) ** 0.5 / 2
    assert mixed == pytest.approx(0.95 + spread)


def test_one_chunk_is_read_against_its_own_bound() -> None:
    """Pooling changes nothing for a single chunk: its bound is its own 95% bound."""
    for stats in ({"median": 0.7, "mean": 0.8, "p95": 1.3}, {"median": 0.7, "p95": 1.3}):
        assert pooled_ceiling([stats]) == pytest.approx(1.3)
    assert pooled_ceiling([{"median": 0.1, "mean": 0.12, "p95": 0.2}]) == 0.5  # the floor


def test_a_long_run_is_read_against_the_mean_not_the_median() -> None:
    """Held-out Deltas are skewed to the right, so their mean sits above their median; the
    bound for a mean over many chunks settles near the mean, never at the median, and keeps
    a share of one chunk's spread that the chunks of one run have in common."""
    stats = {"median": 0.8, "mean": 0.9, "p95": 1.5}
    ceiling = pooled_ceiling([stats] * 10_000)
    assert ceiling is not None
    assert ceiling == pytest.approx(0.9 + 0.6 * RUN_SIMILARITY**0.5, abs=1e-3)
    # A range stored without a mean (built before it was stored) is read at its median.
    assert centre({"median": 0.8, "p95": 1.5}) == 0.8
    # The centre never lies above the 95% bound.
    assert centre({"median": 0.8, "mean": 1.7, "p95": 1.5}) == 1.5


def test_the_mean_is_an_upper_confidence_bound() -> None:
    """The stored centre is the held-out mean plus 1.28 standard errors, with the standard
    error from the effective number of pieces, so pieces from few documents set it higher."""
    values = [0.5, 0.7, 0.9, 1.1, 1.3, 0.6, 0.8, 1.0, 1.2, 1.4]
    spread = statistics.stdev(values) * 1.2816
    apart = [f"d{index}" for index in range(10)]
    assert upper_mean(values, apart) == pytest.approx(statistics.fmean(values) + spread / 10**0.5)
    together = ["a"] * 5 + ["b"] * 5
    assert upper_mean(values, together, icc=0.5) == pytest.approx(
        statistics.fmean(values) + spread / effective_count(values, together, 0.5) ** 0.5
    )
    assert upper_mean(values, together, icc=0.5) > upper_mean(values, apart)
    reference = _reference()
    stored = reference["calibration"]["by_length"]["75"]["delta"]
    assert stored["median"] < stored["mean"] < stored["p95"]


def test_a_profile_from_before_length_calibration_is_refused(tmp_path: Path) -> None:
    for missing in ("chunk_words", "by_length"):
        # Edited to lack a key, so no longer a ReferenceReport.
        reference: Any = _reference()
        del reference["calibration"][missing]
        path = tmp_path / f"no-{missing}.json"
        write_report(reference, path)
        with pytest.raises(StyleProfileError, match="before length-aware verdicts") as error:
            load_reference(path)
        assert error.value.code == "outdated"
        with pytest.raises(StyleProfileError, match="rebuild it"):
            _score([BEES], reference)


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
    assert "under 75 words" in (verdict["reason"] or "")

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


def test_a_reference_from_one_document_does_not_judge_short_texts(corpus: Path) -> None:
    files = sorted((corpus / "writer").glob("*.md"))
    one = "\n\n".join(path.read_text(encoding="utf-8") for path in files[:20])
    profile = sp.build(sp.Text(one, "all.md"), sp.Settings(syntax=False))
    assert profile.report["word_count"] > 15_000 and "calibration" not in profile.report
    # The writer's own paragraphs: none may read "somewhat" or worse; all abstain.
    results = [
        profile.score(sp.Text(piece), syntax=False)
        for path in files[40:60]
        for length in (75, 150)
        for piece in cut_paragraphs(path.read_text(encoding="utf-8"), length)
    ]
    assert len(results) > 40 and {result.verdict for result in results} == {sp.Verdict.TOO_SHORT}
    reason = results[-1].reason
    assert reason is not None and "add more of the writer's documents" in reason
    # Half a window or more is still judged, against fixed steps as before.
    long = cut_paragraphs(_body(files[60]), 400)[0]
    assert profile.score(sp.Text(long), syntax=False).judged


def _prefix(text: str, count: int) -> str | None:
    """The first ``count`` words of a document's prose paragraphs, cut mid-sentence."""
    body = "\n\n".join(
        block for block in re.split(r"\n\s*\n", text) if block.strip() and not block.startswith("#")
    )
    ends = [match.end() for match in re.finditer(r"\S+", body)]
    return body[: ends[count - 1]] if len(ends) >= count else None


@pytest.mark.parametrize("first", [0, 6])
def test_texts_shorter_than_a_window_get_a_wider_range_without_shorter_lengths(
    corpus: Path, first: int
) -> None:
    # Two documents: calibrated windows, but no shorter length. Between half a window and a
    # window, the windows' range is widened by sqrt(window / words); once it read 35% of
    # the writer's own 300-word passages as "somewhat different" or worse.
    files = sorted((corpus / "writer").glob("*.md"))
    profile = sp.build([str(path) for path in files[first : first + 2]], sp.Settings(syntax=False))
    lengths = profile.report["calibration"]["by_length"]
    assert not any(enough(entry) for entry in lengths.values())
    for count in (300, 350):
        texts = [text for path in files[100:160] if (text := _prefix(_body(path), count))]
        results = [profile.score(sp.Text(text), syntax=False) for text in texts]
        judged = [result for result in results if result.judged]
        assert len(judged) >= 50
        somewhat = sum(result.verdict is not sp.Verdict.CLOSE for result in judged)
        assert somewhat / len(judged) <= 0.1, f"{somewhat} of {len(judged)} at {count} words"


def test_a_length_from_too_few_documents_abstains_and_says_why() -> None:
    # Two long documents: plenty of pieces, but only two documents behind them.
    texts = [
        "\n\n".join(
            " ".join(
                f"Document {doc} sentence {index} runs about eight words." for index in range(5)
            )
            for _ in range(50)
        )
        for doc in range(2)
    ]
    chunks = [Chunk(f"d{doc}", f"d{doc}.md", text) for doc, text in enumerate(texts)]
    reference = build_reference(window(chunks, WINDOW), parser=None)
    entry = reference["calibration"]["by_length"]["75"]
    assert entry["pieces"] >= MIN_CALIBRATION_PIECES and entry["documents"] == 2
    assert not enough(entry)
    at = Lengths(reference).at(100)
    assert not at.judged and at.reason is not None
    assert "from 2 documents" in at.reason
    assert f"from {MIN_CALIBRATION_DOCUMENTS} or more documents" in at.reason
    assert "add more of the writer's documents" in at.reason


def test_each_chunk_is_judged_at_its_own_length() -> None:
    essay = (WRITER / "old-maps.md").read_text(encoding="utf-8")
    report = _score([essay, FENCE], _reference())
    long, short = report["chunks"]
    assert long["reference"]["calibration"]["judged"]
    assert not short["reference"]["calibration"]["judged"]
    assert chunk_level(short) is None and chunk_likeness_level(short) is None
    verdict = report["reference"]["verdict"]
    # The short chunk is left out of the verdict and the means, and the warning names it.
    assert verdict["judged"] and verdict["chunks_judged"] == 1
    assert report["reference"]["delta_mean"] == pytest.approx(long["reference"]["delta"])
    assert (
        "left out of the verdict and the means as too short to judge: t1 (36 words, under 75)"
        in report["warnings"]
    )


def test_chunks_left_out_add_nothing_to_the_differences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = tmp_path / "writer.json"
    write_report(_reference(), reference)
    for name, text in (("fence.md", FENCE), ("bees.md", BEES)):
        (tmp_path / name).write_text(text, encoding="utf-8")
    samples = [str(tmp_path / "fence.md"), str(tmp_path / "bees.md"), str(DRAFT)]

    def differences(paths: list[str]) -> str:
        assert main(["score", "--no-syntax", *paths, str(reference)]) == 0
        return capsys.readouterr().out.split("Biggest differences", 1)[1].split("\n\n", 1)[0]

    assert differences(samples) == differences(samples[1:])
    # -q gives each document its own line (plan PR 7): the short one abstains, and last.
    assert main(["score", "-q", *samples, str(reference)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[-1] == f"{tmp_path / 'fence.md'}: too short to judge (36 words)"
    assert not any("too short" in line for line in lines[:-1])


def test_without_shorter_lengths_only_texts_near_a_window_are_judged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Windows too short to cut into pieces.
    reference = _reference()
    reference["calibration"]["by_length"] = {}
    lengths = Lengths(reference)
    assert lengths.at(330).judged, "half a window or more reads the windows' own range"
    at = lengths.at(200)
    assert not at.judged
    assert at.reason is not None and "rebuild it with window_words 113 or more" in at.reason
    # The command line names its flag instead.
    path = tmp_path / "writer.json"
    write_report(reference, path)
    sample = tmp_path / "part.md"
    sample.write_text(cut_paragraphs(DRAFT.read_text(encoding="utf-8"), 200)[0], encoding="utf-8")
    assert main(["score", "--no-syntax", str(sample), str(path)]) == 0
    assert "rebuild it with --window-words 113 or more" in capsys.readouterr().out


def test_the_demo_draft_gets_a_sensible_verdict() -> None:
    report = _score([DRAFT.read_text(encoding="utf-8")], _reference())
    verdict = report["reference"]["verdict"]
    assert verdict["judged"]
    # Mostly the writer's voice with two paragraphs that slip into the LLM register.
    assert verdict["delta"]["level"] <= 1
    assert verdict["likeness"]["level"] <= 1


# Acceptance: false positives and detection at each length, on held-out text.

# (cutter, length) -> [judged, above the bound, clearly different or worse, flagged]
Counts = dict[tuple[str, int], list[int]]


def _count(reference: ReferenceReport, texts: Sequence[str], counts: Counts) -> None:
    for name, cutter in CUTTERS.items():
        for length in CALIBRATION_LENGTHS:
            pieces = [piece for text in texts for piece in cutter(text, length)]
            if not pieces:
                continue
            report = _score(pieces, reference)
            totals = counts.setdefault((name, length), [0, 0, 0, 0])
            for row in report["chunks"]:
                level = chunk_level(row)
                if level is None:
                    continue
                likeness = chunk_likeness_level(row) or 0
                totals[0] += 1
                totals[1] += level >= SOMEWHAT
                totals[2] += level >= CLEARLY
                totals[3] += level >= CLEARLY or likeness >= LEANS


Rates = tuple[Counts, Counts]


@pytest.fixture(scope="module")
def examples_rates() -> Rates:
    """Leave one document out: each essay scored against the other six, and each LLM draft
    against a contrast of the other four."""
    writer = load_chunks([str(WRITER)])
    drafts = load_chunks([str(DRAFTS)])
    own: Counts = {}
    llm: Counts = {}
    for held in writer:
        rest = [chunk for chunk in writer if chunk is not held]
        reference = build_reference(
            window(rest, WINDOW), parser=None, contrast=window(drafts, WINDOW)
        )
        _count(reference, [held.text], own)
    for held in drafts:
        rest = [chunk for chunk in drafts if chunk is not held]
        reference = build_reference(
            window(writer, WINDOW), parser=None, contrast=window(rest, WINDOW)
        )
        _count(reference, [held.text], llm)
    return own, llm


@pytest.fixture(scope="module")
def synthetic_reference(corpus: Path) -> ReferenceReport:
    """A synthetic writer of 80 documents with 20 contrast drafts."""
    writer = sorted((corpus / "writer").glob("*.md"))
    drafts = sorted((corpus / "contrast").glob("*.md"))
    return build_reference(
        window(load_chunks([str(path) for path in writer[:80]]), WINDOW),
        parser=None,
        contrast=window(load_chunks([str(path) for path in drafts[:20]]), WINDOW),
    )


@pytest.fixture(scope="module")
def synthetic_rates(corpus: Path, synthetic_reference: ReferenceReport) -> Rates:
    """The synthetic writer scored on 40 held-out documents and 10 held-out drafts."""
    writer = sorted((corpus / "writer").glob("*.md"))
    drafts = sorted((corpus / "contrast").glob("*.md"))
    reference = synthetic_reference
    own: Counts = {}
    llm: Counts = {}
    _count(reference, [path.read_text(encoding="utf-8") for path in writer[80:120]], own)
    _count(reference, [path.read_text(encoding="utf-8") for path in drafts[20:30]], llm)
    return own, llm


@pytest.mark.parametrize("corpus_rates", ["examples_rates", "synthetic_rates"])
@pytest.mark.parametrize("cutter", list(CUTTERS))
@pytest.mark.parametrize("length", CALIBRATION_LENGTHS)
def test_held_out_writer_text_is_rarely_clearly_different(
    corpus_rates: str, cutter: str, length: int, request: pytest.FixtureRequest
) -> None:
    judged, above, clearly, _ = request.getfixturevalue(corpus_rates)[0][(cutter, length)]
    assert judged >= 10, "too few judged pieces to measure a rate"
    assert clearly / judged <= MAX_FALSE_POSITIVES, f"{clearly} of {judged} pieces"
    assert above / judged <= MAX_ABOVE_BOUND, f"{above} of {judged} pieces above the bound"


@pytest.mark.parametrize("corpus_rates", ["examples_rates", "synthetic_rates"])
@pytest.mark.parametrize("cutter", list(CUTTERS))
def test_llm_drafts_cut_to_150_words_are_still_flagged(
    corpus_rates: str, cutter: str, request: pytest.FixtureRequest
) -> None:
    judged, _, _, flagged = request.getfixturevalue(corpus_rates)[1][(cutter, 150)]
    assert judged >= 10
    assert flagged / judged >= MIN_DETECTION, f"{flagged} of {judged} pieces"


def test_min_judged_words_is_the_shortest_length() -> None:
    assert min(CALIBRATION_LENGTHS) == MIN_JUDGED_WORDS


# Acceptance: the pooled verdict over many short chunks of mixed length.

# How many chunks each batch pools, and how many batches of each size are drawn.
BATCHES = (5, 20, 50, 200)
TRIALS = 200
# At most this share of batches of the writer's own chunks may read "somewhat different"
# or worse overall, or "leans LLM" or worse on likeness; and, from 20 chunks on, in any
# area. A pooled verdict must not be harsher than the chunks it pools.
MAX_POOLED_FALSE_POSITIVES = 0.05
# At least this share of batches of LLM chunks must still read "clearly different" or
# worse, or "leans LLM" or worse.
MIN_POOLED_DETECTION = 0.95


def cut_mixed(text: str, rng: random.Random, *, whole: bool) -> list[str]:
    """Pieces of mixed length, each aiming at a length drawn log-uniformly from 75 to 300
    words: whole paragraphs, or greedy runs of sentences that split paragraphs."""
    low, high = MIN_JUDGED_WORDS, 4 * MIN_JUDGED_WORDS

    def aim() -> float:
        return low * (high / low) ** rng.random()

    pieces: list[str] = []
    current: list[list[str]] = []
    count, target = 0, aim()

    def flush() -> None:
        nonlocal current, count, target
        pieces.append("\n\n".join(" ".join(part) for part in current if part))
        current, count, target = [], 0, aim()

    for paragraph in _paragraphs(text):
        if whole:
            current.append(paragraph)
            count += sum(len(sentence.split()) for sentence in paragraph)
            if count >= target:
                flush()
            continue
        current.append([])
        for sentence in paragraph:
            if count >= target:
                flush()
                current.append([])
            current[-1].append(sentence)
            count += len(sentence.split())
    return pieces


Judged = tuple[list[ScoredChunk], list[ScoredChunk]]


@pytest.fixture(scope="module")
def mixed_chunks(corpus: Path, synthetic_reference: ReferenceReport) -> dict[str, Judged]:
    """Per cutter, the judged chunks of 80 held-out writer documents and 20 held-out LLM
    drafts, cut into pieces of mixed length and each scored once."""
    writer = sorted((corpus / "writer").glob("*.md"))[80:160]
    drafts = sorted((corpus / "contrast").glob("*.md"))[20:40]
    found: dict[str, Judged] = {}
    for name, whole in (("sentences", False), ("paragraphs", True)):
        rng = random.Random(1)
        sides = []
        for paths in (writer, drafts):
            chunks = [
                Chunk(f"{path.stem}#{index}", path.stem, piece)
                for path in paths
                for index, piece in enumerate(cut_mixed(path.read_text(), rng, whole=whole))
            ]
            report = score(chunks, synthetic_reference, parser=None)
            sides.append(
                [row for row in report["chunks"] if row["reference"]["calibration"]["judged"]]
            )
        found[name] = (sides[0], sides[1])
    return found


def _pooled_rates(rows: Sequence[ScoredChunk], count: int) -> dict[str, float]:
    """How often batches of ``count`` of ``rows`` read somewhat different or worse overall
    and in any area, leaning LLM, and clearly different or leaning LLM."""
    rng = random.Random(count)
    totals = dict.fromkeys(("overall", "area", "likeness", "flagged"), 0)
    for _ in range(TRIALS):
        pooled = verdict(rng.sample(list(rows), count), "LLM")
        level = pooled["delta"]["level"] or 0
        likeness = (pooled["likeness"] or {}).get("level") or 0
        totals["overall"] += level >= SOMEWHAT
        totals["area"] += any(
            (area["level"] or 0) >= SOMEWHAT for area in pooled["by_group"].values()
        )
        totals["likeness"] += likeness >= LEANS
        totals["flagged"] += level >= CLEARLY or likeness >= LEANS
    return {key: value / TRIALS for key, value in totals.items()}


@pytest.mark.parametrize("cutter", ["sentences", "paragraphs"])
def test_pooled_verdicts_over_the_writers_short_chunks_stay_close(
    cutter: str, mixed_chunks: dict[str, Judged]
) -> None:
    """Batches of 5 to 200 of the writer's own held-out chunks, 75 to 300 words each: the
    pooled verdict reads "somewhat different" or worse at most 5% of the time. It read so
    for 62% of batches of 200 when the bound for a mean closed in on the pieces' median,
    which lies below their mean."""
    own, _ = mixed_chunks[cutter]
    assert len(own) >= 300
    for row in own:
        # Each chunk alone is judged against its own 95% bound, as before pooling changed.
        stored = row["reference"]["calibration"]["delta"]
        assert stored is not None
        expected = delta_level(row["reference"]["delta"] or 0.0, max(stored["p95"], MIN_CEILING))
        assert chunk_level(row) == expected
    for count in BATCHES:
        rates = _pooled_rates(own, count)
        assert rates["overall"] <= MAX_POOLED_FALSE_POSITIVES, (count, rates)
        assert rates["likeness"] <= MAX_POOLED_FALSE_POSITIVES, (count, rates)
        if count >= 20:
            assert rates["area"] <= MAX_POOLED_FALSE_POSITIVES, (count, rates)


@pytest.mark.parametrize("cutter", ["sentences", "paragraphs"])
def test_pooled_verdicts_over_llm_chunks_are_still_flagged(
    cutter: str, mixed_chunks: dict[str, Judged]
) -> None:
    _, llm = mixed_chunks[cutter]
    assert len(llm) >= 50
    for count in (5, 20, 50):
        rates = _pooled_rates(llm, count)
        assert rates["flagged"] >= MIN_POOLED_DETECTION, (count, rates)


def test_the_writers_own_comments_pool_to_close() -> None:
    """The case that found the bug, without pooling: 3,000 generated comments of about 50
    words (bench/gen.py), joined ten to a document, make the reference, and 300 held-out
    comments are scored one by one. 38 are long enough to judge and 35 of them read close
    alone; the headline over all 38 read "somewhat different" (Delta 1.17 against a bound
    of 1.12, the pieces' median 0.99 plus their spread over sqrt(38)). It reads close."""
    bench = _bench()
    material = bench._pool(WRITER)
    rng = random.Random(5)
    train = [bench._comment(rng, material, 50) for _ in range(3000)]
    seen = set(train)
    held: list[str] = []
    while len(held) < 300:
        text = bench._comment(rng, material, 50)
        if text not in seen:
            seen.add(text)
            held.append(text)
    documents = [
        Chunk(f"w{index}", f"w{index}", "\n\n".join(train[index : index + 10]))
        for index in range(0, len(train), 10)
    ]
    reference = build_reference(window(documents, WINDOW), parser=None)
    report = _score(held, reference)
    judged = [row for row in report["chunks"] if row["reference"]["calibration"]["judged"]]
    levels = [chunk_level(row) for row in judged]
    assert len(judged) == 38
    assert levels.count(0) == 35
    assert report["reference"]["verdict"]["verdict"] == sp.Verdict.CLOSE


# spaCy.


def _parser() -> Any:
    pytest.importorskip("spacy")
    from styleprofile.syntax import SyntaxUnavailableError, load_parser

    try:
        return load_parser()
    except SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")


def test_pieces_are_parsed_once_with_their_windows() -> None:
    parser = _parser()
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


def test_a_span_is_measured_on_its_own_sentences() -> None:
    from styleprofile.syntax import _sentences  # pyright: ignore[reportPrivateUsage]

    parser = _parser()
    doc = parser.nlp("First sentence here is long enough. Second one follows right after it.")
    # The span starts mid-sentence; spaCy's Span.sents would give both whole sentences.
    span = doc[3:]
    assert [sentence.text for sentence in _sentences(span)] == [
        "is long enough.",
        "Second one follows right after it.",
    ]
    assert [sentence.text for sentence in _sentences(doc)] == [
        sentence.text for sentence in doc.sents
    ]
