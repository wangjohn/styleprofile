"""Where a draft drifts: paragraphs with their lines, spans, the flag rules, and the output.

The acceptance tests run without spaCy on the demo corpus (``bench/gen.py --corpus demo``,
remixed from the seven essays): draft.md's two planted paragraphs drift and nothing
else is, and the writer's own held-out documents get no flags. docs/method.md ("Where a
draft drifts") has the fuller evidence.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile.cli import main
from styleprofile.drift import (
    SPAN_WORDS,
    Paragraph,
    Passage,
    Trait,
    calibrate,
    excerpt,
    fit_tail,
    judge,
    judge_span,
    level,
    paragraphs,
    plan,
    threshold,
    thresholds,
)
from styleprofile.profile import load_report
from styleprofile.surface import classify

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "examples" / "draft.md"
# The paragraphs of draft.md written in the LLM register (examples/README.md).
PLANTED = [(9, 9), (15, 15)]
HAS_SPACY = importlib.util.find_spec("spacy") is not None

sys.path.insert(0, str(ROOT / "bench"))
gen: Any = importlib.import_module("gen")


@cache
def _demo(tmp: Path) -> tuple[list[Path], Path]:
    folder = gen.generate("demo", out=tmp)
    return sorted((folder / "writer").glob("*.md")), folder / "contrast"


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Path], Path]:
    return _demo(tmp_path_factory.mktemp("demo"))


@pytest.fixture(scope="module")
def demo_profile(demo: tuple[list[Path], Path]) -> sp.Profile:
    documents, contrast = demo
    return sp.build(documents, sp.Settings(syntax=False), contrast=contrast)


# Blocks and paragraphs.


def test_blocks_record_the_line_they_start_on() -> None:
    text = "---\ntitle: x\n---\n\n# Title\r\n\r\nOne line\r\nand two.\n\n```\ncode\n```\n\nEnd.\n"
    found = [(block.line, block.end_line, block.raw.split("\n")[0]) for block in classify(text)]
    assert found == [(5, 5, "# Title"), (7, 8, "One line"), (10, 12, "```"), (14, 14, "End.")]


def test_a_bare_unclosed_fence_keeps_the_lines_after_it() -> None:
    blocks = classify("First.\n\n```\n\nSecond.\n")
    assert [(block.line, block.raw) for block in blocks] == [(1, "First."), (5, "Second.")]


def test_paragraphs_carry_headings_above_them_but_their_lines_are_their_prose() -> None:
    text = (
        "# Title\n\nOpening words here.\n\n## Part\n\n- one item\n- two\n\n"
        "    more of item two\n\nClosing.\n\n```\nx = 1\n```\n"
    )
    found = paragraphs(text)
    # Headings and code are measured with a paragraph but never widen its lines.
    assert [(p.line, p.end_line) for p in found] == [(3, 3), (7, 10), (12, 12)]
    assert [p.words for p in found] == [3, 7, 1]
    assert found[0].excerpt == "Opening words here."
    assert found[1].raw.startswith("## Part") and not found[1].own.startswith("## Part")
    assert found[2].raw.endswith("```") and found[2].own == "Closing."


def test_excerpts_are_cut_at_a_word() -> None:
    text = "word " * 30
    cut = excerpt(text, 20)
    assert cut.endswith("…") and len(cut) <= 21 and "wor…" not in cut
    assert excerpt("short one") == "short one"


# Spans.


def test_every_paragraph_rests_on_two_spans() -> None:
    sizes = [70, 85, 84, 60, 86, 63, 33, 61]
    layout = plan(sizes, 100)
    for start, end in layout.spans:
        assert sum(sizes[start:end]) >= 100
    for index, (one, two) in enumerate(layout.sides):
        assert one is not None and two is not None and one != two
        for position in (one, two):
            start, end = layout.spans[position]
            assert start <= index < end
    # Near the end, the paragraph at line 15 (33 words) takes the last span forward.
    one, two = layout.sides[6]
    assert one is not None and two is not None
    assert layout.spans[one] == (5, 8) and layout.spans[two] == (4, 7)
    # At the ends, the one span is widened by the neighbouring paragraph.
    assert {layout.spans[p] for p in layout.sides[0] if p is not None} == {(0, 2), (0, 3)}
    assert {layout.spans[p] for p in layout.sides[7] if p is not None} == {(5, 8), (4, 8)}


def test_a_long_paragraph_takes_its_shorter_neighbour_for_a_second_span() -> None:
    layout = plan([80, 150, 40, 90], 100)
    assert {layout.spans[p] for p in layout.sides[1] if p is not None} == {(1, 2), (1, 3)}


def test_a_document_of_one_span_rests_on_it_alone() -> None:
    layout = plan([60, 60], 100)
    assert layout.spans == [(0, 2)] and layout.sides == [(0, 0), (0, 0)]


def test_a_document_shorter_than_a_span_has_none() -> None:
    assert plan([40, 30], 100).spans == []
    assert plan([], 100).spans == []


# The statistic, the null and the flags.


def _span(value: float) -> dict[str, Any]:
    return {
        "by": "delta",
        "judged": True,
        "relative": value,
        "delta": value,
        "level": 1 if value > 1 else 0,
        "likeness": None,
        "likeness_level": None,
    }


LIMITS: dict[str, float | None] = {"two": 1.5, "one": None}


def _entries(
    sizes: list[int], values: dict[tuple[int, int], float], limits: Any = LIMITS
) -> list[dict[str, Any]]:
    found = [
        Paragraph(f"p{index}", index * 2 + 1, index * 2 + 1, size, f"p{index}", f"p{index}")
        for index, size in enumerate(sizes)
    ]
    layout = plan(sizes, 100)
    judged = [_span(values.get(span, 0.5)) for span in layout.spans]
    return judge(found, layout, judged, limits, [[] for _ in found])


def _drifts(sizes: list[int], values: dict[tuple[int, int], float], limits: Any = LIMITS):
    return [entry["drifts"] for entry in _entries(sizes, values, limits)]


def test_a_paragraph_drifts_when_both_its_spans_pass_the_threshold() -> None:
    # Four paragraphs of 60 words; paragraph 1's spans are (1,3) and (0,2).
    assert _drifts([60] * 4, {(0, 2): 1.6, (1, 3): 1.6}) == [False, True, False, False]


def test_a_low_span_beside_it_clears_a_neighbour() -> None:
    # Paragraph 2 drifts: its spans (2,4) and (1,3) are high, paragraph 1's (0,2) is not.
    assert _drifts([60] * 4, {(1, 3): 1.6, (2, 4): 1.6}) == [False, False, True, False]


def test_nothing_drifts_without_thresholds() -> None:
    assert _drifts([60] * 4, {(1, 3): 9.0, (2, 4): 9.0}, limits=None) == [False] * 4


def test_a_paragraph_whose_spans_hold_one_that_drifts_is_explained_by_it() -> None:
    # The first paragraph's spans (0,2) and (0,3) are high because of the second, which
    # is stronger and whose own spans (1,3) and (0,2) do not both hold the first.
    values = {(0, 2): 1.8, (0, 3): 1.6, (1, 3): 1.8}
    entries = _entries([60] * 4, values)
    assert [entry["drifts"] for entry in entries] == [False, True, False, False]
    assert entries[0]["note"] == "both its spans hold line 3"


def test_a_weaker_paragraph_never_explains_a_stronger_one() -> None:
    # The first paragraph (5x) has the second (1.6x) in both its spans, but the second is
    # weaker, so it cannot explain the first away: both drift.
    values = {(0, 2): 5.0, (0, 3): 5.0, (1, 3): 1.6}
    entries = _entries([60] * 4, values)
    assert [entry["drifts"] for entry in entries] == [True, True, False, False]
    assert entries[0]["note"] is None


def test_a_paragraph_on_one_span_does_not_drift_without_its_own_null() -> None:
    entries = _entries([60, 60], {(0, 2): 3.0})
    assert [entry["drifts"] for entry in entries] == [False, False]
    assert entries[0]["note"] == "rests on one span, which this reference cannot calibrate"


def test_the_threshold_rises_with_the_number_of_paragraphs() -> None:
    tail = fit_tail([index / 100 for index in range(1000)])
    assert tail is not None and tail["share"] == pytest.approx(0.1, abs=0.01)
    few, many = threshold(tail, 5), threshold(tail, 80)
    assert tail["threshold"] < few < many
    assert fit_tail(list(range(100))) is None  # too few values above the tail's start


def _document(shift: float, count: int = 30) -> list[tuple[float, int, int] | None]:
    return [(shift + index / 100, 2, 0) for index in range(count)]


def test_a_small_reference_gets_no_thresholds() -> None:
    null = calibrate(5, [_document(0.3)] * 20, "likeness")
    assert null["within"] is None and thresholds(null, 10, "likeness") is None


def test_the_threshold_is_set_off_from_the_top_of_the_writers_levels() -> None:
    documents = [_document(0.2 + index / 100) for index in range(20)]
    null = calibrate(20, documents, "likeness")
    assert null["levels"] is not None and null["within"] is not None
    assert null["levels"]["high"] > null["levels"]["median"]
    spread = [_document(0.2 + index / 20) for index in range(20)]
    wider = calibrate(20, spread, "likeness")
    limits, wide = thresholds(null, 10, "likeness"), thresholds(wider, 10, "likeness")
    assert limits is not None and wide is not None
    # Documents whose levels spread further set a higher bar.
    assert (wide["two"] or 0.0) > (limits["two"] or 0.0) >= 1.0
    # A null for likeness sets no thresholds for spans judged by Delta.
    assert thresholds(null, 10, "delta") is None
    assert level([(0.2, 2, 0), (0.4, 2, 0), (9.0, 1, 0), None]) == pytest.approx(0.3)


def test_spans_are_judged_by_likeness_when_there_is_a_range_for_it() -> None:
    scored: dict[str, Any] = {
        "delta": 1.1,
        "likeness": 0.9,
        "calibration": {
            "judged": True,
            "delta": {"median": 0.5, "p95": 1.0},
            "likeness": {"median": 0.2, "p95": 0.6, "target": 5.0},
            "length_scale": {},
            "stretch": 1.0,
        },
    }
    judged = judge_span(scored)
    assert judged["by"] == "likeness" and judged["relative"] == pytest.approx(1.5)
    scored["calibration"]["likeness"] = None
    judged = judge_span(scored)
    assert judged["by"] == "delta" and judged["relative"] == pytest.approx(1.1)


# Acceptance, without spaCy.


def test_the_planted_paragraphs_of_the_draft_are_found_and_nothing_else(
    demo_profile: sp.Profile,
) -> None:
    result = demo_profile.score(DRAFT)
    drifting = [passage.lines for passage in result.passages if passage.drifts]
    assert drifting == PLANTED
    assert all(passage.document == "draft.md" for passage in result.passages)
    assert len(result.passages) == 8


def test_the_writers_own_held_out_documents_do_not_drift(
    tmp_path: Path, demo: tuple[list[Path], Path]
) -> None:
    documents, contrast = demo
    profile = sp.build(documents[:30], sp.Settings(syntax=False), contrast=contrast)
    drifting = []
    for path in documents[30:]:
        result = profile.score(path)
        assert result.passages
        drifting += [(path.name, p.lines) for p in result.passages if p.drifts]
    assert drifting == []


def test_a_thin_reference_flags_nothing_rather_than_guess() -> None:
    # Seven essays calibrate paragraph-sized spans too loosely to single a paragraph out.
    examples = ROOT / "examples"
    profile = sp.build(
        examples / "writer", sp.Settings(syntax=False), contrast=examples / "llm-drafts"
    )
    result = profile.score(DRAFT)
    assert result.passages and not any(passage.drifts for passage in result.passages)
    (document,) = result.report["passages"] or []
    assert not document["sensitive"] and document["thresholds"] is None
    null = profile.report["calibration"].get("drift")
    assert null is not None and null["documents"] == 7 and null["within"] is None
    text = result.to_text()
    assert "Where it drifts: the reference is too small to set paragraph thresholds" in text
    assert "no paragraph drifts" not in text
    assert "too small to set paragraph thresholds" in result.to_text(by_paragraph=True)


def test_a_reference_calibrated_for_paragraphs_is_sensitive(demo_profile: sp.Profile) -> None:
    result = demo_profile.score(ROOT / "examples" / "writer" / "old-maps.md")
    (document,) = result.report["passages"] or []
    assert document["sensitive"] and document["thresholds"] is not None
    null = demo_profile.report["calibration"].get("drift")
    assert null is not None and null["by"] == "likeness" and null["documents"] >= 10
    text = result.to_text()
    assert "Where it drifts: no paragraph drifts (7 paragraphs checked)." in text
    assert "too small" not in text


def test_a_reference_without_short_calibration_says_so() -> None:
    essays = sorted((ROOT / "examples" / "writer").glob("*.md"))[:2]
    profile = sp.build(essays, sp.Settings(syntax=False))
    result = profile.score(DRAFT)
    (document,) = result.report["passages"] or []
    assert not document["judged"] and not document["sensitive"]
    assert "the reference has no range for passages this short" in result.to_text()


def test_signals_under_one_sd_get_no_dangling_entry(demo_profile: sp.Profile) -> None:
    text = demo_profile.score(DRAFT).to_text()
    signals = next(line for line in text.splitlines() if "Strongest LLM signals" in line)
    assert " ," not in signals and not signals.endswith(" ")


# The library, the report and the command line.


def test_passages_are_typed_for_the_library(demo_profile: sp.Profile) -> None:
    result = demo_profile.score(DRAFT)
    passage = next(passage for passage in result.passages if passage.drifts)
    assert isinstance(passage, Passage) and sp.Passage is Passage
    assert passage.lines == (9, 9) and passage.words == 60
    assert passage.excerpt.startswith("But the store is more than a place")
    assert passage.verdict in sp.Verdict and passage.likeness is not None
    assert passage.likeness_verdict in sp.LikenessVerdict
    assert passage.traits and all(isinstance(trait, Trait) for trait in passage.traits)
    assert "punctuation.em_dashes_per_1k" in {trait.metric for trait in passage.traits}


def test_several_documents_are_read_in_parts_only_when_asked(demo_profile: sp.Profile) -> None:
    other = ROOT / "examples" / "writer" / "old-maps.md"
    assert demo_profile.score([DRAFT, other]).report["passages"] is None
    assert demo_profile.score([DRAFT, other]).passages == ()
    asked = demo_profile.score([DRAFT, other], passages=True)
    assert {passage.document for passage in asked.passages} == {"draft.md", "old-maps.md"}
    assert demo_profile.score(DRAFT, passages=False).passages == ()


def test_a_document_too_short_for_a_span_abstains(demo_profile: sp.Profile) -> None:
    result = demo_profile.score(sp.Text("A short note. " * 20, name="note"))
    (document,) = result.report["passages"] or []
    assert not document["judged"] and document["paragraphs"] == []
    assert f"under {SPAN_WORDS} words" in (document["reason"] or "")
    assert result.passages == ()
    assert "Where it drifts: too short to check paragraphs (under 100 words)." in result.to_text()
    assert "Not read in parts: under" in result.to_text(by_paragraph=True)


def test_windows_do_not_split_a_document_read_in_parts(demo_profile: sp.Profile) -> None:
    # Two copies of the draft run past one window; paragraphs keep the document's lines.
    text = DRAFT.read_text() + "\n" + DRAFT.read_text()
    result = demo_profile.score(sp.Text(text, name="twice"))
    assert result.chunk_count == 2
    lines = [passage.lines for passage in result.passages]
    assert lines[:8] == [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13), (15, 15), (17, 17)]
    assert lines[8] == (21, 21)


def test_saved_reports_show_where_it_drifts(tmp_path: Path, demo_profile: sp.Profile) -> None:
    path = tmp_path / "score.json"
    result = demo_profile.score(DRAFT)
    result.save(path)
    saved = load_report(path)
    assert saved["kind"] == "score"
    assert json.loads(path.read_text())["passages"][0]["paragraphs"][3]["drifts"]
    text = sp.ScoreResult(saved).to_text(by_paragraph=True)
    assert "Where it drifts" in text and "By paragraph" in text


def test_the_command_line_shows_passages_and_lists_paragraphs(
    tmp_path: Path, demo_profile: sp.Profile, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = tmp_path / "demo.json"
    demo_profile.save(reference)
    assert main(["score", "-q", str(DRAFT), str(reference)]) == 0
    assert capsys.readouterr().out.strip().endswith("drifts at lines 9, 15")
    assert main(["score", str(DRAFT), str(reference)]) == 0
    out = capsys.readouterr().out
    assert "Where it drifts" in out and "By paragraph" not in out
    assert out.index("By area") < out.index("Where it drifts") < out.index("Biggest")
    assert main(["score", "--by-paragraph", str(DRAFT), str(reference)]) == 0
    out = capsys.readouterr().out
    assert "By paragraph" in out and "* drifts, above" in out


@pytest.mark.skipif(not HAS_SPACY, reason="spaCy is not installed")
def test_spans_across_windows_use_the_documents_parse() -> None:
    from styleprofile.api import _default_parser
    from styleprofile.profile import _Joined

    parser = _default_parser()
    texts = ["One sentence here. Another one.", "A second window starts. It ends."]
    joined = _Joined("\n\n".join(texts), texts, list(parser.docs(texts)))
    text = joined.text
    start = text.index("Another")
    end = text.index("starts.") + len("starts.")
    span = joined.span(start, end)
    assert span is not None and span.text.split() == text[start:end].split()


def test_without_a_contrast_set_it_does_not_reassure(demo: tuple[list[Path], Path]) -> None:
    documents, _ = demo
    profile = sp.build(documents, sp.Settings(syntax=False))
    result = profile.score(DRAFT)
    (document,) = result.report["passages"] or []
    assert document["by"] == "delta"
    text = result.to_text()
    assert "paragraph checks need a reference built with --contrast" in text
    assert "no paragraph drifts" not in text
    assert "need a reference built with --contrast" in result.to_text(by_paragraph=True)


def test_a_document_drifting_throughout_says_so(
    demo_profile: sp.Profile, capsys: pytest.CaptureFixture[str]
) -> None:
    draft = ROOT / "examples" / "llm-drafts" / "mud-season.md"
    result = demo_profile.score(draft)
    drifting = sum(passage.drifts for passage in result.passages)
    assert drifting * 2 > len(result.passages)
    text = result.to_text()
    assert f"drifts throughout ({drifting} of {len(result.passages)} paragraphs" in text
    assert "stand out" not in text


def test_traits_are_the_paragraphs_own(demo_profile: sp.Profile) -> None:
    # A prose paragraph beside a list does not borrow the list's traits.
    text = (
        "# Notes\n\n"
        + (DRAFT.read_text().split("\n\n")[1])
        + "\n\n"
        + "\n".join(f"- **Point {n}:** a crucial and vital item" for n in range(8))
        + "\n\n"
        + (DRAFT.read_text().split("\n\n")[2])
        + "\n"
    )
    result = demo_profile.score(sp.Text(text, name="mixed"))
    prose_paragraph = result.passages[0]
    metrics = {trait.metric for trait in prose_paragraph.traits}
    assert "markdown.list_items_per_1k" not in metrics
    assert "markdown.bold_per_1k" not in metrics


def test_html_line_numbers_are_said_to_be_of_the_converted_text(
    tmp_path: Path, demo_profile: sp.Profile
) -> None:
    paragraphs_html = "".join(f"<p>{block}</p>" for block in DRAFT.read_text().split("\n\n")[1:])
    page = tmp_path / "page.html"
    page.write_text(f"<html><body><article>{paragraphs_html}</article></body></html>")
    result = demo_profile.score(page)
    (document,) = result.report["passages"] or []
    assert document["converted"]
    assert "line numbers are of the text converted from html" in result.to_text().lower()
    assert (
        "line numbers are of the text converted from html"
        in result.to_text(by_paragraph=True).lower()
    )


def test_an_llm_opening_is_not_hidden_behind_its_neighbours(demo_profile: sp.Profile) -> None:
    # The review's chain: lines 3, 7 and 11 of an LLM draft were each explained away by the
    # next, down to a weaker paragraph. Explanation now only runs from stronger to weaker.
    result = demo_profile.score(ROOT / "examples" / "llm-drafts" / "mud-season.md")
    drifting = {passage.lines[0] for passage in result.passages if passage.drifts}
    assert {3, 7, 11} <= drifting


def test_pooled_records_are_not_read_in_parts(tmp_path: Path, demo_profile: sp.Profile) -> None:
    records = tmp_path / "records.jsonl"
    blocks = DRAFT.read_text().split("\n\n")[1:] * 3
    records.write_text(
        "".join(json.dumps({"id": str(i), "text": b}) + "\n" for i, b in enumerate(blocks))
    )
    result = demo_profile.score(records, pool=True)
    documents = result.report["passages"] or []
    assert documents and all(d["pooled"] and not d["judged"] for d in documents)
    assert result.passages == ()
    line = "Where it drifts: paragraph checks don't apply to pooled records"
    assert line in result.to_text()
    path = tmp_path / "score.json"
    result.save(path)
    saved = sp.ScoreResult(load_report(path))
    assert line in saved.to_text() and all(d["pooled"] for d in saved.report["passages"] or [])
    # Unpooled, each record is its own document and none is pooled.
    plain = demo_profile.score(records, pool=False, passages=True)
    assert not any(d["pooled"] for d in plain.report["passages"] or [])
