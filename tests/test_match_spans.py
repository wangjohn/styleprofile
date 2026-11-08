"""``match_spans`` and ``opener_span`` find exactly what the surface metrics count."""

from __future__ import annotations

import pytest

from styleprofile import surface
from styleprofile.surface import (
    ELLIPSIS,
    EM_DASH,
    match_spans,
    opener_span,
    prose,
    sentences,
    surface_metrics,
)

COUNT_METRICS = (
    "commas",
    "semicolons",
    "colons",
    "parentheses",
    "questions",
    "exclamations",
    "em_dashes",
    "ellipses",
    "hedges",
    "boosters",
    "llm_markers",
    "not_just_but",
    "corrective_contrasts",
    "parallel_triads",
)

Q = "\N{RIGHT SINGLE QUOTATION MARK}"
TEXTS = (
    f"I think it{Q}s worth noting that we{Q}re"
    f" probably wrong\N{EM DASH}clearly, very wrong. It{Q}s not just"
    f" the map but the road. I{Q}m not sure; maybe later...",
    "The posts shift--the stones do not. A word - word dash, a word -- word one, and a"
    " hyphenated well-known word. Wait\N{HORIZONTAL ELLIPSIS} really? Yes! (Usually.)",
    f"Moreover, it{Q}s crucial: in today{Q}s"
    " ever-evolving landscape we delve into robust, seamless tools. Furthermore, it is worth"
    " noting that this is not only fast but cheap. Kind of. Sort of? It seems so...",
    f"This isn{Q}t a report, it{Q}s a map. It{Q}s not about speed. It{Q}s about trust. We"
    " listen, we learn, and we act; we left early, quietly, and happily.",
    "Plain sentence with nothing to find in it at all",
    "",
)


def _metric_values(text: str) -> dict[str, float | None]:
    return {
        name: value for group in surface_metrics(text).values() for name, value in group.items()
    }


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("metric", COUNT_METRICS)
def test_span_count_equals_rate_count(text: str, metric: str) -> None:
    parsed = prose(text)
    values = _metric_values(text)
    words = values["words"]
    rate = values[f"{metric}_per_1k"]
    spans = match_spans(parsed.text, metric)
    if not words:
        assert rate is None
        return
    assert rate is not None
    assert len(spans) == round(rate * words / 1000)


def test_spans_slice_to_matched_text() -> None:
    curly = Q
    text = f"It{curly}s I{curly}m not sure{curly} maybe\N{EM DASH}clearly. I{curly}m not sure."
    assert [text[a:b] for a, b in match_spans(text, "hedges")] == [
        f"I{curly}m not sure",
        "maybe",
        f"I{curly}m not sure",
    ]
    assert [text[a:b] for a, b in match_spans(text, "boosters")] == ["clearly"]
    assert [text[a:b] for a, b in match_spans(text, "em_dashes")] == ["\N{EM DASH}"]
    dashes = "one--two, three - four, five -- six, well-known"
    assert [dashes[a:b] for a, b in match_spans(dashes, "em_dashes")] == ["--", " - ", " -- "]
    dots = "Wait... no\N{HORIZONTAL ELLIPSIS} yes."
    assert [dots[a:b] for a, b in match_spans(dots, "ellipses")] == [
        "...",
        "\N{HORIZONTAL ELLIPSIS}",
    ]
    marker = f"In today{curly}s world, it{curly}s worth noting this."
    assert [marker[a:b] for a, b in match_spans(marker, "llm_markers")] == [
        f"In today{curly}s",
        f"it{curly}s worth noting",
    ]
    contrast = "It is not just fast but cheap."
    assert [contrast[a:b] for a, b in match_spans(contrast, "not_just_but")] == [
        "not just fast but"
    ]
    punctuation = "a, b; c: (d)? e!"
    for metric, mark in [
        ("commas", ","),
        ("semicolons", ";"),
        ("colons", ":"),
        ("parentheses", "("),
        ("questions", "?"),
        ("exclamations", "!"),
    ]:
        assert [punctuation[a:b] for a, b in match_spans(punctuation, metric)] == [mark]


def test_corrective_contrast_spans() -> None:
    text = (
        f"This isn{Q}t a report, it{Q}s a map. It{Q}s not about speed. It{Q}s about trust."
        " The plan is not ready, it is scheduled for May. It is not a cat, it is not a dog."
    )
    assert [text[a:b] for a, b in match_spans(text, "corrective_contrasts")] == [
        f"This isn{Q}t a report, it{Q}s",
        f"It{Q}s not about speed. It{Q}s",
    ]
    # Same spans whether apostrophes are curly or straight.
    straight = text.replace(Q, "'")
    assert match_spans(straight, "corrective_contrasts") == match_spans(
        text, "corrective_contrasts"
    )


def test_parallel_triad_spans() -> None:
    text = (
        "We listen, we learn, and we act. I bought eggs, milk, bread, and jam. When we met,"
        " Ann, Bo and Cy came; we left early, quietly, and happily.\n\nRed, green and blue."
    )
    assert [text[a:b] for a, b in match_spans(text, "parallel_triads")] == [
        "We listen, we learn, and we act",
        "we left early, quietly, and happily",
        "Red, green and blue",
    ]


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown count metric"):
        match_spans("text", "commas_per_1k")
    with pytest.raises(ValueError, match="unknown opener metric"):
        opener_span("However, no.", "hedges")


def test_opener_span() -> None:
    assert opener_span("However, it rained.", "transition_openers") == (0, 7)
    assert opener_span("In fact it rained.", "transition_openers") == (0, 7)
    assert opener_span("It rained, however.", "transition_openers") is None
    assert opener_span("But it rained.", "and_but_so_openers") == (0, 3)
    assert opener_span("Sometimes it rained.", "and_but_so_openers") is None


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("metric", ["transition_openers", "and_but_so_openers"])
def test_opener_count_equals_share(text: str, metric: str) -> None:
    parsed = prose(text)
    found = [s for block in parsed.blocks for s in sentences(block)]
    share = _metric_values(text)[f"{metric}_pct"]
    hits = sum(opener_span(s, metric) is not None for s in found)
    if not found:
        assert share is None
        return
    assert share == pytest.approx(100 * hits / len(found))


def test_public_patterns_keep_private_aliases() -> None:
    assert surface._EM_DASH is EM_DASH  # pyright: ignore[reportPrivateUsage]
    assert surface._ELLIPSIS is ELLIPSIS  # pyright: ignore[reportPrivateUsage]


def test_offsets_count_code_points_past_multibyte_characters() -> None:
    # An emoji (4 UTF-8 bytes, 2 UTF-16 units), an accented letter and curly quotes before
    # each match: the offsets still slice the Python string to the matched text.
    text = (
        f"\N{GRINNING FACE} Caf\N{LATIN SMALL LETTER E WITH ACUTE} \N{LEFT SINGLE QUOTATION MARK}"
        f"I{Q}m not sure{Q}\N{EM DASH}it{Q}s very late\N{HORIZONTAL ELLIPSIS} maybe."
    )
    spans = {metric: match_spans(text, metric) for metric in COUNT_METRICS}
    assert spans["hedges"] == [(8, 20), (38, 43)]
    assert [text[a:b] for a, b in spans["hedges"]] == [f"I{Q}m not sure", "maybe"]
    assert [text[a:b] for a, b in spans["em_dashes"]] == ["\N{EM DASH}"]
    assert [text[a:b] for a, b in spans["boosters"]] == ["very"]
    assert [text[a:b] for a, b in spans["ellipses"]] == ["\N{HORIZONTAL ELLIPSIS}"]


def test_surface_metrics_counts_through_the_span_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    # The rates are counted from the spans themselves, so the two cannot drift apart.
    text = "Maybe, maybe not; probably fine."
    before = _metric_values(text)
    found = surface._spans  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(surface, "_spans", lambda *args: [*found(*args), (0, 0)])
    after = _metric_values(text)
    for metric in COUNT_METRICS:
        old, new = before[f"{metric}_per_1k"], after[f"{metric}_per_1k"]
        assert old is not None and new is not None and new > old
