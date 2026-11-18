"""Comma series, balanced semicolon lines and colon reveals: reported, not scored."""

from __future__ import annotations

import time

import pytest

from styleprofile.metrics import UNSCORED_GROUPS
from styleprofile.surface import match_spans, surface_metrics


def spans(text: str, metric: str) -> list[str]:
    return [text[a:b] for a, b in match_spans(text, metric)]


@pytest.mark.parametrize(
    "text",
    [
        "Dignity, privacy, and autonomy.",
        "Made, ignored, or forgotten.",
        "More capable, more independent and more abundant.",
        "Robots can now use tools, check progress, recover from mistakes, and ask for help.",
        "We plan, we build, we test, we ship, and we learn.",
    ],
)
def test_comma_series_counts_lists_of_three_or_more(text: str) -> None:
    assert spans(text, "comma_series") == [text.rstrip(".")]


@pytest.mark.parametrize(
    "text",
    [
        "We tried two things, and both failed.",
        "When we met, Ann, Bo, and Cy came.",
        "However, we tried, we failed, and we left.",
        "Apples, pears and\nplums.",
    ],
)
def test_comma_series_rejects(text: str) -> None:
    assert spans(text, "comma_series") == []


def test_comma_series_includes_every_parallel_triad() -> None:
    text = "We listen, we learn, and we act. They came; we saw, we heard, and we left."
    assert set(match_spans(text, "parallel_triads")) <= set(match_spans(text, "comma_series"))


@pytest.mark.parametrize(
    "text",
    [
        "The anxiety is new for engineers; for writers, it's the water they've always swum in.",
        "The barrier to entry collapsed; the distribution of attention did not.",
    ],
)
def test_semicolon_balance_counts_two_short_halves(text: str) -> None:
    assert spans(text, "semicolon_balances") == [text.rstrip(".")]


@pytest.mark.parametrize(
    "text",
    [
        "Yes; no.",
        "We had apples, pears, plums; they had none of them at all.",
        "One half here; a second half; and a third half too.",
        "This first half of the sentence runs on for well over fourteen words before it ever"
        " reaches the semicolon; then it stops.",
    ],
)
def test_semicolon_balance_rejects(text: str) -> None:
    assert spans(text, "semicolon_balances") == []


def test_colon_reveal_counts_a_clause_then_a_short_ending() -> None:
    text = (
        "This is a standard problem with solutions that require compensation: they require"
        " too much precision."
    )
    assert spans(text, "colon_reveals") == [text.rstrip(".")]


@pytest.mark.parametrize(
    "text",
    [
        "The answer: no.",
        'When I tried to negotiate, I was given the party line: "all salaries are the same."',
        "We need four things from the team: speed, care, accuracy, and patience.",
        "Is this a question with a colon in it: really now?",
        "The meeting moved to a later time slot at 3:30 today.",
    ],
)
def test_colon_reveal_rejects(text: str) -> None:
    assert spans(text, "colon_reveals") == []


def test_rhetorical_shapes_are_reported_but_not_scored() -> None:
    text = "The barrier collapsed for us; the attention did not follow. Made, ignored, and lost."
    rhetoric = surface_metrics(text)["rhetoric"]
    words = 14
    assert rhetoric["semicolon_balances_per_1k"] == 1000 / words
    assert rhetoric["comma_series_per_1k"] == 1000 / words
    assert rhetoric["colon_reveals_per_1k"] == 0
    assert "rhetoric" in UNSCORED_GROUPS


@pytest.mark.parametrize(
    "text",
    [
        "a, " * 3000 + "b",
        "word " * 5000 + ";" + " word" * 5000,
        "word " * 5000 + ":" + " word" * 5000,
        ("a b c d e: f g, " * 2000),
        "I.\x1f" * 10922,
        "One two three. " * 3000,
        "One two three four five six: seven eight nine. " * 1000,
        "One two three; four five six. " * 1000,
    ],
)
def test_rhetorical_shapes_stay_fast_on_adversarial_text(text: str) -> None:
    started = time.perf_counter()
    for metric in ("comma_series", "semicolon_balances", "colon_reveals"):
        match_spans(text, metric)
    assert time.perf_counter() - started < 0.5
