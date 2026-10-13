"""Typography never changes a verdict: the transforms in ``bench/invariance.py``.

Each transform changes how a text was typed or converted, not what it says. See "What
styleprofile ignores" in ``docs/method.md``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from styleprofile.surface import char_trigrams, surface_metrics

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))  # the bench scripts import each other as top-level modules
invariance: Any = importlib.import_module("invariance")

HAS_SPACY = importlib.util.find_spec("spacy") is not None

STRAIGHT = (
    'She said, "Don\'t go." He\'d "already" gone -- well... almost.\n\n'
    "It's the writer's own 'habit', isn't it?"
)
TYPOGRAPHIC = (
    "She said, \u201cDon\u2019t go.\u201d He\u2019d \u201calready\u201d gone \u2014 well\u2026 "
    "almost.\n\n"
    "It\u2019s the writer\u2019s own \u2018habit\u2019, isn\u2019t it?"
)


def test_metrics_and_trigrams_do_not_depend_on_the_glyphs_used() -> None:
    straight, typographic = surface_metrics(STRAIGHT), surface_metrics(TYPOGRAPHIC)
    assert straight == typographic
    assert char_trigrams(STRAIGHT.replace("--", "-").replace("...", "...")) == char_trigrams(
        TYPOGRAPHIC
    )


def test_no_metric_records_only_the_encoding() -> None:
    names = {name for group in surface_metrics(STRAIGHT).values() for name in group}
    assert "curly_apostrophe_pct" not in names


def _failures(syntax: str) -> list[str]:
    return [
        f"{item.transform} [{item.side}] {failure}"
        for item in invariance.run(syntax)
        for failure in item.failures
    ]


@pytest.mark.parametrize(
    "syntax",
    [
        "off",
        pytest.param(
            "auto",
            marks=pytest.mark.skipif(not HAS_SPACY, reason="the parser-based metrics need spaCy"),
        ),
    ],
)
def test_no_transform_changes_a_verdict(syntax: str) -> None:
    assert _failures(syntax) == []
