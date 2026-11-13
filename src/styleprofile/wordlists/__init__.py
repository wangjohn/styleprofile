"""Weighted word lists and the rate a text scores against one.

A list maps words (casefolded, as ``styleprofile.surface.words`` gives them) to weights; a
text's rate is 1000 times the sum of its tokens' weights over its token count, so words not
on the list count as zero. ``edited_register`` is the list behind the
``edited_register_per_1k`` metric; its JSON file says how it was learned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from types import MappingProxyType


@cache
def load(name: str) -> Mapping[str, float]:
    """The weights of a shipped list, by name (``edited_register``), read-only: the result is
    cached and shared, so a caller can't change what later calls get."""
    text = resources.files("styleprofile.wordlists").joinpath(f"{name}.json").read_text("utf-8")
    weights = json.loads(text)["weights"]
    return MappingProxyType({str(word): float(weight) for word, weight in weights.items()})


def weighted_rate(tokens: Sequence[str], weights: Mapping[str, float]) -> float | None:
    """1000 times the sum of the tokens' weights over the token count; None with no tokens."""
    if not tokens:
        return None
    return 1000.0 * sum(weights.get(token, 0.0) for token in tokens) / len(tokens)
