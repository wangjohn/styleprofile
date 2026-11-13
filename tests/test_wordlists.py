import json
from importlib import resources

import pytest

from styleprofile.surface import prose, surface_metrics, words
from styleprofile.wordlists import load, weighted_rate


def test_weighted_rate_is_per_thousand_tokens():
    assert weighted_rate(["a", "b", "c", "d"], {"a": 2.0, "b": -1.0}) == pytest.approx(250.0)
    assert weighted_rate([], {"a": 1.0}) is None


def test_shipped_list_has_both_sides_and_casefolded_words():
    weights = load("edited_register")
    assert len(weights) == 300
    assert sum(w > 0 for w in weights.values()) == 150
    assert all(word == word.casefold() and word.replace("'", "").isalpha() for word in weights)
    data = json.loads(
        resources.files("styleprofile.wordlists")
        .joinpath("edited_register.json")
        .read_text("utf-8")
    )
    assert data["name"] == "edited_register" and "learned on all" in data["about"]


def test_surface_metric_matches_the_rate_over_its_tokens():
    text = "Basically, it is pretty good stuff. We should explore practical judgment here."
    register = surface_metrics(text)["register"]
    expected = weighted_rate(words(prose(text).text), load("edited_register"))
    assert register["edited_register_per_1k"] == pytest.approx(expected)
    assert surface_metrics("")["register"]["edited_register_per_1k"] is None


def test_register_is_reported_but_not_scored():
    from styleprofile.metrics import UNSCORED_GROUPS

    assert "register" in UNSCORED_GROUPS


def test_loaded_list_is_read_only():
    with pytest.raises(TypeError):
        load("edited_register")["practical"] = 0.0  # type: ignore[index]
