"""The contrast AUC's document bootstrap: exact reweighting, early stopping, and the cases that
need no resampling at all."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from styleprofile import weighting
from styleprofile.evaluate import evaluate_rewording
from styleprofile.profile import build_reference, load_chunks, window
from styleprofile.weighting import (
    BOOTSTRAP_CHECKPOINTS,
    auc,
    auc_interval,
    bootstrap_auc,
    bootstrap_interval,
    quantile,
    resampled_aucs,
    settle_tolerance,
)


def _documents(rng: random.Random, count: int, mean: float, ties: bool) -> list[list[float]]:
    """``count`` documents of 1-4 chunk scores; rounded to one decimal, they tie often."""
    documents = []
    for _ in range(count):
        scores = [rng.gauss(mean, 1.0) for _ in range(rng.randint(1, 4))]
        documents.append([round(score, 1) for score in scores] if ties else scores)
    return documents


@pytest.mark.parametrize("ties", [False, True])
def test_each_resample_is_the_rank_auc_of_the_documents_drawn(ties: bool) -> None:
    rng = random.Random(3)
    reference = _documents(rng, 12, 0.0, ties)
    contrast = _documents(rng, 7, 0.6, ties)
    # Replay the draws: reference documents, then contrast documents, per resample.
    replay = random.Random(5)
    for value in bootstrap_auc(reference, contrast, resamples=50, seed=5):
        drawn_reference = replay.choices(range(len(reference)), k=len(reference))
        drawn_contrast = replay.choices(range(len(contrast)), k=len(contrast))
        negatives = [score for document in drawn_reference for score in reference[document]]
        positives = [score for document in drawn_contrast for score in contrast[document]]
        assert value == pytest.approx(auc(positives, negatives), abs=1e-12)


def _one_chunk(rng: random.Random, count: int, mean: float) -> list[list[float]]:
    return [[rng.gauss(mean, 1.0)] for _ in range(count)]


def test_early_stop_is_deterministic_and_close_to_the_full_interval() -> None:
    # Many documents make a narrow interval that settles well before the maximum.
    rng = random.Random(11)
    reference, contrast = _one_chunk(rng, 1500, 0.0), _one_chunk(rng, 300, 2.5)

    found = bootstrap_interval(reference, contrast)
    assert bootstrap_interval(reference, contrast) == found
    assert found.method == "bootstrap"
    assert found.resamples in BOOTSTRAP_CHECKPOINTS
    assert found.resamples < BOOTSTRAP_CHECKPOINTS[-1]

    full = bootstrap_auc(reference, contrast, resamples=BOOTSTRAP_CHECKPOINTS[-1])
    assert found.bounds == pytest.approx([quantile(full, 0.025), quantile(full, 0.975)], abs=0.005)
    flat = auc(
        [score for scores in contrast for score in scores],
        [score for scores in reference for score in scores],
    )
    assert flat is not None and found.bounds[0] < flat < found.bounds[1]


def _expected_stop(stream: list[float]) -> int:
    """The stop rule, restated: the first checkpoint where both ends moved less than the
    tolerance at it and at the checkpoint before."""
    previous, settled = None, 0
    for checkpoint in BOOTSTRAP_CHECKPOINTS:
        current = [quantile(stream[:checkpoint], share) for share in (0.025, 0.975)]
        tolerance = min(0.005, max((current[1] - current[0]) * (1 / 50), 0.001))
        assert tolerance == settle_tolerance(current)
        if previous and all(abs(a - b) < tolerance for a, b in zip(current, previous, strict=True)):
            settled += 1
            if settled == 2:
                return checkpoint
        else:
            settled = 0
        previous = current
    return BOOTSTRAP_CHECKPOINTS[-1]


@pytest.mark.parametrize("seed", range(8))
def test_resampling_stops_at_the_first_twice_settled_checkpoint(seed: int) -> None:
    rng = random.Random(seed)
    # From a handful of documents (wide, runs to the maximum) to hundreds (stops early).
    size = (4, 12, 1000, 1500)[seed % 4]
    if size < 100:
        reference = _documents(rng, size, 0.0, ties=seed % 2 == 0)
        contrast = _documents(rng, size, 0.7, ties=seed % 2 == 0)
    else:
        reference, contrast = (
            _one_chunk(rng, size, 0.0),
            _one_chunk(rng, size // 5, 3.0 if size == 1000 else 2.5),
        )
    found = bootstrap_interval(reference, contrast, seed=seed)

    # The resamples are a prefix of the same endless stream.
    stream = bootstrap_auc(reference, contrast, resamples=BOOTSTRAP_CHECKPOINTS[-1], seed=seed)
    assert found.resamples == _expected_stop(stream)
    assert found.bounds == [quantile(stream[: found.resamples], s) for s in (0.025, 0.975)]


def test_early_stop_is_about_as_accurate_as_the_maximum() -> None:
    """A cheap version of the 200-seed measurement in the PR: on 15 against 12 documents
    with ties (a wide interval), the early stop's 90th-percentile error from a 20,000-resample
    interval stays within 1.25 times that of always drawing 2,000."""
    rng = random.Random(11)
    reference = [[rng.randint(0, 5) for _ in range(rng.randint(1, 4))] for _ in range(15)]
    contrast = [[rng.randint(1, 6) for _ in range(rng.randint(1, 4))] for _ in range(12)]
    truth_draws = bootstrap_auc(reference, contrast, resamples=20_000, seed=999_999)
    truth = [quantile(truth_draws, 0.025), quantile(truth_draws, 0.975)]

    def error(bounds: list[float]) -> float:
        return max(abs(a - b) for a, b in zip(bounds, truth, strict=True))

    early, fixed = [], []
    for seed in range(60):
        early.append(error(bootstrap_interval(reference, contrast, seed=seed).bounds))
        draws = bootstrap_auc(reference, contrast, resamples=2000, seed=seed)
        fixed.append(error([quantile(draws, 0.025), quantile(draws, 0.975)]))
    assert quantile(early, 0.9) <= 1.25 * quantile(fixed, 0.9)


def test_perfect_separation_needs_no_resampling(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("resampled")

    monkeypatch.setattr(weighting, "resampled_aucs", never)
    reference = [[0.1, 0.2], [0.3], [0.25, 0.05]]
    contrast = [[0.9], [0.31, 0.8]]
    assert bootstrap_interval(reference, contrast) == ([1.0, 1.0], 0, "exact")
    assert bootstrap_interval(contrast, reference) == ([0.0, 0.0], 0, "exact")
    assert auc_interval(reference, contrast) == [1.0, 1.0]
    # One score throughout: every pair ties, so every resample is 0.5.
    assert bootstrap_interval([[0.4], [0.4, 0.4]], [[0.4], [0.4]]) == ([0.5, 0.5], 0, "exact")


def test_a_tie_across_the_sides_is_not_perfect_separation() -> None:
    # The top reference chunk ties the bottom contrast chunk: the AUC is below 1, and only
    # resamples that leave out one of the tied documents reach 1.
    reference = [[0.1], [0.2], [0.5]]
    contrast = [[0.5], [0.9], [0.8]]
    flat_auc = auc([0.5, 0.9, 0.8], [0.1, 0.2, 0.5])
    assert flat_auc is not None and flat_auc < 1.0
    interval, used, method = bootstrap_interval(reference, contrast)
    assert used > 0 and method == "bootstrap"
    assert interval[0] < 1.0 and interval[1] == 1.0

    # The same holds with the sides swapped.
    interval, used, method = bootstrap_interval(contrast, reference)
    assert used > 0 and method == "bootstrap" and interval[0] == 0.0


def test_the_stream_does_not_depend_on_how_much_is_taken() -> None:
    rng = random.Random(2)
    reference = _documents(rng, 10, 0.0, ties=True)
    contrast = _documents(rng, 10, 0.5, ties=True)
    stream = resampled_aucs(reference, contrast, seed=9)
    first = [next(stream) for _ in range(300)]
    assert bootstrap_auc(reference, contrast, resamples=120, seed=9) == first[:120]
    assert bootstrap_auc(reference, contrast, resamples=300, seed=9) == first


def test_the_method_says_how_the_interval_was_found() -> None:
    separated = weighting.auc_confidence([[0.1], [0.2]], [[0.8], [0.9]])
    assert separated == {
        "auc_ci": [1.0, 1.0],
        "bootstrap": {
            "method": "exact",
            "resamples": 0,
            "unit": "document",
            "reference_documents": 2,
            "contrast_documents": 2,
        },
    }
    rng = random.Random(4)
    overlapping = weighting.auc_confidence(
        _documents(rng, 20, 0.0, ties=False), _documents(rng, 20, 0.5, ties=False)
    )
    assert overlapping["auc_ci"] is not None
    assert overlapping["bootstrap"]["method"] == "bootstrap"
    assert overlapping["bootstrap"]["resamples"] in BOOTSTRAP_CHECKPOINTS
    # One document on a side: no interval, so no method either.
    assert weighting.auc_confidence([[0.1, 0.2]], [[0.8], [0.9]]) == {
        "auc_ci": None,
        "bootstrap": {
            "method": None,
            "resamples": 0,
            "unit": "document",
            "reference_documents": 1,
            "contrast_documents": 2,
        },
    }


def test_references_and_evaluations_record_the_method() -> None:
    examples = Path(__file__).resolve().parent.parent / "examples"
    writer = window(load_chunks([str(examples / "writer")]), 150)
    drafts = window(load_chunks([str(examples / "llm-drafts")]), 150)
    calibration = build_reference(writer, contrast=drafts, parser=None)["contrast"]["calibration"]
    exact = calibration["auc"] in (0.0, 1.0)
    assert calibration["bootstrap"]["method"] == ("exact" if exact else "bootstrap")

    result = evaluate_rewording(writer, drafts, {"same": drafts}, parser=None, retrain=True)
    retrain = result["retrain"]
    for entry in [*result["sets"].values(), retrain, *retrain["by_set"].values()]:
        method, resamples = entry["bootstrap"]["method"], entry["bootstrap"]["resamples"]
        assert (method is None) == (entry["auc_ci"] is None)
        assert method != "exact" or resamples == 0
        assert method != "bootstrap" or resamples in BOOTSTRAP_CHECKPOINTS
