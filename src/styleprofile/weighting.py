"""How metrics are weighted: reliability-weighted Delta and a contrast-learned likeness score.

Delta answers "how far is this from the reference overall?". Every stylistic area counts
equally, and within an area a metric counts less the more it swings in the reference's own
writing, measured on held-out chunks (each document scored against a profile built from the
other documents). That keeps a habit the writer uses only now and then, such as semicolons,
from dominating, without capping it.

The likeness score answers "does this look like the contrast set (say, LLM drafts) rather
than the reference?". Each metric's weight is its effect size squared: the gap between the
contrast set's mean z and the reference's held-out mean z, over the reference's held-out
spread. Only deviations in the contrast direction count, so writing fewer em dashes than
usual does not make text look less like an LLM, and more does make it look more like one.

Every z uses a spread of at least half of one occurrence per chunk for counts
(``resolution``), or 5% of the mean for other metrics: a spread computed from values that
almost never vary can be arbitrarily small, and would otherwise turn a single semicolon
into dozens of standard deviations.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from styleprofile.surface import Metrics

Key = tuple[str, str]
ZScores = dict[Key, float]
UNSCORED_GROUPS = frozenset({"size"})
# A metric without a natural resolution that the reference never varies on has no spread
# and so no true z-score: a chunk that differs scores this value (the "3+ sd" level) and
# a matching chunk 0. Metrics without a measured reliability are capped at it in Delta.
UNSEEN_Z = 3.0
SIGNALS_SHOWN = 5


# What each percentage is a share of; any other *_pct metric is a share of sentences.
_SHARE_OF: dict[str, str] = {
    "long_words_pct": "words",
    "one_sentence_paragraphs_pct": "paragraphs",
    "curly_apostrophe_pct": "apostrophes",
}


def resolution(name: str, counts: Mapping[str, float]) -> float:
    """Half of one occurrence per chunk, in the metric's units: the smallest meaningful spread.

    ``counts`` are the reference's typical words, sentences, paragraphs and apostrophes per
    chunk. A rate per 1,000 words moves by 1000 / words per occurrence; a percentage by
    100 / (whatever it is a share of). Other metrics have no count resolution.
    """
    if name.endswith("_per_1k"):
        units = counts.get("words", 0.0)
        return 500.0 / units if units else 0.0
    if name.endswith("_pct"):
        units = counts.get(_SHARE_OF.get(name, "sentences"), 0.0)
        return 50.0 / units if units else 0.0
    return 0.0


# Metrics without a count resolution (lengths, ratios, diversity) get a spread of at least
# this share of their mean: far below real variation, but enough that a reference which
# happens never to vary cannot turn a small difference into dozens of standard deviations.
RELATIVE_FLOOR = 0.05


def floors(summary: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[Key, float]:
    """Each metric's smallest meaningful spread for chunks the size of the reference's."""
    counts = {name: stats.get("mean") or 0.0 for name, stats in summary.get("size", {}).items()}
    result: dict[Key, float] = {}
    for group, names in summary.items():
        if group in UNSCORED_GROUPS:
            continue
        for name, stats in names.items():
            floor = resolution(name, counts)
            result[(group, name)] = floor or RELATIVE_FLOOR * abs(stats.get("mean") or 0.0)
    return result


def z_score(
    value: float, mean: float, sd: float | None, n: int, floor: float = 0.0
) -> float | None:
    """One metric's z against reference statistics, or None when it cannot be scored."""
    if n < 2:
        return None
    # Saved profiles round to 6 decimals, so equal at that precision is equal.
    if math.isclose(value, mean, rel_tol=1e-6, abs_tol=1e-6):
        return 0.0
    spread = max(sd or 0.0, floor)
    if spread > 0:
        return (value - mean) / spread
    return math.copysign(UNSEEN_Z, value - mean)


def _values(metrics: Metrics) -> dict[Key, float]:
    return {
        (group, name): value
        for group, values in metrics.items()
        if group not in UNSCORED_GROUPS
        for name, value in values.items()
        if value is not None
    }


def held_out_z(
    chunk_metrics: Sequence[Metrics], sources: Sequence[str], floor: Mapping[Key, float]
) -> list[ZScores]:
    """Each chunk's z-scores against the chunks from every other source.

    Running sums keep this linear in the number of chunks, so a corpus of thousands of
    comments costs about as much as profiling it.
    """
    values = [_values(metrics) for metrics in chunk_metrics]
    total, parts = _sums(values, sources)
    scored: list[ZScores] = []
    for chunk_values, source in zip(values, sources, strict=True):
        own = parts[source]
        chunk_z: ZScores = {}
        for key, value in chunk_values.items():
            n = total[key][0] - own[key][0]
            if n < 2:
                continue
            mean = (total[key][1] - own[key][1]) / n
            variance = (total[key][2] - own[key][2] - n * mean * mean) / (n - 1)
            sd = math.sqrt(variance) if variance > 1e-12 * max(1.0, mean * mean) else 0.0
            z = z_score(value, mean, sd, int(n), floor.get(key, 0.0))
            if z is not None:
                chunk_z[key] = z
        scored.append(chunk_z)
    return scored


def reliability(held: Sequence[ZScores], sources: Sequence[str]) -> dict[Key, float]:
    """Root-mean-square held-out z per metric: about 1 for a well-behaved metric, more for
    one that swings unpredictably in the reference's own writing."""
    return _rms(_sums(held, sources)[0])


def delta_weights(rms: Mapping[Key, float]) -> dict[Key, float]:
    """1 / rms^2, floored at 1 so steady metrics are not boosted above an ideal one."""
    return {key: 1.0 / max(value, 1.0) ** 2 for key, value in rms.items()}


def delta(z_scores: ZScores, weights: Mapping[Key, float]) -> tuple[float | None, dict[str, float]]:
    """Weighted mean |z| per area, then the plain mean over areas (every area counts once).

    A metric without a measured reliability counts with weight 1 and its |z| capped at
    UNSEEN_Z, since nothing else would stop a tiny spread from dominating.
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (group, name), z in z_scores.items():
        weight = weights.get((group, name))
        size = abs(z) if weight is not None else min(abs(z), UNSEEN_Z)
        weight = 1.0 if weight is None else weight
        sums[group][0] += weight * size
        sums[group][1] += weight
    by_group = {group: total / weight for group, (total, weight) in sums.items() if weight}
    return (statistics.fmean(by_group.values()) if by_group else None), by_group


def quantile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[round(share * (len(ordered) - 1))]


def likeness(
    z_scores: ZScores, effects: Mapping[Key, float], rms: Mapping[Key, float]
) -> tuple[float, list[dict[str, Any]]]:
    """Effect-size-squared weighted mean of each deviation in the contrast direction.

    Each z is measured in the reference's held-out units (z / max(rms, 1)), so a metric that
    swings wildly in the reference's own writing cannot dominate, and deviations away from
    the contrast direction count as 0. Only metrics the sample has count, so a sample without
    syntax metrics is judged on the rest. Returns the score and its largest contributors.
    """
    present = {key: effect for key, effect in effects.items() if key in z_scores}
    denominator = sum(effect * effect for effect in present.values())
    if not denominator:
        return 0.0, []
    contributions: list[tuple[float, Key, float]] = []
    for key, effect in present.items():
        z = z_scores[key]
        toward = max(0.0, math.copysign(1.0, effect) * z) / max(rms.get(key, 1.0), 1.0)
        if toward:
            contributions.append((effect * effect * toward / denominator, key, z))
    contributions.sort(reverse=True)
    signals = [
        {"metric": f"{group}.{name}", "z": z, "contribution": share}
        for share, (group, name), z in contributions[:SIGNALS_SHOWN]
    ]
    return sum(share for share, _, _ in contributions), signals


Sums = dict[Key, list[float]]


def _sums(
    rows: Sequence[Mapping[Key, float]], sources: Sequence[str]
) -> tuple[Sums, dict[str, Sums]]:
    """[count, sum, sum of squares] per metric, overall and per source."""
    total: Sums = defaultdict(lambda: [0.0, 0.0, 0.0])
    by_source: dict[str, Sums] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0]))
    for chunk_z, source in zip(rows, sources, strict=True):
        for key, z in chunk_z.items():
            for sums in (total[key], by_source[source][key]):
                sums[0] += 1
                sums[1] += z
                sums[2] += z * z
    return total, by_source


def _minus(total: Sums, part: Mapping[Key, list[float]]) -> Sums:
    return {
        key: [value - part.get(key, [0.0, 0.0, 0.0])[index] for index, value in enumerate(sums)]
        for key, sums in total.items()
    }


def _rms(sums: Sums) -> dict[Key, float]:
    return {key: math.sqrt(squares / n) for key, (n, _, squares) in sums.items() if n >= 1}


def _fold(reference: Sums, contrast: Sums) -> tuple[dict[Key, float], dict[Key, float]]:
    """Effects (contrast mean z - reference mean z) / max(rms, 1), and the rms they used."""
    rms = _rms(reference)
    effects = {
        key: (contrast[key][1] / contrast[key][0] - total / n) / max(rms[key], 1.0)
        for key, (n, total, _) in reference.items()
        if n >= 1 and key in contrast and contrast[key][0] >= 1
    }
    return effects, rms


def _auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Probability a positive outscores a negative (ties count half), by ranking once."""
    if not positives or not negatives:
        return None
    ranked = sorted([(value, 1) for value in positives] + [(value, 0) for value in negatives])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(label for _, label in ranked[index : end + 1])
        index = end + 1
    count = len(positives)
    return (rank_sum - count * (count + 1) / 2) / (count * len(negatives))


def calibrate_delta(held: Sequence[ZScores], sources: Sequence[str]) -> dict[str, Any] | None:
    """The reference's own Delta range on held-out chunks, overall and per area.

    Each source is scored with reliability weights learned without it, so the range shows
    how Delta behaves on the writer's new text rather than on text the weights have seen.
    """
    total, parts = _sums(held, sources)
    folds = {
        source: delta_weights(_rms(_minus(total, parts.get(source, {})))) for source in set(sources)
    }
    overall: list[float] = []
    areas: dict[str, list[float]] = defaultdict(list)
    for chunk_z, source in zip(held, sources, strict=True):
        value, by_group = delta(chunk_z, folds[source])
        if value is None:
            continue
        overall.append(value)
        for group, amount in by_group.items():
            areas[group].append(amount)
    if not overall:
        return None
    return {
        "median": statistics.median(overall),
        "p95": quantile(overall, 0.95),
        "max": max(overall),
        "by_group": {
            group: {"median": statistics.median(values), "p95": quantile(values, 0.95)}
            for group, values in areas.items()
        },
    }


def learn_contrast(
    reference_held: Sequence[ZScores],
    reference_sources: Sequence[str],
    contrast_z: Sequence[ZScores],
    contrast_sources: Sequence[str],
) -> dict[str, Any]:
    """Learn effect-size weights and calibrate the likeness score by cross-validation.

    Each reference chunk is scored with weights learned without its own source, and each
    contrast chunk with weights learned without its own source, so the calibration ranges
    show how the score behaves on text the weights have not seen. (Contrast z-scores are
    measured against the full reference, so a held-out reference source still shapes them
    slightly; with many sources the effect is negligible.)
    """
    reference_total, reference_parts = _sums(reference_held, reference_sources)
    contrast_total, contrast_parts = _sums(contrast_z, contrast_sources)
    effects, rms = _fold(reference_total, contrast_total)

    reference_folds = {
        source: _fold(_minus(reference_total, reference_parts.get(source, {})), contrast_total)
        for source in set(reference_sources)
    }
    reference_scores = [
        likeness(chunk_z, *reference_folds[source])[0]
        for chunk_z, source in zip(reference_held, reference_sources, strict=True)
    ]
    cross_validated = len(set(contrast_sources)) > 1
    contrast_folds = {
        source: (
            _fold(reference_total, _minus(contrast_total, contrast_parts.get(source, {})))
            if cross_validated
            else (effects, rms)
        )
        for source in set(contrast_sources)
    }
    contrast_scores = [
        likeness(chunk_z, *contrast_folds[source])[0]
        for chunk_z, source in zip(contrast_z, contrast_sources, strict=True)
    ]
    return {
        "effects": nest(effects),
        "calibration": {
            "reference": {
                "median": statistics.median(reference_scores),
                "p95": quantile(reference_scores, 0.95),
                "max": max(reference_scores),
            },
            "contrast": {
                "median": statistics.median(contrast_scores),
                "min": min(contrast_scores),
            },
            "auc": _auc(contrast_scores, reference_scores),
            "cross_validated": cross_validated,
        },
    }


def flatten(nested: Mapping[str, Mapping[str, float]]) -> dict[Key, float]:
    return {
        (group, name): value for group, values in nested.items() for name, value in values.items()
    }


def nest(flat: Mapping[Key, float]) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = defaultdict(dict)
    for (group, name), value in flat.items():
        nested[group][name] = value
    return dict(nested)
