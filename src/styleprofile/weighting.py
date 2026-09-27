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
(``Metric.resolution``, from each metric's unit and denominator in the registry), or 5% of
the mean for other metrics: a spread computed from values that almost never vary can be
arbitrarily small, and would otherwise turn a single semicolon into dozens of standard
deviations.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from styleprofile.metrics import UNSCORED_GROUPS, resolution
from styleprofile.surface import Metrics

Key = tuple[str, str]
ZScores = dict[Key, float]
# A metric without a natural resolution that the reference never varies on has no spread
# and so no true z-score: a chunk that differs scores this value (the "3+ sd" level) and
# a matching chunk 0. Metrics without a measured reliability are capped at it in Delta.
UNSEEN_Z = 3.0
SIGNALS_SHOWN = 5
# The contrast AUC's confidence interval resamples whole documents this many times.
BOOTSTRAP_RESAMPLES = 2000
# Word count alone separating the contrast set this well means likeness may be partly length.
LENGTH_AUC_WARNING = 0.75


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


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
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


def bootstrap_auc(
    reference: Sequence[Sequence[float]],
    contrast: Sequence[Sequence[float]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> list[float]:
    """The contrast-over-reference AUC on documents resampled with replacement.

    Each argument holds one list of chunk scores per document. Reference and contrast
    documents are resampled separately and a document's chunks always move together, so
    the spread reflects how many documents there are rather than how many windows they were
    cut into. The chunks are ranked once; each resample then only reweights them, so a
    resample costs one pass over the chunks.
    """
    ranked = sorted(
        [(score, 0, document) for document, scores in enumerate(reference) for score in scores]
        + [(score, 1, document) for document, scores in enumerate(contrast) for score in scores]
    )
    # Tied scores form one group: (reference documents, contrast documents), one entry per chunk.
    groups: list[tuple[list[int], list[int]]] = []
    previous: float | None = None
    for score, is_contrast, document in ranked:
        if score != previous:
            groups.append(([], []))
            previous = score
        groups[-1][is_contrast].append(document)
    reference_sizes = [len(scores) for scores in reference]
    contrast_sizes = [len(scores) for scores in contrast]
    rng = random.Random(seed)
    aucs: list[float] = []
    for _ in range(resamples):
        reference_draws = _draws(rng, len(reference))
        contrast_draws = _draws(rng, len(contrast))
        below = wins = 0.0
        for reference_documents, contrast_documents in groups:
            tied = sum(reference_draws[document] for document in reference_documents)
            if contrast_documents:
                wins += sum(contrast_draws[document] for document in contrast_documents) * (
                    below + tied / 2
                )
            below += tied
        positives = sum(n * size for n, size in zip(contrast_draws, contrast_sizes, strict=True))
        negatives = sum(n * size for n, size in zip(reference_draws, reference_sizes, strict=True))
        aucs.append(wins / (positives * negatives))
    return aucs


def _draws(rng: random.Random, count: int) -> list[int]:
    """How many times each of ``count`` documents is drawn in one resample."""
    drawn = [0] * count
    for _ in range(count):
        drawn[rng.randrange(count)] += 1
    return drawn


def auc_interval(
    reference: Sequence[Sequence[float]], contrast: Sequence[Sequence[float]]
) -> list[float] | None:
    """The 95% document-bootstrap interval of the AUC, or None with fewer than 2 documents
    on either side (resampling one document cannot show document-to-document variation)."""
    if len(reference) < 2 or len(contrast) < 2:
        return None
    aucs = bootstrap_auc(reference, contrast)
    return [quantile(aucs, 0.025), quantile(aucs, 0.975)]


def length_baseline(
    reference_words: Sequence[float], contrast_words: Sequence[float]
) -> dict[str, Any] | None:
    """How well chunk word count alone separates the contrast set, in its stronger direction.

    A likeness AUC is only informative if it clearly beats this: otherwise the contrast set
    may simply be longer or shorter than the reference.
    """
    value = auc(contrast_words, reference_words)
    if value is None:
        return None
    return {
        "auc": max(value, 1 - value),
        "direction": "contrast longer" if value >= 0.5 else "contrast shorter",
        "reference_median_words": statistics.median(reference_words),
        "contrast_median_words": statistics.median(contrast_words),
    }


def by_document(scores: Sequence[float], sources: Sequence[str]) -> list[list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for score, source in zip(scores, sources, strict=True):
        grouped[source].append(score)
    return list(grouped.values())


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


Fold = tuple[dict[Key, float], dict[Key, float]]


@dataclass(frozen=True)
class CrossValidated:
    """Contrast weights learned on everything, and the likeness of every chunk scored with
    weights learned without its own document.

    ``contrast_folds`` holds, per contrast document, the (effects, rms) learned without it:
    any other text derived from that document (an edited copy, say) must be scored with that
    fold, or its own original would have shaped the weights that judge it.
    """

    effects: dict[Key, float]
    rms: dict[Key, float]
    reference_scores: list[float]
    contrast_scores: list[float]
    contrast_folds: dict[str, Fold]
    cross_validated: bool


def cross_validate(
    reference_held: Sequence[ZScores],
    reference_sources: Sequence[str],
    contrast_z: Sequence[ZScores],
    contrast_sources: Sequence[str],
) -> CrossValidated:
    """Leave-one-document-out likeness scores for the reference and the contrast chunks."""
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
    return CrossValidated(
        effects=effects,
        rms=rms,
        reference_scores=reference_scores,
        contrast_scores=fold_scores(contrast_z, contrast_sources, contrast_folds),
        contrast_folds=contrast_folds,
        cross_validated=cross_validated,
    )


def fold_scores(
    z_scores: Sequence[ZScores], documents: Sequence[str], folds: Mapping[str, Fold]
) -> list[float]:
    """Each chunk's likeness under the fold that excluded ``documents[i]``.

    ``documents`` names the contrast document each chunk belongs to or was derived from; a
    chunk whose document has no fold is an error rather than silently scored in-sample.
    """
    scores: list[float] = []
    for chunk_z, document in zip(z_scores, documents, strict=True):
        if document not in folds:
            raise KeyError(f"no held-out fold for contrast document {document!r}")
        scores.append(likeness(chunk_z, *folds[document])[0])
    return scores


def learn_contrast(
    reference_held: Sequence[ZScores],
    reference_sources: Sequence[str],
    contrast_z: Sequence[ZScores],
    contrast_sources: Sequence[str],
    reference_words: Sequence[float],
    contrast_words: Sequence[float],
) -> dict[str, Any]:
    """Learn effect-size weights and calibrate the likeness score by cross-validation.

    Each reference chunk is scored with weights learned without its own source, and each
    contrast chunk with weights learned without its own source, so the calibration ranges
    show how the score behaves on text the weights have not seen. (Contrast z-scores are
    measured against the full reference, so a held-out reference source still shapes them
    slightly; with many sources the effect is negligible.)
    """
    learned = cross_validate(reference_held, reference_sources, contrast_z, contrast_sources)
    return summarize_contrast(
        learned, reference_sources, contrast_sources, reference_words, contrast_words
    )


def summarize_contrast(
    learned: CrossValidated,
    reference_sources: Sequence[str],
    contrast_sources: Sequence[str],
    reference_words: Sequence[float],
    contrast_words: Sequence[float],
) -> dict[str, Any]:
    """The stored effects and calibration of a cross-validated contrast.

    The AUC gets a document-bootstrap 95% interval, and ``*_words`` (each chunk's prose word
    count) give the AUC of length alone, the baseline the likeness AUC should beat.
    """
    reference_scores, contrast_scores = learned.reference_scores, learned.contrast_scores
    return {
        "effects": nest(learned.effects),
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
            "auc": auc(contrast_scores, reference_scores),
            "auc_ci": auc_interval(
                by_document(reference_scores, reference_sources),
                by_document(contrast_scores, contrast_sources),
            ),
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "unit": "document"},
            "length_baseline": length_baseline(reference_words, contrast_words),
            "cross_validated": learned.cross_validated,
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
