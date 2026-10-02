"""How metrics are weighted: reliability-weighted Delta and a contrast-learned likeness score.

Delta answers "how far is this from the reference overall?". Every stylistic area counts
equally, and within an area a metric counts less the more it swings in the reference's own
writing, measured on held-out chunks (each document scored against a profile built from the
other documents). That keeps a habit the writer uses only now and then, such as semicolons,
from dominating. Each metric's |z| is also capped at 5 in both scores, so an extreme
value cannot decide the verdict on its own.

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

import functools
import math
import random
import statistics
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby, islice, repeat
from operator import itemgetter, mul
from typing import Any, Literal, NamedTuple, overload

from styleprofile.core import DISTANCES, LIKENESSES, Verdict
from styleprofile.metrics import UNSCORED_GROUPS, resolution
from styleprofile.rows import MetricRows
from styleprofile.rows import key as _key
from styleprofile.schema import (
    AucConfidence,
    Bootstrap,
    LearnedContrast,
    LengthBaseline,
    LengthLikeness,
    LikenessSignal,
)
from styleprofile.surface import Metrics

Key = tuple[str, str]
ZScores = dict[Key, float]
# A metric without a natural resolution that the reference never varies on has no spread
# and so no true z-score: a chunk that differs scores this value (the "3+ sd" level) and
# a matching chunk 0. Metrics without a measured reliability are capped at it in Delta.
UNSEEN_Z = 3.0
Z_CAP = 5.0
SIGNALS_SHOWN = 5
# The narrowest "close" band a calibrated Delta verdict uses, in mean |z|: half a standard
# deviation per metric, half the uncalibrated close threshold of 1.0. Without it an area the
# writer never varies in (Markdown in plain essays) has a held-out range near zero, and any
# trace of it would read as very different.
MIN_CEILING = 0.5
# The same guard for the likeness score. Likeness averages only the part of each z that
# points toward the contrast set, and for noise that one-sided part is on average half of
# |z| (E max(0, z) = E|z| / 2), so the equivalent band is half as wide. It only binds when
# the reference's own held-out likeness is near zero or averaged over very many chunks.
LIKENESS_MIN_CEILING = MIN_CEILING / 2
# With few documents the within-document similarity of pieces (their intraclass
# correlation) is poorly estimated even pooled over the lengths, and an estimate that
# happens to be low calls the pieces more independent than they are. Below FLOOR_DOCUMENTS
# documents it is taken as at least SIMILARITY_FLOOR, a little above the 0.17 measured on
# the synthetic corpus. In a simulation of 3 to 9 documents with true similarities of 0 to
# 0.4, among references that pass as calibrated, the share whose bound is exceeded more
# than 7.5% of the time falls from 7.5% with each length's own estimate to 2.6% (worst case
# 49% to 12%); a floor of 0.3 would reach 0.9% but calibrate half as many small
# references (see docs/method.md).
FLOOR_DOCUMENTS = 10
SIMILARITY_FLOOR = 0.2
# How alike the chunks of one run are taken to be, beyond chance: the share of a chunk's
# variation that every chunk of the run shares, so that it does not average away. Each
# range stores its own (``run_similarity``): the intraclass correlation by document of that
# range's held-out values (its windows' or its pieces' Delta, overall or in one area, or
# their likeness), floored as above below FLOOR_DOCUMENTS documents, and never below
# MIN_RUN_SIMILARITY. A run of one document, or one topic, shares that document's or
# topic's offset; and every run shares whatever sets its text apart from the calibration
# pieces (its format, how it was cut): on the synthetic corpus, cutting held-out text as
# whole paragraphs or as runs of sentences moves an area's mean by up to a tenth of its
# spread, which MIN_RUN_SIMILARITY (sqrt 0.22) covers (see docs/method.md).
MIN_RUN_SIMILARITY = 0.05
# Calibration pieces come many to a document, so their 95th percentile is read as an upper
# confidence bound: one-sided, at this many standard errors of the quantile's share (90%).
UPPER_Z = 1.2816
# The contrast AUC's confidence interval resamples whole documents, checking at each of
# these counts whether both ends of the interval have settled: moved less than a fiftieth of
# its width (at least 0.001, at most 0.005) since the previous checkpoint, at this and the
# previous checkpoint. It resamples at most the last count.
BOOTSTRAP_CHECKPOINTS = (250, 500, 1000, 2000)
BOOTSTRAP_RESAMPLES = BOOTSTRAP_CHECKPOINTS[-1]
BOOTSTRAP_TOLERANCE = 0.005
BOOTSTRAP_TOLERANCE_FLOOR = 0.001
BOOTSTRAP_RELATIVE = 1 / 50
BOOTSTRAP_SETTLED = 2
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
        _key(group, name): value
        for group, values in metrics.items()
        if group not in UNSCORED_GROUPS
        for name, value in values.items()
        if value is not None
    }


def held_out_z(
    chunk_metrics: Sequence[Metrics],
    sources: Sequence[str],
    floor: Mapping[Key, float],
    *,
    others: tuple[Sequence[Metrics], Sequence[str]] | None = None,
) -> Sequence[ZScores]:
    """Each chunk's z-scores against the chunks from every other source.

    With ``others`` (more texts and their sources, such as shorter pieces cut from these
    chunks), those texts are scored instead, each against the chunks from every source but
    its own. Running sums keep this linear in the number of chunks, so a corpus of
    thousands of comments costs about as much as profiling it.
    """
    if others is None:
        return held_out(chunk_metrics, sources, floor)[0]
    return held_out(chunk_metrics, sources, floor, others)[1]


def held_out(
    chunk_metrics: Sequence[Metrics],
    sources: Sequence[str],
    floor: Mapping[Key, float],
    others: tuple[Sequence[Metrics], Sequence[str]] | None = None,
) -> tuple[ZRows, list[ZScores]]:
    """``held_out_z`` of the chunks and, given ``others``, of the others too, from one pass
    over the chunks' sums. The chunks' z-scores come back as ``ZRows``, and each chunk's
    values are made from its metrics when needed rather than kept for every chunk."""
    values = _LazyValues(chunk_metrics)
    total, parts = _sums(values, sources)
    held = ZRows(_scored_keys(chunk_metrics[0]) if chunk_metrics else ())
    _held_z(values, sources, total, parts, floor, held, own_rows=True)
    pieces: list[ZScores] = []
    if others is not None and others[0]:
        _held_z(_LazyValues(others[0]), others[1], total, parts, floor, pieces)
    return held, pieces


def _scored_keys(metrics: Metrics) -> list[Key]:
    """The scored metrics' keys, in the order a chunk's values and z-scores list them."""
    return [
        _key(group, name)
        for group, values in metrics.items()
        if group not in UNSCORED_GROUPS
        for name in values
    ]


class _LazyValues(Sequence[dict[Key, float]]):
    """Each chunk's scored values (``_values``), made when asked for rather than kept (from
    ``MetricRows`` without rebuilding each chunk's dicts)."""

    def __init__(self, metrics: Sequence[Metrics]) -> None:
        self._metrics = metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self) -> Iterator[dict[Key, float]]:
        return map(self.__getitem__, range(len(self._metrics)))

    def _one(self, index: int) -> dict[Key, float]:
        if isinstance(self._metrics, MetricRows):
            found = self._metrics.scored_values(index)
            if found is not None:
                return found
        return _values(self._metrics[index])

    @overload
    def __getitem__(self, index: int) -> dict[Key, float]: ...
    @overload
    def __getitem__(self, index: slice) -> list[dict[Key, float]]: ...
    def __getitem__(self, index: int | slice) -> dict[Key, float] | list[dict[Key, float]]:
        if isinstance(index, slice):
            return [self._one(position) for position in range(*index.indices(len(self)))]
        return self._one(index)


class ZRows(Sequence[ZScores]):
    """Rows of z-scores held compactly: an array of doubles per row over ``keys``, NaN where
    the row has no z. A row reads back as exactly the dict it was made from (same keys, same
    order, same floats); one whose keys do not follow ``keys``' order is kept as its dict.
    As dicts, 20,000 chunks' z-scores took about 150 MB; as arrays, 25 MB."""

    def __init__(self, keys: Sequence[Key]) -> None:
        self.keys = tuple(keys)
        self._index = {key: position for position, key in enumerate(self.keys)}
        self._blank = array("d", [math.nan]) * len(self.keys)
        self._rows: list[array[float] | ZScores] = []
        # ``_sums`` of these rows, by (by_source, sources).
        self.sums: dict[tuple[bool, tuple[str, ...]], tuple[Sums, _Parts]] = {}

    def append(self, z_scores: ZScores) -> None:
        self.sums.clear()
        row = array("d", self._blank)
        last = -1
        for key, z in z_scores.items():
            position = self._index.get(key)
            if position is None or position <= last or z != z:
                self._rows.append(dict(z_scores))
                return
            row[position] = z
            last = position
        self._rows.append(row)

    def array(self, index: int) -> array[float] | None:
        """Row ``index`` as its array (NaN where it has no z), or None when it is kept as its
        dict."""
        row = self._rows[index]
        return None if isinstance(row, dict) else row

    def pairs(self, index: int) -> list[tuple[Key, float]]:
        """Row ``index``'s (key, z) pairs in order, without making its dict."""
        row = self._rows[index]
        if isinstance(row, dict):
            return list(row.items())
        return [(key, z) for key, z in zip(self.keys, row, strict=True) if z == z]

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> ZScores: ...
    @overload
    def __getitem__(self, index: slice) -> list[ZScores]: ...
    def __getitem__(self, index: int | slice) -> ZScores | list[ZScores]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        row = self._rows[index]
        if isinstance(row, dict):
            return row
        return {key: z for key, z in zip(self.keys, row, strict=True) if z == z}


def _held_z(
    values: Sequence[Mapping[Key, float]],
    sources: Sequence[str],
    total: Sums,
    parts: _Parts,
    floor: Mapping[Key, float],
    out: ZRows | list[ZScores],
    *,
    own_rows: bool = False,
) -> None:
    """Each row's z-scores against ``total`` less its own source's part: the arithmetic of
    ``z_score`` over the leave-one-source-out mean and spread, written out, since it runs
    for every metric of every chunk."""
    stats = {key: (sums[0], sums[1], sums[2], floor.get(key, 0.0)) for key, sums in total.items()}
    isclose, sqrt, copysign = math.isclose, math.sqrt, math.copysign
    for index, (chunk_values, source) in enumerate(zip(values, sources, strict=True)):
        # A source of one row is summed from that row as its part would be (0.0 + v): the
        # row being scored when these are the rows summed (``own_rows``).
        found_index = parts.single_index(source)
        single = (
            None
            if found_index is None
            else chunk_values
            if own_rows and found_index == index
            else parts.row(found_index)
        )
        own = {} if single is not None else parts.get(source, {})
        chunk_z: ZScores = {}
        if single is chunk_values:
            _held_z_alone(chunk_values, stats, chunk_z)
            out.append(chunk_z)
            continue
        for key, value in chunk_values.items():
            found = stats.get(key)
            if found is None:
                continue
            count, first, second, least = found
            if single is not None:
                mine_value = single.get(key)
                if mine_value is None:
                    n, sum1, sum2 = count, first, second
                else:
                    n = count - 1.0
                    sum1 = first - (0.0 + mine_value)
                    sum2 = second - (0.0 + mine_value * mine_value)
            else:
                mine = own.get(key, _EMPTY)
                n, sum1, sum2 = count - mine[0], first - mine[1], second - mine[2]
            if n < 2:
                continue
            mean = sum1 / n
            variance = (sum2 - n * mean * mean) / (n - 1)
            sd = sqrt(variance) if variance > 1e-12 * max(1.0, mean * mean) else 0.0
            # z_score, as it reads for n of 2 or more.
            if isclose(value, mean, rel_tol=1e-6, abs_tol=1e-6):
                chunk_z[key] = 0.0
                continue
            spread = max(sd or 0.0, least)
            if spread > 0:
                chunk_z[key] = (value - mean) / spread
            else:
                chunk_z[key] = copysign(UNSEEN_Z, value - mean)
        out.append(chunk_z)


def _held_z_alone(
    values: Mapping[Key, float],
    stats: Mapping[Key, tuple[float, float, float, float]],
    out: ZScores,
) -> None:
    """``_held_z`` for a row that is its source's only one, so its part is itself: the same
    arithmetic, in a loop of its own since it is most of the work for a corpus of short
    documents. ``max(a, b)`` is written ``b if b > a else a``, which is what it returns."""
    isclose, sqrt, copysign = math.isclose, math.sqrt, math.copysign
    for key, value in values.items():
        found = stats.get(key)
        if found is None:
            continue
        count, first, second, least = found
        n = count - 1.0
        if n < 2:
            continue
        mean = (first - (0.0 + value)) / n
        variance = ((second - (0.0 + value * value)) - n * mean * mean) / (n - 1)
        square = mean * mean
        sd = sqrt(variance) if variance > 1e-12 * (square if square > 1.0 else 1.0) else 0.0
        if isclose(value, mean, rel_tol=1e-6, abs_tol=1e-6):
            out[key] = 0.0
            continue
        spread = sd or 0.0
        if least > spread:
            spread = least
        if spread > 0:
            out[key] = (value - mean) / spread
        else:
            out[key] = copysign(UNSEEN_Z, value - mean)


def reliability(held: Sequence[ZScores], sources: Sequence[str]) -> dict[Key, float]:
    """Root-mean-square held-out z per metric: about 1 for a well-behaved metric, more for
    one that swings unpredictably in the reference's own writing."""
    return _rms(_sums(held, sources, by_source=False)[0])


def delta_weights(rms: Mapping[Key, float]) -> dict[Key, float]:
    """1 / rms^2, floored at 1 so steady metrics are not boosted above an ideal one."""
    return {key: 1.0 / max(value, 1.0) ** 2 for key, value in rms.items()}


def delta(z_scores: ZScores, weights: Mapping[Key, float]) -> tuple[float | None, dict[str, float]]:
    """Weighted mean |z| per area, then the plain mean over areas (every area counts once).

    Every metric's |z| is capped at Z_CAP. A metric without a measured reliability counts
    with weight 1 and the stricter UNSEEN_Z cap.
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for (group, name), z in z_scores.items():
        weight = weights.get((group, name))
        size = min(abs(z), Z_CAP if weight is not None else UNSEEN_Z)
        weight = 1.0 if weight is None else weight
        sums[group][0] += weight * size
        sums[group][1] += weight
    by_group = {group: total / weight for group, (total, weight) in sums.items() if weight}
    return (statistics.fmean(by_group.values()) if by_group else None), by_group


def quantile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[round(share * (len(ordered) - 1))]


def _group_size(groups: Sequence[str]) -> float:
    """n0, the usual adjusted mean group size of a one-way analysis of variance."""
    count = len(groups)
    sizes = [len(index) for index in _by_source(groups).values()]
    if len(sizes) < 2:
        return float(count)
    return (count - sum(size * size for size in sizes) / count) / (len(sizes) - 1)


def intraclass_correlation(values: Sequence[float], groups: Sequence[str]) -> float | None:
    """How alike values from one group (one document) are: the intraclass correlation from
    a one-way analysis of variance, clipped to [0, 1]; None with fewer than two groups or
    no group of two or more."""
    count = len(values)
    positions = _by_source(groups)
    if len(positions) < 2 or count <= len(positions):
        return None
    mean = statistics.fmean(values)
    means = {
        group: statistics.fmean(values[i] for i in index) for group, index in positions.items()
    }
    between = sum(len(index) * (means[group] - mean) ** 2 for group, index in positions.items())
    within = sum(
        (values[i] - means[group]) ** 2 for group, index in positions.items() for i in index
    )
    mean_between = between / (len(positions) - 1)
    mean_within = within / (count - len(positions))
    spread = mean_between + (_group_size(groups) - 1) * mean_within
    return min(max((mean_between - mean_within) / spread, 0.0), 1.0) if spread > 0 else 0.0


def effective_count(
    values: Sequence[float], groups: Sequence[str], icc: float | None = None
) -> float:
    """How many independent values ``values`` are worth, given that values from one group
    (one document) are alike: n over the design effect 1 + (n0 - 1) x ICC, with n0 the
    adjusted mean group size. ``icc`` is the intraclass correlation to use; by default it
    is estimated from the values themselves (``intraclass_correlation``). Values from one
    group count as one."""
    count = len(values)
    if len(set(groups)) < 2:
        return float(min(count, 1))
    if icc is None:
        icc = intraclass_correlation(values, groups) or 0.0
    return count / (1 + (_group_size(groups) - 1) * icc)


def upper_quantile(
    values: Sequence[float],
    groups: Sequence[str],
    share: float,
    z: float = UPPER_Z,
    icc: float | None = None,
) -> float:
    """An upper confidence bound on the ``share`` quantile of ``values``: the quantile at
    share + z x sqrt(share (1 - share) / n_eff), with n_eff from ``effective_count``, so a
    bound read from few documents is set higher rather than trusted as exact. For the 95th
    percentile that is the sample maximum below about 31 effective values."""
    effective = max(effective_count(values, groups, icc), 1.0)
    return quantile(values, min(share + z * math.sqrt(share * (1 - share) / effective), 1.0))


def t_quantile(share: float, df: int) -> float:
    """The ``share`` quantile of Student's t with ``df`` degrees of freedom (at least 1):
    exact for 1 and 2, and the Cornish-Fisher expansion in 1/df beyond (Abramowitz and
    Stegun 26.7.5), within 0.002 of the exact value from 3 at the shares used here."""
    df = max(df, 1)
    if df == 1:
        return math.tan(math.pi * (share - 0.5))
    if df == 2:
        return (2 * share - 1) / math.sqrt(2 * share * (1 - share))
    z = statistics.NormalDist().inv_cdf(share)
    terms = (
        (z**3 + z) / 4,
        (5 * z**5 + 16 * z**3 + 3 * z) / 96,
        (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384,
        (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160,
    )
    return z + sum(term / df ** (power + 1) for power, term in enumerate(terms))


# The one-sided share an upper confidence bound covers, as UPPER_Z does for a normal.
UPPER_SHARE = 0.9


def upper_mean(values: Sequence[float], groups: Sequence[str], icc: float | None = None) -> float:
    """An upper confidence bound on the mean of ``values``: the mean plus t standard errors,
    the standard error from n_eff (``effective_count``) and t the 90% quantile of Student's t
    with one degree of freedom fewer than there are documents (``groups``), since documents,
    not values, are what vary independently. It is the centre a mean over many chunks is
    read against (``pooled_ceiling``): held-out Deltas are skewed to the right, so their
    mean sits above their median, and a mean over many chunks settles on the mean. A mean
    estimated from few documents is set higher rather than trusted as exact, as
    ``upper_quantile`` does for the 95th percentile."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean
    effective = max(effective_count(values, groups, icc), 1.0)
    t = t_quantile(UPPER_SHARE, len(set(groups)) - 1)
    return mean + t * statistics.stdev(values) / math.sqrt(effective)


def document_similarity(values: Sequence[float], groups: Sequence[str]) -> float:
    """How alike values from one document are (``intraclass_correlation``, 0 when it cannot
    be estimated), at least SIMILARITY_FLOOR below FLOOR_DOCUMENTS documents: a few
    documents estimate it poorly, and a lucky low estimate would call their values more
    independent than they are."""
    estimate = intraclass_correlation(values, groups) or 0.0
    return max(estimate, SIMILARITY_FLOOR) if len(set(groups)) < FLOOR_DOCUMENTS else estimate


def run_similarity(values: Sequence[float], groups: Sequence[str]) -> float:
    """How much of their variation the chunks of one run are taken to share, for one range
    (see MIN_RUN_SIMILARITY): the ``document_similarity`` of its held-out values, and never
    below MIN_RUN_SIMILARITY.

    It is the estimate itself, not an upper confidence bound on it: with a few values per
    document the bound is two to three times the estimate, and on the synthetic corpora it
    cut the share of batches with 10% of LLM chunks that read "somewhat different" or worse
    from 47-100% to 3-100%. Pieces are cut from each window twice (as paragraphs and as
    excerpts), and the two cuts overlap, which makes pieces of one document look a little
    more alike than they are: an error on the side of a looser bound."""
    return max(document_similarity(values, groups), MIN_RUN_SIMILARITY)


@dataclass(frozen=True)
class HeldDeltas:
    """Held-out Deltas of chunks (or pieces), overall and per area, with the source each
    came from (see ``held_out_deltas``)."""

    overall: list[float]
    sources: list[str]
    areas: dict[str, list[float]]
    area_sources: dict[str, list[str]]


def likeness(
    z_scores: ZScores, effects: Mapping[Key, float], rms: Mapping[Key, float]
) -> tuple[float, list[LikenessSignal]]:
    """Effect-size-squared weighted mean of each deviation in the contrast direction.

    Each z is capped at Z_CAP in size, then measured in the reference's held-out units
    (z / max(rms, 1)), so a metric that swings wildly in the reference's own writing cannot
    dominate, and deviations away from
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
        toward = min(Z_CAP, max(0.0, math.copysign(1.0, effect) * z)) / max(rms.get(key, 1.0), 1.0)
        if toward:
            contributions.append((effect * effect * toward / denominator, key, z))
    contributions.sort(reverse=True)
    signals: list[LikenessSignal] = [
        {"metric": f"{group}.{name}", "z": z, "contribution": share}
        for share, (group, name), z in contributions[:SIGNALS_SHOWN]
    ]
    return sum(share for share, _, _ in contributions), signals


Sums = dict[Key, list[float]]
_EMPTY = (0.0, 0.0, 0.0)


class _Parts(Mapping[str, Mapping[Key, Sequence[float]]]):
    """Each source's [count, sum, sum of squares] per metric (see ``_sums``). A source with
    one row, which is every source of a corpus of comments, is summed from that row when
    asked for rather than kept: kept for 20,000 sources, the sums took gigabytes."""

    def __init__(
        self,
        rows: Sequence[Mapping[Key, float]],
        several: dict[str, Sums],
        single: dict[str, int],
    ) -> None:
        self._rows = rows
        self._several = several
        self._single = single

    def single_index(self, source: str) -> int | None:
        """The row of a source that has only one, by position, else None."""
        return self._single.get(source)

    def row(self, index: int) -> Mapping[Key, float]:
        return self._rows[index]

    def __getitem__(self, source: str) -> Mapping[Key, Sequence[float]]:
        part = self._several.get(source)
        if part is not None:
            return part
        index = self._single.get(source)
        if index is None:
            return {}
        # As summing from zero gives them: 0.0 + z, not z (they differ for -0.0).
        return {key: (1.0, 0.0 + z, 0.0 + z * z) for key, z in self._rows[index].items()}

    def __iter__(self) -> Iterator[str]:
        yield from self._several
        yield from self._single

    def __len__(self) -> int:
        return len(self._several) + len(self._single)


def _sums(
    rows: Sequence[Mapping[Key, float]], sources: Sequence[str], *, by_source: bool = True
) -> tuple[Sums, _Parts]:
    """[count, sum, sum of squares] per metric, overall and (unless not ``by_source``) per
    source (``_Parts``). For ``ZRows`` they are worked out once per set of sources: the
    reference's held-out z-scores are summed by the Delta range, the reliability and the
    contrast alike. Callers only read them."""
    if isinstance(rows, ZRows):
        memo = (by_source, tuple(sources))
        found = rows.sums.get(memo)
        if found is None:
            found = rows.sums[memo] = _sums_of(rows, sources, by_source=by_source)
        return found
    return _sums_of(rows, sources, by_source=by_source)


def _sums_of(
    rows: Sequence[Mapping[Key, float]], sources: Sequence[str], *, by_source: bool
) -> tuple[Sums, _Parts]:
    total: Sums = defaultdict(lambda: [0.0, 0.0, 0.0])
    for chunk_z in _row_pairs(rows):
        for key, z in chunk_z:
            sums = total[key]
            sums[0] += 1
            sums[1] += z
            sums[2] += z * z
    several: dict[str, Sums] = {}
    single: dict[str, int] = {}
    if by_source:
        for source, positions in _by_source(sources).items():
            if len(positions) == 1:
                single[source] = positions[0]
                continue
            own: Sums = defaultdict(lambda: [0.0, 0.0, 0.0])
            for index in positions:
                for key, z in rows.pairs(index) if isinstance(rows, ZRows) else rows[index].items():
                    sums = own[key]
                    sums[0] += 1
                    sums[1] += z
                    sums[2] += z * z
            several[source] = own
    elif len(rows) != len(sources):
        raise ValueError("rows and sources differ in length")
    return total, _Parts(rows, several, single)


def _row_pairs(rows: Sequence[Mapping[Key, float]]) -> Iterator[Iterable[tuple[Key, float]]]:
    """Each row's (key, value) pairs in order; ``ZRows`` without making their dicts."""
    if isinstance(rows, ZRows):
        return map(rows.pairs, range(len(rows)))
    return (row.items() for row in rows)


def _less(sums: Sequence[float], part: Mapping[Key, Sequence[float]], key: Key) -> list[float]:
    """``sums`` without one source's ``part`` of metric ``key``."""
    removed = part.get(key)
    return list(sums) if removed is None else [a - b for a, b in zip(sums, removed, strict=True)]


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


def _fold_without(
    reference: Sums,
    contrast: Sums,
    full: tuple[dict[Key, float], dict[Key, float]],
    reference_part: Mapping[Key, Sequence[float]] | None = None,
    contrast_part: Mapping[Key, Sequence[float]] | None = None,
) -> tuple[dict[Key, float], dict[Key, float]]:
    """``_fold`` of the totals less one source's part, given ``full``, the fold of the whole
    totals. Only the metrics the part has change, so only they are recomputed: with thousands
    of one-comment sources that is a few metrics each rather than every metric."""
    reference_part, contrast_part = reference_part or {}, contrast_part or {}
    effects, rms = dict(full[0]), dict(full[1])
    sqrt = math.sqrt
    for key in {*reference_part, *contrast_part}:
        sums = reference.get(key)
        if sums is None:
            continue
        # Removing a part only lowers counts, so a metric either keeps its place in the full
        # fold (and the order likeness sums in) or drops out; none is added.
        removed = reference_part.get(key)
        if removed is None:
            n, total, squares = sums
        else:
            n, total, squares = sums[0] - removed[0], sums[1] - removed[1], sums[2] - removed[2]
        if n < 1:
            effects.pop(key, None)
            rms.pop(key, None)
            continue
        spread = rms[key] = sqrt(squares / n)
        other = contrast.get(key, _EMPTY)
        removed = contrast_part.get(key)
        if removed is None:
            count, contrast_total = other[0], other[1]
        else:
            count, contrast_total = other[0] - removed[0], other[1] - removed[1]
        if count >= 1:
            effects[key] = (contrast_total / count - total / n) / max(spread, 1.0)
        else:
            effects.pop(key, None)
    return effects, rms


def held_out_effects(
    reference_held: Sequence[ZScores],
    reference_sources: Sequence[str],
    contrast_z: Sequence[ZScores],
    learned: CrossValidated,
) -> Callable[[str], dict[Key, float]]:
    """The likeness effects learned without one reference source, as ``likeness_range``
    scores its pieces: a function of the source, each computed once."""
    reference_total, reference_parts = _sums(reference_held, reference_sources)
    contrast_total = _sums(contrast_z, [""] * len(contrast_z), by_source=False)[0]

    @functools.cache
    def without(source: str) -> dict[Key, float]:
        return _fold_without(
            reference_total,
            contrast_total,
            (learned.effects, learned.rms),
            reference_part=reference_parts.get(source, {}),
        )[0]

    return without


def _rms_without(
    total: Sums, full: Mapping[Key, float], part: Mapping[Key, Sequence[float]]
) -> dict[Key, float]:
    """``_rms`` of ``total`` less one source's ``part``, given ``full``, the rms of the whole
    total; only the metrics the part has are recomputed."""
    rms = dict(full)
    for key in part:
        n, _, squares = _less(total[key], part, key)
        if n < 1:
            rms.pop(key, None)
        else:
            rms[key] = math.sqrt(squares / n)
    return rms


def _by_source(sources: Sequence[str]) -> dict[str, list[int]]:
    """The positions of each source's rows, in order of first appearance."""
    positions: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        positions[source].append(index)
    return positions


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
    """The contrast-over-reference AUC on ``resamples`` document resamples (see
    ``resampled_aucs``)."""
    return list(islice(resampled_aucs(reference, contrast, seed), resamples))


def resampled_aucs(
    reference: Sequence[Sequence[float]], contrast: Sequence[Sequence[float]], seed: int = 0
) -> Iterator[float]:
    """The contrast-over-reference AUC on documents resampled with replacement, endlessly.

    Each argument holds one list of chunk scores per document. Reference and contrast
    documents are resampled separately and a document's chunks always move together, so
    the spread reflects how many documents there are rather than how many windows they were
    cut into. The chunks are ranked once; each resample then only reweights them by how
    often their document was drawn, so a resample is a few passes over the chunks in C
    (``map`` and ``accumulate``) rather than a Python loop. For a seed, the n-th value is
    the same however many are taken.
    """
    # Ties rank reference chunks first, so at a contrast chunk the running reference weight
    # counts every tied reference chunk; half of each tie is taken back below.
    ranked = sorted(
        [(score, 0, document) for document, scores in enumerate(reference) for score in scores]
        + [(score, 1, document) for document, scores in enumerate(contrast) for score in scores]
    )
    # The reference chunks' documents in rank order, and for each contrast chunk its document
    # and how many reference chunks rank below it (ties rank reference chunks first).
    reference_count, contrast_count = len(reference), len(contrast)
    reference_documents: list[int] = []
    contrast_documents: list[int] = []
    contrast_below: list[int] = []
    for _, is_contrast, document in ranked:
        if is_contrast:
            contrast_documents.append(document)
            contrast_below.append(len(reference_documents))
        else:
            reference_documents.append(document)
    # Groups of tied scores holding both sides: (reference documents, contrast documents).
    mixed: list[tuple[list[int], list[int]]] = []
    for _, tied in groupby(ranked, key=itemgetter(0)):
        sides: tuple[list[int], list[int]] = ([], [])
        for _, is_contrast, document in tied:
            sides[is_contrast].append(document)
        if sides[0] and sides[1]:
            mixed.append(sides)
    reference_range, contrast_range = range(reference_count), range(contrast_count)
    rng = random.Random(seed)
    while True:
        # How often each document was drawn (a document not drawn is not counted: 0).
        reference_draws = _draws(rng, reference_range).get
        contrast_draws = _draws(rng, contrast_range).get
        # The running reference weight below each contrast chunk, from one running sum over
        # the reference chunks: the same whole numbers as summing over every chunk.
        below = [0, *accumulate(map(reference_draws, reference_documents, repeat(0)))]
        above = list(map(contrast_draws, contrast_documents, repeat(0)))
        wins = float(sum(map(mul, above, map(below.__getitem__, contrast_below))))
        for reference_tied, contrast_tied in mixed:
            tied = sum(map(reference_draws, reference_tied, repeat(0)))
            wins -= sum(map(contrast_draws, contrast_tied, repeat(0))) * tied / 2
        yield wins / (sum(above) * below[-1])


def _draws(rng: random.Random, documents: range) -> Counter[int]:
    """How many times each document is drawn in one resample of ``len(documents)``; a
    document not drawn is not in it."""
    # What ``rng.choices(documents, k=count)`` draws (one ``random()`` each, floored times
    # the count), counted in C rather than in a Python loop.
    count = len(documents)
    return Counter(map(int, map(mul, islice(iter(rng.random, None), count), repeat(count + 0.0))))


class Interval(NamedTuple):
    """An AUC interval and how it was found: ``method`` is ``"exact"`` when every resample
    would give the same AUC, so none were drawn, or ``"bootstrap"``."""

    bounds: list[float]
    resamples: int
    method: Literal["exact", "bootstrap"]


def settle_tolerance(interval: Sequence[float]) -> float:
    """How little both ends must move between checkpoints for the interval to count as
    settled: a fiftieth of its width, between ``BOOTSTRAP_TOLERANCE_FLOOR`` and
    ``BOOTSTRAP_TOLERANCE``."""
    width = interval[1] - interval[0]
    return min(BOOTSTRAP_TOLERANCE, max(width * BOOTSTRAP_RELATIVE, BOOTSTRAP_TOLERANCE_FLOOR))


def bootstrap_interval(
    reference: Sequence[Sequence[float]], contrast: Sequence[Sequence[float]], seed: int = 0
) -> Interval:
    """The 95% document-bootstrap interval of the AUC, the resamples it took, and its method.

    With perfect separation (every contrast chunk above every reference chunk, or every one
    below) every resample has the same AUC, 1 or 0, so the interval is exact and needs none;
    the same holds when every chunk has one score (0.5). A tie across the sides is not
    perfect separation: resamples that drop the tied documents can reach 1 or 0, others
    cannot.

    Otherwise resampling stops at the first of ``BOOTSTRAP_CHECKPOINTS`` where both ends have
    moved less than ``settle_tolerance`` since the previous checkpoint, and did at the
    checkpoint before too. Consecutive estimates share most of their draws, so one small
    move understates the error; asking for two keeps the early stop about as accurate as
    always drawing the maximum (see docs/method.md).
    """
    reference_scores = [score for scores in reference for score in scores]
    contrast_scores = [score for scores in contrast for score in scores]
    if min(contrast_scores) > max(reference_scores):
        return Interval([1.0, 1.0], 0, "exact")
    if max(contrast_scores) < min(reference_scores):
        return Interval([0.0, 0.0], 0, "exact")
    if (
        min(contrast_scores)
        == max(contrast_scores)
        == min(reference_scores)
        == max(reference_scores)
    ):
        # One score throughout: every pair ties, so every resample's AUC is one half.
        return Interval([0.5, 0.5], 0, "exact")
    draws = resampled_aucs(reference, contrast, seed)
    aucs: list[float] = []
    previous: list[float] | None = None
    settled = 0
    interval: list[float] = []
    for checkpoint in BOOTSTRAP_CHECKPOINTS:
        aucs.extend(islice(draws, checkpoint - len(aucs)))
        interval = [quantile(aucs, 0.025), quantile(aucs, 0.975)]
        tolerance = settle_tolerance(interval)
        if previous is not None and all(
            abs(now - before) < tolerance for now, before in zip(interval, previous, strict=True)
        ):
            settled += 1
            if settled == BOOTSTRAP_SETTLED:
                break
        else:
            settled = 0
        previous = interval
    return Interval(interval, len(aucs), "bootstrap")


def auc_interval(
    reference: Sequence[Sequence[float]], contrast: Sequence[Sequence[float]]
) -> list[float] | None:
    """The 95% document-bootstrap interval of the AUC, or None with fewer than 2 documents
    on either side (resampling one document cannot show document-to-document variation)."""
    return auc_confidence(reference, contrast)["auc_ci"]


def auc_confidence(
    reference: Sequence[Sequence[float]], contrast: Sequence[Sequence[float]]
) -> AucConfidence:
    """``auc_ci``, the AUC's 95% document-bootstrap interval, and ``bootstrap``, how it was
    found: ``method`` is ``"exact"`` when every resample would give the same AUC (perfect
    separation, or one score throughout) so none were drawn, ``"bootstrap"`` when it was
    resampled, and None, with no interval, when either side has fewer than 2 documents
    (resampling one document cannot show document-to-document variation). ``resamples`` is
    how many were drawn, and ``*_documents`` how many documents each side had."""
    found: Bootstrap = {
        "method": None,
        "resamples": 0,
        "unit": "document",
        "reference_documents": len(reference),
        "contrast_documents": len(contrast),
    }
    if len(reference) < 2 or len(contrast) < 2:
        return {"auc_ci": None, "bootstrap": found}
    interval = bootstrap_interval(reference, contrast)
    found.update(method=interval.method, resamples=interval.resamples)
    return {"auc_ci": interval.bounds, "bootstrap": found}


def length_baseline(
    reference_words: Sequence[float], contrast_words: Sequence[float]
) -> LengthBaseline | None:
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


def held_out_deltas(held: Sequence[ZScores], sources: Sequence[str]) -> HeldDeltas:
    """Each chunk's Delta, overall and per area, with reliability weights learned without
    its own source, so the values show how Delta behaves on the writer's new text rather
    than on text the weights have seen."""
    total, parts = _sums(held, sources)
    full = delta_weights(_rms(total))
    scored: list[tuple[float | None, dict[str, float]]] = [(None, {})] * len(held)
    alone = _DeltaWithoutOwn(total, held.keys if isinstance(held, ZRows) else None)
    for source, positions in _by_source(sources).items():
        if len(positions) == 1:
            index = positions[0]
            scored[index] = alone.delta(held, index)
            continue
        # Weights learned without this source: only the metrics it has differ from the full
        # weights, so only they are recomputed.
        weights = dict(full)
        part = parts[source]
        for key, removed in part.items():
            sums = total[key]
            n, squares = sums[0] - removed[0], sums[2] - removed[2]
            if n >= 1:
                weights[key] = 1.0 / max(math.sqrt(squares / n), 1.0) ** 2
            else:
                weights.pop(key, None)
        for index in positions:
            scored[index] = delta(held[index], weights)
    overall: list[float] = []
    overall_sources: list[str] = []
    areas: dict[str, list[float]] = defaultdict(list)
    area_sources: dict[str, list[str]] = defaultdict(list)
    for (value, by_group), source in zip(scored, sources, strict=True):
        if value is None:
            continue
        overall.append(value)
        overall_sources.append(source)
        for group, amount in by_group.items():
            areas[group].append(amount)
            area_sources[group].append(source)
    return HeldDeltas(overall, overall_sources, dict(areas), dict(area_sources))


def _pairs(rows: Sequence[Mapping[Key, float]], index: int) -> Iterable[tuple[Key, float]]:
    return rows.pairs(index) if isinstance(rows, ZRows) else rows[index].items()


class _DeltaWithoutOwn:
    """``delta`` of a source's only chunk with the weights learned without it: what
    ``held_out_deltas`` computes by copying the weights and replacing every metric the chunk
    has, done in one pass over the chunk, with each metric's totals looked up once. The
    arithmetic is the same, step for step."""

    def __init__(self, total: Sums, keys: Sequence[Key] | None) -> None:
        self._total = total
        # For rows kept as arrays: each metric's (group, count without the chunk, sum of
        # squares), by position; None for a metric no row has.
        self._columns: list[tuple[str, float, float] | None] | None = None
        if keys is not None:
            self._columns = [
                (key[0], counts[0] - 1.0, counts[2]) if (counts := total.get(key)) else None
                for key in keys
            ]

    def delta(
        self, rows: Sequence[Mapping[Key, float]], index: int
    ) -> tuple[float | None, dict[str, float]]:
        sqrt = math.sqrt
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        if self._columns is not None and isinstance(rows, ZRows) and (row := rows.array(index)):
            for column, z in zip(self._columns, row, strict=True):
                if z != z or column is None:
                    continue
                group, n, squares = column
                if n >= 1:
                    weight: float = 1.0 / max(sqrt((squares - (0.0 + z * z)) / n), 1.0) ** 2
                    size = min(abs(z), Z_CAP)
                else:
                    weight, size = 1.0, min(abs(z), UNSEEN_Z)
                entry = sums[group]
                entry[0] += weight * size
                entry[1] += weight
        else:
            for key, z in _pairs(rows, index):
                counts = self._total[key]
                n = counts[0] - 1.0
                if n >= 1:
                    weight = 1.0 / max(sqrt((counts[2] - (0.0 + z * z)) / n), 1.0) ** 2
                    size = min(abs(z), Z_CAP)
                else:
                    weight, size = 1.0, min(abs(z), UNSEEN_Z)
                entry = sums[key[0]]
                entry[0] += weight * size
                entry[1] += weight
        by_group = {group: amount / weight for group, (amount, weight) in sums.items() if weight}
        return (statistics.fmean(by_group.values()) if by_group else None), by_group


def calibrate_delta(
    held: Sequence[ZScores],
    sources: Sequence[str],
    *,
    upper: bool = False,
    icc: float | None = None,
    deltas: HeldDeltas | None = None,
) -> dict[str, Any] | None:
    """The reference's own Delta range on held-out chunks (``held_out_deltas``, or
    ``deltas`` when already computed), overall and per area.

    Each range has its median, an upper confidence bound on its mean (``upper_mean``), its
    95th percentile, and ``similarity``: the share of their variation the chunks of one run
    share, from that range's values (``run_similarity``). The overall mean uses the
    intraclass correlation ``icc``, by default ``document_similarity`` of the overall
    values; each area's mean uses its own values' ``document_similarity``, as its run
    similarity does, since an area can vary by document far more than the whole.

    With ``upper`` (for calibration pieces, many per document), each 95th percentile is an
    upper confidence bound (``upper_quantile``) with ``icc`` too, the overall 99th
    percentile is added, and ``effective`` records how many independent pieces the overall
    values are worth.
    """
    deltas = deltas or held_out_deltas(held, sources)
    overall, overall_sources = deltas.overall, deltas.sources
    if not overall:
        return None
    if icc is None:
        icc = document_similarity(overall, overall_sources)

    def p95(values: list[float], groups: list[str]) -> float:
        if upper:
            return upper_quantile(values, groups, 0.95, icc=icc)
        return quantile(values, 0.95)

    extra: dict[str, float] = {}
    if upper:
        extra = {
            "p99": quantile(overall, 0.99),
            "effective": effective_count(overall, overall_sources, icc),
        }
    return {
        "median": statistics.median(overall),
        "mean": upper_mean(overall, overall_sources, icc=icc),
        "p95": p95(overall, overall_sources),
        "similarity": run_similarity(overall, overall_sources),
        **extra,
        "max": max(overall),
        "by_group": {
            group: {
                "median": statistics.median(values),
                "mean": upper_mean(
                    values,
                    deltas.area_sources[group],
                    icc=document_similarity(values, deltas.area_sources[group]),
                ),
                "p95": p95(values, deltas.area_sources[group]),
                "similarity": run_similarity(values, deltas.area_sources[group]),
            }
            for group, values in deltas.areas.items()
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
    contrast_folds: Mapping[str, Fold]
    cross_validated: bool


def cross_validate(
    reference_held: Sequence[ZScores],
    reference_sources: Sequence[str],
    contrast_z: Sequence[ZScores],
    contrast_sources: Sequence[str],
) -> CrossValidated:
    """Leave-one-document-out likeness scores for the reference and the contrast chunks.

    Each reference chunk is scored with weights learned without its own source, and each
    contrast chunk with weights learned without its own source, so calibration shows how the
    score behaves on text the weights have not seen. (Contrast z-scores are measured against
    the full reference, so a held-out reference source still shapes them slightly; with many
    sources the effect is negligible.)
    """
    reference_total, reference_parts = _sums(reference_held, reference_sources)
    contrast_total, contrast_parts = _sums(contrast_z, contrast_sources)
    effects, rms = _fold(reference_total, contrast_total)

    # Each reference source's fold is used for its own chunks and then dropped, so thousands
    # of one-comment sources never hold thousands of folds at once.
    reference_scores = [0.0] * len(reference_held)
    alone = _LikenessWithoutOwn(effects, reference_total, contrast_total, reference_held)
    for source, positions in _by_source(reference_sources).items():
        if len(positions) == 1:
            reference_scores[positions[0]] = alone.score(reference_held, positions[0])
            continue
        fold = _fold_without(
            reference_total, contrast_total, (effects, rms), reference_part=reference_parts[source]
        )
        for index in positions:
            reference_scores[index] = likeness(reference_held[index], *fold)[0]
    cross_validated = len(set(contrast_sources)) > 1
    contrast_folds = _Folds(
        set(contrast_sources),
        reference_total,
        contrast_total,
        contrast_parts,
        (effects, rms),
        cross_validated=cross_validated,
    )
    return CrossValidated(
        effects=effects,
        rms=rms,
        reference_scores=reference_scores,
        contrast_scores=fold_scores(contrast_z, contrast_sources, contrast_folds),
        contrast_folds=contrast_folds,
        cross_validated=cross_validated,
    )


class _Folds(Mapping[str, Fold]):
    """Each contrast document's fold (``_fold_without`` its part), made when asked for rather
    than kept: a contrast set of thousands of drafts would otherwise hold thousands of
    folds. Without ``cross_validated`` every document gets the full fold."""

    def __init__(
        self,
        sources: set[str],
        reference: Sums,
        contrast: Sums,
        parts: _Parts,
        full: Fold,
        *,
        cross_validated: bool,
    ) -> None:
        self._sources = sources
        self._reference, self._contrast, self._parts = reference, contrast, parts
        self._full = full
        self._cross_validated = cross_validated

    def __getitem__(self, source: str) -> Fold:
        if source not in self._sources:
            raise KeyError(source)
        if not self._cross_validated:
            return self._full
        return _fold_without(
            self._reference, self._contrast, self._full, contrast_part=self._parts[source]
        )

    def __contains__(self, source: object) -> bool:
        return source in self._sources

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


class _LikenessWithoutOwn:
    """``likeness(z_scores, *fold)[0]`` for a reference source's only chunk, where ``fold`` is
    ``_fold_without`` of that source (``effects`` being the full fold's): every metric the
    chunk has is refitted without it, so only those count, in the full fold's order. The
    arithmetic is the same, step for step, with what does not depend on the chunk (its
    metric's totals less one count, the contrast's mean) worked out once; the contributions
    are summed largest first, as ``likeness`` sorts them."""

    def __init__(
        self,
        effects: Mapping[Key, float],
        reference: Sums,
        contrast: Sums,
        rows: Sequence[Mapping[Key, float]],
    ) -> None:
        # (key, count without the chunk, sum, sum of squares, contrast mean) of each metric
        # that can have an effect without one chunk, in the full fold's order.
        self._metrics: list[tuple[Key, float, float, float, float]] = []
        for key in effects:
            counts = reference[key]
            n = counts[0] - 1.0
            other = contrast.get(key, _EMPTY)
            if n < 1 or other[0] < 1:
                continue
            self._metrics.append((key, n, counts[1], counts[2], other[1] / other[0]))
        # With rows kept as arrays, each metric's z is read by position, all in one call.
        self._pick: Callable[[array[float]], Sequence[float]] | None = None
        if isinstance(rows, ZRows) and len(self._metrics) >= 2:
            index = {key: position for position, key in enumerate(rows.keys)}
            positions = [index.get(key) for key, *_ in self._metrics]
            if None not in positions:
                self._pick = itemgetter(*positions)  # type: ignore[arg-type]

    def score(self, rows: Sequence[Mapping[Key, float]], index: int) -> float:
        sqrt, copysign = math.sqrt, math.copysign
        present: list[tuple[float, float, float]] = []
        row = rows.array(index) if isinstance(rows, ZRows) else None
        values: Sequence[float]
        if row is not None and self._pick is not None:
            values = self._pick(row)
        else:
            z_scores = dict(_pairs(rows, index))
            values = [z_scores.get(key, math.nan) for key, *_ in self._metrics]
        for (_, n, first, second, contrast_mean), z in zip(self._metrics, values, strict=True):
            if z != z:
                continue
            total = first - (0.0 + z)
            spread = sqrt((second - (0.0 + z * z)) / n)
            # max(spread, 1.0), as max returns it (the first of equals).
            scale = 1.0 if spread < 1.0 else spread
            effect = (contrast_mean - total / n) / scale
            present.append((effect, z, scale))
        denominator = sum(effect * effect for effect, _, _ in present)
        if not denominator:
            return 0.0
        shares = []
        for effect, z, scale in present:
            signed = copysign(1.0, effect) * z
            toward = min(Z_CAP, max(0.0, signed)) / scale
            if toward:
                shares.append(effect * effect * toward / denominator)
        shares.sort(reverse=True)
        return sum(shares)


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


def likeness_range(
    reference_held: Sequence[ZScores],
    reference_sources: Sequence[str],
    contrast_z: Sequence[ZScores],
    piece_held: Sequence[ZScores],
    piece_sources: Sequence[str],
    contrast_piece_z: Sequence[ZScores],
    contrast_piece_sources: Sequence[str],
    learned: CrossValidated,
    *,
    icc: float | None = None,
) -> LengthLikeness:
    """The likeness range of shorter pieces: the reference's held-out median and 95th
    percentile, and the contrast pieces' median.

    Effects stay those learned on whole chunks, each piece scored with the fold that left
    out its own document, as ``cross_validate`` does; the rms that scales each z is the
    pieces' own, since that is what a text of their length is scored with. A reference
    piece's rms leaves out its document too. ``reference_held``, ``reference_sources`` and
    ``contrast_z`` are the whole chunks ``learned`` came from; folds are made only for the
    documents that have pieces, one at a time.
    """
    reference_total, reference_parts = _sums(reference_held, reference_sources)
    contrast_total = _sums(contrast_z, [""] * len(contrast_z), by_source=False)[0]
    piece_total, piece_parts = _sums(piece_held, piece_sources)
    rms = _rms(piece_total)
    reference_scores = [0.0] * len(piece_held)
    for source, positions in _by_source(piece_sources).items():
        effects = _fold_without(
            reference_total,
            contrast_total,
            (learned.effects, learned.rms),
            reference_part=reference_parts.get(source, {}),
        )[0]
        own_rms = _rms_without(piece_total, rms, piece_parts[source])
        for index in positions:
            reference_scores[index] = likeness(piece_held[index], effects, own_rms)[0]
    contrast_scores = [
        likeness(chunk_z, learned.contrast_folds[source][0], rms)[0]
        for chunk_z, source in zip(contrast_piece_z, contrast_piece_sources, strict=True)
    ]
    return {
        "reference": {
            "median": statistics.median(reference_scores),
            "mean": upper_mean(reference_scores, piece_sources, icc=icc),
            "p95": upper_quantile(reference_scores, piece_sources, 0.95, icc=icc),
            "similarity": run_similarity(reference_scores, piece_sources),
        },
        "contrast": {"median": statistics.median(contrast_scores)},
    }


def summarize_contrast(
    learned: CrossValidated,
    reference_sources: Sequence[str],
    contrast_sources: Sequence[str],
    reference_words: Sequence[float],
    contrast_words: Sequence[float],
) -> LearnedContrast:
    """The stored effects and calibration of a cross-validated contrast.

    The AUC gets a document-bootstrap 95% interval (``auc_confidence``), and ``*_words``
    (each chunk's prose word count) give the AUC of length alone, the baseline the likeness
    AUC should beat.
    """
    reference_scores, contrast_scores = learned.reference_scores, learned.contrast_scores
    return {
        "effects": nest(learned.effects),
        "calibration": {
            "reference": {
                "median": statistics.median(reference_scores),
                "mean": upper_mean(
                    reference_scores,
                    reference_sources,
                    icc=document_similarity(reference_scores, reference_sources),
                ),
                "p95": quantile(reference_scores, 0.95),
                "similarity": run_similarity(reference_scores, reference_sources),
                "max": max(reference_scores),
            },
            "contrast": {
                "median": statistics.median(contrast_scores),
                "min": min(contrast_scores),
            },
            "auc": auc(contrast_scores, reference_scores),
            **auc_confidence(
                by_document(reference_scores, reference_sources),
                by_document(contrast_scores, contrast_sources),
            ),
            "length_baseline": length_baseline(reference_words, contrast_words),
            "cross_validated": learned.cross_validated,
        },
    }


# Verdicts: where a Delta or likeness score sits relative to the reference's own range.

DISTANCE_WORDS: tuple[Verdict, ...] = DISTANCES
# What every verdict says for a text too short to judge (see ``calibration``).
TOO_SHORT = Verdict.TOO_SHORT


def delta_level(delta: float, ceiling: float | None = None) -> int:
    """0-3 for Delta, relative to a ceiling from the reference's held-out range when known.

    Up to that ceiling (see ``mean_ceiling``) reads as close; 1.5x and 2x mark the next
    steps. Without a calibrated reference, fixed steps suit a writer whose own text scores
    about 0.8.
    """
    if ceiling:
        return (
            0
            if delta <= ceiling
            else 1
            if delta <= 1.5 * ceiling
            else 2
            if delta <= 2 * ceiling
            else 3
        )
    return 0 if delta < 1.0 else 1 if delta < 1.5 else 2 if delta < 2.5 else 3


def centre(stats: Mapping[str, Any]) -> float:
    """Where a mean over many chunks settles, as a range stores it: its upper bound on the
    held-out mean (``upper_mean``). (A reference from before ranges stored it is refused as
    outdated, ``profile.check_version``.)

    It is not capped at the 95% bound. A heavy-tailed range (rare large values, such as an
    area that is usually absent) can have a mean above its 95th percentile, and a pooled
    mean still converges to the mean, so capping it would flag a long run of the writer's
    own text; ``pooled_ceiling`` then reads a long run against the mean itself."""
    return stats["mean"]


def mean_ceiling(stats: Mapping[str, Any], count: int, floor: float = MIN_CEILING) -> float | None:
    """The usual upper bound for an average over ``count`` chunks (``pooled_ceiling`` of
    ``count`` equal ranges)."""
    if stats.get("p95") is None:
        return None
    return pooled_ceiling([stats] * max(count, 1), floor)


def pooled_ceiling(stats: Sequence[Mapping[str, Any]], floor: float = MIN_CEILING) -> float | None:
    """The usual upper bound for the mean of several chunks, each with the range for its
    own length.

    One chunk is unusual above its 95% bound, p95, which is its bound. A mean over n chunks
    settles on the mean of their centres (``centre``): held-out Deltas are skewed to the
    right, so their mean sits above their median, and a bound that closed in on the median
    would call the writer's own text different once enough of it is pooled. Each chunk's
    spread about its centre is s = p95 - centre (none when the centre is higher), and the
    chunks of one run share a part r of their variation that does not average away: each
    range's ``similarity`` (``run_similarity``). As for equicorrelated values, the mean's
    spread is sqrt(sum of (1 - r) s^2 + (sum of sqrt(r) s)^2) / n, which for n chunks of one
    range is s x sqrt(r + (1 - r) / n): it never closes in on the centre entirely. The
    bound is never below ``floor``, so a near-zero held-out range cannot make a negligible
    deviation look large.
    """
    if not stats or any(item.get("p95") is None for item in stats):
        return None
    if len(stats) == 1:
        return max(stats[0]["p95"], floor)
    centres = [centre(item) for item in stats]
    spreads = [max(item["p95"] - middle, 0.0) for item, middle in zip(stats, centres, strict=True)]
    alike = [item["similarity"] for item in stats]
    independent = sum((1 - r) * s * s for r, s in zip(alike, spreads, strict=True))
    shared = sum(math.sqrt(r) * s for r, s in zip(alike, spreads, strict=True))
    spread = math.sqrt(independent + shared * shared)
    return max(statistics.fmean(centres) + spread / len(stats), floor)


def likeness_level(score: float, calibration: Mapping[str, Any], count: int = 1) -> int:
    """0-3 from the reference's own held-out range up to the contrast set's typical score."""
    ceiling = (
        mean_ceiling(calibration["reference"], count, LIKENESS_MIN_CEILING) or LIKENESS_MIN_CEILING
    )
    return likeness_step(score, ceiling, calibration["contrast"]["median"])


def likeness_step(score: float, ceiling: float, target: float) -> int:
    """0-3 for a likeness score, given the top of the reference's usual range (``ceiling``)
    and the contrast set's typical score (``target``)."""
    if score <= ceiling:
        return 0
    if target <= ceiling:
        # The contrast drafts score no higher than the reference: the score cannot tell
        # them apart, so it never claims more than a few traits.
        return 1
    return 1 if score < (ceiling + target) / 2 else 2 if score < target else 3


def likeness_words(level: int, label: str) -> str:
    """A likeness level (see ``likeness_level``) in words naming the contrast set."""
    return LIKENESSES[level].words(label)


def flatten(nested: Mapping[str, Mapping[str, float]]) -> dict[Key, float]:
    return {
        (group, name): value for group, values in nested.items() for name, value in values.items()
    }


def nest(flat: Mapping[Key, float]) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = defaultdict(dict)
    for (group, name), value in flat.items():
        nested[group][name] = value
    return dict(nested)
