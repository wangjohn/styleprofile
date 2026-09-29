"""Length-aware verdicts: how the writer's own text varies at the length being judged.

Every z-score is measured against the spread of the reference's windows (about 500 words by
default). A shorter text varies far more than a window by chance alone: a 75-word paragraph
has a handful of sentences, so one semicolon or one long sentence moves its rates a long way,
and judged against the windows' range the writer's own paragraphs would read as "very
different".

So ``build`` also cuts the reference's windows into pieces of about 75, 150 and 300 words at
sentence or block boundaries, and measures each piece as it would a draft: its z-scores
against the windows of every other document. Per length it stores each metric's held-out
rms, the Delta range overall and per area, and the likeness range of reference and contrast
pieces cut the same way (``calibration.by_length``), with the number of pieces behind each.

``score`` judges each chunk at its own length (``Lengths.at``): stored values are
interpolated linearly in log word count, with the windows themselves as the longest anchor.
A chunk under ``MIN_JUDGED_WORDS`` words, or shorter than any length the reference has enough
pieces for, gets no verdict ("too short to judge") and its numbers are only indicative.
``verdict`` pools a run's chunks into the headline, each chunk read against its own range.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, cast

from styleprofile.schema import (
    ChunkCalibration,
    GroupRange,
    LengthCalibration,
    LengthDelta,
    LikenessAtLength,
    ScoreVerdict,
    VerdictArea,
    VerdictDelta,
    VerdictLikeness,
)
from styleprofile.surface import Metrics, classify, plain_sentences
from styleprofile.weighting import (
    DISTANCE_WORDS,
    FLOOR_DOCUMENTS,
    LIKENESS_MIN_CEILING,
    MIN_CEILING,
    SIMILARITY_FLOOR,
    TOO_SHORT,
    HeldDeltas,
    Key,
    ZScores,
    calibrate_delta,
    centre,
    delta_level,
    flatten,
    held_out_deltas,
    held_out_z,
    intraclass_correlation,
    likeness_step,
    likeness_words,
    nest,
    pooled_ceiling,
    reliability,
)

# The shorter lengths the reference is calibrated at, in prose words.
CALIBRATION_LENGTHS = (75, 150, 300)
# Below this a text gets no verdict: a few sentences vary too much by chance to judge.
MIN_JUDGED_WORDS = 75
# A length is calibrated only with pieces from at least this many documents, and worth at
# least MIN_CALIBRATION_PIECES independent pieces (``weighting.effective_count``: pieces of
# one document are alike, so they count for less). The verdict's bound is an upper bound on
# the pieces' 95th percentile; with 20 independent pieces one lies above it, so it is
# observed rather than set by the single largest piece.
MIN_CALIBRATION_DOCUMENTS = 3
MIN_CALIBRATION_PIECES = 20
# The contrast's likeness range at a length is only its median, which needs fewer pieces.
MIN_CONTRAST_PIECES = 5
# At most this many words of pieces are measured per length, taken evenly across the corpus,
# so calibration costs a bounded amount on a large corpus: 30,000 words are about 400 pieces
# of 75 words and 100 of 300. The contrast only needs its median at each length, so it gets
# a smaller share.
CALIBRATION_WORDS = 30_000
CONTRAST_CALIBRATION_WORDS = 10_000
# Pieces are cut from a chunk only when it holds at least this many of them, so a piece is
# a genuine part of a window rather than the window itself.
PIECES_PER_CHUNK = 1.5
# A piece ends at a paragraph break rather than mid-paragraph when the break is within this
# share of a piece of the ideal cut.
PARAGRAPH_PREFERENCE = 0.25
# rms values are stored to this many decimals, which keeps ``by_length`` to about 20 KB.
RMS_DIGITS = 3


# Cutting windows into pieces.


@dataclass(frozen=True)
class _Unit:
    """The smallest part a piece is made of: a sentence of a plain paragraph, or a whole
    block (code, heading, list, quote or table)."""

    raw: str
    words: int
    block: int  # units of one block join with a space, other units with a blank line
    sticky: bool  # a list continuation, which a cut never separates from its item


def _units(markdown: str) -> list[_Unit]:
    """The chunk's units in order. Their sizes only place the cuts, so whitespace-separated
    tokens stand in for prose words (a piece's true count is measured afterwards)."""
    units: list[_Unit] = []
    for index, block in enumerate(classify(markdown)):
        split = plain_sentences(block)
        if split is None:
            size = 0 if block.code else len(block.raw.split())
            units.append(_Unit(block.raw, size, index, block.continues_list))
        else:
            units += [_Unit(sentence, len(sentence.split()), index, False) for sentence in split]
    return units


def _cuts(units: Sequence[_Unit], length: int, *, excerpts: bool) -> list[tuple[int, int]]:
    """Where to cut ``units`` into pieces of about ``length`` words: (start, end) unit ranges.

    The units are divided into round(words / length) pieces of equal size, each cut at the
    sentence or block boundary nearest its share, so no text is left over. Unless
    ``excerpts``, a paragraph break a little further from the share wins over a cut inside a
    paragraph. None when the units hold fewer than 1.5 pieces.
    """
    starts = [0]
    for unit in units:
        starts.append(starts[-1] + unit.words)
    total = starts[-1]
    count = round(total / length)
    if count < 2 or total < PIECES_PER_CHUNK * length:
        return []
    penalty = 0.0 if excerpts else PARAGRAPH_PREFERENCE * length

    def cost(index: int, target: float) -> float:
        inside = units[index].block == units[index - 1].block
        return abs(starts[index] - target) + (penalty if inside else 0.0)

    cuts = [0]
    for part in range(1, count):
        target = part * total / count
        allowed = [index for index in range(cuts[-1] + 1, len(units)) if not units[index].sticky]
        if not allowed:
            break
        cuts.append(min(allowed, key=lambda index: cost(index, target)))
    cuts.append(len(units))
    return list(pairwise(cuts))


def _joined(units: Sequence[_Unit], start: int, end: int) -> str:
    text = units[start].raw
    for previous, unit in pairwise(units[start:end]):
        text += (" " if unit.block == previous.block else "\n\n") + unit.raw
    return text


def cut(markdown: str, length: int, *, excerpts: bool = False) -> list[str]:
    """Cut one chunk into pieces of about ``length`` prose words, as Markdown.

    Pieces end at sentence or block boundaries: by default at a paragraph break when one is
    nearly as near as the ideal cut, like a paragraph quoted whole; with ``excerpts``, at the
    nearest sentence, like a passage lifted from the middle of a text. A chunk shorter than
    1.5 pieces gives none.
    """
    units = _units(markdown)
    return [_joined(units, start, end) for start, end in _cuts(units, length, excerpts=excerpts)]


def plan_pieces(
    texts: Sequence[str], sizes: Sequence[int], lengths: Sequence[int], budget: int
) -> list[tuple[int, int, str]]:
    """(chunk index, length, Markdown) of every piece to measure for calibration.

    Only lengths the median chunk holds 1.5 times are cut. Every chunk is cut both ways
    (``cut`` and ``cut(excerpts=True)``), since a short text may be a whole paragraph or an
    excerpt; when the pieces of a length hold more than ``budget`` words, every k-th piece
    is kept, so every document still contributes in proportion to its text however few and
    large the chunks are.
    """
    if not texts:
        return []
    median = statistics.median(sizes)
    usable = [length for length in lengths if PIECES_PER_CHUNK * length <= median]
    if not usable:
        return []
    shortest = PIECES_PER_CHUNK * min(usable)
    units = {
        index: _units(text)
        for index, (text, size) in enumerate(zip(texts, sizes, strict=True))
        if size >= shortest
    }
    planned: list[tuple[int, int, str]] = []
    for length in usable:
        ranges = [
            (index, start, end)
            for index, chunk_units in units.items()
            for excerpts in (False, True)
            for start, end in _cuts(chunk_units, length, excerpts=excerpts)
        ]
        words = sum(sum(unit.words for unit in units[i][a:b]) for i, a, b in ranges)
        step = max(1, math.ceil(words / budget))
        planned += [
            (index, length, _joined(units[index], start, end))
            for index, start, end in ranges[::step]
        ]
    return planned


# Build time: the stored calibration per length.


def held_out_pieces(
    window_metrics: Sequence[Metrics],
    window_documents: Sequence[str],
    piece_metrics: Sequence[Metrics],
    piece_documents: Sequence[str],
    floor: Mapping[Key, float],
) -> list[ZScores]:
    """Each piece's z-scores against the windows of every other document, as a draft of its
    length is scored against the whole reference."""
    return held_out_z(
        window_metrics, window_documents, floor, others=(piece_metrics, piece_documents)
    )


Pieces = tuple[Sequence[ZScores], Sequence[str], Sequence[float]]


def calibrate_lengths(pieces: Mapping[int, Pieces]) -> tuple[dict[int, LengthCalibration], float]:
    """Every length's stored calibration (``calibrate_length``), from its pieces' held-out
    z-scores (``held_out_pieces``), documents and word counts; and the intraclass
    correlation their bounds share (``similarity``)."""
    deltas = {
        length: held_out_deltas(held, documents) for length, (held, documents, _) in pieces.items()
    }
    documents = {document for _, sources, _ in pieces.values() for document in sources}
    icc = similarity([(found.overall, found.sources) for found in deltas.values()], len(documents))
    entries = {
        length: calibrate_length(held, sources, words, deltas=deltas[length], icc=icc)
        for length, (held, sources, words) in pieces.items()
    }
    return entries, icc


def similarity(values: Sequence[tuple[Sequence[float], Sequence[str]]], documents: int) -> float:
    """How alike one document's pieces are: the intraclass correlation of their Deltas,
    pooled over the lengths (weighted by pieces), since one length's estimate from a few
    documents swings widely; and, below ``FLOOR_DOCUMENTS`` documents, at least
    ``SIMILARITY_FLOOR``, so that a lucky low estimate cannot pass a length as calibrated
    with an optimistic bound."""
    estimates = [
        (estimate, len(found))
        for found, sources in values
        if (estimate := intraclass_correlation(found, sources)) is not None
    ]
    total = sum(count for _, count in estimates)
    pooled = sum(estimate * count for estimate, count in estimates) / total if total else 0.0
    return max(pooled, SIMILARITY_FLOOR) if documents < FLOOR_DOCUMENTS else pooled


def calibrate_length(
    held: Sequence[ZScores],
    piece_documents: Sequence[str],
    words: Sequence[float],
    *,
    deltas: HeldDeltas | None = None,
    icc: float | None = None,
) -> LengthCalibration:
    """One length's stored calibration, from its pieces' held-out z-scores
    (``held_out_pieces``), their documents and their word counts, with the intraclass
    correlation ``icc`` for its bounds (estimated from these pieces alone when None).

    The entry always records how many pieces and documents it rests on, and how many
    independent pieces they are worth (``effective``); the rms and the Delta range only when
    there are enough (``enough``).
    """
    entry: LengthCalibration = {
        "pieces": len(held),
        "documents": len(set(piece_documents)),
        "words": statistics.median(words) if words else 0.0,
    }
    if len(held) < MIN_CALIBRATION_PIECES or entry["documents"] < MIN_CALIBRATION_DOCUMENTS:
        return entry
    # Every 95th percentile is an upper confidence bound; the 99th percentile is for
    # flagging one passage among many (plan PR 12's drift localization).
    delta = calibrate_delta(held, piece_documents, upper=True, icc=icc, deltas=deltas)
    if delta is None:
        return entry
    delta.pop("max", None)
    entry["effective"] = delta.pop("effective")
    if not enough(entry):
        return entry
    entry["reliability"] = nest(
        {key: round(value, RMS_DIGITS) for key, value in reliability(held, piece_documents).items()}
    )
    # With ``upper``, less ``max`` and ``effective``: median, p95, p99 and the areas.
    entry["delta"] = cast(LengthDelta, delta)
    return entry


def enough(entry: Mapping[str, Any]) -> bool:
    """Whether a length has pieces from enough documents, worth enough independent pieces,
    to be calibrated."""
    return (
        entry.get("documents", 0) >= MIN_CALIBRATION_DOCUMENTS
        and entry.get("effective", 0) >= MIN_CALIBRATION_PIECES
    )


def shortfall(entry: Mapping[str, Any]) -> str:
    """What a length rests on, e.g. "28 pieces from 7 documents, worth 12 independent ones"."""
    documents = entry.get("documents", 0)
    text = f"{entry.get('pieces', 0)} pieces from {documents} document{'s' * (documents != 1)}"
    if "effective" in entry and entry["effective"] < entry.get("pieces", 0) - 0.5:
        text += f", worth {entry['effective']:.0f} independent ones"
    return text


# Score time: the calibration for one chunk's length.


@dataclass(frozen=True)
class AtLength:
    """What judging a chunk of ``words`` words reads from the reference.

    ``rms`` weights Delta and scales likeness, ``scale`` is how many times more each metric
    swings at this length than in a window (only factors above 1), and ``delta``,
    ``by_group`` and ``likeness`` are the ranges the verdicts read (None or empty without
    calibration). When ``judged`` is False, ``reason`` says why and the rest is the nearest
    calibrated length's, for indicative numbers only.
    """

    words: int
    judged: bool
    reason: str | None
    rms: dict[Key, float]
    scale: dict[Key, float]
    delta: GroupRange | None
    by_group: dict[str, GroupRange]
    likeness: LikenessAtLength | None
    # Without calibration, how many times wider than a window's the fixed steps are read
    # for a shorter text (``stretch``); 1 at a window's length or more.
    stretch: float = 1.0

    def row(self) -> ChunkCalibration:
        """What a score report keeps per chunk, so a saved report can be shown again."""
        return {
            "judged": self.judged,
            "reason": self.reason,
            "delta": self.delta,
            "delta_by_group": self.by_group,
            "likeness": self.likeness,
            "length_scale": nest(self.scale),
            "stretch": self.stretch,
        }


@dataclass(frozen=True)
class _Anchor:
    words: float
    covers: float  # the shortest text this anchor judges without a shorter one
    rms: dict[Key, float]
    delta: GroupRange | None
    by_group: dict[str, GroupRange]
    likeness: LikenessAtLength | None


def _likeness_range(stored: Mapping[str, Any] | None) -> LikenessAtLength | None:
    if not stored or "reference" not in stored:
        return None
    reference = _range(stored["reference"])
    if reference is None:
        return None
    return {**reference, "target": stored["contrast"]["median"]}


def _range(stats: Mapping[str, Any] | None) -> GroupRange | None:
    """A stored range as a verdict reads it: its median (what the output calls typical),
    the centre a mean over many chunks settles on (``weighting.centre``), its 95% bound,
    and the share of their variation a run's chunks share (``weighting.run_similarity``)."""
    if not stats or stats.get("p95") is None:
        return None
    return {
        "median": stats.get("median", stats["p95"]),
        "mean": centre(stats),
        "p95": stats["p95"],
        "similarity": stats["similarity"],
    }


def _interpolate(points: Sequence[tuple[float, float]], words: float) -> float:
    """Linear in log word count between the anchors either side; the nearest one outside."""
    if words <= points[0][0]:
        return points[0][1]
    for (low, below), (high, above) in pairwise(points):
        if words <= high:
            if high <= low:
                return above
            share = math.log(words / low) / math.log(high / low)
            return below + share * (above - below)
    return points[-1][1]


def stretch(anchor_words: float, words: float) -> float:
    """How much wider a range read at ``anchor_words`` should be for a text of ``words``:
    sqrt(anchor_words / words) below it, since a rate's chance variation grows as the text
    shrinks; 1 at or above it."""
    if words <= 0 or words >= anchor_words:
        return 1.0
    return math.sqrt(anchor_words / words)


class Lengths:
    """A reference's calibration anchors, read once and then asked for any chunk length."""

    def __init__(self, reference: Mapping[str, Any]) -> None:
        calibration = reference.get("calibration") or {}
        contrast = reference.get("contrast") or {}
        # Without held-out calibration (a reference from one document) there are no ranges
        # at all: verdicts use fixed steps, and only the minimum length applies.
        self.calibrated = bool(calibration.get("delta"))
        # Its windows' length, when there is no calibration to read it from.
        size = ((reference.get("summary") or {}).get("size") or {}).get("words") or {}
        self.uncalibrated_words = (reference.get("settings") or {}).get("window_words") or (
            size.get("mean") or 0.0
        )
        self.window_rms = flatten(reference.get("reliability", {}))
        self.stored: dict[int, dict[str, Any]] = {
            int(length): entry for length, entry in calibration.get("by_length", {}).items()
        }
        window_words = calibration.get("chunk_words") or 0.0
        anchors = [
            _Anchor(
                words=window_words,
                # window() makes no window shorter than half a window.
                covers=window_words / 2,
                rms=self.window_rms,
                delta=_range(calibration.get("delta")),
                by_group={
                    group: stats
                    for group, values in (calibration.get("delta") or {})
                    .get("by_group", {})
                    .items()
                    if (stats := _range(values))
                },
                likeness=_likeness_range(contrast.get("calibration")),
            )
        ]
        for length, entry in sorted(self.stored.items()):
            if "delta" in entry and 0 < entry["words"] < window_words:
                anchors.append(
                    _Anchor(
                        words=entry["words"],
                        covers=length,
                        rms=flatten(entry.get("reliability", {})),
                        delta=_range(entry["delta"]),
                        by_group={
                            group: stats
                            for group, values in entry["delta"].get("by_group", {}).items()
                            if (stats := _range(values))
                        },
                        likeness=_likeness_range(entry.get("likeness")),
                    )
                )
        self.anchors = sorted(anchors, key=lambda anchor: anchor.words)

    def _value(self, words: float, value: Callable[[_Anchor], float | None]) -> float | None:
        """``value`` at ``words``, interpolated between the anchors that have it."""
        points = [
            (anchor.words, found) for anchor in self.anchors if (found := value(anchor)) is not None
        ]
        return _interpolate(points, words) if points else None

    def _stats(
        self, words: float, stats: Callable[[_Anchor], GroupRange | None]
    ) -> GroupRange | None:
        """A range at ``words``, interpolated between the anchors that have it. Below the
        shortest of them it is widened by ``stretch``: a text shorter than every anchor
        varies more than any of them, by about sqrt(anchor / words), as the noise in a rate
        does."""
        points = [anchor for anchor in self.anchors if stats(anchor)]
        if not points:
            return None
        median = self._value(words, lambda anchor: (stats(anchor) or {}).get("median"))
        mean = self._value(words, lambda anchor: (stats(anchor) or {}).get("mean"))
        p95 = self._value(words, lambda anchor: (stats(anchor) or {}).get("p95"))
        shared = self._value(words, lambda anchor: (stats(anchor) or {}).get("similarity"))
        if median is None or mean is None or p95 is None or shared is None:
            return None
        widen = stretch(points[0].words, words)
        # A share, which widening leaves as it is.
        return {
            "median": median * widen,
            "mean": mean * widen,
            "p95": p95 * widen,
            "similarity": shared,
        }

    def at(self, words: int) -> AtLength:
        """The calibration for a chunk of ``words`` prose words."""
        reason = self._too_short(words)
        if not self.calibrated:
            return AtLength(
                words=words,
                judged=reason is None,
                reason=reason,
                rms=dict(self.window_rms),
                scale={},
                delta=None,
                by_group={},
                likeness=None,
                stretch=stretch(self.uncalibrated_words, words),
            )
        # Between the shortest anchor and the windows. Below the shortest (a text under
        # 75 words, which is not judged), its values are the least inflated available.
        keys = {key for anchor in self.anchors for key in anchor.rms}
        rms: dict[Key, float] = {}
        for key in keys:
            value = self._value(words, lambda anchor, key=key: anchor.rms.get(key))
            if value is not None:
                rms[key] = value
        scale = {
            key: factor
            for key, value in rms.items()
            if key in self.window_rms
            and (factor := max(value, 1.0) / max(self.window_rms[key], 1.0)) > 1.0 + 1e-9
        }
        groups = {group for anchor in self.anchors for group in anchor.by_group}
        by_group = {
            group: stats
            for group in sorted(groups)
            if (stats := self._stats(words, lambda anchor, group=group: anchor.by_group.get(group)))
        }
        likeness: LikenessAtLength | None = None
        likeness_stats = self._stats(words, lambda anchor: anchor.likeness)
        target = self._value(words, lambda anchor: (anchor.likeness or {}).get("target"))
        if likeness_stats and target is not None:
            likeness = {**likeness_stats, "target": target}
        return AtLength(
            words=words,
            judged=reason is None,
            reason=reason,
            rms=rms,
            scale=scale,
            delta=self._stats(words, lambda anchor: anchor.delta),
            by_group=by_group,
            likeness=likeness,
        )

    def _too_short(self, words: int) -> str | None:
        """Why a chunk of ``words`` words gets no verdict, or None when it gets one."""
        if words < MIN_JUDGED_WORDS:
            return (
                f"under {MIN_JUDGED_WORDS} words, the writer's own text varies too much by "
                "chance to judge"
            )
        if not self.calibrated:
            # No held-out range at all: only its windows' own length can be read against
            # fixed steps, as for a window.
            if words >= self.uncalibrated_words / 2:
                return None
            return (
                "the reference has no held-out range (it comes from fewer than two documents), "
                f"so texts under half its window ({self.uncalibrated_words / 2:.0f} words) "
                "cannot be judged; add more of the writer's documents"
            )
        if any(anchor.covers <= words for anchor in self.anchors):
            return None
        shorter = [length for length in self.stored if length <= words]
        if not shorter:
            shortest = math.ceil(PIECES_PER_CHUNK * min(CALIBRATION_LENGTHS))
            return (
                "the reference's windows are too short to cut into calibration pieces, so it "
                f"has no range for texts this short; rebuild it with window_words {shortest} "
                "or more"
            )
        entry = self.stored[max(shorter)]
        return (
            f"to calibrate texts this short the reference has pieces of about {max(shorter)} "
            f"words, but only {shortfall(entry)}, and needs {MIN_CALIBRATION_PIECES} "
            f"independent pieces from {MIN_CALIBRATION_DOCUMENTS} or more documents; add more "
            "of the writer's documents"
        )


# The pooled verdict of one run.


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def verdict(rows: Sequence[Mapping[str, Any]], contrast_label: str | None) -> ScoreVerdict:
    """The headline verdict over a score report's chunk rows.

    Only chunks long enough to judge count; if none is, the verdict is "too short to judge"
    and its numbers, over every chunk, are indicative. Each chunk is read against the range
    for its own length, and the mean of several against the pooled range of the means
    (``pooled_ceiling``), so a run mixing a long chunk and a short one is judged as both.

    The mean of many chunks is what it judges: a few very different chunks among many close
    ones move it little. So it also counts the judged chunks flagged on their own
    (``chunk_flagged``), which every front end reports beside the headline.
    """
    judged = [row for row in rows if row["reference"]["calibration"]["judged"]]
    used = judged or list(rows)
    scored = [row["reference"] for row in used]
    words = sum(int(row["metrics"]["size"]["words"] or 0) for row in rows)
    reason: str | None = None
    if not judged:
        reasons = {row["reference"]["calibration"]["reason"] for row in rows}
        reason = (
            reasons.pop()
            if len(reasons) == 1
            else f"none of the {len(rows)} chunks is long enough to judge"
        )
    deltas = [score["delta"] for score in scored if score["delta"] is not None]
    delta = _mean(deltas)
    ranges = [score["calibration"]["delta"] for score in scored if score["delta"] is not None]
    ceiling = pooled_ceiling(ranges, MIN_CEILING) if all(ranges) else None
    widen = _mean([score["calibration"].get("stretch", 1.0) for score in scored]) or 1.0
    level = None
    if judged and delta is not None:
        level = delta_level(delta, ceiling) if ceiling else delta_level(delta / widen)
    overall: VerdictDelta = {
        "value": delta,
        "typical": _mean([item["median"] for item in ranges if item]) if all(ranges) else None,
        "p95": _mean([item["p95"] for item in ranges if item]) if all(ranges) else None,
        "ceiling": ceiling,
        "level": level,
    }
    return {
        "judged": bool(judged),
        "words": words,
        "chunks": len(rows),
        "chunks_judged": len(judged),
        "reason": reason,
        # As for ``Note.setting``: the setting the reason names, for a front end to swap in
        # its own name for it (the CLI's flag).
        "setting": "window_words" if reason and "window_words" in reason else None,
        "delta": overall,
        # Plain strings, as a saved report reads them back.
        "verdict": str(TOO_SHORT if not judged or level is None else DISTANCE_WORDS[level]),
        "by_group": _by_group(scored, judged=bool(judged), widen=widen),
        "likeness": (
            _likeness(scored, contrast_label, judged=bool(judged))
            if contrast_label is not None
            else None
        ),
        "flagged": sum(chunk_flagged(row) for row in judged),
    }


def _by_group(
    scored: Sequence[Mapping[str, Any]], *, judged: bool, widen: float = 1.0
) -> dict[str, VerdictArea]:
    groups = sorted({group for score in scored for group in score["delta_by_group"]})
    result: dict[str, VerdictArea] = {}
    for group in groups:
        present = [score for score in scored if group in score["delta_by_group"]]
        value = statistics.fmean(score["delta_by_group"][group] for score in present)
        ranges = [score["calibration"]["delta_by_group"].get(group) for score in present]
        ceiling = pooled_ceiling(ranges, MIN_CEILING) if all(ranges) else None
        relative = round(value / ceiling, 2) if ceiling else None
        level = None
        if judged:
            level = (
                delta_level(relative, 1.0) if relative is not None else delta_level(value / widen)
            )
        result[group] = {
            "value": value,
            "typical": _mean([item["median"] for item in ranges if item]) if all(ranges) else None,
            "ceiling": ceiling,
            "relative": relative,
            "level": level,
        }
    return result


def _likeness(
    scored: Sequence[Mapping[str, Any]], label: str, *, judged: bool
) -> VerdictLikeness | None:
    present = [score for score in scored if score.get("likeness") is not None]
    if not present:
        return None
    value = statistics.fmean(score["likeness"] for score in present)
    ranges = [score["calibration"]["likeness"] for score in present]
    calibrated = all(ranges)
    ceiling = pooled_ceiling(ranges, LIKENESS_MIN_CEILING) if calibrated else None
    target = _mean([item["target"] for item in ranges if item]) if calibrated else None
    level = None
    if judged and ceiling is not None and target is not None:
        level = likeness_step(value, ceiling, target)
    return {
        "value": value,
        "typical": _mean([item["median"] for item in ranges if item]) if calibrated else None,
        "p95": _mean([item["p95"] for item in ranges if item]) if calibrated else None,
        "target": target,
        "ceiling": ceiling,
        "level": level,
        "verdict": str(TOO_SHORT) if level is None else likeness_words(level, label),
    }


# A chunk is flagged on its own (``chunk_flagged``) at "clearly different" or worse, or at
# "leans <contrast>" or more: levels the writer's own chunks rarely reach (0-0.3% of held-out
# chunks, see docs/method.md), unlike "somewhat different", which about one in twenty of
# them reads by design. A long run still has many chances: runs of 200 of the writer's own
# chunks name one in 15-27% of batches.
FLAGGED_DISTANCE = 2
FLAGGED_LIKENESS = 2


def chunk_flagged(row: Mapping[str, Any]) -> bool:
    """Whether one chunk, judged at its own length, reads clearly different or worse, or
    leans toward the contrast set or more (``FLAGGED_DISTANCE``, ``FLAGGED_LIKENESS``)."""
    level = chunk_level(row)
    if level is None:
        return False
    return level >= FLAGGED_DISTANCE or (chunk_likeness_level(row) or 0) >= FLAGGED_LIKENESS


def flagged_text(flagged: int, judged: int, chunks: int, label: str | None) -> str:
    """ "4 of 40 chunks read clearly different or lean LLM" ("1 of 9 chunks reads ... or
    leans LLM"; "judged chunks" when some of ``chunks`` were not judged; without the likeness
    half when there is no contrast set ``label``): how many judged chunks are flagged on
    their own (``chunk_flagged``), in words every front end uses."""
    one = flagged == 1
    judged_word = "judged " if judged < chunks else ""
    lean = f" or lean{'s' * one} {label}" if label else ""
    return f"{flagged} of {judged} {judged_word}chunks read{'s' * one} clearly different{lean}"


def chunk_level(row: Mapping[str, Any]) -> int | None:
    """One chunk's Delta level against the range for its own length, or None if it is too
    short to judge."""
    scored = row["reference"]
    calibration = scored["calibration"]
    if not calibration["judged"]:
        return None
    ceiling = pooled_ceiling([calibration["delta"]], MIN_CEILING) if calibration["delta"] else None
    if not ceiling:
        return delta_level((scored["delta"] or 0.0) / calibration.get("stretch", 1.0))
    return delta_level(scored["delta"] or 0.0, ceiling)


def chunk_likeness_level(row: Mapping[str, Any]) -> int | None:
    """One chunk's likeness level against the range for its own length, or None if it is
    too short to judge or its length has no likeness range."""
    scored = row["reference"]
    calibration = scored["calibration"]
    ranges = calibration.get("likeness")
    if not calibration["judged"] or not ranges or scored.get("likeness") is None:
        return None
    ceiling = pooled_ceiling([ranges], LIKENESS_MIN_CEILING) or LIKENESS_MIN_CEILING
    return likeness_step(scored["likeness"], ceiling, ranges["target"])


def too_short_text(result: Mapping[str, Any]) -> str:
    """ "too short to judge (36 words)", for a verdict that is not judged."""
    if result["chunks"] == 1:
        return f"{TOO_SHORT} ({result['words']:,} words)"
    return f"{TOO_SHORT} ({result['chunks']} chunks, {result['words']:,} words)"
