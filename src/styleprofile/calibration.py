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
from typing import Any

from styleprofile.surface import Metrics, classify, plain_sentences
from styleprofile.weighting import (
    DISTANCE_WORDS,
    LIKENESS_MIN_CEILING,
    MIN_CEILING,
    TOO_SHORT,
    Key,
    ZScores,
    calibrate_delta,
    delta_level,
    flatten,
    held_out_z,
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
# A length is calibrated only with at least this many pieces (from two or more documents).
# The verdict's bound is the pieces' 95th percentile; with 20 pieces one lies above it, so
# it is observed rather than set by the single largest piece.
MIN_CALIBRATION_PIECES = 20
# The contrast's likeness range at a length is only its median, which needs fewer pieces.
MIN_CONTRAST_PIECES = 5
# At most this many words of windows are cut into pieces, spread evenly over the corpus, so
# calibration costs a bounded amount on a large corpus: 30,000 words give about 400 pieces
# of 75 words and 100 of 300, plenty for a 95th percentile. The contrast only needs its
# median at each length, so it gets a smaller share.
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


def cut(markdown: str, length: int) -> list[str]:
    """Cut one chunk into pieces of about ``length`` prose words, as Markdown.

    The chunk's prose is divided into round(words / length) pieces of equal size, each cut
    at the sentence or block boundary nearest its share (a paragraph break if one is
    nearly as near), so every piece is close to ``length`` and no text is left over. A
    chunk shorter than 1.5 pieces gives none.
    """
    units = _units(markdown)
    starts = [0]
    for unit in units:
        starts.append(starts[-1] + unit.words)
    total = starts[-1]
    count = round(total / length)
    if count < 2 or total < PIECES_PER_CHUNK * length:
        return []

    def cost(index: int, target: float) -> float:
        # A cut inside a paragraph leaves two part-paragraphs, which a real short text
        # rarely is, so a paragraph break a little further from the target wins.
        inside = units[index].block == units[index - 1].block
        return abs(starts[index] - target) + (PARAGRAPH_PREFERENCE * length if inside else 0)

    cuts = [0]
    for part in range(1, count):
        target = part * total / count
        allowed = [index for index in range(cuts[-1] + 1, len(units)) if not units[index].sticky]
        if not allowed:
            break
        cuts.append(min(allowed, key=lambda index: cost(index, target)))
    cuts.append(len(units))
    pieces = []
    for start, end in pairwise(cuts):
        text = units[start].raw
        for previous, unit in pairwise(units[start:end]):
            text += (" " if unit.block == previous.block else "\n\n") + unit.raw
        pieces.append(text)
    return pieces


def plan_pieces(
    texts: Sequence[str], sizes: Sequence[int], lengths: Sequence[int], budget: int
) -> list[tuple[int, int, str]]:
    """(chunk index, length, Markdown) of every piece to measure for calibration.

    Only lengths the median chunk holds 1.5 times are cut. When the chunks hold more than
    ``budget`` words, every k-th chunk is cut, so the pieces still span the whole corpus.
    """
    if not texts:
        return []
    median = statistics.median(sizes)
    usable = [length for length in lengths if PIECES_PER_CHUNK * length <= median]
    step = max(1, math.ceil(sum(sizes) / budget))
    return [
        (index, length, piece)
        for index in range(0, len(texts), step)
        for length in usable
        for piece in cut(texts[index], length)
    ]


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


def calibrate_length(
    held: Sequence[ZScores], piece_documents: Sequence[str], words: Sequence[float]
) -> dict[str, Any]:
    """One length's stored calibration, from its pieces' held-out z-scores
    (``held_out_pieces``), their documents and their word counts.

    The entry always records how many pieces and documents it rests on; the rms and the
    Delta range only when there are enough.
    """
    entry: dict[str, Any] = {
        "pieces": len(held),
        "documents": len(set(piece_documents)),
        "words": statistics.median(words) if words else 0.0,
    }
    if not enough(entry):
        return entry
    # The 99th percentile is for flagging one passage among many (plan PR 12's drift
    # localization); with fewer than 100 pieces it is the largest or second largest.
    delta = calibrate_delta(held, piece_documents, p99=True)
    if delta is None:
        return entry
    delta.pop("max", None)
    entry["reliability"] = nest(
        {key: round(value, RMS_DIGITS) for key, value in reliability(held, piece_documents).items()}
    )
    entry["delta"] = delta
    return entry


def enough(entry: Mapping[str, Any]) -> bool:
    """Whether a length has enough pieces, from enough documents, to be calibrated."""
    return entry.get("pieces", 0) >= MIN_CALIBRATION_PIECES and entry.get("documents", 0) >= 2


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
    delta: dict[str, float] | None
    by_group: dict[str, dict[str, float]]
    likeness: dict[str, float] | None

    def row(self) -> dict[str, Any]:
        """What a score report keeps per chunk, so a saved report can be shown again."""
        return {
            "judged": self.judged,
            "reason": self.reason,
            "delta": self.delta,
            "delta_by_group": self.by_group,
            "likeness": self.likeness,
            "length_scale": nest(self.scale),
        }


@dataclass(frozen=True)
class _Anchor:
    words: float
    covers: float  # the shortest text this anchor judges without a shorter one
    rms: dict[Key, float]
    delta: dict[str, float] | None
    by_group: dict[str, dict[str, float]]
    likeness: dict[str, float] | None


def _likeness_range(stored: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not stored or "reference" not in stored:
        return None
    return {
        "median": stored["reference"]["median"],
        "p95": stored["reference"]["p95"],
        "target": stored["contrast"]["median"],
    }


def _range(stats: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not stats or stats.get("p95") is None:
        return None
    return {"median": stats.get("median", stats["p95"]), "p95": stats["p95"]}


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


class Lengths:
    """A reference's calibration anchors, read once and then asked for any chunk length."""

    def __init__(self, reference: Mapping[str, Any]) -> None:
        calibration = reference.get("calibration") or {}
        contrast = reference.get("contrast") or {}
        # Without held-out calibration (a reference from one document) there are no ranges
        # at all: verdicts use fixed steps, and only the minimum length applies.
        self.calibrated = bool(calibration.get("delta"))
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
        self, words: float, stats: Callable[[_Anchor], Mapping[str, float] | None]
    ) -> dict[str, float] | None:
        median = self._value(words, lambda anchor: (stats(anchor) or {}).get("median"))
        p95 = self._value(words, lambda anchor: (stats(anchor) or {}).get("p95"))
        if median is None or p95 is None:
            return None
        return {"median": median, "p95": p95}

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
        likeness = None
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
            return None
        if any(anchor.covers <= words for anchor in self.anchors):
            return None
        shorter = [length for length in self.stored if length <= words]
        if not shorter:
            # Windows too short to cut, or a profile built before length calibration.
            shortest = math.ceil(PIECES_PER_CHUNK * min(CALIBRATION_LENGTHS))
            return (
                "the reference has no calibration for texts this short; rebuild it, with "
                f"windows of at least {shortest} words"
            )
        entry = self.stored[max(shorter)]
        return (
            f"the reference has {entry['pieces']} pieces of about {max(shorter)} words from "
            f"{entry['documents']} document(s) to calibrate texts this short, and needs "
            f"{MIN_CALIBRATION_PIECES} from 2 or more; add more of the writer's text"
        )


# The pooled verdict of one run.


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def verdict(rows: Sequence[Mapping[str, Any]], contrast_label: str | None) -> dict[str, Any]:
    """The headline verdict over a score report's chunk rows.

    Only chunks long enough to judge count; if none is, the verdict is "too short to judge"
    and its numbers, over every chunk, are indicative. Each chunk is read against the range
    for its own length, and the mean of several against the pooled range of the means
    (``pooled_ceiling``), so a run mixing a long chunk and a short one is judged as both.
    """
    judged = [row for row in rows if row["reference"]["calibration"]["judged"]]
    used = judged or list(rows)
    scored = [row["reference"] for row in used]
    words = sum(int(row["metrics"]["size"]["words"] or 0) for row in rows)
    result: dict[str, Any] = {
        "judged": bool(judged),
        "words": words,
        "chunks": len(rows),
        "chunks_judged": len(judged),
        "reason": None,
    }
    if not judged:
        reasons = {row["reference"]["calibration"]["reason"] for row in rows}
        result["reason"] = (
            reasons.pop()
            if len(reasons) == 1
            else f"none of the {len(rows)} chunks is long enough to judge"
        )
    deltas = [score["delta"] for score in scored if score["delta"] is not None]
    delta = _mean(deltas)
    ranges = [score["calibration"]["delta"] for score in scored if score["delta"] is not None]
    ceiling = pooled_ceiling(ranges, MIN_CEILING) if all(ranges) else None
    level = delta_level(delta, ceiling) if judged and delta is not None else None
    result["delta"] = {
        "value": delta,
        "typical": _mean([item["median"] for item in ranges if item]) if all(ranges) else None,
        "p95": _mean([item["p95"] for item in ranges if item]) if all(ranges) else None,
        "ceiling": ceiling,
        "level": level,
    }
    result["verdict"] = TOO_SHORT if not judged or level is None else DISTANCE_WORDS[level]
    result["by_group"] = _by_group(scored, judged=bool(judged))
    result["likeness"] = None
    if contrast_label is not None:
        result["likeness"] = _likeness(scored, contrast_label, judged=bool(judged))
    return result


def _by_group(scored: Sequence[Mapping[str, Any]], *, judged: bool) -> dict[str, Any]:
    groups = sorted({group for score in scored for group in score["delta_by_group"]})
    result: dict[str, Any] = {}
    for group in groups:
        present = [score for score in scored if group in score["delta_by_group"]]
        value = statistics.fmean(score["delta_by_group"][group] for score in present)
        ranges = [score["calibration"]["delta_by_group"].get(group) for score in present]
        ceiling = pooled_ceiling(ranges, MIN_CEILING) if all(ranges) else None
        relative = round(value / ceiling, 2) if ceiling else None
        level = None
        if judged:
            level = delta_level(relative, 1.0) if relative is not None else delta_level(value)
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
) -> dict[str, Any] | None:
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
        "verdict": TOO_SHORT if level is None else likeness_words(level, label),
    }


def chunk_level(row: Mapping[str, Any]) -> int | None:
    """One chunk's Delta level against the range for its own length, or None if it is too
    short to judge."""
    scored = row["reference"]
    calibration = scored["calibration"]
    if not calibration["judged"]:
        return None
    ceiling = pooled_ceiling([calibration["delta"]], MIN_CEILING) if calibration["delta"] else None
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
