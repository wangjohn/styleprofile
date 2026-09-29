"""Where a draft drifts: which of its paragraphs read unlike the writer.

One Delta or likeness over a whole draft dilutes a paragraph or two that slip into another
register: a 542-word draft with two such paragraphs still reads "close". So a document is
also read in spans of at least ``SPAN_WORDS`` words cut at paragraph breaks (``plan``), two
per paragraph: the span that starts at it and runs forward, and the span that ends at it
and runs back (near the ends, the document's first or last span; where the two coincide, as
for a paragraph of ``SPAN_WORDS`` words or more, the second takes in a neighbour). Each span
is judged against the writer's own held-out range at its length (``calibration.Lengths``),
as a chunk is: by likeness when the reference has a contrast set, else by Delta, as that
score over its 95% bound (``judge_span``).

A paragraph's statistic is the lower of its two spans (``statistics``). A span high only
because of a neighbouring paragraph has the neighbour's other span beside it, without the
drifting paragraph, so the neighbour's statistic stays low; and a paragraph whose two spans
both hold another that drifts, while it is not in both of that one's, is explained by it
(``judge``). Every paragraph rests on two spans, however short or long its paragraphs.

Checking every paragraph of a long document invites false alarms, so the threshold is set
per document from the writer's own documents: ``build`` reads held-out reference documents
the same way (each scored against a reference without it, ``profile._calibrate_drift``),
splits each paragraph's statistic into its document's level (the median, ``level``) and its
excess over it, and stores the spread of the levels and the tail of the excess
(``calibrate``). A document of n paragraphs lets a paragraph drift only above the level of
the writer's documents at the top of their spread plus the 1 - ``ALPHA`` / n quantile of the
excess, its tail's scale taken at an upper confidence bound (``thresholds``), so the chance
of any paragraph of the writer's drifting is at most about ``ALPHA`` whatever the document's
length, and the margin comes from how far the writer's documents move rather than from a
constant. A reference too small to estimate these lets nothing drift, and says so.

This module plans, judges and calibrates; ``profile`` measures the spans.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from styleprofile.calibration import chunk_level, chunk_likeness_level
from styleprofile.core import DISTANCES, LIKENESSES, LikenessVerdict, Verdict
from styleprofile.metrics import METRICS
from styleprofile.surface import Block, block_word_count, classify, prose
from styleprofile.weighting import LIKENESS_MIN_CEILING, MIN_CEILING, UNSEEN_Z, UPPER_Z

# Spans hold at least this many prose words (docs/method.md, "Where a draft drifts").
SPAN_WORDS = 100
# The chance of any false flag in a document of the writer's own, whatever its length.
ALPHA = 0.05
# The null's tail above its (1 - TAIL_SHARE) quantile is modelled as exponential (a
# generalized Pareto tail of shape 0), from at least MIN_TAIL values above it; the null
# needs paragraphs from at least MIN_DOCUMENTS documents.
TAIL_SHARE = 0.1
MIN_TAIL = 20
MIN_DOCUMENTS = 10
# A paragraph never drifts at or below its span's 95% bound (MINIMUM times it). The null
# of its excess over a document's level is set off from the level of the writer's documents
# at the top of their spread, the LEVEL_SHARE quantile (``thresholds``): the margin comes
# from how far the writer's own documents move, not a constant (docs/method.md, "Where a
# draft drifts").
MINIMUM = 1.0
LEVEL_SHARE = 0.9
# At most this many words of reference documents are read for the null, taken evenly.
CALIBRATION_WORDS = 40_000
# Traits kept per paragraph, and the length of its excerpt in characters.
TRAITS_KEPT = 3
EXCERPT_CHARS = 60
# A span's Delta and likeness count each z of the parser's metrics (the ``SPAN_Z_GROUPS``)
# up to this size (the cap an unseen value gets, ``weighting.UNSEEN_Z``), in scoring spans
# and in their null alike. A span of 100 words has 5 to 8 sentences, so one parse decision
# (a sentence tagged as opening on a verb, where the writer almost never does) can move such
# a rate a dozen standard deviations or more, which alone would carry the span. Capping every
# metric also caps the surface tells an LLM paragraph is found by (em dashes, LLM marker
# words), and found fewer inserts for no fewer false drifts (docs/method.md).
SPAN_Z = UNSEEN_Z
SPAN_Z_GROUPS: frozenset[str] | None = frozenset(
    metric.group for metric in METRICS if metric.syntax
)
# The two scores a span can be judged by.
DELTA = "delta"
LIKENESS = "likeness"


@dataclass(frozen=True)
class Paragraph:
    """A paragraph of a document, measured with any heading or code block just above it.

    A list's indented continuation stays with its list. ``line`` and ``end_line`` are its
    prose's, so headings and code never widen the range; ``words`` counts prose words, as
    ``prose`` does; ``excerpt`` is the start of its prose, for a reader to find it."""

    raw: str
    line: int
    end_line: int
    words: int
    excerpt: str
    # Its own blocks, without the headings and code above it, for its traits.
    own: str


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The first ``limit`` characters of ``text`` or so, cut at a word, with "…" when cut."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit + 1].rsplit(" ", 1)[0]
    return (cut if len(cut) >= limit // 2 else text[:limit]).rstrip(" ,;:") + "…"


def paragraphs(markdown: str) -> list[Paragraph]:
    """The document's paragraphs in order: every block with prose, each carrying the blocks
    without prose above it (a heading, code) and its list continuations. Blocks without prose
    after the last paragraph join its text, not its lines."""
    found: list[Paragraph] = []
    pending: list[Block] = []  # blocks without prose, waiting for the next paragraph
    for block in classify(markdown):
        size = block_word_count(block)
        if block.continues_list and found:
            last = found[-1]
            found[-1] = Paragraph(
                "\n\n".join([last.raw, *(item.raw for item in pending), block.raw]),
                last.line,
                block.end_line,
                last.words + size,
                last.excerpt,
                f"{last.own}\n\n{block.raw}",
            )
            pending = []
        elif size:
            text = " ".join(prose(block.raw).blocks)
            found.append(
                Paragraph(
                    "\n\n".join(item.raw for item in [*pending, block]),
                    block.line,
                    block.end_line,
                    size,
                    excerpt(text),
                    block.raw,
                )
            )
            pending = []
        else:
            pending.append(block)
    if pending and found:
        last = found[-1]
        found[-1] = Paragraph(
            "\n\n".join([last.raw, *(item.raw for item in pending)]),
            last.line,
            last.end_line,
            last.words,
            last.excerpt,
            last.own,
        )
    return found


@dataclass(frozen=True)
class Plan:
    """The spans to score, as (start, end) paragraph ranges in order, and for each paragraph
    the positions in ``spans`` of its two spans (``plan``); the two are the same only when
    one span is the whole document, and None for a document under a span's length."""

    spans: list[tuple[int, int]]
    sides: list[tuple[int | None, int | None]]


def _run(sizes: Sequence[int], start: int, step: int, length: int) -> tuple[int, int] | None:
    """The shortest run of whole paragraphs from ``start`` in direction ``step`` holding
    ``length`` words, as (start, end); None when the document runs out first."""
    total = 0
    index = start
    while 0 <= index < len(sizes):
        total += sizes[index]
        if total >= length:
            return (start, index + 1) if step > 0 else (index, start + 1)
        index += step
    return None


def _widen(span: tuple[int, int], sizes: Sequence[int]) -> tuple[int, int]:
    """``span`` with one more paragraph: on the side with the shorter neighbour (the one
    that dilutes it least), or the side there is one; the span itself when it is the whole
    document."""
    start, end = span
    before = sizes[start - 1] if start > 0 else None
    after = sizes[end] if end < len(sizes) else None
    if before is None and after is None:
        return span
    if after is None or (before is not None and before < after):
        return (start - 1, end)
    return (start, end + 1)


def plan(sizes: Sequence[int], length: int = SPAN_WORDS) -> Plan:
    """The spans of a document with paragraphs of ``sizes`` words. Every paragraph gets two:
    the shortest run of whole paragraphs from it forward that holds ``length`` words, and
    the shortest back from it. Where the document runs out first, it takes the document's
    last (or first) such run instead; and where the two are one span (a paragraph of
    ``length`` words or more, or at an end), the second is that span widened by one
    neighbouring paragraph (``_widen``), so every paragraph rests on two distinct spans
    unless one span is the whole document. A document under ``length`` words has none."""
    count = len(sizes)
    forward = [_run(sizes, index, 1, length) for index in range(count)]
    backward = [_run(sizes, index, -1, length) for index in range(count)]
    if not any(forward):
        return Plan([], [(None, None)] * count)
    last = backward[-1] or next(span for span in reversed(forward) if span)
    first = forward[0] or next(span for span in backward if span)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ahead, behind in zip(forward, backward, strict=True):
        one, two = ahead or last, behind or first
        if one == two:
            two = _widen(one, sizes)
        pairs.append((one, two))
    spans = sorted({span for pair in pairs for span in pair})
    position = {span: index for index, span in enumerate(spans)}
    return Plan(spans, [(position[one], position[two]) for one, two in pairs])


def judge_span(scored: Mapping[str, Any]) -> dict[str, Any]:
    """What a span's scores (``profile._score``) say: it is judged by likeness when the
    reference has a likeness range at its length, else by Delta (``by``), and ``relative``
    is that score over its 95% bound, with the verdicts' floors."""
    row = {"reference": scored}
    calibration = scored["calibration"]
    delta = scored["delta"]
    judged = bool(calibration["judged"]) and calibration["delta"] is not None and delta is not None
    ceiling = max(calibration["delta"]["p95"], MIN_CEILING) if judged else None
    likeness = scored.get("likeness")
    ranges = calibration.get("likeness")
    likeness_ceiling = None
    if judged and ranges and likeness is not None:
        likeness_ceiling = max(ranges["p95"], LIKENESS_MIN_CEILING)
    by = LIKENESS if likeness_ceiling is not None else DELTA
    value, bound = (likeness, likeness_ceiling) if by == LIKENESS else (delta, ceiling)
    return {
        "by": by,
        "judged": judged,
        "relative": value / bound if judged and value is not None and bound else None,
        "delta": delta,
        "ceiling": ceiling,
        "level": chunk_level(row),
        "likeness": likeness,
        "likeness_level": chunk_likeness_level(row),
        "likeness_ceiling": likeness_ceiling,
    }


def statistics(
    layout: Plan, relatives: Sequence[float | None]
) -> list[tuple[float, int, int] | None]:
    """Each paragraph's statistic: the lower of the relative scores of its two spans, or
    the one it has; with how many distinct judged spans it rests on (1 or 2) and the
    position of the lower span. None when no span of it is judged."""
    found: list[tuple[float, int, int] | None] = []
    for sides in layout.sides:
        own = sorted(
            {
                position
                for position in sides
                if position is not None and relatives[position] is not None
            }
        )
        if not own:
            found.append(None)
            continue
        lowest = min(own, key=lambda position: relatives[position] or 0.0)
        found.append((relatives[lowest] or 0.0, len(own), lowest))
    return found


def level(stats: Sequence[tuple[float, int, int] | None]) -> float | None:
    """A document's own level: the median statistic of its paragraphs on two spans."""
    values = sorted(stat[0] for stat in stats if stat is not None and stat[1] == 2)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def fit_tail(values: Sequence[float]) -> dict[str, float] | None:
    """The null's upper tail: the value ``TAIL_SHARE`` of ``values`` exceed, and the mean
    excess above it (the scale of an exponential tail). None with fewer than ``MIN_TAIL``
    values above it."""
    ordered = sorted(values)
    above = int(len(ordered) * TAIL_SHARE)
    if above < MIN_TAIL:
        return None
    start = ordered[len(ordered) - above - 1]
    excess = [value - start for value in ordered[len(ordered) - above :]]
    return {
        "count": len(ordered),
        "threshold": start,
        "share": above / len(ordered),
        "scale": max(sum(excess) / len(excess), 1e-6),
    }


def upper_scale(tail: Mapping[str, float]) -> float:
    """An upper confidence bound on the tail's scale: the mean of k exponential excesses
    is a chi-square with 2k degrees of freedom over 2k, so its bound is 2k / the chi-square's
    lower ``UPPER_Z`` quantile (Wilson-Hilferty) times the mean. Few values above the tail's
    start give a wider bound: 1.24 times the mean for 40, 1.16 for 80."""
    freedom = 2 * max(round(tail["count"] * tail["share"]), 1)
    step = 2 / (9 * freedom)
    lower = freedom * (1 - step - UPPER_Z * math.sqrt(step)) ** 3
    return tail["scale"] * freedom / lower


def threshold(tail: Mapping[str, float], count: int, alpha: float = ALPHA) -> float:
    """The 1 - alpha / count quantile of a null with ``tail``, its scale taken at its upper
    confidence bound (``upper_scale``): a document of ``count`` paragraphs has at most
    about an ``alpha`` chance of any paragraph above it by chance."""
    chance = alpha / max(count, 1)
    if chance >= tail["share"]:
        return tail["threshold"]
    return tail["threshold"] + upper_scale(tail) * math.log(tail["share"] / chance)


def _quantile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))]


def calibrate(
    documents: int,
    found: Sequence[Sequence[tuple[float, int, int] | None]],
    by: str,
) -> dict[str, Any]:
    """The stored null (``calibration.drift``) from held-out reference documents, each read
    in parts (``found``: its paragraphs' statistics).

    A paragraph's statistic is split into its document's level (``level``) and its excess
    over that level. ``within`` is the tail of the excess of paragraphs on two spans;
    ``levels`` the spread of the documents' levels (median and ``LEVEL_SHARE`` quantile);
    ``one`` the tail of the statistic of paragraphs on one span (a document that is one
    span), usually too few to fit. A tail is None when the reference is too small."""
    enough = documents >= MIN_DOCUMENTS
    levels: list[float] = []
    excess: list[float] = []
    for stats in found:
        own = level(stats)
        if own is None:
            continue
        levels.append(own)
        excess += [stat[0] - own for stat in stats if stat is not None and stat[1] == 2]
    one = [stat[0] for stats in found for stat in stats if stat is not None and stat[1] == 1]
    return {
        "by": by,
        "span_words": SPAN_WORDS,
        "alpha": ALPHA,
        "documents": documents,
        "paragraphs": sum(1 for stats in found for stat in stats if stat is not None),
        "levels": (
            {"median": _quantile(levels, 0.5), "high": _quantile(levels, LEVEL_SHARE)}
            if enough and levels
            else None
        ),
        "within": fit_tail(excess) if enough else None,
        "one": fit_tail(one) if enough else None,
    }


def thresholds(
    null: Mapping[str, Any] | None, count: int, by: str
) -> dict[str, float | None] | None:
    """A document's thresholds, for paragraphs on two spans and on one (None: such a
    paragraph never drifts), or None when the reference has no null for the score its
    spans are judged by.

    A paragraph on two spans must stand out by the 1 - alpha / n quantile of the writer's
    excess over a document's level (``threshold``) above the level of the writer's
    documents at the top of their spread (``levels.high``): a draft is judged as if it sat
    as high as the writer's documents go. Never at or below ``MINIMUM``."""
    if not null or null.get("by") != by or not null.get("within") or not null.get("levels"):
        return None
    excess = threshold(null["within"], count, null["alpha"])
    one = null.get("one")
    return {
        "two": max(null["levels"]["high"] + excess, MINIMUM),
        "one": max(threshold(one, count, null["alpha"]), MINIMUM) if one else None,
    }


def judge(
    found: Sequence[Paragraph],
    layout: Plan,
    judged: Sequence[Mapping[str, Any]],
    limits: Mapping[str, float | None] | None,
    traits: Sequence[Sequence[Mapping[str, Any]]],
    alone: Sequence[float | None] | None = None,
) -> list[dict[str, Any]]:
    """Each paragraph's entry: its lines, words and excerpt, its statistic and the figures
    of its lower span, its own traits, and whether it ``drifts``: above its threshold
    (``limits``), unless a paragraph at least as strong that still drifts lies in both its
    spans while it does not lie in both of that one's, which then explains its spans
    (``note`` says so). A paragraph that reads unlike the writer on its own (its score
    ``alone`` above its bound) is never explained away. Nothing drifts without
    thresholds."""
    stats = statistics(layout, [span["relative"] for span in judged])
    entries: list[dict[str, Any]] = []
    for index, paragraph in enumerate(found):
        entry: dict[str, Any] = {
            "lines": [paragraph.line, paragraph.end_line],
            "words": paragraph.words,
            "excerpt": paragraph.excerpt,
            "spans": 0,
            "by": None,
            "relative": None,
            "threshold": None,
            "delta": None,
            "level": None,
            "likeness": None,
            "likeness_level": None,
            "traits": list(traits[index]),
            "alone": alone[index] if alone is not None else None,
            "drifts": False,
            "note": None,
        }
        stat = stats[index]
        if stat is not None:
            value, spans, lowest = stat
            span = judged[lowest]
            limit = (limits["two" if spans == 2 else "one"] if limits else None) or None
            entry |= {
                "spans": spans,
                "by": span["by"],
                "relative": value,
                "threshold": limit,
                "delta": span["delta"],
                "level": span["level"],
                "likeness": span["likeness"],
                "likeness_level": span["likeness_level"],
                "drifts": limit is not None and value > limit,
            }
            if limits and limit is None and value > 1.0:
                entry["note"] = "rests on one span, which this reference cannot calibrate"
        entries.append(entry)

    def inside(index: int) -> set[int]:
        """The paragraphs in every span of ``index``."""
        sides = [position for position in layout.sides[index] if position is not None]
        ranges = [set(range(*layout.spans[position])) for position in sides]
        return set.intersection(*ranges) if ranges else set()

    # Strongest first: a paragraph is explained only by one at least as strong that still
    # drifts, so an explanation never passes along a chain to a weaker paragraph.
    drifting = sorted(
        (index for index, entry in enumerate(entries) if entry["drifts"]),
        key=lambda index: -(entries[index]["relative"] or 0.0),
    )
    kept: list[int] = []
    for index in drifting:
        own = entries[index]["alone"]
        if own is not None and own > 1.0:
            # It reads unlike the writer by itself: nothing next to it explains that.
            kept.append(index)
            continue
        explains = next(
            (other for other in kept if other in inside(index) and index not in inside(other)),
            None,
        )
        if explains is None:
            kept.append(index)
            continue
        entries[index]["drifts"] = False
        entries[index]["note"] = f"both its spans hold line {found[explains].line}"
    return entries


def traits(
    z: Mapping[str, Mapping[str, float]],
    scale: Mapping[str, Mapping[str, float]],
    metrics: Mapping[str, Mapping[str, float | None]],
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    signals: Sequence[Mapping[str, Any]] | None,
    kept: int = TRAITS_KEPT,
) -> list[dict[str, Any]]:
    """What sets a paragraph itself apart, each z in standard deviations of the writer's own
    text at its length (over its ``length_scale``), as "Biggest differences" counts them:
    its strongest contrast ``signals`` when its spans are judged by likeness, else the
    metrics furthest from the writer."""

    def scaled(group: str, name: str) -> float:
        return z[group][name] / scale.get(group, {}).get(name, 1.0)

    if signals is not None:
        keys = [tuple(signal["metric"].split(".", 1)) for signal in signals]
    else:
        keys = sorted(
            ((group, name) for group, values in z.items() for name in values),
            key=lambda key: -abs(scaled(*key)),
        )
    return [
        {
            "metric": f"{group}.{name}",
            "z": scaled(group, name),
            "value": metrics[group][name],
            "reference": summary[group][name]["mean"],
        }
        for group, name in keys[:kept]
        if scaled(group, name)
    ]


@dataclass(frozen=True)
class Trait:
    """One way a passage differs from the writer: ``z`` is in standard deviations of the
    writer's own text at the passage's length, ``value`` the passage's and ``reference``
    the writer's mean."""

    metric: str
    z: float
    value: float | None
    reference: float | None


@dataclass(frozen=True)
class Passage:
    """One paragraph of a scored document and how it reads against the writer.

    ``lines`` is the (first, last) line of its prose in the document as read (for HTML, its
    Markdown conversion); ``document`` is the document's name, as ``documents`` gives it.
    ``delta``, ``verdict``, ``likeness`` and ``likeness_verdict`` are those of the lower of
    its two spans of 100 words or more (see ``styleprofile.drift``), None or ``TOO_SHORT``
    when no span covers it; ``likeness_verdict`` is None without a contrast set. ``traits``
    are the paragraph's own. ``drifts`` is True when its statistic is above the threshold
    that keeps the chance of any paragraph of the writer's own documents drifting near 5%
    (a paragraph's own flag, not a chunk's ``flagged``); ``note`` says why not when it is
    above its range but does not drift.
    """

    document: str
    lines: tuple[int, int]
    words: int
    excerpt: str
    delta: float | None
    verdict: Verdict
    likeness: float | None
    likeness_verdict: LikenessVerdict | None
    traits: tuple[Trait, ...]
    drifts: bool
    note: str | None

    @classmethod
    def from_report(cls, document: str, entry: Mapping[str, Any], contrast: bool) -> Passage:
        level = entry["level"]
        likeness_level = entry["likeness_level"]
        likeness_verdict = None
        if contrast:
            likeness_verdict = (
                LIKENESSES[likeness_level]
                if likeness_level is not None
                else LikenessVerdict.TOO_SHORT
            )
        return cls(
            document=document,
            lines=(entry["lines"][0], entry["lines"][1]),
            words=entry["words"],
            excerpt=entry["excerpt"],
            delta=entry["delta"],
            verdict=DISTANCES[level] if level is not None else Verdict.TOO_SHORT,
            likeness=entry["likeness"],
            likeness_verdict=likeness_verdict,
            traits=tuple(
                Trait(item["metric"], item["z"], item["value"], item["reference"])
                for item in entry["traits"]
            ),
            drifts=entry["drifts"],
            note=entry["note"],
        )
