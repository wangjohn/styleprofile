"""Human-readable terminal rendering of a style profile report.

The JSON report is the complete record; this view names each metric in plain English, shows
units, trims statistics to the ones a reader needs, and ends with a short verdict when the
report was scored against a reference.
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from styleprofile.calibration import (
    FLAGGED_DISTANCE,
    MIN_CALIBRATION_DOCUMENTS,
    MIN_CALIBRATION_PIECES,
    MIN_JUDGED_WORDS,
    chunk_level,
    chunk_likeness_level,
    enough,
    flagged_text,
    shortfall,
    too_short_text,
)
from styleprofile.core import DISTANCES, LIKENESSES, LikenessVerdict, Verdict, warning_text
from styleprofile.drift import DELTA, LIKENESS
from styleprofile.drift import excerpt as excerpt_text
from styleprofile.metrics import (
    DISTRIBUTION_LABELS,
    KEY_VIEW,
    PCT,
    PER_1K,
    PLAIN,
    UNSCORED_GROUPS,
    WORDS,
    label,
)
from styleprofile.metrics import title as group_title
from styleprofile.profile import average_z, base_id, document_label, summarize
from styleprofile.schema import (
    AucResult,
    Baseline,
    BaselineLength,
    Contrast,
    DocumentEntry,
    DocumentPassages,
    EvaluationReport,
    LengthBaseline,
    MetricStats,
    ParagraphScore,
    ReferenceReport,
    ReportBase,
    ScoredChunk,
    ScoreReport,
    ScoreVerdict,
    Survival,
    VerdictDelta,
    VerdictLikeness,
)
from styleprofile.terminal import glyphs
from styleprofile.weighting import (
    DISTANCE_WORDS,
    LENGTH_AUC_WARNING,
    TOO_SHORT,
    delta_level,
    likeness_words,
)

# The verdict helpers live in ``weighting``; these names stay importable from here.
from styleprofile.weighting import likeness_level as likeness_level
from styleprofile.weighting import mean_ceiling as mean_ceiling

# Wide enough for the longest metric label, so value columns stay aligned.
LABEL_WIDTH = 46
VALUE_WIDTH = 12
DIFFERENCES_SHOWN = 8
NOTABLE_Z = 1.0
CHUNKS_SHOWN = 3
# The document table: names are cut in the middle to fit (``shorten``) between these
# widths; the verdict column fits the longest verdict; past CLOSE_ROWS close documents the
# rest are counted instead of listed, unless every row is asked for (--all).
NAME_WIDTH = 36
MIN_NAME_WIDTH = 12
VERDICT_WIDTH = 18
CLOSE_ROWS = 10
# The width views are fitted to when the terminal's is unknown (not a terminal).
DEFAULT_WIDTH = 80
LIKENESS_CELLS = {
    LikenessVerdict.LIKE_REFERENCE: "like reference",
    LikenessVerdict.FEW_TRAITS: "a few traits",
    LikenessVerdict.LEANS: "leans",
    LikenessVerdict.LIKE_DRAFTS: "like drafts",
    # Judged on Delta, but the reference has no likeness range at its length.
    LikenessVerdict.TOO_SHORT: "too short",
}
# The paragraph section's title: experimental, and shown only when asked (--by-paragraph),
# since on topics the reference never saw it can find a writer's own paragraph drifting
# (docs/method.md, "Where a draft drifts").
DRIFT_TITLE = "Where it drifts (experimental)"
# Paragraphs that drift "Where it drifts" shows, and the excerpt length in "By paragraph".
PASSAGES_SHOWN = 3
EXCERPT_SHOWN = 40
BAR_WIDTH = 20
# Bars grow with log(1 + amount), full at BAR_MAX times an area's usual held-out range (or a
# raw Delta of BAR_MAX without calibration). A linear bar full at 3x filled 7 of 9 bars for an
# LLM draft, whose areas run from 1.3x to over 20x, so it ranked nothing; on this scale
# they spread from 5 to 18 cells, while the verdict steps stay apart (1x fills 4 cells, 1.5x
# 5, 2x 6) and a close area stays short.
BAR_MAX = 32.0


# How far from the reference, as one orange ramp (pale -> deep). "Close" stays uncolored so
# color only appears where there is something to look at. Validated as an ordinal ramp on
# light and dark surfaces; no green or red, since distance is not right or wrong. Each
# color is paired with a word or arrows, so the report reads the same without color.
DISTANCE_RGB: tuple[tuple[int, int, int], ...] = ((217, 149, 106), (220, 111, 52), (184, 70, 26))
DISTANCE_256: tuple[int, ...] = (173, 166, 130)


def z_level(z: float) -> int:
    """0-3 for one metric: under 1, 1-2, 2-3, or 3+ standard deviations away."""
    size = abs(z)
    return 0 if size < 1 else 1 if size < 2 else 2 if size < 3 else 3


class _Style:
    def __init__(self, color: bool, truecolor: bool = False) -> None:
        self.color = color
        self.truecolor = truecolor

    def distance(self, text: str, level: int) -> str:
        """Color a mark by distance level; level 0 (close) stays plain."""
        if not self.color or level <= 0:
            return text
        if self.truecolor:
            red, green, blue = DISTANCE_RGB[level - 1]
            code = f"38;2;{red};{green};{blue}"
        else:
            code = f"38;5;{DISTANCE_256[level - 1]}"
        return self._wrap(f"1;{code}" if level == 3 else code, text)

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def warn(self, text: str) -> str:
        return self._wrap("33", text)


def value(number: float | None, unit: str) -> str:
    if number is None:
        return "-"
    if unit == PCT:
        return f"{number:.0f}%" if abs(number) >= 10 else f"{number:.1f}%"
    digits = 0 if abs(number) >= 100 else 2 if abs(number) < 10 and unit == PLAIN else 1
    text = f"{number:,.{digits}f}"
    if unit == PER_1K:
        return f"{text} /1k"
    if unit == WORDS:
        return f"{text} words"
    return text


def _range(stats: MetricStats, unit: str) -> str:
    if stats["sd"] is None or stats["mean"] is None:
        return ""
    low = max(0.0, stats["mean"] - stats["sd"])
    return f"{value(low, unit).split(' ')[0]} - {value(stats['mean'] + stats['sd'], unit)}"


def _arrow(z: float | None) -> str:
    if z is None:
        return ""
    size = abs(z)
    count = 0 if size < 1 else 1 if size < 2 else 2 if size < 3 else 3
    return (glyphs()["up"] if z > 0 else glyphs()["down"]) * count


def _bar(amount: float, style: _Style, level: int) -> str:
    """Filled length carries the amount, on a log scale (``BAR_MAX``); only the filled part
    is colored."""
    share = math.log1p(max(amount, 0.0)) / math.log1p(BAR_MAX)
    filled = max(0, min(BAR_WIDTH, round(share * BAR_WIDTH)))
    return style.distance(glyphs()["bar"] * filled, level) + style.dim(
        glyphs()["empty"] * (BAR_WIDTH - filled)
    )


def describe_delta(delta: float, ceiling: float | None = None) -> Verdict:
    """A reading of Delta; see ``delta_level``."""
    return DISTANCE_WORDS[delta_level(delta, ceiling)]


def _row(text: str, *cells: str) -> str:
    return (
        f"  {text:{LABEL_WIDTH}}" + "".join(f"{cell:>{VALUE_WIDTH}}" for cell in cells)
    ).rstrip()


def _key_rows(report: ReportBase) -> list[tuple[str, list[tuple[str, str]]]]:
    sections = []
    for title, metrics in KEY_VIEW:
        present = [
            (group, name) for group, name in metrics if name in report["summary"].get(group, {})
        ]
        if present:
            sections.append((title, present))
    return sections


def _profile_view(report: ReferenceReport | ScoreReport, style: _Style, full: bool) -> list[str]:
    multi = report["chunk_count"] > 1
    header = _row("", "typical") + ("   usual range" if multi else "")
    sections = (
        [
            (group_title(group), [(group, name) for name in metrics])
            for group, metrics in report["summary"].items()
            if group not in UNSCORED_GROUPS
        ]
        if full
        else _key_rows(report)
    )
    lines = ["", style.dim(header)]
    for title, rows in sections:
        lines += ["", style.bold(title)]
        for group, name in rows:
            stats = report["summary"][group][name]
            text, unit = label(name)
            row = _row(text, value(stats["mean"], unit))
            lines.append(f"{row}   {_range(stats, unit)}".rstrip() if multi else row)
    if "reference" in report:
        # A score report shown on its own is not a reference.
        return lines
    return lines + _reference_lines(report, style)


def _reference_lines(report: ReferenceReport, style: _Style) -> list[str]:
    """How the profile works as a reference: its held-out Delta range and any contrast."""
    lines: list[str] = []
    # Both need chunks from two or more documents; contrast also needs --contrast.
    calibration = report.get("calibration")
    if calibration:
        held = calibration["delta"]
        lines += [
            "",
            style.bold("As a reference")
            + style.dim(
                f"   held-out Delta {held['median']:.2f} typically, 95% under {held['p95']:.2f} "
                f"({calibration['sources']} documents)"
            ),
        ]
        lines.append(style.dim(f"  {_lengths_line(calibration['by_length'])}"))
    contrast = report.get("contrast")
    if contrast:
        lines += _contrast_summary(contrast, style)
    return lines


def _lengths_line(lengths: Mapping[str, BaselineLength]) -> str:
    """Which shorter lengths the reference is calibrated for, and on how many pieces."""
    calibrated = [
        f"{length} words ({entry['pieces']} pieces)"
        for length, entry in lengths.items()
        if "delta" in entry
    ]
    thin = [
        f"{length} words ({shortfall(entry)})"
        for length, entry in lengths.items()
        if not enough(entry)
    ]
    if not lengths:
        text = "Shorter texts: not calibrated (its chunks are too short to cut into pieces)"
    else:
        parts = []
        if calibrated:
            parts.append("calibrated at " + ", ".join(calibrated))
        if thin:
            parts.append(
                ("not at " if calibrated else "not calibrated at ")
                + ", ".join(thin)
                + f", as a length needs {MIN_CALIBRATION_PIECES} independent pieces from "
                f"{MIN_CALIBRATION_DOCUMENTS} or more documents"
            )
        text = "Shorter texts: " + "; ".join(parts)
    return f"{text}. Under {MIN_JUDGED_WORDS} words, no verdict."


def _contrast_summary(contrast: Contrast, style: _Style) -> list[str]:
    name = contrast["label"]
    calibration = contrast["calibration"]
    effects = [
        (effect, metric)
        for group, values in contrast["effects"].items()
        for metric, effect in values.items()
    ]
    top = sorted(effects, key=lambda item: -abs(item[0]))[:4]
    auc = calibration["auc"]
    interval = calibration["auc_ci"]
    separation = ""
    if auc is not None:
        separation = f"; separates them from the reference with AUC {auc:.2f}"
        exact = _exact(calibration)
        if exact:
            separation += f" ({exact})"
        elif interval:
            low, high = interval
            separation += f" (95% CI {low:.2f}–{high:.2f}, resampling whole documents)"  # noqa: RUF001
    lines = [
        "",
        style.bold(f"Contrast: {name} drafts")
        + style.dim(
            f"   {contrast['chunk_count']} chunks from {contrast['sources']} documents" + separation
        ),
        style.dim(
            f"  {name}-likeness on held-out text: reference "
            f"{calibration['reference']['median']:.2f} typically "
            f"(95% under {calibration['reference']['p95']:.2f}), "
            f"{name} drafts {calibration['contrast']['median']:.2f}"
        ),
        f"  Compared with the reference, {name} drafts have: "
        + ", ".join(
            f"{label(metric)[0]} {glyphs()['up'] if effect > 0 else glyphs()['down']}"
            for effect, metric in top
        ),
    ]
    length = calibration["length_baseline"]
    if length:
        lines.insert(3, style.dim(f"  {_length_line(length, name)}"))
    return lines


def _length_line(length: LengthBaseline, name: str) -> str:
    """How well word count alone separates the contrast set, and what that means."""
    auc = length["auc"]
    if auc >= LENGTH_AUC_WARNING:
        verdict = f"{name}-likeness may partly reflect length"
    elif auc >= 0.6:
        verdict = "some length difference"
    else:
        verdict = "not a length effect"
    return (
        f"Length alone: AUC {auc:.2f} (reference {length['reference_median_words']:.0f} words "
        f"per chunk, {name} drafts {length['contrast_median_words']:.0f}): {verdict}"
    )


def _judged_view(report: ScoreReport) -> ScoreReport:
    """The report as far as its verdict goes: only the chunks long enough to judge, with
    their own summary, or every chunk when none is (then everything shown is indicative).
    A chunk left out of the verdict adds nothing to the differences, arrows or signals."""
    rows = [row for row in report["chunks"] if row["reference"]["calibration"]["judged"]]
    if not rows or len(rows) == len(report["chunks"]):
        return report
    return {
        **report,
        "chunks": rows,
        "chunk_count": len(rows),
        "summary": summarize([row["metrics"] for row in rows]),
    }


def _chunk_z(report: ScoreReport) -> dict[tuple[str, str], list[float]]:
    """Each metric's z per chunk, over how many times more that metric swings at the
    chunk's length than in a reference window (``length_scale``), so a short chunk's arrows
    count standard deviations of the writer's own text at its length, as ``average_z``
    reads them."""
    totals: dict[tuple[str, str], list[float]] = {}
    for chunk in report["chunks"]:
        scale = chunk["reference"]["calibration"]["length_scale"]
        for group, values in chunk["reference"]["z"].items():
            for name, z in values.items():
                factor = scale.get(group, {}).get(name, 1.0)
                totals.setdefault((group, name), []).append(z / factor)
    return totals


def _average_z(report: ScoreReport, reference: Baseline) -> dict[tuple[str, str], float]:
    """Each metric's z against the reference, averaged over all the scored chunks."""
    return average_z(report["chunks"], reference["summary"])


def _comparison_rows(report: ScoreReport, reference: Baseline, style: _Style) -> list[str]:
    averaged = _average_z(report, reference)
    lines = ["", style.bold("All metrics"), style.dim(_row("", "this text", "reference"))]
    for group, metrics in report["summary"].items():
        if group in UNSCORED_GROUPS:
            continue
        lines += ["", style.bold(group_title(group))]
        for name, stats in metrics.items():
            text, unit = label(name)
            # The reference may lack a metric the sample has (syntax, say).
            ref = reference["summary"].get(group, {}).get(name)
            z = averaged.get((group, name))
            row = _row(text, value(stats["mean"], unit), value(ref["mean"] if ref else None, unit))
            # No trailing spaces on rows without arrows.
            if z is not None and (arrow := _arrow(z)):
                row += f"   {style.distance(arrow, z_level(z))}"
            lines.append(row)
    return lines


def _differences(
    report: ScoreReport, reference: Baseline, style: _Style, *, judged: bool = True
) -> list[str]:
    """Metrics whose average z over chunks sits furthest from the reference, in standard
    deviations of the writer's own text at each chunk's length (see ``_chunk_z``).

    A metric the reference never varies on scores a capped z when a chunk differs and 0
    when it matches, so its arrows reflect how many chunks depart from the reference.
    """
    averaged = _average_z(report, reference)
    differing = {key: sum(z != 0 for z in zs) for key, zs in _chunk_z(report).items()}
    ranked = sorted(
        ((key, z) for key, z in averaged.items() if abs(z) >= NOTABLE_Z),
        key=lambda item: -abs(item[1]),
    )[:DIFFERENCES_SHOWN]
    indicative = "" if judged else "; indicative only"
    lines = [
        style.bold("Biggest differences")
        + style.dim(
            f"   (each {glyphs()['up']} or {glyphs()['down']} is one standard deviation, "
            f"up to 3{indicative})"
        ),
    ]
    if not ranked:
        return [*lines, f"  No metric differs by {NOTABLE_Z:.0f} sd or more on average."]
    lines.append(style.dim(_row("", "this text", "reference")))
    for (group, name), z in ranked:
        text, unit = label(name)
        here = report["summary"][group][name]["mean"]
        ref = reference["summary"][group][name]
        # With one chunk, "1 of 1 chunks differ" says nothing the row does not.
        count = report["chunk_count"]
        note = (
            ""
            if ref["sd"]
            else style.dim(
                "  reference never varies"
                + (f"; {differing[(group, name)]} of {count} chunks differ" if count > 1 else "")
            )
        )
        lines.append(
            _row(text, value(here, unit), value(ref["mean"], unit))
            + f"   {style.distance(_arrow(z), z_level(z))}{note}"
        )
    return lines


def _at_length(report: ScoreReport, reference: Baseline) -> str:
    """How the verdict's ranges were matched to length, for the lines that quote them."""
    if reference["calibration"] is None:
        return ""
    counts = [int(row["metrics"]["size"]["words"] or 0) for row in report["chunks"]]
    if len(counts) == 1:
        return f" at this length ({counts[0]:,} words)"
    return f" at these lengths ({min(counts):,}-{max(counts):,} words per chunk)"


def _likeness(
    report: ScoreReport, reference: Baseline, style: _Style, verdict: ScoreVerdict
) -> list[str]:
    entry = verdict["likeness"]
    contrast = reference["contrast"]
    if entry is None or contrast is None:
        return []
    name = contrast["label"]
    level = entry["level"]
    shares: dict[str, list[float]] = {}
    zs: dict[str, list[float]] = {}
    for chunk in report["chunks"]:
        scale = chunk["reference"]["calibration"]["length_scale"]
        for signal in chunk["reference"].get("likeness_signals", []):
            group, metric = signal["metric"].split(".", 1)
            shares.setdefault(signal["metric"], []).append(signal["contribution"])
            factor = scale.get(group, {}).get(metric, 1.0)
            zs.setdefault(signal["metric"], []).append(signal["z"] / factor)
    ranked = sorted(shares, key=lambda metric: -sum(shares[metric]))[:3]
    signals = []
    for metric in ranked:
        z = sum(zs[metric]) / len(zs[metric])
        # A signal under one standard deviation has no arrow to show, so it is left out.
        if arrow := _arrow(z):
            signals.append(
                f"{label(metric.split('.', 1)[1])[0]} {style.distance(arrow, z_level(z))}"
            )
    words = style.bold(entry["verdict"])
    lines = [
        style.bold(f"{name}-likeness: ")
        + (style.distance(words, level) if level is not None else words)
        + f"   {entry['value']:.2f}",
    ]
    # A text too short to judge has no range of its own length to quote.
    if entry["typical"] is not None and level is not None:
        lines.append(
            style.dim(
                f"  The reference's own writing{_at_length(report, reference)} "
                f"scores {entry['typical']:.2f} typically ({_bound(report, entry)}); "
                f"the {name} drafts {entry['target']:.2f}."
            )
        )
    if signals and level is None:
        lines.append(f"  Indicative {name} signals: " + ", ".join(signals))
    elif signals and level:
        lines.append(f"  Strongest {name} signals: " + ", ".join(signals))
    return lines


def _delta_baseline(report: ScoreReport, reference: Baseline, entry: VerdictDelta) -> str:
    if entry["typical"] is None:
        return "Text by the reference's own writer usually scores around 0.8."
    return (
        f"The reference's own held-out writing{_at_length(report, reference)} scores "
        f"{entry['typical']:.2f} typically ({_bound(report, entry)})."
    )


def _bound(report: ScoreReport, entry: VerdictDelta | VerdictLikeness) -> str:
    """The bound the verdict reads: one chunk's 95% bound, or the tighter one for a mean
    over several chunks (``pooled_ceiling``), which is what the verdict words compare with.
    Only quoted with a range (``typical`` not None), which has both."""
    ceiling, p95 = entry["ceiling"], entry["p95"]
    if report["chunk_count"] > 1:
        return f"a mean over {report['chunk_count']} chunks, up to {ceiling:.2f}"
    if ceiling is not None and p95 is not None and ceiling > p95 + 5e-3:
        # The floor that keeps a near-zero range from inflating verdicts.
        return f"95% under {p95:.2f}; close up to {ceiling:.2f}"
    return f"95% under {p95:.2f}"


def _flagged_note(report: ScoreReport, reference: Baseline, style: _Style) -> str:
    """What the headline adds when some judged chunks are flagged on their own
    (``calibration.chunk_flagged``): "; 4 of 40 chunks read clearly different or lean LLM on
    their own (see below)". The headline judges the chunks' mean, which a few such chunks
    move little, so it says so whenever the chunk lists below are shown to name them. When
    every document is one chunk, the document table and its count line already do."""
    verdict = report["reference"]["verdict"]
    # With every document one chunk, the document table and its count line say it already.
    if not verdict["flagged"] or report["chunk_count"] <= max(len(report["documents"]), 1):
        return ""
    contrast = reference["contrast"]
    words = flagged_text(
        verdict["flagged"],
        verdict["chunks_judged"],
        verdict["chunks"],
        contrast["label"] if contrast else None,
    )
    own = "its" if verdict["flagged"] == 1 else "their"
    return style.distance(f"; {words} on {own} own (see below)", FLAGGED_DISTANCE)


def _flagged_chunks(
    report: ScoreReport, reference: Baseline, style: _Style, *, full: bool = False
) -> list[str]:
    """Up to a few chunks per list (every one with ``full``), only those that are flagged,
    each judged at its own length, the furthest by level first so that every chunk flagged
    on its own is among the first; chunks too short to judge are never listed."""
    lines: list[str] = []
    contrast = reference["contrast"]
    if contrast and report["reference"]["verdict"]["likeness"]:
        name = contrast["label"]
        # Against a contrast reference every chunk has a likeness.
        liked = [
            (chunk, score)
            for chunk in report["chunks"]
            if (score := chunk["reference"].get("likeness")) is not None
        ]
        most = sorted(
            (
                (chunk, score, level)
                for chunk, score in liked
                if (level := chunk_likeness_level(chunk))
            ),
            key=lambda item: (-item[2], -item[1]),
        )
        if most:
            lines += ["", style.bold(f"Most {name}-like chunks")]
            for chunk, score, level in most[: None if full else CHUNKS_SHOWN]:
                word = style.distance(f"{likeness_words(level, name):22}", level)
                lines.append(f"  {score:5.2f}  {word}  {_chunk_label(chunk)}")
            lines += _more(len(most), full, style)
    flagged = sorted(
        ((chunk, level) for chunk in report["chunks"] if (level := chunk_level(chunk))),
        key=lambda item: (-item[1], -(item[0]["reference"]["delta"] or 0)),
    )
    if flagged:
        lines += ["", style.bold("Least like the reference")]
        for chunk, level in flagged[: None if full else CHUNKS_SHOWN]:
            word = style.distance(f"{DISTANCE_WORDS[level]:22}", level)
            lines.append(
                f"  {chunk['reference']['delta'] or 0.0:5.2f}  {word}  {_chunk_label(chunk)}"
            )
        lines += _more(len(flagged), full, style)
    return lines


def _more(count: int, full: bool, style: _Style) -> list[str]:
    """The line under a chunk list that shows only the first ``CHUNKS_SHOWN`` of ``count``."""
    hidden = 0 if full else count - CHUNKS_SHOWN
    return [style.dim(f"  … and {hidden} more (--all lists them)")] if hidden > 0 else []


def _chunk_label(row: ScoredChunk) -> str:
    """A chunk as its document is named (``document_label``), plus its window: ``posts/a.md#w2``
    rather than the bare id ``a.md#w2``, which two documents can share."""
    base = base_id(row["id"])
    return document_label(row["source"], base) + row["id"][len(base) :]


@dataclass(frozen=True)
class _Area:
    """One area's Delta for the "By area" view, read against the reference's held-out range."""

    group: str
    delta: float
    # Delta divided by the top of the area's usual held-out range at the text's length (the
    # bound of "close", see ``pooled_ceiling``), rounded as shown; None without calibration.
    relative: float | None
    # None when the text is too short to judge.
    level: int | None
    # The reference's median held-out Delta for the area and the top of its usual range.
    typical: float | None
    ceiling: float | None


def _areas(verdict: ScoreVerdict) -> list[_Area]:
    """Each scored area, most different first: by verdict, then by Delta relative to the
    top of the area's usual held-out range at the text's length.

    Areas vary by different amounts on the writer's own text, so raw area Deltas do not
    compare: 1.45 can be usual for sentence shape while 1.34 is unusual for voice. The
    relative value is what the verdict reads, and the verdict is taken from it as rounded
    for display (``calibration.verdict``), so the order, the numbers and the words agree.
    """
    areas = [
        _Area(
            group,
            entry["value"],
            entry["relative"],
            entry["level"],
            entry["typical"],
            entry["ceiling"],
        )
        for group, entry in verdict["by_group"].items()
    ]
    return sorted(
        areas,
        key=lambda area: (
            -(area.level or 0),
            -(area.relative if area.relative is not None else area.delta),
        ),
    )


def _area_lines(areas: list[_Area], style: _Style, *, judged: bool = True) -> list[str]:
    """The "By area" block: each area's Delta over the top of its usual held-out range, or
    the raw Delta when the reference has no calibration for it. A text too short to judge
    gets the numbers without verdict words."""
    calibrated = any(area.relative is not None for area in areas)
    about = (
        f"   Delta {glyphs()['divide']} the top of the reference's usual range in each area; "
        "bars on a log scale"
        if calibrated
        else "   Delta in each area"
    )
    lines = [style.bold("By area") + style.dim(about + ("" if judged else "; indicative only"))]
    if calibrated and judged:
        lines.append(
            style.dim(
                "  close up to 1x, somewhat different to 1.5x, clearly different to 2x, "
                "very different above"
            )
        )
    for area in areas:
        relative = area.relative
        amount = area.delta if relative is None else relative
        cell = f"Delta {amount:.2f}" if relative is None else f"{amount:.2f}x"
        level = area.level or 0
        line = f"  {group_title(area.group):24}{cell:>10}  " + _bar(amount, style, level)
        if area.level is not None:
            line += "  " + style.distance(DISTANCE_WORDS[level], level)
        lines.append(line)
    return lines


def _area_deltas(areas: list[_Area], style: _Style) -> list[str]:
    """The raw area Deltas behind "By area", with the held-out values they are read against.

    Without calibration "By area" already shows the raw Deltas, so this adds nothing.
    """
    if not any(area.ceiling is not None for area in areas):
        return []
    lines = [
        "",
        style.bold("Delta by area")
        + style.dim("   (the reference's held-out median, and the top of its usual range)"),
        style.dim(_row("", "this text", "median", "range top")),
    ]
    for area in areas:
        lines.append(
            _row(
                group_title(area.group),
                f"{area.delta:.2f}",
                f"{area.typical:.2f}" if area.typical is not None else "-",
                f"{area.ceiling:.2f}" if area.ceiling is not None else "-",
            )
        )
    return lines


# Where a draft drifts (``drift``).

SMALL_REFERENCE = (
    "the reference is too small to set paragraph thresholds (they need 10 or more documents "
    "and about 200 paragraphs); "
    "add more of the writer's documents"
)
NEEDS_CONTRAST = (
    "paragraph checks need a reference built with --contrast; judged by Delta alone they "
    "rarely catch an LLM passage"
)
CONVERTED = "line numbers are of the text converted from HTML"
POOLED = (
    "paragraph checks don't apply to pooled records; score without --pool for each "
    "record's own verdict in the document table"
)


def _line_range(entry: ParagraphScore) -> str:
    first, last = entry["lines"]
    return f"line {first}" if first == last else f"lines {first}-{last}"


def _passage_verdict(entry: ParagraphScore, label: str | None, style: _Style) -> str:
    """How a paragraph reads: likeness words first when a contrast set judges it, then Delta,
    each with its value and colored by its level."""
    parts = []
    if label is not None and entry["likeness"] is not None:
        level = entry["likeness_level"] or 0
        words = likeness_words(level, label)
        parts.append(style.distance(words, level) + f" ({entry['likeness']:.2f})")
    if entry["delta"] is not None:
        level = entry["level"] or 0
        parts.append(
            style.distance(str(DISTANCE_WORDS[level]), level) + f" (Delta {entry['delta']:.2f})"
        )
    return ", ".join(parts)


def _trait_line(entry: ParagraphScore, style: _Style) -> str:
    """A paragraph's own traits with their arrows; a trait under one standard deviation has
    none, and is left out."""
    return ", ".join(
        f"{label(trait['metric'].split('.', 1)[1])[0]} "
        + style.distance(_arrow(trait["z"]), z_level(trait["z"]))
        for trait in entry["traits"]
        if _arrow(trait["z"])
    )


def _document_label(document: DocumentPassages, several: bool, names: Mapping[str, str]) -> str:
    """How the document table and ``-q`` name a document, before its lines when several."""
    return f"{names.get(document['name'], document['name'])}, " if several else ""


def throughout(document: DocumentPassages) -> bool:
    """Whether more than half of a document's paragraphs drift: it drifts throughout, and
    listing them would say less than that."""
    paragraphs = document["paragraphs"]
    return sum(entry["drifts"] for entry in paragraphs) * 2 > len(paragraphs) > 0


def _caveats(documents: Sequence[DocumentPassages]) -> list[str]:
    """What limits the paragraph checks of these documents: no contrast set, a reference
    too small for thresholds, lines of converted HTML."""
    found = []
    if any(document["by"] == DELTA for document in documents):
        found.append(NEEDS_CONTRAST)
    elif any(not document["sensitive"] for document in documents if document["judged"]):
        found.append(SMALL_REFERENCE)
    if any(document["converted"] for document in documents):
        found.append(CONVERTED)
    return found


def _where_it_drifts(
    documents: list[DocumentPassages],
    label: str | None,
    style: _Style,
    names: Mapping[str, str],
) -> list[str]:
    """Up to ``PASSAGES_SHOWN`` paragraphs that drift, most unlike the writer first by their
    statistic, shown in document order; a document drifting in more than half its
    paragraphs is one line, "drifts throughout". Nothing when none drifts."""
    several = len(documents) > 1
    lines: list[str] = []
    whole = [document for document in documents if throughout(document)]
    for document in whole:
        count = sum(entry["drifts"] for entry in document["paragraphs"])
        name = _document_label(document, several, names).rstrip(", ")
        lines.append(
            f"  {name + ': ' if name else ''}drifts throughout ({count} of "
            f"{len(document['paragraphs'])} paragraphs read unlike the writer)"
        )
    drifting = [
        (document, entry)
        for document in documents
        if document not in whole
        for entry in document["paragraphs"]
        if entry["drifts"]
    ]
    if not drifting and not whole:
        return []
    shown = sorted(drifting, key=lambda item: -(item[1]["relative"] or 0.0))[:PASSAGES_SHOWN]
    order = {id(entry): index for index, (_, entry) in enumerate(drifting)}
    shown.sort(key=lambda item: order[id(item[1])])
    total = sum(len(document["paragraphs"]) for document in documents if document not in whole)
    header = style.bold(DRIFT_TITLE)
    if drifting:
        count = len(drifting)
        header += style.dim(
            f"   {count} of {total} paragraphs drift{'s' * (count == 1)}, read in spans of at "
            f"least {documents[0]['span_words']} words"
        )
    places = [
        _document_label(document, several, names) + _line_range(entry) for document, entry in shown
    ]
    # Capitalized only when it starts with "line", not a document's name.
    places = [place if several else place[:1].upper() + place[1:] for place in places]
    width = max(14, *(len(place) for place in places)) + 2 if places else 0
    indent = " " * (width + 2)
    for (_, entry), place in zip(shown, places, strict=True):
        lines.append(f"  {place:{width}}{_passage_verdict(entry, label, style)}")
        lines.append(style.dim(f'{indent}"{entry["excerpt"]}"'))
        if traits := _trait_line(entry, style):
            lines.append(f"{indent}{traits}")
    if len(drifting) > len(shown):
        lines.append(
            style.dim(
                f"  and {len(drifting) - len(shown)} more; --by-paragraph lists every paragraph."
            )
        )
    lines += [style.dim(f"  {caveat[0].upper()}{caveat[1:]}.") for caveat in _caveats(documents)]
    return ["", header, *lines]


def _paragraph_level(entry: ParagraphScore, label: str | None) -> tuple[str, str, int]:
    """A paragraph's statistic (its lower span's score over its 95% bound: what its
    threshold applies to), and that span's verdict words and level, by the score its spans
    are judged by."""
    value = f"{entry['relative']:.2f}x" if entry["relative"] is not None else "-"
    if entry["by"] == LIKENESS and label is not None and entry["likeness"] is not None:
        level = entry["likeness_level"] or 0
        return value, likeness_words(level, label), level
    if entry["delta"] is not None:
        level = entry["level"] or 0
        return value, str(DISTANCE_WORDS[level]), level
    return value, "no span", 0


def _by_paragraph(
    documents: list[DocumentPassages],
    label: str | None,
    style: _Style,
    names: Mapping[str, str],
) -> list[str]:
    """Every paragraph of each document: its statistic (x its range) against the
    document's threshold, its lower span's verdict, and * for one that drifts, with why not
    when it is above its range but does not drift."""
    lines: list[str] = []
    for document in documents:
        name = style.dim(f"   {names.get(document['name'], document['name'])}")
        if not document["judged"]:
            reason = document["reason"] or "no span could be judged"
            lines += ["", style.bold("By paragraph") + name, f"  Not read in parts: {reason}."]
            continue
        by = document["by"] or DELTA
        score = f"{label}-likeness" if by == LIKENESS and label else "Delta"
        limits = document["thresholds"]
        above = ""
        if limits:
            one = limits["one"]
            above = f"; * drifts, above {limits['two']:.2f}x its range" + (
                f" ({one:.2f}x on one span)" if one is not None else ""
            )
        lines += [
            "",
            style.bold("By paragraph")
            + name
            + style.dim(f"   spans of at least {document['span_words']} words{above}"),
        ]
        lines += [
            style.dim(f"  {caveat[0].upper()}{caveat[1:]}.") for caveat in _caveats([document])
        ]
        lines.append(style.dim(f"     {'lines':10}{'words':>6}   {'x range':>7}  {score}"))
        for entry in document["paragraphs"]:
            value, words, level = _paragraph_level(entry, label)
            first, last = entry["lines"]
            where = str(first) if first == last else f"{first}-{last}"
            mark = "*" if entry["drifts"] else " "
            excerpt = excerpt_text(entry["excerpt"], EXCERPT_SHOWN)
            lines.append(
                f"  {mark}  {where:10}{entry['words']:>6}   {value:>7}  "
                + style.distance(f"{words:22}", level)
                + style.dim(f"  {excerpt}")
            )
            if entry["note"]:
                lines.append(style.dim(f"{'':32}does not drift: {entry['note']}"))
    return lines


def _no_drift(documents: list[DocumentPassages]) -> str:
    """The one line "Where it drifts" becomes when no paragraph drifts: that the check ran
    and on how much, or why it could not or says little."""
    judged = [document for document in documents if document["judged"]]
    if any(document["pooled"] for document in documents):
        return f"{DRIFT_TITLE}: {POOLED}."
    if not judged:
        if all(not document["spans"] for document in documents):
            words = documents[0]["span_words"]
            return f"{DRIFT_TITLE}: too short to check paragraphs (under {words} words)."
        return (
            f"{DRIFT_TITLE}: the reference has no range for passages this short, so "
            "paragraphs cannot be checked; add more of the writer's documents."
        )
    caveats = _caveats(judged)
    if caveats and caveats[0] in (NEEDS_CONTRAST, SMALL_REFERENCE):
        return f"{DRIFT_TITLE}: {'; '.join(caveats)}."
    count = sum(len(document["paragraphs"]) for document in judged)
    text = f"{DRIFT_TITLE}: no paragraph drifts ({count} paragraphs checked)"
    return text + "".join(f"; {caveat}" for caveat in caveats) + "."


def _drift_view(
    report: ScoreReport,
    reference: Baseline,
    style: _Style,
    by_paragraph: bool,
    shown: Mapping[str, str],
) -> list[str]:
    """The "Where it drifts" section, and with ``by_paragraph`` every paragraph. Several
    documents come in the document table's order (``worst_first``), named as it names them
    (``shown``)."""
    documents = report.get("passages") or []
    if not documents:
        return []
    order = {doc["name"]: index for index, doc in enumerate(worst_first(report["documents"]))}
    documents = sorted(documents, key=lambda document: order.get(document["name"], len(order)))
    label = reference["contrast"]["label"] if reference["contrast"] else None
    judged = [document for document in documents if document["judged"]]
    lines = _where_it_drifts(judged, label, style, shown) or [
        "",
        style.dim(_no_drift(documents)),
    ]
    if by_paragraph:
        lines += _by_paragraph(documents, label, style, shown)
    return lines


def _comparison_view(
    report: ScoreReport,
    reference: Baseline,
    style: _Style,
    full: bool,
    width: int,
    shown: Mapping[str, str],
    by_paragraph: bool = False,
) -> list[str]:
    scored = report["reference"]
    verdict = scored["verdict"]
    delta = verdict["delta"]["value"]
    if delta is None:
        message = (
            too_short_text(verdict)
            if verdict["verdict"] == str(Verdict.TOO_SHORT)
            else "No metrics could be compared with the reference."
        )
        return ["", style.warn(message)]
    documents = report["documents"]
    lines = (
        _document_table(documents, reference, style, width=width, full=full, shown=shown)
        if len(documents) > 1
        else []
    )
    headline = f"Across {len(documents)} documents: " if len(documents) > 1 else "Overall: "
    judged = verdict["judged"]
    if judged:
        # Always set for a judged verdict with a Delta.
        level = verdict["delta"]["level"] or 0
        # The shading is described only when it is shown; the words carry the same reading.
        shading = " Darker orange is further away." if style.color else ""
        lines += [
            "",
            style.bold(headline)
            + style.distance(style.bold(verdict["verdict"]), level)
            + f"   Delta {delta:.2f}"
            + _flagged_note(report, reference, style),
            style.dim(
                f"  Lower is closer. "
                f"{_delta_baseline(_judged_view(report), reference, verdict['delta'])}"
                f"{shading}"
            ),
        ]
    else:
        # Always set for a verdict that is not judged.
        reason = verdict["reason"] or TOO_SHORT
        lines += [
            "",
            style.bold(headline) + style.bold(too_short_text(verdict)),
            style.dim(
                f"  {reason[0].upper()}{reason[1:]}. Delta {delta:.2f}; the numbers below "
                "are indicative only."
            ),
        ]
    # What the verdict reads: without the chunks it left out.
    view = _judged_view(report)
    if view is not report:
        left_out = report["chunk_count"] - view["chunk_count"]
        verb = "is" if left_out == 1 else "are"
        lines[-1] += style.dim(
            f" {left_out} of {report['chunk_count']} chunks {verb} too short to judge and "
            f"{verb} left out (see the note below)."
        )
    if len(documents) > 1:
        lines.append(
            style.dim(
                "  Pooled over every document from here on; the table above has each one's own."
            )
        )
    if reference["contrast"]:
        lines += ["", *_likeness(view, reference, style, verdict)]
    areas = _areas(verdict)
    lines += ["", *_area_lines(areas, style, judged=judged)]
    lines += _drift_view(report, reference, style, by_paragraph, shown)
    lines += ["", *_differences(view, reference, style, judged=judged)]
    # With every document one chunk, the chunk lists would repeat the document table.
    if report["chunk_count"] > max(len(documents), 1):
        lines += _flagged_chunks(report, reference, style, full=full)
    if full:
        divergences = [
            (name, amount)
            for name, amount in scored["pooled_divergence"].items()
            if amount is not None
        ]
        if divergences:
            lines += [
                "",
                style.bold("Pattern divergence")
                + style.dim("   (Jensen-Shannon: 0 = identical, 1 = nothing shared)"),
            ]
            lines += [
                f"  {DISTRIBUTION_LABELS.get(name, name):{LABEL_WIDTH}}{amount:5.2f}"
                for name, amount in divergences
            ]
        lines += _area_deltas(areas, style)
        lines += _comparison_rows(view, reference, style)
    return lines


def _level(verdict: Verdict) -> int:
    """0-3 for a Delta verdict, as ``delta_level`` gives; 0 for no verdict (not comparable,
    too short to judge), which is never colored."""
    return DISTANCES.index(verdict) if verdict in DISTANCES else 0


def severity(
    verdict: Verdict, likeness: LikenessVerdict | None, delta: float | None
) -> tuple[int, float, int]:
    """A sort key that puts the documents furthest from the reference first: by Delta
    verdict, then Delta, then likeness verdict. Documents with no verdict (not comparable,
    too short to judge) come last."""
    level = DISTANCES.index(verdict) if verdict in DISTANCES else -1
    likeness_level = LIKENESSES.index(likeness) if likeness in LIKENESSES else -1
    return (-level, -(delta or 0.0) if level >= 0 else 0.0, -likeness_level)


def worst_first(documents: Sequence[DocumentEntry]) -> list[DocumentEntry]:
    """A score report's ``documents``, furthest from the reference first (``severity``)."""
    return sorted(
        documents,
        key=lambda doc: severity(
            Verdict(doc["verdict"]),
            LikenessVerdict(doc["likeness_verdict"]) if doc["likeness_verdict"] else None,
            doc["delta"],
        ),
    )


def shorten(text: str, width: int) -> str:
    """``text`` in at most ``width`` characters, cut in the middle so that both its first
    folder and the end of its file name survive: ``spring/…-a-blog-post.md``. Names that
    differ only in their folder, or only at the end, stay apart."""
    if len(text) <= width:
        return text
    if width < 3:
        return text[:width]
    first, slash, _ = text.partition("/")
    head = first + slash if slash and len(first) + 1 <= (width - 1) // 2 else ""
    if not head:
        head = text[: (width - 1) // 3]
    tail = width - len(head) - 1
    return f"{head}…{text[-tail:]}"


def _likeness_cell(likeness: LikenessVerdict) -> str:
    """A likeness verdict in the table's few columns."""
    return LIKENESS_CELLS.get(likeness, str(likeness))


def _counts(documents: Sequence[DocumentEntry]) -> str:
    """How many documents got each verdict, furthest first: ``3 very different, 8 close``."""
    counts: dict[str, int] = {}
    for doc in worst_first(documents):
        counts[doc["verdict"]] = counts.get(doc["verdict"], 0) + 1
    return ", ".join(f"{count} {verdict}" for verdict, count in counts.items())


def _fit(prefix: str, items: list[str], width: int) -> str:
    """``prefix`` and as many of ``items`` as fit in ``width`` visible columns (at least
    one), comma-separated."""
    line = prefix + items[0]
    for item in items[1:]:
        if len(_strip(line + ", " + item)) > width:
            break
        line += ", " + item
    return line


def not_judged(chunks: int, chunks_judged: int) -> str:
    """``; 8 of 10 chunks not judged: too short`` for a verdict that rests on only some of
    its ``chunks``, else ``""`` (also when none is judged, which says so itself)."""
    left_out = chunks - chunks_judged
    if not chunks_judged or not left_out:
        return ""
    return f"; {left_out} of {chunks} chunks not judged: too short"


def _document_table(
    documents: Sequence[DocumentEntry],
    reference: Baseline,
    style: _Style,
    *,
    width: int = DEFAULT_WIDTH,
    full: bool = False,
    shown: Mapping[str, str] | None = None,
) -> list[str]:
    """One row per scored document, furthest from the reference first, fitted to ``width``
    columns, with the metrics that set apart each one that is not close. Each is named by
    ``shown`` when it has it, else by its saved name. Past ``CLOSE_ROWS`` close documents,
    the rest are counted instead, unless ``full``."""
    names = {doc["name"]: (shown or {}).get(doc["name"], doc["name"]) for doc in documents}
    contrast = reference["contrast"]
    name_label = contrast["label"] if contrast else None
    likeness_header = f"{name_label}-likeness" if name_label else ""
    likeness_width = max(len(likeness_header), *map(len, LIKENESS_CELLS.values()))
    # indent, words, verdict, Delta and likeness, with two spaces between columns
    fixed = 2 + 2 + 6 + 2 + VERDICT_WIDTH + 2 + 5 + (2 + likeness_width if name_label else 0)
    longest = max(len("document"), *(len(name) for name in names.values()))
    name_width = max(min(longest, NAME_WIDTH, width - fixed), MIN_NAME_WIDTH)
    header = f"  {'document':{name_width}}  {'words':>6}  {'verdict':{VERDICT_WIDTH}}  {'Delta':>5}"
    if name_label:
        header += f"  {likeness_header}"
    lines = [
        "",
        style.bold("By document") + style.dim("   furthest from the reference first"),
        f"  {len(documents)} documents: {_counts(documents)}",
        style.dim(header),
    ]
    close_shown = 0
    for doc in worst_first(documents):
        verdict = Verdict(doc["verdict"])
        level = _level(verdict)
        if verdict == Verdict.CLOSE and not full:
            close_shown += 1
            if close_shown > CLOSE_ROWS:
                continue
        name = shorten(names[doc["name"]], name_width)
        if verdict == Verdict.NOT_COMPARABLE:
            lines.append(f"  {name:{name_width}}  {doc['words']:>6,}  {verdict}")
            continue
        if not doc["judged"]:
            # Its figures are indicative only, so the row gives none.
            lines.append(f"  {name:{name_width}}  {doc['words']:>6,}  {too_short_text(doc)}")
            continue
        delta = f"{doc['delta']:.2f}" if doc["delta"] is not None else "-"
        row = (
            f"  {name:{name_width}}  {doc['words']:>6,}  "
            + style.distance(f"{verdict:{VERDICT_WIDTH}}", level)
            + f"  {delta:>5}"
        )
        if name_label and doc["likeness_verdict"]:
            likeness = LikenessVerdict(doc["likeness_verdict"])
            shade = LIKENESSES.index(likeness) if likeness in LIKENESSES else 0
            row += "  " + style.distance(_likeness_cell(likeness), shade)
        lines.append(row.rstrip())
        if note := not_judged(doc["chunks"], doc["chunks_judged"]):
            # Its verdict rests on its judged chunks alone.
            lines.append(style.dim(f"      {note[2:]}"))
        notable = [
            (item["metric"], item["z"])
            for item in doc["differences"]
            if abs(item["z"]) >= NOTABLE_Z
        ]
        differences = [
            f"{metric_label(metric)} {style.distance(_arrow(z), z_level(z))}"
            for metric, z in notable
        ]
        if level and differences:
            lines.append(_fit(style.dim("      most different in: "), differences, width))
    hidden = close_shown - CLOSE_ROWS
    if hidden > 0:
        lines.append(style.dim(f"  … and {hidden} more close (--all lists them)"))
    return lines


def metric_label(metric: str) -> str:
    """A ``group.name`` metric key in plain English, as reports print it."""
    return label(metric.split(".", 1)[1])[0]


def format_reference_summary(
    report: ReferenceReport, *, color: bool = False, full: bool = False, verbose: bool = False
) -> str:
    """The short view ``build`` prints: size, held-out range, contrast and warnings; ``full``
    adds every metric (``format_summary`` shows the key ones)."""
    if full:
        return format_summary(report, color=color, full=True, verbose=verbose)
    style = _Style(color, truecolor=False)
    lines = [style.bold("STYLE PROFILE") + style.dim(f"   {_size(report)}")]
    lines += _reference_lines(report, style)
    lines += ["", style.dim("Pass --all, or run `styleprofile show` on it, to see the metrics.")]
    if report["warnings"]:
        lines += [
            "",
            *(
                style.warn(f"Note: {warning_text(warning, verbose=verbose)}")
                for warning in report["warnings"]
            ),
        ]
    return _human_text(lines, verbose=verbose)


def _size(report: ReportBase) -> str:
    chunk_word = "chunk" if report["chunk_count"] == 1 else "chunks"
    return f"{report['chunk_count']} {chunk_word}, {report['word_count']:,} words"


def format_summary(
    report: ReferenceReport | ScoreReport,
    reference: Baseline | None = None,
    *,
    color: bool = False,
    full: bool = False,
    width: int = DEFAULT_WIDTH,
    shown: Mapping[str, str] | None = None,
    by_paragraph: bool = False,
    verbose: bool = False,
) -> str:
    """Terminal view of a report; ``full`` shows every metric instead of the key ones.

    A score report is compared with ``reference``, its ``reference.baseline``; without it,
    or for a reference profile, this shows the report's own metrics. The document table of
    a multi-document score fits ``width`` columns, and names each document by ``shown``
    (its saved name to how the command line names it), or else by its saved name. A score
    that read its documents in parts shows where they drift, and ``by_paragraph`` lists
    every paragraph."""
    truecolor = os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}
    style = _Style(color, truecolor=color and truecolor)
    size = _size(report)
    documents = len(report["documents"]) if "documents" in report else 0
    if documents > 1:
        size = f"{documents} documents, {size}"
    if reference is not None and "reference" in report:
        ref_name = Path(report["reference"]["path"] or "reference").name
        lines = [
            style.bold("STYLE COMPARISON")
            + style.dim(f"   {size}   vs {ref_name} ({reference['chunk_count']} chunks)")
        ]
        lines += _comparison_view(report, reference, style, full, width, shown or {}, by_paragraph)
    else:
        lines = [style.bold("STYLE PROFILE") + style.dim(f"   {size}")]
        lines += _profile_view(report, style, full)
    if not full:
        lines += ["", style.dim("Pass --all to see every metric.")]
    if report["warnings"]:
        lines += [
            "",
            *(
                style.warn(f"Note: {warning_text(warning, verbose=verbose)}")
                for warning in report["warnings"]
            ),
        ]
    return _human_text(lines, verbose=verbose)


def _exact(result: AucResult) -> str | None:
    """How to describe an AUC whose interval is exact rather than resampled, or None.

    Every resample of a perfectly separated sample gives the same AUC, so its "interval" has
    no width; printed as one it would read as certainty however few documents there are.
    """
    found = result["bootstrap"]
    interval = result["auc_ci"]
    if found["method"] != "exact" or not interval:
        return None
    kind = "no difference" if interval[0] == 0.5 else "perfect separation"
    return (
        f"{kind} on {found['reference_documents']} reference and "
        f"{found['contrast_documents']} contrast documents"
    )


def _auc_cell(result: AucResult) -> str:
    auc = result["auc"]
    if auc is None:
        return "-"
    if _exact(result):
        # The table is narrow; the note under it says what "exact" means.
        return f"{auc:.2f} (exact)"
    interval = result["auc_ci"]
    return f"{auc:.2f} ({interval[0]:.2f}-{interval[1]:.2f})" if interval else f"{auc:.2f}"


def _survival(entry: Survival, style: _Style) -> str:
    z, remaining = entry["z"], entry["remaining"]
    if z is None:
        return "-"
    if remaining is None:
        return f"{z:+.1f}"
    removed = 1 - remaining
    # Color by how much of the signal is left: a surviving tell is the thing to look at.
    level = 0 if remaining < 0.25 else 1 if remaining < 0.5 else 2 if remaining < 0.75 else 3
    # An edit can also push a signal further from the reference than the original was.
    change = f"{100 * removed:.0f}% gone" if removed >= 0 else f"{-100 * removed:.0f}% stronger"
    return f"{z:+.1f} " + style.distance(change, level)


def format_evaluation(
    result: EvaluationReport, *, color: bool = False, verbose: bool = False
) -> str:
    """Terminal view of an evaluation report (``styleprofile evaluate``)."""
    truecolor = os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}
    style = _Style(color, truecolor=color and truecolor)
    name = result["label"]
    sets = result["sets"]
    lines = [
        style.bold("REWORDING STRESS TEST")
        + style.dim(
            f"   {result['contrast']['drafts']} {name} drafts vs "
            f"{result['reference']['documents']} reference documents"
        ),
        style.dim(
            f"  Each draft is scored with {name}-likeness weights learned without it, and each "
            "edited draft with the weights that left out its original."
        ),
        "",
        style.dim(f"  {'':12}{'AUC (95% CI)':>20}{'median likeness':>18}   drafts still flagged"),
    ]
    for set_label, entry in sets.items():
        median = entry["likeness_median_chunks"]
        flagged = f"{entry['flagged']} of {entry['drafts']}"
        lines.append(
            f"  {set_label:12}{_auc_cell(entry):>20}"
            + (f"{median:>18.2f}" if median is not None else f"{'-':>18}")
            + f"   {flagged}"
        )
    lines.append(
        style.dim(
            f"  AUC 1.0 separates every draft from the reference, 0.5 is chance. The median is "
            f"over chunks; flagged drafts read "
            f'"{likeness_words(2, name)}" or "{likeness_words(3, name)}".'
        )
    )
    # The originals have no edit statistics.
    edits = [
        (set_label, stats) for set_label, entry in sets.items() if (stats := entry.get("edits"))
    ]
    if edits:
        lines += ["", style.bold("How much the edits changed")]
        for set_label, stats in edits:
            changed = stats["ngram13_changed_median"]
            ratio = stats["word_ratio_median"]
            parts = [
                f"{100 * changed:.0f}% of 13-word sequences rewritten"
                if changed is not None
                else "",
                f"length x{ratio:.2f}" if ratio is not None else "",
            ]
            lines.append(f"  {set_label:12}" + ", ".join(part for part in parts if part))
    edited_labels = [item for item in sets if item != "original"]
    # A partial set is compared with the originals it covers, not the "original" column.
    headers = {item: item + ("*" if sets[item].get("partial") else "") for item in edited_labels}
    lines += [
        "",
        style.bold("Signal survival")
        + style.dim(
            f"   mean z of the strongest {name} signals; how much of the original gap from "
            "the reference each edit removed (or added)"
        ),
        style.dim(
            f"  {'':{LABEL_WIDTH}}{'reference':>10}{'original':>10}"
            + "".join(f"{headers[item]:>20}" for item in edited_labels)
        ),
    ]
    for signal in result["signals"]:
        text = label(signal["metric"].split(".", 1)[1])[0]
        original = signal["original_z"]
        cells = [
            f"{signal['reference_z']:+10.1f}",
            f"{original:+10.1f}" if original is not None else f"{'-':>10}",
        ]
        survival = [_survival(signal["edited"][item], style) for item in edited_labels]
        # Pad by visible width, since color codes do not take up columns.
        padded = [" " * max(0, 20 - len(_strip(cell))) + cell for cell in survival]
        lines.append(f"  {text:{LABEL_WIDTH}}" + "".join(cells + padded))
    partial = [item for item in edited_labels if sets[item].get("partial")]
    if partial:
        lines.append(
            style.dim(
                f"  * {', '.join(partial)} edited only some drafts; its share is measured "
                "against the original z of the drafts it covers, not the column above."
            )
        )
    retrain = result["retrain"]
    if retrain:
        lines += [
            "",
            style.bold("Retrained with the edited drafts in the contrast set")
            + style.dim(f"   AUC {_auc_cell(retrain)} over all drafts"),
        ]
        for item, entry in retrain["by_set"].items():
            before = entry["before"]
            lines.append(
                f"  {item:12}{_auc_cell(entry):>20}"
                + (style.dim(f"   was {before:.2f}") if before is not None else "")
            )
    entries = [*sets.values(), *([retrain, *retrain["by_set"].values()] if retrain else [])]
    if any(_exact(entry) for entry in entries):
        lines += [
            "",
            style.dim(
                '"exact": every resample gives the same AUC (the drafts and the reference\'s '
                "chunks never overlap, or every chunk scores the same), so there is no interval "
                "to show. That is not certainty: with few documents, new drafts may differ."
            ),
        ]
    lines += [
        "",
        style.dim(
            f"Verdicts on edited text are weaker evidence: editing removes the {name} habits "
            "the score relies on, so a draft that reads like the reference may still be a "
            f"lightly edited {name} draft."
        ),
    ]
    if result["warnings"]:
        lines += [
            "",
            *(
                style.warn(f"Note: {warning_text(warning, verbose=verbose)}")
                for warning in result["warnings"]
            ),
        ]
    return _human_text(lines, verbose=verbose)


def _strip(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _human_text(lines: list[str], *, verbose: bool) -> str:
    """Keep prose readable on a normal terminal; aligned verdict tables keep their layout.

    Verbose retains the original rendering, including its explanatory paragraphs.
    """
    if verbose:
        return "\n".join(lines)
    rendered = []
    for line in lines:
        plain = _strip(line)
        table = re.search(r"^  .*\S {3,}\S", plain) is not None
        if len(plain) <= 120 or table:
            rendered.append(line)
        else:
            indent = plain[: len(plain) - len(plain.lstrip(" "))]
            rendered.append(
                textwrap.fill(
                    line,
                    width=120,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(rendered)
