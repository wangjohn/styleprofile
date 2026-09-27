"""Human-readable terminal rendering of a style profile report.

The JSON report is the complete record; this view names each metric in plain English, shows
units, trims statistics to the ones a reader needs, and ends with a short verdict when the
report was scored against a reference.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from styleprofile.core import DISTANCES, Verdict
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
from styleprofile.weighting import (
    LENGTH_AUC_WARNING,
    likeness_level,
    likeness_words,
    mean_ceiling,
)

# Wide enough for the longest metric label, so value columns stay aligned.
LABEL_WIDTH = 46
VALUE_WIDTH = 12
DIFFERENCES_SHOWN = 8
NOTABLE_Z = 1.0
CHUNKS_SHOWN = 3
BAR_WIDTH = 20
# A full bar is 3x an area's usual held-out range, or a raw Delta of 3 without calibration.
BAR_SCALE = 3.0


# How far from the reference, as one orange ramp (pale -> deep). "Close" stays uncolored so
# color only appears where there is something to look at. Validated as an ordinal ramp on
# light and dark surfaces; no green or red, since distance is not right or wrong. Each
# color is paired with a word or arrows, so the report reads the same without color.
DISTANCE_RGB: tuple[tuple[int, int, int], ...] = ((217, 149, 106), (220, 111, 52), (184, 70, 26))
DISTANCE_256: tuple[int, ...] = (173, 166, 130)
DISTANCE_WORDS: tuple[Verdict, ...] = DISTANCES


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


def _range(stats: dict[str, Any], unit: str) -> str:
    if stats.get("sd") is None or stats.get("mean") is None:
        return ""
    low = max(0.0, stats["mean"] - stats["sd"])
    return f"{value(low, unit).split(' ')[0]} - {value(stats['mean'] + stats['sd'], unit)}"


def _arrow(z: float | None) -> str:
    if z is None:
        return ""
    size = abs(z)
    count = 0 if size < 1 else 1 if size < 2 else 2 if size < 3 else 3
    return ("▲" if z > 0 else "▼") * count


def _bar(amount: float, style: _Style, level: int) -> str:
    """Filled length carries the amount; only the filled part is colored."""
    filled = max(0, min(BAR_WIDTH, round(amount / BAR_SCALE * BAR_WIDTH)))
    return style.distance("█" * filled, level) + style.dim("░" * (BAR_WIDTH - filled))


def describe_delta(delta: float, ceiling: float | None = None) -> Verdict:
    """A reading of Delta; see ``delta_level``."""
    return DISTANCE_WORDS[delta_level(delta, ceiling)]


def _row(text: str, *cells: str) -> str:
    return (
        f"  {text:{LABEL_WIDTH}}" + "".join(f"{cell:>{VALUE_WIDTH}}" for cell in cells)
    ).rstrip()


def _key_rows(report: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    sections = []
    for title, metrics in KEY_VIEW:
        present = [
            (group, name) for group, name in metrics if name in report["summary"].get(group, {})
        ]
        if present:
            sections.append((title, present))
    return sections


def _profile_view(report: dict[str, Any], style: _Style, full: bool) -> list[str]:
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
    return lines + _reference_lines(report, style)


def _reference_lines(report: dict[str, Any], style: _Style) -> list[str]:
    """How the profile works as a reference: its held-out Delta range and any contrast."""
    lines: list[str] = []
    held = (report.get("calibration") or {}).get("delta")
    if held:
        lines += [
            "",
            style.bold("As a reference")
            + style.dim(
                f"   held-out Delta {held['median']:.2f} typically, 95% under {held['p95']:.2f} "
                f"({report['calibration']['sources']} documents)"
            ),
        ]
    if report.get("contrast"):
        lines += _contrast_summary(report["contrast"], style)
    return lines


def _contrast_summary(contrast: dict[str, Any], style: _Style) -> list[str]:
    name = contrast["label"]
    calibration = contrast["calibration"]
    effects = [
        (effect, metric)
        for group, values in contrast["effects"].items()
        for metric, effect in values.items()
    ]
    top = sorted(effects, key=lambda item: -abs(item[0]))[:4]
    auc = calibration.get("auc")
    interval = calibration.get("auc_ci")
    separation = ""
    if auc is not None:
        separation = f"; separates them from the reference with AUC {auc:.2f}"
        if interval:
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
        + ", ".join(f"{label(metric)[0]} {'▲' if effect > 0 else '▼'}" for effect, metric in top),
    ]
    length = calibration.get("length_baseline")
    if length:
        lines.insert(3, style.dim(f"  {_length_line(length, name)}"))
    return lines


def _length_line(length: dict[str, Any], name: str) -> str:
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


def _chunk_z(report: dict[str, Any]) -> dict[tuple[str, str], list[float]]:
    totals: dict[tuple[str, str], list[float]] = {}
    for chunk in report["chunks"]:
        for group, values in chunk["reference"]["z"].items():
            for name, z in values.items():
                totals.setdefault((group, name), []).append(z)
    return totals


def _average_z(report: dict[str, Any], reference: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Each metric's z against the reference, averaged over the scored chunks.

    For a metric the reference never varies on, every differing chunk scores the cap in one
    direction or the other; average the size instead so opposite differences cannot cancel.
    """
    averaged: dict[tuple[str, str], float] = {}
    for (group, name), zs in _chunk_z(report).items():
        if reference["summary"].get(group, {}).get(name, {}).get("sd"):
            averaged[(group, name)] = sum(zs) / len(zs)
        else:
            size = sum(abs(z) for z in zs) / len(zs)
            here = report["summary"][group][name]["mean"] or 0.0
            there = reference["summary"][group][name]["mean"] or 0.0
            averaged[(group, name)] = -size if here < there else size
    return averaged


def _comparison_rows(report: dict[str, Any], reference: dict[str, Any], style: _Style) -> list[str]:
    averaged = _average_z(report, reference)
    lines = ["", style.bold("All metrics"), style.dim(_row("", "this text", "reference"))]
    for group, metrics in report["summary"].items():
        if group in UNSCORED_GROUPS:
            continue
        lines += ["", style.bold(group_title(group))]
        for name, stats in metrics.items():
            text, unit = label(name)
            ref = reference["summary"].get(group, {}).get(name, {})
            z = averaged.get((group, name))
            row = _row(text, value(stats["mean"], unit), value(ref.get("mean"), unit))
            # No trailing spaces on rows without arrows.
            if z is not None and (arrow := _arrow(z)):
                row += f"   {style.distance(arrow, z_level(z))}"
            lines.append(row)
    return lines


def _differences(report: dict[str, Any], reference: dict[str, Any], style: _Style) -> list[str]:
    """Metrics whose average z over chunks sits furthest from the reference.

    A metric the reference never varies on scores a capped z when a chunk differs and 0
    when it matches, so its arrows reflect how many chunks depart from the reference.
    """
    averaged = _average_z(report, reference)
    differing = {key: sum(z != 0 for z in zs) for key, zs in _chunk_z(report).items()}
    ranked = sorted(
        ((key, z) for key, z in averaged.items() if abs(z) >= NOTABLE_Z),
        key=lambda item: -abs(item[1]),
    )[:DIFFERENCES_SHOWN]
    lines = [
        style.bold("Biggest differences")
        + style.dim("   (each ▲ or ▼ is one standard deviation, up to 3)"),
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
            if ref.get("sd")
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


def _likeness(report: dict[str, Any], contrast: dict[str, Any], style: _Style) -> list[str]:
    scored = report["reference"]
    score = scored.get("likeness_mean")
    if score is None:
        return []
    name = contrast["label"]
    calibration = contrast["calibration"]
    level = likeness_level(score, calibration, report["chunk_count"])
    shares: dict[str, list[float]] = {}
    zs: dict[str, list[float]] = {}
    for chunk in report["chunks"]:
        for signal in chunk["reference"].get("likeness_signals", []):
            shares.setdefault(signal["metric"], []).append(signal["contribution"])
            zs.setdefault(signal["metric"], []).append(signal["z"])
    ranked = sorted(shares, key=lambda metric: -sum(shares[metric]))[:3]
    signals = []
    for metric in ranked:
        z = sum(zs[metric]) / len(zs[metric])
        signals.append(
            f"{label(metric.split('.', 1)[1])[0]} {style.distance(_arrow(z), z_level(z))}"
        )
    lines = [
        style.bold(f"{name}-likeness: ")
        + style.distance(style.bold(likeness_words(level, name)), level)
        + f"   {score:.2f}",
        style.dim(
            f"  The reference's own writing scores {calibration['reference']['median']:.2f} "
            f"typically (95% under {calibration['reference']['p95']:.2f}); "
            f"the {name} drafts {calibration['contrast']['median']:.2f}."
        ),
    ]
    if signals and level:
        lines.append(f"  Strongest {name} signals: " + ", ".join(signals))
    return lines


def _delta_baseline(reference: dict[str, Any]) -> str:
    held = (reference.get("calibration") or {}).get("delta")
    if not held:
        return "Text by the reference's own writer usually scores around 0.8."
    return (
        f"The reference's own held-out writing scores {held['median']:.2f} typically, "
        f"95% under {held['p95']:.2f}."
    )


def _flagged_chunks(
    report: dict[str, Any], reference: dict[str, Any], ceiling: float | None, style: _Style
) -> list[str]:
    """Up to a few chunks per list, only those that are actually flagged."""
    lines: list[str] = []
    contrast = reference.get("contrast")
    if contrast and report["reference"].get("likeness_mean") is not None:
        name = contrast["label"]
        ranked = sorted(report["chunks"], key=lambda chunk: -chunk["reference"]["likeness"])
        flagged = [
            (chunk, level)
            for chunk in ranked
            if (level := likeness_level(chunk["reference"]["likeness"], contrast["calibration"]))
        ][:CHUNKS_SHOWN]
        if flagged:
            lines += ["", style.bold(f"Most {name}-like chunks")]
            for chunk, level in flagged:
                word = style.distance(f"{likeness_words(level, name):22}", level)
                lines.append(f"  {chunk['reference']['likeness']:5.2f}  {word}  {chunk['id']}")
    ranked = sorted(report["chunks"], key=lambda chunk: -(chunk["reference"]["delta"] or 0))
    flagged = [
        (chunk, level)
        for chunk in ranked
        if (level := delta_level(chunk["reference"]["delta"] or 0.0, ceiling))
    ][:CHUNKS_SHOWN]
    if flagged:
        lines += ["", style.bold("Least like the reference")]
        for chunk, level in flagged:
            word = style.distance(f"{DISTANCE_WORDS[level]:22}", level)
            lines.append(f"  {chunk['reference']['delta'] or 0.0:5.2f}  {word}  {chunk['id']}")
    return lines


@dataclass(frozen=True)
class _Area:
    """One area's Delta for the "By area" view, read against the reference's held-out range."""

    group: str
    delta: float
    # Delta divided by the top of the area's usual held-out range (the bound of "close",
    # see ``mean_ceiling``), rounded as shown; None without calibration.
    relative: float | None
    level: int
    # The reference's median held-out Delta for the area and the top of its usual range.
    typical: float | None
    ceiling: float | None


def _areas(report: dict[str, Any], reference: dict[str, Any]) -> list[_Area]:
    """Each scored area, most different first: by verdict, then by Delta relative to the
    top of the area's usual held-out range.

    Areas vary by different amounts on the writer's own text, so raw area Deltas do not
    compare: 1.45 can be usual for sentence shape while 1.34 is unusual for voice. The
    relative value is what the verdict reads, and the verdict is taken from it as rounded
    for display, so the order, the numbers and the words agree.
    """
    held = (reference.get("calibration") or {}).get("delta", {}).get("by_group", {})
    areas = []
    for group, amount in report["reference"]["delta_by_group_mean"].items():
        if amount is None:
            continue
        stats = held.get(group, {})
        ceiling = mean_ceiling(stats, report["chunk_count"])
        if ceiling:
            relative = round(amount / ceiling, 2)
            level = delta_level(relative, 1.0)
        else:
            relative, level = None, delta_level(amount)
        areas.append(_Area(group, amount, relative, level, stats.get("median"), ceiling))
    return sorted(areas, key=lambda area: (-area.level, -(area.relative or area.delta)))


def _area_lines(areas: list[_Area], style: _Style) -> list[str]:
    """The "By area" block: each area's Delta over the top of its usual held-out range, or
    the raw Delta when the reference has no calibration for it."""
    calibrated = any(area.relative is not None for area in areas)
    lines = [
        style.bold("By area")
        + style.dim(
            "   Delta ÷ the top of the reference's usual range in each area"
            if calibrated
            else "   Delta in each area"
        )
    ]
    if calibrated:
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
        lines.append(
            f"  {group_title(area.group):24}{cell:>10}  "
            + _bar(amount, style, area.level)
            + "  "
            + style.distance(DISTANCE_WORDS[area.level], area.level)
        )
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


def _comparison_view(
    report: dict[str, Any], reference: dict[str, Any], style: _Style, full: bool
) -> list[str]:
    scored = report["reference"]
    delta = scored["delta_mean"]
    if delta is None:
        return ["", style.warn("No metrics could be compared with the reference.")]
    held = (reference.get("calibration") or {}).get("delta", {})
    count = report["chunk_count"]
    ceiling = mean_ceiling(held, count)
    chunk_ceiling = mean_ceiling(held, 1)
    level = delta_level(delta, ceiling)
    # The shading is described only when it is shown; the words carry the same reading.
    shading = " Darker orange is further away." if style.color else ""
    lines = [
        "",
        style.bold("Overall: ")
        + style.distance(style.bold(describe_delta(delta, ceiling)), level)
        + f"   Delta {delta:.2f}",
        style.dim(f"  Lower is closer. {_delta_baseline(reference)}{shading}"),
    ]
    if reference.get("contrast"):
        lines += ["", *_likeness(report, reference["contrast"], style)]
    areas = _areas(report, reference)
    lines += ["", *_area_lines(areas, style)]
    lines += ["", *_differences(report, reference, style)]
    if report["chunk_count"] > 1:
        lines += _flagged_chunks(report, reference, chunk_ceiling, style)
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
        lines += _comparison_rows(report, reference, style)
    return lines


def format_reference_summary(
    report: dict[str, Any], *, color: bool = False, full: bool = False
) -> str:
    """The short view ``build`` prints: size, held-out range, contrast and warnings; ``full``
    adds every metric (``format_summary`` shows the key ones)."""
    if full:
        return format_summary(report, color=color, full=True)
    style = _Style(color, truecolor=False)
    lines = [style.bold("STYLE PROFILE") + style.dim(f"   {_size(report)}")]
    lines += _reference_lines(report, style)
    lines += ["", style.dim("Pass --all, or run `styleprofile show` on it, to see the metrics.")]
    if report["warnings"]:
        lines += ["", *(style.warn(f"Note: {warning}") for warning in report["warnings"])]
    return "\n".join(lines)


def _size(report: dict[str, Any]) -> str:
    chunk_word = "chunk" if report["chunk_count"] == 1 else "chunks"
    return f"{report['chunk_count']} {chunk_word}, {report['word_count']:,} words"


def format_summary(
    report: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    color: bool = False,
    full: bool = False,
) -> str:
    """Terminal view of a report; ``full`` shows every metric instead of the key ones."""
    truecolor = os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}
    style = _Style(color, truecolor=color and truecolor)
    size = _size(report)
    if reference is not None and "reference" in report:
        ref_name = Path(report["reference"]["path"] or "reference").name
        lines = [
            style.bold("STYLE COMPARISON")
            + style.dim(f"   {size}   vs {ref_name} ({reference.get('chunk_count', '?')} chunks)")
        ]
        lines += _comparison_view(report, reference, style, full)
    else:
        lines = [style.bold("STYLE PROFILE") + style.dim(f"   {size}")]
        lines += _profile_view(report, style, full)
    if not full:
        lines += ["", style.dim("Pass --all to see every metric.")]
    if report["warnings"]:
        lines += ["", *(style.warn(f"Note: {warning}") for warning in report["warnings"])]
    return "\n".join(lines)


def _auc_cell(result: dict[str, Any]) -> str:
    auc = result.get("auc")
    if auc is None:
        return "-"
    interval = result.get("auc_ci")
    return f"{auc:.2f} ({interval[0]:.2f}-{interval[1]:.2f})" if interval else f"{auc:.2f}"


def _survival(entry: dict[str, Any], style: _Style) -> str:
    z, remaining = entry.get("z"), entry.get("remaining")
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


def format_evaluation(result: dict[str, Any], *, color: bool = False) -> str:
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
    edits = [(set_label, entry["edits"]) for set_label, entry in sets.items() if entry.get("edits")]
    if edits:
        lines += ["", style.bold("How much the edits changed")]
        for set_label, stats in edits:
            changed = stats.get("ngram13_changed_median")
            ratio = stats.get("word_ratio_median")
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
    retrain = result.get("retrain")
    if retrain:
        lines += [
            "",
            style.bold("Retrained with the edited drafts in the contrast set")
            + style.dim(f"   AUC {_auc_cell(retrain)} over all drafts"),
        ]
        for item, entry in retrain["by_set"].items():
            before = entry.get("before")
            lines.append(
                f"  {item:12}{_auc_cell(entry):>20}"
                + (style.dim(f"   was {before:.2f}") if before is not None else "")
            )
    lines += [
        "",
        style.dim(
            f"Verdicts on edited text are weaker evidence: editing removes the {name} habits "
            "the score relies on, so a draft that reads like the reference may still be a "
            f"lightly edited {name} draft."
        ),
    ]
    if result["warnings"]:
        lines += ["", *(style.warn(f"Note: {warning}") for warning in result["warnings"])]
    return "\n".join(lines)


def _strip(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)
