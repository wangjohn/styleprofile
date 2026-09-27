"""Human-readable terminal rendering of a style profile report.

The JSON report is the complete record; this view names each metric in plain English, shows
units, trims statistics to the ones a reader needs, and ends with a short verdict when the
report was scored against a reference.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from styleprofile.profile import UNSCORED_GROUPS
from styleprofile.weighting import LENGTH_AUC_WARNING

WORDS = "words"
PCT = "%"
PER_1K = "/1k"
PLAIN = ""

GROUP_TITLES: dict[str, str] = {
    "size": "Size",
    "sentence_shape": "Sentence shape",
    "rhythm": "Rhythm",
    "vocabulary": "Vocabulary",
    "punctuation": "Punctuation",
    "voice": "Voice and stance",
    "markdown": "Formatting",
    "syntax": "Syntax",
    "sentence_openers": "How sentences open",
    "function_words": "Function words",
}

LABELS: dict[str, tuple[str, str]] = {
    "words": ("Words", PLAIN),
    "sentences": ("Sentences", PLAIN),
    "paragraphs": ("Paragraphs", PLAIN),
    "apostrophes": ("Apostrophes", PLAIN),
    "sentence_words_mean": ("Average sentence length", WORDS),
    "sentence_words_median": ("Median sentence length", WORDS),
    "sentence_words_sd": ("Sentence length spread", WORDS),
    "short_sentences_pct": ("Short sentences (8 words or fewer)", PCT),
    "long_sentences_pct": ("Long sentences (30+ words)", PCT),
    "paragraph_sentences_mean": ("Sentences per paragraph", PLAIN),
    "paragraph_words_mean": ("Words per paragraph", WORDS),
    "one_sentence_paragraphs_pct": ("One-sentence paragraphs", PCT),
    "sentence_length_cv": ("Sentence length variation (0 = uniform)", PLAIN),
    "sentence_length_change_mean": ("Length change from one sentence to the next", WORDS),
    "word_chars_mean": ("Average word length (letters)", PLAIN),
    "long_words_pct": ("Long words (7+ letters)", PCT),
    "mattr_100": ("Vocabulary variety (MATTR, 0-1)", PLAIN),
    "mtld": ("Lexical diversity (MTLD)", PLAIN),
    "hapax_share_100": ("Words used only once, per 100", PCT),
    "commas_per_1k": ("Commas", PER_1K),
    "semicolons_per_1k": ("Semicolons", PER_1K),
    "colons_per_1k": ("Colons", PER_1K),
    "em_dashes_per_1k": ("Em dashes", PER_1K),
    "parentheses_per_1k": ("Parentheses", PER_1K),
    "questions_per_1k": ("Question marks", PER_1K),
    "exclamations_per_1k": ("Exclamation marks", PER_1K),
    "ellipses_per_1k": ("Ellipses", PER_1K),
    "quotations_per_1k": ("Quotations", PER_1K),
    "curly_apostrophe_pct": ("Curly apostrophes (vs straight)", PCT),
    "contractions_per_1k": ("Contractions", PER_1K),
    "first_singular_per_1k": ("I / me / my", PER_1K),
    "first_plural_per_1k": ("We / us / our", PER_1K),
    "second_person_per_1k": ("You / your", PER_1K),
    "negations_per_1k": ("Negations", PER_1K),
    "hedges_per_1k": ("Hedges (maybe, I think)", PER_1K),
    "boosters_per_1k": ("Intensifiers (really, clearly)", PER_1K),
    "and_but_so_openers_pct": ("Sentences starting And / But / So", PCT),
    "transition_openers_pct": ("Sentences starting However / Moreover", PCT),
    "llm_markers_per_1k": ("LLM marker words (delve, crucial)", PER_1K),
    "not_just_but_per_1k": ('"Not just X, but Y"', PER_1K),
    "headings_per_1k": ("Headings", PER_1K),
    "list_items_per_1k": ("List items", PER_1K),
    "links_per_1k": ("Links", PER_1K),
    "bold_per_1k": ("Bold phrases", PER_1K),
    "code_spans_per_1k": ("Inline code", PER_1K),
    "parse_depth_mean": ("Parse tree depth (clause nesting)", PLAIN),
    "clauses_per_sentence": ("Subordinate clauses per sentence", PLAIN),
    "passive_sentences_pct": ("Passive sentences", PCT),
    "noun_verb_ratio": ("Nouns per verb", PLAIN),
    "adjectives_per_1k": ("Adjectives", PER_1K),
    "adverbs_per_1k": ("Adverbs", PER_1K),
    "modals_per_1k": ("Modal verbs (can, should)", PER_1K),
    "nominalizations_per_1k": ("Nominalizations (-tion, -ment)", PER_1K),
    "opens_pronoun_pct": ("Pronoun (I, It, You)", PCT),
    "opens_determiner_pct": ("Determiner (The, A, This)", PCT),
    "opens_adverb_or_preposition_pct": ("Adverb or preposition (Still, In)", PCT),
    "opens_subordinator_pct": ("Subordinator (If, When, Because)", PCT),
    "opens_conjunction_pct": ("Conjunction (And, But)", PCT),
    "opens_noun_pct": ("Noun", PCT),
    "opens_verb_pct": ("Verb", PCT),
}

DISTRIBUTION_LABELS: dict[str, str] = {
    "masked_bigram": "Function-word transitions",
    "pos_trigram": "Grammar patterns (POS trigrams)",
    "char_trigram": "Character trigrams",
}

# The default view: a short, stable set that covers each level without the long tail.
KEY_METRICS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Sentences",
        (
            ("sentence_shape", "sentence_words_mean"),
            ("sentence_shape", "short_sentences_pct"),
            ("sentence_shape", "long_sentences_pct"),
            ("rhythm", "sentence_length_cv"),
            ("sentence_shape", "paragraph_words_mean"),
            ("sentence_shape", "one_sentence_paragraphs_pct"),
        ),
    ),
    (
        "Vocabulary",
        (("vocabulary", "long_words_pct"), ("vocabulary", "mattr_100")),
    ),
    (
        "Punctuation",
        (
            ("punctuation", "commas_per_1k"),
            ("punctuation", "em_dashes_per_1k"),
            ("punctuation", "parentheses_per_1k"),
            ("punctuation", "colons_per_1k"),
        ),
    ),
    (
        "Voice",
        (
            ("voice", "contractions_per_1k"),
            ("voice", "first_singular_per_1k"),
            ("voice", "second_person_per_1k"),
            ("voice", "hedges_per_1k"),
            ("voice", "transition_openers_pct"),
            ("voice", "llm_markers_per_1k"),
        ),
    ),
    (
        "Syntax",
        (
            ("syntax", "clauses_per_sentence"),
            ("syntax", "passive_sentences_pct"),
            ("syntax", "nominalizations_per_1k"),
            ("sentence_openers", "opens_pronoun_pct"),
        ),
    ),
)

LABEL_WIDTH = 42
VALUE_WIDTH = 12
DIFFERENCES_SHOWN = 8
NOTABLE_Z = 1.0
CHUNKS_SHOWN = 3
BAR_WIDTH = 20
BAR_SCALE = 3.0


# How far from the reference, as one orange ramp (pale -> deep). "Close" stays uncolored so
# color only appears where there is something to look at. Validated as an ordinal ramp on
# light and dark surfaces; no green or red, since distance is not right or wrong. Each
# color is paired with a word or arrows, so the report reads the same without color.
DISTANCE_RGB: tuple[tuple[int, int, int], ...] = ((217, 149, 106), (220, 111, 52), (184, 70, 26))
DISTANCE_256: tuple[int, ...] = (173, 166, 130)
DISTANCE_WORDS: tuple[str, ...] = (
    "close",
    "somewhat different",
    "clearly different",
    "very different",
)


def delta_level(delta: float, ceiling: float | None = None) -> int:
    """0-3 for Delta, relative to the reference's own held-out 95th percentile when known.

    Up to that ceiling reads as close; 1.5x and 2x mark the next steps. Without a
    calibrated reference, fixed steps suit a writer whose own text scores about 0.8.
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


def _title(group: str) -> str:
    return GROUP_TITLES.get(group, group.replace("_", " ").capitalize())


def label(name: str) -> tuple[str, str]:
    if name.startswith("fw_") and name.endswith("_per_1k"):
        return f'"{name[3:-7]}"', PER_1K
    return LABELS.get(name, (name.replace("_", " ").capitalize(), PLAIN))


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


def describe_delta(delta: float, ceiling: float | None = None) -> str:
    """A reading of Delta; see ``delta_level``."""
    return DISTANCE_WORDS[delta_level(delta, ceiling)]


def _row(text: str, *cells: str) -> str:
    return (
        f"  {text:{LABEL_WIDTH}}" + "".join(f"{cell:>{VALUE_WIDTH}}" for cell in cells)
    ).rstrip()


def _key_rows(report: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    sections = []
    for title, metrics in KEY_METRICS:
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
            (_title(group), [(group, name) for name in metrics])
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
    held = report.get("calibration", {}).get("delta")
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
        lines += ["", style.bold(_title(group))]
        for name, stats in metrics.items():
            text, unit = label(name)
            ref = reference["summary"].get(group, {}).get(name, {})
            z = averaged.get((group, name))
            lines.append(
                _row(text, value(stats["mean"], unit), value(ref.get("mean"), unit))
                + f"   {style.distance(_arrow(z), z_level(z)) if z is not None else ''}"
            )
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
        note = (
            ""
            if ref.get("sd")
            else style.dim(
                f"  reference never varies; {differing[(group, name)]} of "
                f"{report['chunk_count']} chunks differ"
            )
        )
        lines.append(
            _row(text, value(here, unit), value(ref["mean"], unit))
            + f"   {style.distance(_arrow(z), z_level(z))}{note}"
        )
    return lines


def mean_ceiling(stats: dict[str, Any], count: int) -> float | None:
    """The usual upper bound for an average over ``count`` chunks.

    A single chunk is unusual above the held-out 95th percentile; an average over n chunks
    varies about 1/sqrt(n) as much, so its bound sits that much closer to the median.
    """
    if not stats.get("p95"):
        return None
    median = stats.get("median", stats["p95"])
    return median + (stats["p95"] - median) / math.sqrt(max(count, 1))


def likeness_level(score: float, calibration: dict[str, Any], count: int = 1) -> int:
    """0-3 from the reference's own held-out range up to the contrast set's typical score."""
    ceiling = mean_ceiling(calibration["reference"], count) or calibration["reference"]["p95"]
    target = calibration["contrast"]["median"]
    if score <= ceiling:
        return 0
    if target <= ceiling:
        # The contrast drafts score no higher than the reference: the score cannot tell
        # them apart, so it never claims more than a few traits.
        return 1
    return 1 if score < (ceiling + target) / 2 else 2 if score < target else 3


def likeness_words(level: int, label: str) -> str:
    return (
        "like the reference",
        f"a few {label} traits",
        f"leans {label}",
        f"like the {label} drafts",
    )[level]


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
    held = reference.get("calibration", {}).get("delta")
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
                lines.append(f"  {chunk['reference']['likeness']:4.2f}  {word}  {chunk['id']}")
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
            lines.append(f"  {chunk['reference']['delta'] or 0.0:4.2f}  {word}  {chunk['id']}")
    return lines


def _comparison_view(
    report: dict[str, Any], reference: dict[str, Any], style: _Style, full: bool
) -> list[str]:
    scored = report["reference"]
    delta = scored["delta_mean"]
    if delta is None:
        return ["", style.warn("No metrics could be compared with the reference.")]
    held = reference.get("calibration", {}).get("delta", {})
    count = report["chunk_count"]
    ceiling = mean_ceiling(held, count)
    chunk_ceiling = held.get("p95")
    area_ceilings = {
        group: mean_ceiling(stats, count) for group, stats in held.get("by_group", {}).items()
    }
    level = delta_level(delta, ceiling)
    lines = [
        "",
        style.bold("Overall: ")
        + style.distance(style.bold(describe_delta(delta, ceiling)), level)
        + f"   Delta {delta:.2f}",
        style.dim(
            f"  Lower is closer. {_delta_baseline(reference)} Darker orange is further away."
        ),
    ]
    if reference.get("contrast"):
        lines += ["", *_likeness(report, reference["contrast"], style)]
    lines += ["", style.bold("By area")]
    for group, amount in sorted(
        scored["delta_by_group_mean"].items(), key=lambda item: -(item[1] or 0)
    ):
        if amount is not None:
            area_level = delta_level(amount, area_ceilings.get(group))
            lines.append(
                f"  {_title(group):24}{amount:5.2f}  "
                + _bar(amount, style, area_level)
                + "  "
                + style.distance(DISTANCE_WORDS[area_level], area_level)
            )
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
        lines += _comparison_rows(report, reference, style)
    return lines


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
    chunk_word = "chunk" if report["chunk_count"] == 1 else "chunks"
    size = f"{report['chunk_count']} {chunk_word}, {report['word_count']:,} words"
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
