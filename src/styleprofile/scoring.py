"""Compare chunks and documents with a calibrated reference."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from styleprofile import drift
from styleprofile.calibration import MIN_JUDGED_WORDS, AtLength, Lengths, verdict
from styleprofile.core import LIKENESSES, LikenessVerdict, NoteCode
from styleprofile.corpus.ids import _document_names, base_id, chunk_document, document_label
from styleprofile.corpus.reading import root_name
from styleprofile.measure import Measurer, measure_spans, parsed_document, prose_text, span_markdown
from styleprofile.reports import MINOR_VERSION, SCORE, VERSION, check_version
from styleprofile.schema import (
    Baseline,
    BaselineCalibration,
    BaselineLength,
    Calibration,
    ChunkScore,
    DocumentEntry,
    DocumentPassages,
    LikenessSignal,
    ParagraphScore,
    ReferenceReport,
    ScoredChunk,
    ScoreReport,
    SpanScore,
    Summary,
    Unseen,
)
from styleprofile.stats import _base_report, _collapse, _distribution, _measure
from styleprofile.surface import Metrics, jensen_shannon
from styleprofile.syntax import Parser
from styleprofile.weighting import (
    LIKENESS_MIN_CEILING,
    MIN_CEILING,
    UNSCORED_GROUPS,
    Key,
    ZScores,
    delta,
    delta_weights,
    flatten,
    floors,
    held_out_effects,
    likeness,
    nest,
    z_score,
)

if TYPE_CHECKING:
    from styleprofile.corpus.types import Chunk
    from styleprofile.reference import ContrastFit

DEVIATIONS_SHOWN = 8
# Per document, a score report keeps this many of the largest differences and signals.
DOCUMENT_TRAITS = 3


def _z_against(metrics: Metrics, summary: Summary, floor: dict[Key, float]) -> ZScores:
    scored: ZScores = {}
    for group, values in metrics.items():
        if group in UNSCORED_GROUPS:
            continue
        for name, value in values.items():
            stats = summary.get(group, {}).get(name)
            mean = stats.get("mean") if stats else None
            if value is None or not stats or mean is None:
                continue
            z = z_score(
                value,
                mean,
                stats.get("sd"),
                stats.get("n", 0),
                floor.get((group, name), 0.0),
            )
            if z is not None:
                scored[(group, name)] = z
    return scored


@dataclass(frozen=True)
class _Prepared:
    """Everything scoring needs from a reference, computed once rather than per chunk."""

    floor: dict[Key, float]
    effects: dict[Key, float]
    lengths: Lengths


def _prepare(reference: ReferenceReport) -> _Prepared:
    return _Prepared(
        floor=floors(reference["summary"]),
        effects=flatten((reference.get("contrast") or {}).get("effects", {})),
        lengths=Lengths(reference),
    )


def _score(
    metrics: Metrics,
    distributions: dict[str, Counter[str]],
    reference: ReferenceReport,
    prepared: _Prepared,
    at: AtLength,
    cap: float | None = None,
) -> ChunkScore:
    """One chunk's scores, with Delta weights and likeness scaled by the reference's
    held-out rms at the chunk's own length (``at``). With ``cap`` (a span of ``drift``),
    Delta and likeness count each z up to that size."""
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    counted = _capped(z_scores, cap) if cap is not None else z_scores
    summary = reference["summary"]
    unseen: list[Unseen] = [
        {"metric": f"{group}.{name}", "value": metrics[group][name], "reference_value": mean}
        for (group, name), z in z_scores.items()
        if z and not summary[group][name].get("sd")
        for mean in [summary[group][name]["mean"]]
    ]
    overall, by_group = delta(counted, delta_weights(at.rms))
    deviations = sorted(z_scores.items(), key=lambda item: -abs(item[1]))[:DEVIATIONS_SHOWN]
    liked: _Liked = {}
    if prepared.effects:
        score, signals = likeness(counted, prepared.effects, at.rms)
        liked = {"likeness": score, "likeness_signals": signals}
    return {
        "delta": overall,
        "delta_by_group": by_group,
        "divergence": {
            name: jensen_shannon(
                _collapse(_distribution(counts), reference["distributions"][name]),
                reference["distributions"][name],
            )
            for name, counts in distributions.items()
            if name in reference.get("distributions", {})
        },
        "largest_deviations": [
            {
                "metric": f"{group}.{name}",
                "z": z,
                "value": metrics[group][name],
                "reference_mean": summary[group][name]["mean"],
            }
            for (group, name), z in deviations
        ],
        "unseen_in_reference": unseen,
        "z": nest(z_scores),
        **liked,
        "calibration": at.row(),
    }


class _Liked(TypedDict, total=False):
    """A chunk's likeness, against a reference with a contrast set."""

    likeness: float
    likeness_signals: list[LikenessSignal]


# What a score report keeps of each span (``drift.judge_span``).
SPAN_KEYS = ("by", "relative", "delta", "ceiling", "likeness", "likeness_ceiling", "level")


@dataclass(frozen=True)
class _Parts:
    """A document read in parts (``drift``): its paragraphs and spans, the metrics of each
    span, and of each paragraph on its own (for its traits)."""

    paragraphs: list[drift.Paragraph]
    layout: drift.Plan
    spans: list[Metrics]
    own: list[Metrics]


def _read_in_parts(
    text: str,
    windows: Sequence[tuple[str, Any]],
    parser: Parser | None,
    *,
    own: bool = True,
) -> _Parts:
    """Measure a document's spans, and with ``own`` its paragraphs, each once: from its
    Markdown, and with a parser from its part of the document's parse (``windows``: its
    windows' prose and parses, in order), so nothing is parsed twice."""
    found = drift.paragraphs(text)
    layout = drift.plan([paragraph.words for paragraph in found])
    if not layout.spans:
        return _Parts(found, layout, [], [])
    whole = parsed_document(prose_text(text), windows, parser) if parser else None
    spans = measure_spans(span_markdown(found, layout), whole)
    own_metrics = measure_spans([item.own for item in found], whole) if own else []
    return _Parts(found, layout, spans, own_metrics)


def _paragraph_alone(
    metrics: Metrics,
    reference: ReferenceReport,
    prepared: _Prepared,
    by: str,
) -> tuple[list[dict[str, Any]], float | None]:
    """A paragraph read on its own: its traits (its metrics' z at its own length, its
    strongest contrast signals when its spans are judged by likeness), and its score over
    the bound at its length (``drift.judge_span``'s ``relative``, parser metrics capped as a
    span's are; below 75 words the bound is widened, and it is not a verdict)."""
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
    signals = None
    if by == drift.LIKENESS and prepared.effects:
        signals = likeness(z_scores, prepared.effects, at.rms)[1]
    found = drift.traits(nest(z_scores), nest(at.scale), metrics, reference["summary"], signals)
    capped = _capped(z_scores, drift.SPAN_Z)
    relative = None
    if by == drift.LIKENESS and prepared.effects and at.likeness is not None:
        value = likeness(capped, prepared.effects, at.rms)[0]
        relative = value / max(at.likeness["p95"], LIKENESS_MIN_CEILING)
    elif by == drift.DELTA and at.delta is not None:
        value, _ = delta(capped, delta_weights(at.rms))
        relative = value / max(at.delta["p95"], MIN_CEILING) if value is not None else None
    return found, relative


# Why a pooled score is not read in parts (``score``).
POOLED_REASON = "paragraph checks don't apply to pooled records"


def _abstained(name: str, source: str, reason: str, *, pooled: bool = False) -> DocumentPassages:
    """A document not read in parts, and why."""
    return {
        "name": name,
        "source": source,
        "converted": False,
        "pooled": pooled,
        "words": 0,
        "span_words": drift.SPAN_WORDS,
        "judged": False,
        "reason": reason,
        "by": None,
        "sensitive": False,
        "own_level": None,
        "thresholds": None,
        "spans": [],
        "paragraphs": [],
    }


def _passages(
    document: Chunk,
    name: str,
    windows: Sequence[tuple[str, Any]],
    reference: ReferenceReport,
    prepared: _Prepared,
    parser: Parser | None,
) -> DocumentPassages:
    """Where one document drifts (``drift``): its paragraphs, the spans of at least
    ``SPAN_WORDS`` words read across them, and each paragraph's figures and flag, against
    the thresholds the reference's null sets for a document of its length."""
    parts = _read_in_parts(document.text, windows, parser)
    entry: DocumentPassages = {
        "name": name,
        "source": document.source,
        "converted": document.converted,
        "pooled": False,
        "words": sum(paragraph.words for paragraph in parts.paragraphs),
        "span_words": drift.SPAN_WORDS,
        "judged": False,
        "reason": None,
        "by": None,
        "sensitive": False,
        "own_level": None,
        "thresholds": None,
        "spans": [],
        "paragraphs": [],
    }
    if not parts.layout.spans:
        entry["reason"] = f"under {drift.SPAN_WORDS} words, too short to read in parts"
        return entry
    judged: list[dict[str, Any]] = []
    reason = None
    for metrics in parts.spans:
        at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
        reason = reason or at.reason
        scored = _score(metrics, {}, reference, prepared, at, cap=drift.SPAN_Z)
        judged.append(drift.judge_span(scored))
    entry["spans"] = [
        cast(
            SpanScore,
            {
                "paragraphs": [start, end],
                "words": sum(paragraph.words for paragraph in parts.paragraphs[start:end]),
                **{key: span[key] for key in SPAN_KEYS},
            },
        )
        for (start, end), span in zip(parts.layout.spans, judged, strict=True)
    ]
    scored = [span for span in judged if span["judged"]]
    if not scored:
        entry["reason"] = reason
        return entry
    by = scored[0]["by"]
    stats = drift.statistics(parts.layout, [span["relative"] for span in judged])
    limits = drift.thresholds(
        (reference.get("calibration") or {}).get("drift"),
        sum(stat is not None for stat in stats),
        by,
    )
    alone = [_paragraph_alone(metrics, reference, prepared, by) for metrics in parts.own]
    entry["judged"], entry["by"] = True, by
    entry["sensitive"], entry["thresholds"] = limits is not None, cast(Any, limits)
    entry["own_level"] = drift.level(stats)
    entry["paragraphs"] = cast(
        list[ParagraphScore],
        drift.judge(
            parts.paragraphs,
            parts.layout,
            judged,
            limits,
            [traits for traits, _ in alone],
            [relative for _, relative in alone],
        ),
    )
    return entry


def _held_out_relative(
    z_scores: ZScores,
    at: AtLength,
    by: str,
    effects: Callable[[str], dict[Key, float]] | None,
    document: str,
) -> float | None:
    """A held-out span's score over its 95% bound at its length, as ``drift.judge_span``
    reads a draft's, each z capped as a draft's span is (``drift.SPAN_Z``); None when its
    length is not judged or has no range for the score."""
    if not at.judged or at.delta is None:
        return None
    z_scores = _capped(z_scores, drift.SPAN_Z)
    if by == drift.LIKENESS:
        if at.likeness is None or effects is None:
            return None
        value = likeness(z_scores, effects(document), at.rms)[0]
        return value / max(at.likeness["p95"], LIKENESS_MIN_CEILING)
    value, _ = delta(z_scores, delta_weights(at.rms))
    return value / max(at.delta["p95"], MIN_CEILING) if value is not None else None


def _held_out_effects(fit: ContrastFit) -> Callable[[str], dict[Key, float]]:
    """The likeness effects learned without one reference document, as ``likeness_range``
    scores its pieces."""
    return held_out_effects(
        fit.reference_held, fit.reference_documents, fit.contrast_z, fit.learned
    )


def _capped(z_scores: ZScores, cap: float) -> ZScores:
    """Each z of the groups ``drift.SPAN_Z_GROUPS`` names (every group when None) limited
    to ``cap`` either way (``drift.SPAN_Z``)."""
    groups = drift.SPAN_Z_GROUPS
    return {
        key: max(-cap, min(cap, z)) if groups is None or key[0] in groups else z
        for key, z in z_scores.items()
    }


def _mean_of(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def z_against_reference(
    report: ReferenceReport,
    chunks: Sequence[Chunk],
    parser: Parser | None,
    min_words: int,
    measurer: Measurer | None = None,
) -> tuple[list[Chunk], list[ZScores]]:
    """Measure chunks and take their z-scores against a reference report, as the contrast
    drafts are; returns the chunks kept (enough prose), possibly none, and their z-scores."""
    measured = _measure(
        chunks, parser, min_words, allow_empty=True, keep_distributions=False, measurer=measurer
    )
    floor = floors(report["summary"])
    return measured.chunks, [
        _z_against(metrics, report["summary"], floor) for metrics in measured.metrics
    ]


def _baseline(reference: ReferenceReport) -> Baseline:
    """What rendering a score needs from its reference, so a saved score can be shown without
    the reference file: each metric's mean and spread, the held-out Delta range, and the
    contrast set's name and likeness range."""
    contrast = reference.get("contrast")
    return {
        "chunk_count": reference["chunk_count"],
        "summary": {
            group: {
                name: {"mean": stats.get("mean"), "sd": stats.get("sd")}
                for name, stats in metrics.items()
            }
            for group, metrics in reference["summary"].items()
        },
        "calibration": _without_rms(reference.get("calibration")),
        "contrast": (
            {"label": contrast["label"], "calibration": contrast["calibration"]}
            if contrast
            else None
        ),
    }


def _left_out(rows: Sequence[ScoredChunk], shown: int = 5) -> str:
    """Which chunks a verdict leaves out as too short to judge, and why, for its warning.

    A chunk is named by its input (``notes.md``), or with its window (``draft.md#w3``) when
    its input was cut into several."""
    left_out = [row for row in rows if not row["reference"]["calibration"]["judged"]]
    windows = Counter(base_id(row["id"]) for row in rows)
    names = []
    for row in left_out[:shown]:
        words = int(row["metrics"]["size"]["words"] or 0)
        why = (
            f"under {MIN_JUDGED_WORDS}"
            if words < MIN_JUDGED_WORDS
            else "shorter than the reference is calibrated for"
        )
        name = row["id"] if windows[base_id(row["id"])] > 1 else base_id(row["id"])
        names.append(f"{name} ({words:,} words, {why})")
    more = f" and {len(left_out) - shown} more" if len(left_out) > shown else ""
    return NoteCode.LEFT_OUT.message("details", ", ".join(names), more)


def _without_rms(calibration: Calibration | None) -> BaselineCalibration | None:
    """The calibration without each length's per-metric rms, which only scoring reads."""
    if calibration is None:
        return None
    by_length: dict[str, BaselineLength] = {}
    for length, entry in calibration["by_length"].items():
        copied = entry.copy()
        copied.pop("reliability", None)
        by_length[length] = copied
    return {
        "sources": calibration["sources"],
        "delta": calibration["delta"],
        "chunk_words": calibration["chunk_words"],
        "by_length": by_length,
    }


def score(
    chunks: Sequence[Chunk],
    reference: ReferenceReport,
    *,
    parser: Parser | None = None,
    top_k: int = 300,
    min_words: int = 1,
    settings: Mapping[str, Any] | None = None,
    reference_path: Path | None = None,
    read_in_parts: Sequence[Chunk] | None = None,
    pooled: bool = False,
    measurer: Measurer | None = None,
) -> ScoreReport:
    """Profile sample chunks and score each against ``reference``: z-scores, Delta, pattern
    divergence and, when the reference learned a contrast, likeness to the contrast set.

    With ``read_in_parts`` (the chunks before windowing, or some of them), each is also read
    in overlapping spans to show where it drifts (``drift``): ``report["passages"]`` holds one
    entry per document, and is None without ``read_in_parts``. When the chunks are
    ``pooled`` windows, every document abstains instead.

    This is the lower-level step under ``Profile.score``: nothing is inherited from the
    reference, so pass chunks cut into the reference's windows, its ``min_words`` and a
    ``parser`` when it has syntax metrics, or use ``Profile.score``, which does all that.
    A reference of another report version is refused (``check_version``).

    Each chunk is judged at its own length (``calibration``). The means and the
    ``verdict`` cover the chunks long enough to judge; when none is, they cover every
    chunk, and the verdict is "too short to judge"."""
    check_version(reference, "the reference")
    measured = _measure(
        chunks,
        parser,
        min_words,
        measurer=measurer,
        keep_docs=bool(read_in_parts),
        docs_required=True,
    )
    base = _base_report(
        measured, kind=SCORE, parser=parser, top_k=top_k, min_words=min_words, settings=settings
    )
    warnings, totals = base.warnings, base.totals
    prepared = _prepare(reference)
    rows: list[ScoredChunk] = []
    for row, metrics, distributions in zip(
        base.rows, measured.metrics, measured.distributions, strict=True
    ):
        at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
        rows.append({**row, "reference": _score(metrics, distributions, reference, prepared, at)})

    judged = [row for row in rows if row["reference"]["calibration"]["judged"]]
    scored = [row["reference"] for row in judged or rows]
    groups = sorted({group for scores in scored for group in scores["delta_by_group"]})
    reference_settings = reference.get("settings", {})
    if judged and len(judged) < len(rows):
        warnings.append(_left_out(rows))
    if not reference.get("reliability"):
        warnings.append(NoteCode.NO_RELIABILITY.message("0"))
    if prepared.effects:
        present = {
            (group, name)
            for row in rows
            for group, names in row["reference"]["z"].items()
            for name in names
        }
        total_weight = sum(effect * effect for effect in prepared.effects.values())
        missing = sum(
            effect * effect for key, effect in prepared.effects.items() if key not in present
        )
        if total_weight and missing / total_weight > 0.1:
            warnings.append(
                NoteCode.MISSING_LIKENESS_METRICS.message(
                    "0", f"{100 * missing / total_weight:.0f}"
                )
            )
    if reference["chunk_count"] < 2:
        warnings.append(NoteCode.SINGLE_CHUNK.message("0"))
    # 0 and None both mean no windowing.
    own_settings = base.described["settings"]
    own_window = own_settings.get("window_words") or None
    reference_window = reference_settings.get("window_words") or None
    if own_window != reference_window:
        warnings.append(
            NoteCode.WINDOW_MISMATCH.message(
                "0", f"{own_window or 'off'}", f"{reference_window or 'off'}"
            )
        )
    own_syntax = own_settings["syntax_used"]
    reference_syntax = reference_settings.get("syntax_used")
    if own_syntax and not reference_syntax:
        warnings.append(NoteCode.REFERENCE_NO_SYNTAX.message("0"))
    elif reference_syntax and not own_syntax:
        warnings.append(NoteCode.SCORE_NO_SYNTAX.message("0"))
    elif own_syntax and reference_syntax:
        keys = ("model", "model_version")
        if any(own_syntax.get(key) != reference_syntax.get(key) for key in keys):
            warnings.append(
                NoteCode.SYNTAX_MODEL_MISMATCH.message(
                    "0",
                    f"{reference_syntax.get('model')}",
                    f"{reference_syntax.get('model_version')}",
                )
            )
    passages: list[DocumentPassages] | None = None
    if pooled:
        # Pooled windows join records: their paragraphs are records, which the paragraph
        # null was not calibrated for. Each document abstains, and each record's own
        # verdict is in ``documents``.
        keys = list(dict.fromkeys((row["source"], base_id(row["id"])) for row in rows))
        passages = [
            _abstained(name, source, POOLED_REASON, pooled=True)
            for (source, _), name in zip(keys, _document_names(keys), strict=True)
        ]
    elif read_in_parts is not None:
        parsed: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        if measured.docs is not None:
            for chunk, found in zip(measured.chunks, measured.docs, strict=True):
                if found is not None:
                    parsed[chunk_document(chunk)].append(found)
        # Named as ``documents`` names them, so the two can be read together.
        keys = list(dict.fromkeys((row["source"], base_id(row["id"])) for row in rows))
        names = dict(zip(keys, _document_names(keys), strict=True))
        passages = [
            _passages(
                document,
                names.get((document.source, base_id(document.id)))
                or document_label(document.source, base_id(document.id)),
                parsed[chunk_document(document)],
                reference,
                prepared,
                parser,
            )
            for document in read_in_parts
        ]
    return {
        "version": VERSION,
        "minor_version": MINOR_VERSION,
        "kind": SCORE,
        **base.described,
        "warnings": warnings,
        "chunks": rows,
        "documents": documents(rows, reference),
        "passages": passages,
        "reference": {
            # Only the profile's file name: a saved report never reveals where files live.
            "path": root_name(str(reference_path)) if reference_path else None,
            "chunk_count": reference["chunk_count"],
            "delta_mean": _mean_of([scores["delta"] for scores in scored]),
            "likeness_mean": _mean_of([scores.get("likeness") for scores in scored]),
            "delta_by_group_mean": {
                group: _mean_of([scores["delta_by_group"].get(group) for scores in scored])
                for group in groups
            },
            "divergence_mean": {
                name: _mean_of([scores["divergence"].get(name) for scores in scored])
                for name in reference.get("distributions", {})
            },
            "pooled_divergence": {
                name: jensen_shannon(
                    _collapse(_distribution(counts), reference["distributions"][name]),
                    reference["distributions"][name],
                )
                for name, counts in totals.items()
                if name in reference.get("distributions", {})
            },
            "verdict": verdict(rows, (reference.get("contrast") or {}).get("label")),
            "baseline": _baseline(reference),
        },
    }


def average_z(
    rows: Sequence[ScoredChunk], summary: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[tuple[str, str], float]:
    """Each metric's z against the reference, averaged over scored chunk rows. Each chunk's
    z is first divided by how many times more that metric swings at the chunk's length than
    in a reference window (its calibration's ``length_scale``), so a short chunk's z counts
    standard deviations of the writer's own text at its length.

    For a metric the reference never varies on (no ``sd`` in its ``summary``), every
    differing chunk scores the cap in one direction or the other; average the size instead,
    signed by which side of the reference's mean the chunks' mean value falls, so opposite
    differences cannot cancel.
    """
    zs: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        scale = row["reference"]["calibration"]["length_scale"]
        for group, values in row["reference"]["z"].items():
            for name, z in values.items():
                factor = scale.get(group, {}).get(name, 1.0)
                zs.setdefault((group, name), []).append(z / factor)
    averaged: dict[tuple[str, str], float] = {}
    for (group, name), values in zs.items():
        stats: Mapping[str, Any] = summary.get(group, {}).get(name, {})
        if stats.get("sd"):
            averaged[(group, name)] = sum(values) / len(values)
            continue
        size = sum(abs(z) for z in values) / len(values)
        here = _mean_of([(row["metrics"].get(group) or {}).get(name) for row in rows]) or 0.0
        there = stats.get("mean") or 0.0
        averaged[(group, name)] = -size if here < there else size
    return averaged


def _key(metric: str) -> tuple[str, str]:
    group, _, name = metric.partition(".")
    return group, name


def documents(rows: Sequence[ScoredChunk], reference: ReferenceReport) -> list[DocumentEntry]:
    """Scored chunk rows grouped by document (``document_of``), in input order, each judged
    by ``calibration.verdict`` on its own chunks, the function that makes the pooled
    headline, so each document is read at its own chunks' lengths: mean Delta and
    likeness, their verdicts (or "too short to judge", with ``reason``), and the metrics
    that differ most.

    Each is named by ``document_label`` and keeps its saved ``path``, its source.
    """
    grouped: dict[tuple[str, str], list[ScoredChunk]] = {}
    for row in rows:
        grouped.setdefault((row["source"], base_id(row["id"])), []).append(row)
    label = (reference.get("contrast") or {}).get("label")
    entries: list[DocumentEntry] = []
    for name, members in zip(_document_names(list(grouped)), grouped.values(), strict=True):
        judged = verdict(members, label)
        # The figures cover the chunks long enough to judge, as the verdict does.
        used = [row for row in members if row["reference"]["calibration"]["judged"]] or members
        mean_delta = judged["delta"]["value"]
        likeness = judged["likeness"]
        averaged = average_z(used, reference["summary"])
        largest = sorted(averaged.items(), key=lambda item: -abs(item[1]))[:DOCUMENT_TRAITS]
        likeness_verdict: str | None = None
        if likeness and mean_delta is not None:
            level = likeness["level"]
            likeness_verdict = str(
                LikenessVerdict.TOO_SHORT if level is None else LIKENESSES[level]
            )
        entry: DocumentEntry = {
            "name": name,
            "path": members[0]["source"],
            "chunks": len(members),
            "words": judged["words"],
            "judged": judged["judged"],
            "chunks_judged": judged["chunks_judged"],
            "reason": judged["reason"],
            "delta": mean_delta,
            "verdict": judged["verdict"],
            "likeness": likeness["value"] if likeness else None,
            "likeness_verdict": likeness_verdict,
            "flagged": judged["flagged"],
            "differences": [
                {"metric": f"{group}.{metric}", "z": z} for (group, metric), z in largest
            ],
        }
        if likeness:
            # Each metric's mean share of the likeness over the document's chunks.
            shares: dict[str, float] = {}
            for row in used:
                for signal in row["reference"].get("likeness_signals", []):
                    share = signal["contribution"] / len(used)
                    shares[signal["metric"]] = shares.get(signal["metric"], 0.0) + share
            ranked = sorted(shares.items(), key=lambda item: -item[1])[:DOCUMENT_TRAITS]
            entry["signals"] = [
                {"metric": metric, "z": averaged.get(_key(metric), 0.0), "contribution": share}
                for metric, share in ranked
            ]
        entries.append(entry)
    return entries
