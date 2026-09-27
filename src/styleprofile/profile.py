"""Profile arbitrary chunks of prose and optionally score them against a saved profile.

A profile is the per-chunk metrics plus, for every metric, its mean and spread across
chunks. The spread is what makes a reference useful: a new chunk's z-score on each metric
says how unusual it is *relative to how much that writer normally varies*, and the mean
absolute z-score over a group of metrics is Burrows' Delta for that group. How metrics are
weighted into Delta and into the optional contrast-likeness score is in ``weighting``.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from styleprofile.surface import (
    Metrics,
    block_word_count,
    char_trigrams,
    classify,
    jensen_shannon,
    masked_bigrams,
    prose,
    surface_metrics,
    words,
)
from styleprofile.syntax import Parser
from styleprofile.weighting import (
    LENGTH_AUC_WARNING,
    UNSCORED_GROUPS,
    CrossValidated,
    Key,
    ZScores,
    calibrate_delta,
    cross_validate,
    delta,
    delta_weights,
    flatten,
    floors,
    held_out_z,
    likeness,
    nest,
    reliability,
    summarize_contrast,
    z_score,
)

# 3: Delta weights areas equally and metrics by held-out reliability (no cap), and a
# reference built with --contrast scores contrast likeness.
# 4: percentage floors use their true denominators (paragraphs, apostrophes), and list
# continuations extend their item instead of counting as paragraphs.
# 5: the contrast AUC has a document-bootstrap 95% interval and a length-only baseline.
# Every report carries ``kind`` (see ``KINDS``), and score reports a ``baseline`` copy of what
# rendering needs from their reference.
VERSION = 5
REFERENCE = "reference"
SCORE = "score"
# Written by ``styleprofile evaluate`` (see the evaluate module); shown, never scored against.
EVALUATION = "evaluation"
KINDS = (REFERENCE, SCORE, EVALUATION)
UNREADABLE = "not a style profile this version of styleprofile can read; rebuild it"
TEXT_FIELDS: tuple[str, ...] = ("text", "body_markdown", "output", "content", "body")
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
# Directory walks skip dot-directories (.git, .venv) and these vendored ones.
SKIPPED_DIRS = frozenset({"node_modules", "__pycache__", "site-packages"})
OTHER = "<other>"
DEVIATIONS_SHOWN = 8
SHORT_CHUNK_WORDS = 150


class StyleProfileError(ValueError):
    """Style profile inputs or a reference profile are unusable.

    ``code`` names the kind of problem so a front end can add its own advice (such as the
    command-line flag that fixes it); the messages themselves never mention flags.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def _decode(data: bytes, name: str | Path) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise StyleProfileError(f"{name} is not UTF-8 text ({error.reason})") from error
    # Universal newlines, as text-mode reading gives.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text(path: Path) -> str:
    return _decode(path.read_bytes(), path)


def _text_fields(text_field: str | Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(text_field, str):
        return (text_field,)
    return tuple(text_field) if text_field else TEXT_FIELDS


def _jsonl_chunks(path: Path, text_field: str | Sequence[str] | None) -> list[Chunk]:
    fields = _text_fields(text_field)
    chunks: list[Chunk] = []
    for line_number, line in enumerate(_read_text(path).split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise StyleProfileError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise StyleProfileError(f"{path}:{line_number}: expected a JSON object")
        field = next((name for name in fields if isinstance(record.get(name), str)), None)
        if field is None:
            raise StyleProfileError(
                f"{path}:{line_number}: no string field among {', '.join(fields)}",
                code="text_field",
            )
        record_id = record.get("id")
        # An id of 0 is kept; a missing, null or empty id falls back to the line.
        chunk_id = f"{path.name}:{line_number}" if record_id in (None, "") else str(record_id)
        chunks.append(Chunk(chunk_id, str(path), record[field]))
    return chunks


def load_chunks(
    inputs: Sequence[str], text_field: str | Sequence[str] | None = None
) -> list[Chunk]:
    """Read JSONL records, Markdown/text files, directories of them, or ``-`` for stdin.

    ``text_field`` names the JSONL field that holds the text, or several to try in order;
    by default the first of ``TEXT_FIELDS`` that a record has."""
    chunks: list[Chunk] = []
    for value in inputs:
        if value == "-":
            chunks.append(Chunk("stdin", "stdin", _decode(sys.stdin.buffer.read(), "stdin")))
            continue
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            files = sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and item.suffix.lower() in {*TEXT_SUFFIXES, ".jsonl"}
                and not any(
                    part.startswith(".") or part in SKIPPED_DIRS
                    for part in item.relative_to(path).parts[:-1]
                )
            )
            if not files:
                raise StyleProfileError(f"{path} contains no .md, .markdown, .txt, or .jsonl files")
            for item in files:
                if item.suffix.lower() == ".jsonl":
                    chunks.extend(_jsonl_chunks(item, text_field))
                else:
                    chunks.append(Chunk(str(item.relative_to(path)), str(item), _read_text(item)))
        elif path.suffix.lower() == ".jsonl":
            chunks.extend(_jsonl_chunks(path, text_field))
        else:
            chunks.append(Chunk(path.name, str(path), _read_text(path)))
    return chunks


def window(chunks: Sequence[Chunk], window_words: int) -> list[Chunk]:
    """Split chunks into roughly ``window_words``-word pieces at Markdown block boundaries.

    Sizes count prose words only (code, URLs and markup excluded). Fenced code stays whole
    and a list's indented continuation stays with its list. A window closes early rather
    than grow past one and a half windows. A remainder under half a window joins the
    previous piece, so no window is shorter than half a window unless its whole chunk is;
    that merge, or a single long block, can make a window longer than one and a half.
    """
    limit = window_words * 1.5
    windows: list[Chunk] = []
    for chunk in chunks:
        blocks = classify(chunk.text)
        counts = [block_word_count(block) for block in blocks]
        pieces: list[tuple[list[str], int]] = []
        current: list[str] = []
        count = 0
        for index, (block, block_words) in enumerate(zip(blocks, counts, strict=True)):
            closes_early = count >= window_words / 2 and count + block_words > limit
            if current and closes_early and not block.continues_list:
                pieces.append((current, count))
                current, count = [], 0
            current.append(block.raw)
            count += block_words
            following = blocks[index + 1] if index + 1 < len(blocks) else None
            if count >= window_words and not (following and following.continues_list):
                pieces.append((current, count))
                current, count = [], 0
        if current and pieces and count < window_words / 2:
            pieces[-1] = (pieces[-1][0] + current, pieces[-1][1] + count)
        elif current or not pieces:
            # A chunk with no prose stays as one window, so it is counted as skipped.
            pieces.append((current, count))
        windows.extend(
            Chunk(f"{chunk.id}#w{index}", chunk.source, "\n\n".join(raw_blocks))
            for index, (raw_blocks, _) in enumerate(pieces, start=1)
        )
    return windows


def _merge(target: Metrics, extra: Metrics) -> Metrics:
    return {**target, **extra}


def _distribution(counts: Counter[str], top_k: int | None = None) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    if top_k is None:
        return {key: value / total for key, value in counts.items()}
    top = counts.most_common(top_k)
    shares = {key: value / total for key, value in top}
    other = 1 - sum(shares.values())
    if other > 1e-12:
        shares[OTHER] = other
    return shares


def _collapse(sample: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    """Fold sample keys the reference only tracks as ``<other>`` into that bucket."""
    collapsed: dict[str, float] = {}
    for key, value in sample.items():
        target = key if key in reference and key != OTHER else OTHER
        collapsed[target] = collapsed.get(target, 0.0) + value
    return collapsed


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "cv": None, "min": None, "max": None}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "cv": sd / abs(mean) if sd is not None and mean else None,
        "min": min(values),
        "max": max(values),
    }


def summarize(chunk_metrics: Sequence[Metrics]) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for metrics in chunk_metrics:
        for group, values in metrics.items():
            for name in values:
                summary.setdefault(group, {}).setdefault(name, {})
    for group, names in summary.items():
        for name in names:
            values = [
                value
                for metrics in chunk_metrics
                if (value := metrics.get(group, {}).get(name)) is not None
            ]
            names[name] = _stats(values)
    return summary


_WINDOW_SUFFIX = re.compile(r"#w\d+$")


def document_of(source: str, chunk_id: str) -> str:
    """The document a chunk came from: its file plus its id, with a window suffix removed.

    Windows ``post#w3`` share their document; files with the same name in different
    folders, and JSONL records with the same id in different files, stay separate.
    """
    return f"{source}\x1f{base_id(chunk_id)}"


def base_id(chunk_id: str) -> str:
    """A chunk's id without the ``#wN`` suffix that windowing adds."""
    return _WINDOW_SUFFIX.sub("", chunk_id)


def _z_against(metrics: Metrics, summary: dict[str, Any], floor: dict[Key, float]) -> ZScores:
    scored: ZScores = {}
    for group, values in metrics.items():
        if group in UNSCORED_GROUPS:
            continue
        for name, value in values.items():
            stats = summary.get(group, {}).get(name)
            if value is None or not stats or stats.get("mean") is None:
                continue
            z = z_score(
                value,
                stats["mean"],
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
    rms: dict[Key, float]
    weights: dict[Key, float]
    effects: dict[Key, float]


def _prepare(reference: dict[str, Any]) -> _Prepared:
    rms = flatten(reference.get("reliability", {}))
    return _Prepared(
        floor=floors(reference["summary"]),
        rms=rms,
        weights=delta_weights(rms),
        effects=flatten((reference.get("contrast") or {}).get("effects", {})),
    )


def _score(
    metrics: Metrics,
    distributions: dict[str, Counter[str]],
    reference: dict[str, Any],
    prepared: _Prepared,
) -> dict[str, Any]:
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    summary = reference["summary"]
    unseen = [
        {"metric": f"{group}.{name}", "value": metrics[group][name], "reference_value": mean}
        for (group, name), z in z_scores.items()
        if z and not summary[group][name].get("sd")
        for mean in [summary[group][name]["mean"]]
    ]
    overall, by_group = delta(z_scores, prepared.weights)
    deviations = sorted(z_scores.items(), key=lambda item: -abs(item[1]))[:DEVIATIONS_SHOWN]
    scored: dict[str, Any] = {
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
    }
    if prepared.effects:
        score, signals = likeness(z_scores, prepared.effects, prepared.rms)
        scored["likeness"] = score
        scored["likeness_signals"] = signals
    return scored


def _mean_of(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def report_kind(report: dict[str, Any]) -> str:
    """``"reference"``, ``"score"`` or ``"evaluation"``, from the report's ``kind``."""
    kind = report.get("kind")
    if kind not in KINDS:
        raise StyleProfileError(
            f"the report has no known kind, so it is {UNREADABLE}", code="outdated"
        )
    return kind


def load_report(path: Path) -> dict[str, Any]:
    """Read any styleprofile report: reference, score or evaluation."""
    if path.is_dir():
        raise StyleProfileError(f"{path} is a directory, not a profile", code="directory")
    if not path.exists():
        raise StyleProfileError(f"{path} not found", code="not_found")
    try:
        report = json.loads(_read_text(path))
    except (json.JSONDecodeError, StyleProfileError):
        report = None
    if not isinstance(report, dict) or ("summary" not in report and "kind" not in report):
        raise StyleProfileError(f"{path} is not a style profile", code="not_a_profile")
    kind = report.get("kind")
    if kind not in KINDS or (kind != EVALUATION and "summary" not in report):
        # A pre-release report without ``kind``, or a kind this version does not know.
        raise StyleProfileError(f"{path} is {UNREADABLE}", code="outdated")
    return report


def load_reference(path: Path) -> dict[str, Any]:
    """Read a reference profile, refusing a score report (a sample scored against one)."""
    reference = load_report(path)
    if report_kind(reference) == EVALUATION:
        raise StyleProfileError(
            f"{path} is an evaluation report, not a reference profile; build a reference "
            "from the writer's own texts",
            code="score_as_reference",
        )
    if report_kind(reference) == SCORE:
        raise StyleProfileError(
            f"{path} is a score report (a sample scored against a reference), not a "
            "reference profile; build a reference from the writer's own texts",
            code="score_as_reference",
        )
    return reference


@dataclass(frozen=True)
class _Measured:
    chunks: list[Chunk]
    metrics: list[Metrics]
    distributions: list[dict[str, Counter[str]]]
    empty: int
    below: int


def _measure(
    chunks: Sequence[Chunk], parser: Parser | None, min_words: int, *, allow_empty: bool = False
) -> _Measured:
    """Parse each chunk once, drop chunks without enough prose, and compute every metric.

    With no chunk left this is an error, unless ``allow_empty``."""
    parsed_all = [prose(chunk.text) for chunk in chunks]
    sizes = [len(words(parsed.text)) for parsed in parsed_all]
    empty = sum(not size for size in sizes)
    below = sum(0 < size < min_words for size in sizes)
    kept = [
        (chunk, parsed)
        for chunk, parsed, size in zip(chunks, parsed_all, sizes, strict=True)
        if size and size >= min_words
    ]
    texts = [parsed.text for _, parsed in kept]
    if not kept and allow_empty:
        return _Measured([], [], [], empty, below)
    if not kept:
        raise StyleProfileError(
            f"no chunks with at least {max(min_words, 1)} prose word(s) to profile "
            f"({empty} had no prose, {below} were shorter)"
        )
    chunk_metrics = [surface_metrics(chunk.text, parsed) for chunk, parsed in kept]
    chunk_distributions: list[dict[str, Counter[str]]] = [
        {"masked_bigram": masked_bigrams(text), "char_trigram": char_trigrams(text)}
        for text in texts
    ]
    if parser is not None:
        for index, (metrics, trigrams) in enumerate(parser.parse(texts)):
            chunk_metrics[index] = _merge(chunk_metrics[index], metrics)
            chunk_distributions[index]["pos_trigram"] = trigrams
    return _Measured([chunk for chunk, _ in kept], chunk_metrics, chunk_distributions, empty, below)


def _calibrate(report: dict[str, Any], chunk_metrics: Sequence[Metrics]) -> list[ZScores] | None:
    """Held-out reliability and Delta range, when the chunks span at least two documents.

    Without them the profile still works as a reference, but Delta falls back to capped z
    and --contrast is unavailable; scoring against such a reference warns about it.
    """
    documents = [document_of(row["source"], row["id"]) for row in report["chunks"]]
    if len(set(documents)) < 2:
        return None
    held = held_out_z(chunk_metrics, documents, floors(report["summary"]))
    calibration = calibrate_delta(held, documents)
    if calibration is None:
        return None
    report["reliability"] = nest(reliability(held, documents))
    report["calibration"] = {"sources": len(set(documents)), "delta": calibration}
    return held


@dataclass(frozen=True)
class ContrastFit:
    """What a contrast reference's likeness weights were learned from.

    ``learned`` holds every held-out fold, so more text derived from a contrast document
    (an edited copy, say) can be scored with the fold that left that document out, and
    judged against the same reference scores and calibration the report stores.
    """

    reference_held: list[ZScores]
    reference_documents: list[str]
    contrast_chunks: list[Chunk]
    contrast_z: list[ZScores]
    contrast_documents: list[str]
    learned: CrossValidated


def _learn_contrast(
    report: dict[str, Any],
    held: list[ZScores] | None,
    contrast: Sequence[Chunk],
    label: str,
    parser: Parser | None,
    min_words: int,
) -> tuple[dict[str, Any], ContrastFit]:
    if held is None:
        raise StyleProfileError(
            "a contrast set needs a reference drawn from at least two documents with enough "
            "chunks to measure the reference's own variation on held-out writing",
            code="contrast_needs_documents",
        )
    measured = _measure(contrast, parser, min_words)
    if measured.empty or measured.below:
        report["warnings"].append(
            f"skipped {measured.empty + measured.below} contrast chunk(s) with no prose or "
            f"fewer than {min_words} prose words"
        )
    floor = floors(report["summary"])
    contrast_z = [_z_against(metrics, report["summary"], floor) for metrics in measured.metrics]
    contrast_sources = [document_of(chunk.source, chunk.id) for chunk in measured.chunks]
    reference_sources = [document_of(row["source"], row["id"]) for row in report["chunks"]]
    fit = ContrastFit(
        reference_held=held,
        reference_documents=reference_sources,
        contrast_chunks=measured.chunks,
        contrast_z=contrast_z,
        contrast_documents=contrast_sources,
        learned=cross_validate(held, reference_sources, contrast_z, contrast_sources),
    )
    learned = summarize_contrast(
        fit.learned,
        reference_sources,
        contrast_sources,
        [row["metrics"]["size"]["words"] for row in report["chunks"]],
        # Measured chunks all have prose, so words is never None.
        [metrics["size"]["words"] or 0.0 for metrics in measured.metrics],
    )
    calibration = learned["calibration"]
    if not calibration["cross_validated"]:
        report["warnings"].append(
            "the contrast set is one document, so its likeness range is measured in-sample "
            "and is optimistic; add more contrast documents"
        )
    if calibration["auc_ci"] is None:
        report["warnings"].append(
            "the reference or contrast set has fewer than 2 documents, so the contrast AUC "
            "has no confidence interval"
        )
    length = calibration["length_baseline"]
    if length and length["auc"] >= LENGTH_AUC_WARNING:
        report["warnings"].append(
            f"the contrast set differs strongly in length (length alone separates it with "
            f"AUC {length['auc']:.2f}), so {label}-likeness may partly reflect length; match "
            "lengths or split both sets into windows of the same size"
        )
    return {
        "label": label,
        "chunk_count": len(measured.chunks),
        "sources": len(set(contrast_sources)),
        **learned,
    }, fit


def _base_report(
    measured: _Measured,
    *,
    kind: str,
    parser: Parser | None,
    top_k: int,
    min_words: int,
    settings: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    """The part of every report that describes its own chunks, plus the pooled counts."""
    chunk_metrics = measured.metrics
    totals: dict[str, Counter[str]] = {}
    for distributions in measured.distributions:
        for name, counts in distributions.items():
            totals.setdefault(name, Counter()).update(counts)

    warnings: list[str] = []
    if measured.empty:
        warnings.append(
            f"skipped {measured.empty} chunk(s) with no prose (only code, tables or markup)"
        )
    if measured.below:
        warnings.append(
            f"skipped {measured.below} chunk(s) with fewer than {min_words} prose words"
        )
    short = sum((metrics["size"]["words"] or 0) < SHORT_CHUNK_WORDS for metrics in chunk_metrics)
    if short:
        warnings.append(
            f"{short} chunk(s) have fewer than {SHORT_CHUNK_WORDS} words; their rates are noisy"
        )
    report: dict[str, Any] = {
        "version": VERSION,
        "kind": kind,
        "settings": {
            **(settings or {}),
            "top_k": top_k,
            "syntax": (
                {
                    "model": parser.model,
                    "model_version": parser.model_version,
                    "spacy_version": parser.spacy_version,
                }
                if parser
                else None
            ),
        },
        "chunk_count": len(measured.chunks),
        "word_count": sum(int(metrics["size"]["words"] or 0) for metrics in chunk_metrics),
        "summary": summarize(chunk_metrics),
        "distributions": {name: _distribution(counts, top_k) for name, counts in totals.items()},
        "chunks": [
            {"id": chunk.id, "source": chunk.source, "metrics": metrics}
            for chunk, metrics in zip(measured.chunks, chunk_metrics, strict=True)
        ],
        "warnings": warnings,
    }
    return report, totals


def build_reference(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None,
    top_k: int = 300,
    min_words: int = 1,
    settings: dict[str, Any] | None = None,
    contrast: Sequence[Chunk] | None = None,
    contrast_label: str = "LLM",
) -> dict[str, Any]:
    """Profile a writer's chunks as a reference: each metric's mean and spread, its held-out
    reliability when the chunks span two or more documents and, given ``contrast`` chunks
    (for example LLM drafts), the weights that score likeness to them."""
    return _build_reference(
        chunks,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
        contrast=contrast,
        contrast_label=contrast_label,
    )[0]


def build_contrast_reference(
    chunks: Sequence[Chunk],
    contrast: Sequence[Chunk],
    *,
    parser: Parser | None,
    top_k: int = 300,
    min_words: int = 1,
    settings: dict[str, Any] | None = None,
    contrast_label: str = "LLM",
) -> tuple[dict[str, Any], ContrastFit]:
    """``build_reference(chunks, contrast=contrast)``, plus what its weights were learned
    from, for scoring more text with the same folds."""
    report, fit = _build_reference(
        chunks,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
        contrast=contrast,
        contrast_label=contrast_label,
    )
    assert fit is not None  # a contrast always produces a fit or raises
    return report, fit


def z_against_reference(
    report: dict[str, Any], chunks: Sequence[Chunk], parser: Parser | None, min_words: int
) -> tuple[list[Chunk], list[ZScores]]:
    """Measure chunks and take their z-scores against a reference report, as the contrast
    drafts are; returns the chunks kept (enough prose), possibly none, and their z-scores."""
    measured = _measure(chunks, parser, min_words, allow_empty=True)
    floor = floors(report["summary"])
    return measured.chunks, [
        _z_against(metrics, report["summary"], floor) for metrics in measured.metrics
    ]


def _build_reference(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None,
    top_k: int,
    min_words: int,
    settings: dict[str, Any] | None,
    contrast: Sequence[Chunk] | None,
    contrast_label: str,
) -> tuple[dict[str, Any], ContrastFit | None]:
    measured = _measure(chunks, parser, min_words)
    report, _ = _base_report(
        measured,
        kind=REFERENCE,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
    )
    held = _calibrate(report, measured.metrics)
    fit: ContrastFit | None = None
    if contrast is not None:
        report["contrast"], fit = _learn_contrast(
            report, held, contrast, contrast_label, parser, min_words
        )
    return report, fit


def _baseline(reference: dict[str, Any]) -> dict[str, Any]:
    """What rendering a score needs from its reference, so a saved score can be shown without
    the reference file: each metric's mean and spread, the held-out Delta range, and the
    contrast set's name and likeness range."""
    contrast = reference.get("contrast")
    return {
        "chunk_count": reference.get("chunk_count"),
        "summary": {
            group: {
                name: {"mean": stats.get("mean"), "sd": stats.get("sd")}
                for name, stats in metrics.items()
            }
            for group, metrics in reference["summary"].items()
        },
        "calibration": reference.get("calibration"),
        "contrast": (
            {"label": contrast["label"], "calibration": contrast["calibration"]}
            if contrast
            else None
        ),
    }


def score(
    chunks: Sequence[Chunk],
    reference: dict[str, Any],
    *,
    parser: Parser | None,
    top_k: int = 300,
    min_words: int = 1,
    settings: dict[str, Any] | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    """Profile sample chunks and score each against ``reference``: z-scores, Delta, pattern
    divergence and, when the reference learned a contrast, likeness to the contrast set."""
    measured = _measure(chunks, parser, min_words)
    report, totals = _base_report(
        measured,
        kind=SCORE,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
    )
    warnings: list[str] = report["warnings"]
    rows: list[dict[str, Any]] = report["chunks"]
    prepared = _prepare(reference)
    for row, metrics, distributions in zip(
        rows, measured.metrics, measured.distributions, strict=True
    ):
        row["reference"] = _score(metrics, distributions, reference, prepared)

    scored = [row["reference"] for row in rows]
    groups = sorted({group for scores in scored for group in scores["delta_by_group"]})
    reference_settings = reference.get("settings", {})
    if reference.get("version") != VERSION:
        warnings.append(
            f"the reference was built by report version {reference.get('version')} "
            f"(this is {VERSION}); rebuild it so Delta is comparable"
        )
    if not reference.get("reliability"):
        warnings.append(
            "the reference has no held-out reliability (it needs chunks from at least two "
            "documents and a current report version), so Delta caps each metric at 3 "
            "instead of weighting it by reliability"
        )
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
                f"this run lacks metrics that carry {100 * missing / total_weight:.0f}% of "
                "the likeness weight (for example syntax); likeness uses the rest"
            )
    if reference.get("chunk_count", 0) < 2:
        warnings.append("the reference has one chunk, so it has no spread; split it into windows")
    if reference_settings.get("window_words") != report["settings"].get("window_words"):
        warnings.append(
            "window sizes differ from the reference "
            f"({report['settings'].get('window_words') or 'off'} vs "
            f"{reference_settings.get('window_words') or 'off'}); "
            "z-scores assume equal-sized chunks"
        )
    own_syntax = report["settings"]["syntax"]
    reference_syntax = reference_settings.get("syntax")
    if own_syntax and not reference_syntax:
        warnings.append("the reference has no syntax metrics, so syntax is not scored")
    elif reference_syntax and not own_syntax:
        warnings.append(
            "the reference has syntax metrics but this run does not, so Delta leaves out "
            "syntax and sentence openers; its value is not comparable with syntax runs"
        )
    elif own_syntax and reference_syntax:
        keys = ("model", "model_version")
        if any(own_syntax.get(key) != reference_syntax.get(key) for key in keys):
            warnings.append(
                "the reference was parsed with a different spaCy model "
                f"({reference_syntax.get('model')} {reference_syntax.get('model_version')}); "
                "syntax metrics may not be comparable"
            )
    report["reference"] = {
        "path": str(reference_path) if reference_path else None,
        "chunk_count": reference.get("chunk_count"),
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
        "baseline": _baseline(reference),
    }
    return report


def dumps_report(report: dict[str, Any]) -> str:
    """A report as the JSON text ``write_report`` saves, with floats rounded."""
    return json.dumps(_round(report), ensure_ascii=False, indent=2) + "\n"


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_report(report)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
