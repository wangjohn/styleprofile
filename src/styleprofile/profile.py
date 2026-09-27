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
VERSION = 5
TEXT_FIELDS: tuple[str, ...] = ("text", "body_markdown", "output", "content", "body")
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
# Directory walks skip dot-directories (.git, .venv) and these vendored ones.
SKIPPED_DIRS = frozenset({"node_modules", "__pycache__", "site-packages"})
OTHER = "<other>"
DEVIATIONS_SHOWN = 8
SHORT_CHUNK_WORDS = 150


class StyleProfileError(ValueError):
    """Style profile inputs or a reference profile are unusable."""


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


def _jsonl_chunks(path: Path, text_field: str | None) -> list[Chunk]:
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
        fields = (text_field,) if text_field else TEXT_FIELDS
        field = next((name for name in fields if isinstance(record.get(name), str)), None)
        if field is None:
            raise StyleProfileError(
                f"{path}:{line_number}: no string field among {', '.join(fields)}; "
                "pass --text-field"
            )
        record_id = record.get("id")
        # An id of 0 is kept; a missing, null or empty id falls back to the line.
        chunk_id = f"{path.name}:{line_number}" if record_id in (None, "") else str(record_id)
        chunks.append(Chunk(chunk_id, str(path), record[field]))
    return chunks


def load_chunks(inputs: Sequence[str], text_field: str | None = None) -> list[Chunk]:
    """Read JSONL records, Markdown/text files, directories of them, or ``-`` for stdin."""
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


def load_reference(path: Path) -> dict[str, Any]:
    try:
        reference = json.loads(_read_text(path))
    except json.JSONDecodeError as error:
        raise StyleProfileError(f"{path} is not a style profile: {error}") from error
    if not isinstance(reference, dict) or "summary" not in reference:
        raise StyleProfileError(f"{path} is not a style profile (no summary)")
    return reference


@dataclass(frozen=True)
class _Measured:
    chunks: list[Chunk]
    metrics: list[Metrics]
    distributions: list[dict[str, Counter[str]]]
    empty: int
    below: int


def _measure(chunks: Sequence[Chunk], parser: Parser | None, min_words: int) -> _Measured:
    """Parse each chunk once, drop chunks without enough prose, and compute every metric."""
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
            "--contrast needs a reference drawn from at least two documents with enough "
            "chunks to measure the reference's own variation on held-out writing"
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
            "lengths or window the drafts with --window-words"
        )
    return {
        "label": label,
        "chunk_count": len(measured.chunks),
        "sources": len(set(contrast_sources)),
        **learned,
    }, fit


def build_profile(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None,
    top_k: int = 300,
    min_words: int = 1,
    reference: dict[str, Any] | None = None,
    reference_path: Path | None = None,
    settings: dict[str, Any] | None = None,
    contrast: Sequence[Chunk] | None = None,
    contrast_label: str = "LLM",
) -> dict[str, Any]:
    """Profile chunks; score them against ``reference``, or, when building a reference,
    learn likeness weights from ``contrast`` chunks (for example LLM drafts)."""
    return _build_profile(
        chunks,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        reference=reference,
        reference_path=reference_path,
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
    """``build_profile(chunks, contrast=contrast)``, plus what its weights were learned from."""
    report, fit = _build_profile(
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
    drafts are; returns the chunks kept (enough prose) and their z-scores."""
    measured = _measure(chunks, parser, min_words)
    floor = floors(report["summary"])
    return measured.chunks, [
        _z_against(metrics, report["summary"], floor) for metrics in measured.metrics
    ]


def _build_profile(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None,
    top_k: int = 300,
    min_words: int = 1,
    reference: dict[str, Any] | None = None,
    reference_path: Path | None = None,
    settings: dict[str, Any] | None = None,
    contrast: Sequence[Chunk] | None = None,
    contrast_label: str = "LLM",
) -> tuple[dict[str, Any], ContrastFit | None]:
    if contrast is not None and reference is not None:
        raise StyleProfileError(
            "--contrast builds a reference profile; score samples against that reference "
            "in a separate run"
        )
    measured = _measure(chunks, parser, min_words)
    chunks, chunk_metrics, chunk_distributions = (
        measured.chunks,
        measured.metrics,
        measured.distributions,
    )
    empty, below = measured.empty, measured.below
    totals: dict[str, Counter[str]] = {}
    for distributions in chunk_distributions:
        for name, counts in distributions.items():
            totals.setdefault(name, Counter()).update(counts)

    warnings: list[str] = []
    if empty:
        warnings.append(f"skipped {empty} chunk(s) with no prose (only code, tables or markup)")
    if below:
        warnings.append(f"skipped {below} chunk(s) with fewer than {min_words} prose words")
    short = sum((metrics["size"]["words"] or 0) < SHORT_CHUNK_WORDS for metrics in chunk_metrics)
    if short:
        warnings.append(
            f"{short} chunk(s) have fewer than {SHORT_CHUNK_WORDS} words; their rates are noisy"
        )

    prepared = _prepare(reference) if reference is not None else None
    rows: list[dict[str, Any]] = []
    for chunk, metrics, distributions in zip(
        chunks, chunk_metrics, chunk_distributions, strict=True
    ):
        row: dict[str, Any] = {"id": chunk.id, "source": chunk.source, "metrics": metrics}
        if reference is not None and prepared is not None:
            row["reference"] = _score(metrics, distributions, reference, prepared)
        rows.append(row)

    report: dict[str, Any] = {
        "version": VERSION,
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
        "chunk_count": len(chunks),
        "word_count": sum(int(metrics["size"]["words"] or 0) for metrics in chunk_metrics),
        "summary": summarize(chunk_metrics),
        "distributions": {name: _distribution(counts, top_k) for name, counts in totals.items()},
        "chunks": rows,
        "warnings": warnings,
    }
    fit: ContrastFit | None = None
    if reference is None:
        held = _calibrate(report, chunk_metrics)
        if contrast is not None:
            report["contrast"], fit = _learn_contrast(
                report, held, contrast, contrast_label, parser, min_words
            )
    if reference is not None:
        scored = [row["reference"] for row in rows]
        groups = sorted({group for score in scored for group in score["delta_by_group"]})
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
        if prepared is not None and prepared.effects:
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
            warnings.append("the reference has one chunk, so it has no spread; use --window-words")
        if reference_settings.get("window_words") != report["settings"].get("window_words"):
            warnings.append(
                "window sizes differ from the reference "
                f"({report['settings'].get('window_words')} vs "
                f"{reference_settings.get('window_words')}); z-scores assume equal-sized chunks"
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
            "delta_mean": _mean_of([score["delta"] for score in scored]),
            "likeness_mean": _mean_of([score.get("likeness") for score in scored]),
            "delta_by_group_mean": {
                group: _mean_of([score["delta_by_group"].get(group) for score in scored])
                for group in groups
            },
            "divergence_mean": {
                name: _mean_of([score["divergence"].get(name) for score in scored])
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
        }
    return report, fit


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
    content = json.dumps(_round(report), ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
