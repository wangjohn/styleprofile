"""Measured report summaries shared by references and scores."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

from styleprofile.calibration import CALIBRATION_WORDS
from styleprofile.core import NoteCode, Phase
from styleprofile.corpus.ids import chunk_document
from styleprofile.measure import Measured as _Measured
from styleprofile.measure import Measurer, measure
from styleprofile.reports import REFERENCE
from styleprofile.rows import MetricRows
from styleprofile.schema import ChunkRow, MetricStats, ReportSettings, Summary
from styleprofile.surface import Metrics
from styleprofile.syntax import Parser

if TYPE_CHECKING:
    from styleprofile.corpus.types import Chunk

OTHER = "<other>"
SHORT_CHUNK_WORDS = 150


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


def _stats(values: Sequence[float]) -> MetricStats:
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


def summarize(chunk_metrics: Sequence[Metrics]) -> Summary:
    if isinstance(chunk_metrics, MetricRows) and chunk_metrics.uniform:
        # Every chunk has the metrics of the layout, in its order: each is one column.
        return {
            group: {name: _stats(chunk_metrics.present(group, name)) for name in group_names}
            for group, group_names in chunk_metrics.layout
        }
    # Every metric any chunk has, in the order chunks first have them.
    names: dict[str, dict[str, None]] = {}
    for metrics in chunk_metrics:
        for group, values in metrics.items():
            for name in values:
                names.setdefault(group, {})[name] = None
    return {
        group: {
            name: _stats(
                [
                    value
                    for metrics in chunk_metrics
                    if (value := metrics.get(group, {}).get(name)) is not None
                ]
            )
            for name in group_names
        }
        for group, group_names in names.items()
    }


def _measure(
    chunks: Sequence[Chunk],
    parser: Parser | None,
    min_words: int,
    *,
    allow_empty: bool = False,
    piece_lengths: Sequence[int] = (),
    piece_words: int = CALIBRATION_WORDS,
    keep_distributions: bool = True,
    measurer: Measurer | None = None,
    phase: Phase = Phase.MEASURE,
    keep_docs: bool | Collection[str] = False,
    docs_required: bool = False,
) -> _Measured:
    """Parse each chunk once, drop chunks without enough prose, and compute every metric,
    with calibration pieces of ``piece_lengths``, keeping the parses ``keep_docs`` asks for
    (see ``measure.measure``)."""
    return measure(
        chunks,
        parser,
        min_words,
        allow_empty=allow_empty,
        piece_lengths=piece_lengths,
        piece_words=piece_words,
        keep_distributions=keep_distributions,
        measurer=measurer,
        phase=phase,
        documents=_documents(chunks),
        keep_docs=keep_docs,
        docs_required=docs_required,
    )


def _words(chunk_metrics: Sequence[Metrics]) -> list[float]:
    """Each chunk's prose word count; measured chunks all have prose, so never None."""
    if isinstance(chunk_metrics, MetricRows):
        column = chunk_metrics.column("size", "words")
    else:
        column = [metrics["size"]["words"] for metrics in chunk_metrics]
    return [value or 0.0 for value in column]


def _documents(chunks: Sequence[Chunk]) -> list[str]:
    return [chunk_document(chunk) for chunk in chunks]


class _Described(TypedDict):
    """The keys every reference and score report has between ``kind`` and ``chunks``."""

    settings: ReportSettings
    chunk_count: int
    document_count: int
    word_count: int
    summary: Summary
    distributions: dict[str, dict[str, float]]


@dataclass(frozen=True)
class _Base:
    """The part of every report that describes its own chunks, plus the pooled counts."""

    described: _Described
    rows: list[ChunkRow]
    warnings: list[str]
    totals: dict[str, Counter[str]]


def _base_report(
    measured: _Measured,
    *,
    kind: str,
    parser: Parser | None,
    top_k: int,
    min_words: int,
    settings: Mapping[str, Any] | None,
    rows: bool = True,
) -> _Base:
    """The part of every report of ``kind`` that describes its own chunks, plus (with
    ``rows``) a row of metrics per chunk: score reports need them to show which chunks stand
    out, while a reference stores only its summary unless asked to keep them, and making
    them for thousands of chunks takes hundreds of megabytes."""
    chunk_metrics = measured.metrics
    totals = measured.totals

    warnings: list[str] = []
    if measured.empty:
        warnings.append(NoteCode.EMPTY_CHUNKS.message("0", f"{measured.empty}"))
    if measured.below:
        warnings.append(NoteCode.BELOW_MIN_WORDS.message("0", f"{measured.below}", f"{min_words}"))
    # A score judges each chunk at its own length instead (``calibration``).
    words = _words(chunk_metrics)
    short = sum(count < SHORT_CHUNK_WORDS for count in words)
    if short and kind == REFERENCE:
        warnings.append(NoteCode.NOISY_CHUNKS.message("0", f"{short}", f"{SHORT_CHUNK_WORDS}"))
    # The settings are recorded as given, whatever their keys.
    recorded = cast(
        ReportSettings,
        {
            **(settings or {}),
            "top_k": top_k,
            # What was used, as opposed to the ``syntax`` setting, which is what was asked.
            "syntax_used": parser.used() if parser else None,
        },
    )
    described: _Described = {
        "settings": recorded,
        "chunk_count": len(measured.chunks),
        "document_count": len(set(_documents(measured.chunks))),
        "word_count": sum(int(count) for count in words),
        "summary": summarize(chunk_metrics),
        "distributions": {name: _distribution(counts, top_k) for name, counts in totals.items()},
    }
    chunk_rows: list[ChunkRow] = (
        [
            {"id": chunk.id, "source": chunk.source, "metrics": metrics}
            for chunk, metrics in zip(measured.chunks, chunk_metrics, strict=True)
        ]
        if rows
        else []
    )
    return _Base(described, chunk_rows, warnings, totals)
