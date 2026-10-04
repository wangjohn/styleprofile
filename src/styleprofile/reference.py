"""Build reference summaries and held-out calibration."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from styleprofile import drift
from styleprofile.calibration import (
    CALIBRATION_LENGTHS,
    CONTRAST_CALIBRATION_WORDS,
    MIN_CALIBRATION_DOCUMENTS,
    MIN_CALIBRATION_PIECES,
    MIN_CONTRAST_PIECES,
    Lengths,
    calibrate_lengths,
)
from styleprofile.core import Note, NoteCode, Phase, Progress, StyleProfileError
from styleprofile.corpus.ids import chunk_document, is_record
from styleprofile.corpus.preparation import _PART
from styleprofile.measure import Measured as _Measured
from styleprofile.measure import Measurer, measure_in_parts, prose_text
from styleprofile.measure import Piece as _Piece
from styleprofile.reports import MINOR_VERSION, REFERENCE, VERSION
from styleprofile.runtime import _plural
from styleprofile.schema import (
    Contrast,
    DeltaRange,
    DriftCalibration,
    LengthCalibration,
    ReferenceReport,
    Summary,
)
from styleprofile.scoring import _held_out_effects, _held_out_relative, _z_against
from styleprofile.settings import (
    DEFAULT_WINDOW_WORDS,
    ENOUGH_CHUNKS,
    ENOUGH_DOCUMENTS,
    ENOUGH_WORDS,
)
from styleprofile.stats import _base_report, _documents, _measure, _words
from styleprofile.syntax import Parser
from styleprofile.weighting import (
    LENGTH_AUC_WARNING,
    CrossValidated,
    Key,
    ZScores,
    calibrate_delta,
    cross_validate,
    floors,
    held_out,
    held_out_z,
    likeness_range,
    nest,
    reliability,
    summarize_contrast,
)

if TYPE_CHECKING:
    from styleprofile.corpus.types import Chunk
    from styleprofile.settings import Settings


def _calibrate_drift(
    report: ReferenceReport,
    measured: _Measured,
    fit: ContrastFit | None,
    parser: Parser | None,
    chosen: Sequence[str],
    measurer: Measurer | None = None,
) -> None:
    """The null of the paragraph statistic (``drift``), from reference documents each read
    in parts and scored held out, as a draft of theirs would be: z-scores against the
    windows of every other document, likeness with the contrast fold that left the document
    out, both at the span's length. Up to ``drift.CALIBRATION_WORDS`` words of documents are
    read, taken evenly. Stored as ``calibration.drift``."""
    calibration = report.get("calibration")
    if not calibration or not calibration.get("by_length"):
        return
    documents = _documents(measured.chunks)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        grouped[document].append(index)
    lengths = Lengths(report)
    floor = floors(report["summary"])
    by = drift.LIKENESS if fit is not None else drift.DELTA
    effects = _held_out_effects(fit) if fit is not None else None
    documents_read: list[str] = []
    to_read: list[tuple[str, list[tuple[str, Any | None]]]] = []
    for document in chosen:
        indices = grouped.get(document)
        if not indices:
            continue  # all its windows were dropped for min_words
        # Its windows' prose and, where this process parsed them, their parses.
        windows: list[tuple[str, Any | None]] = []
        for index in indices:
            window = measured.docs[index] if measured.docs is not None else None
            windows.append(window or (prose_text(measured.chunks[index].text), None))
        documents_read.append(document)
        to_read.append(("\n\n".join(measured.chunks[index].text for index in indices), windows))
    read = list(zip(documents_read, measure_in_parts(to_read, parser, measurer), strict=True))
    span_metrics = [metrics for _, (_, spans) in read for metrics in spans]
    span_documents = [document for document, (_, spans) in read for _ in spans]
    held = held_out_z(measured.metrics, documents, floor, others=(span_metrics, span_documents))
    relatives: list[float | None] = []
    for metrics, z_scores, document in zip(span_metrics, held, span_documents, strict=True):
        at = lengths.at(int(metrics["size"]["words"] or 0))
        relatives.append(_held_out_relative(z_scores, at, by, effects, document))
    found: list[list[tuple[float, int, int] | None]] = []
    position = 0
    for _, (layout, spans) in read:
        count = len(spans)
        found.append(drift.statistics(layout, relatives[position : position + count]))
        position += count
    calibration["drift"] = cast(
        DriftCalibration,
        drift.calibrate(len(read), found, by),
    )


def _drift_documents(chunks: Sequence[Chunk]) -> list[str]:
    """Every k-th reference document (``chunk_document``), so the ones read for the
    paragraph null hold about ``drift.CALIBRATION_WORDS`` words (counted roughly, as
    whitespace-separated tokens)."""
    sizes: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        sizes[chunk_document(chunk)] += len(chunk.text.split())
    step = max(1, math.ceil(sum(sizes.values()) / drift.CALIBRATION_WORDS))
    return list(sizes)[::step]


@dataclass(frozen=True)
class _Calibrated:
    """The reference's held-out z-scores: of its chunks, and of the pieces of each
    calibrated length (``calibration``) with the documents they came from."""

    held: Sequence[ZScores]
    pieces: dict[int, tuple[list[ZScores], list[str]]]
    # The report's ``calibration.by_length``, which a contrast set adds its ranges to.
    by_length: dict[str, LengthCalibration]
    # The intraclass correlation the pieces' bounds use (``calibration.similarity``).
    icc: float = 0.0


def _calibrate(
    report: ReferenceReport, measured: _Measured, measurer: Measurer | None = None
) -> _Calibrated | None:
    """Held-out reliability and Delta range, when the chunks span at least two documents,
    for the chunks and for shorter pieces cut from them (``calibration.by_length``).

    Without them the profile still works as a reference, but Delta falls back to capped z
    and --contrast is unavailable; scoring against such a reference warns about it.
    """
    documents = _documents(measured.chunks)
    if len(set(documents)) < 2:
        return None
    if measurer is not None:
        measurer.report(Progress(Phase.CALIBRATE))
    floor = floors(report["summary"])
    # The windows' and every length's pieces' held-out z-scores, from one pass of sums.
    held, all_held = held_out(
        measured.metrics,
        documents,
        floor,
        (
            [piece.metrics for piece in measured.pieces],
            [documents[piece.chunk] for piece in measured.pieces],
        ),
    )
    calibration = calibrate_delta(held, documents)
    if calibration is None:
        return None
    report["reliability"] = nest(reliability(held, documents))
    by_length: dict[str, LengthCalibration] = {}
    pieces: dict[int, tuple[list[ZScores], list[str]]] = {}
    grouped: dict[int, tuple[list[ZScores], list[str], list[float]]] = {}
    for length in sorted({piece.length for piece in measured.pieces}):
        chosen = [index for index, piece in enumerate(measured.pieces) if piece.length == length]
        grouped[length] = (
            [all_held[index] for index in chosen],
            [documents[measured.pieces[index].chunk] for index in chosen],
            [measured.pieces[index].metrics["size"]["words"] or 0.0 for index in chosen],
        )
    entries, icc = calibrate_lengths(grouped)
    for length, entry in entries.items():
        by_length[str(length)] = entry
        if "delta" in entry:
            pieces[length] = grouped[length][:2]
    report["calibration"] = {
        "sources": len(set(documents)),
        # Without ``upper``, calibrate_delta gives a window's range: median, p95, max.
        "delta": cast(DeltaRange, calibration),
        # The windows' own length: the longest anchor of the length calibration.
        "chunk_words": statistics.median(_words(measured.metrics)),
        "by_length": by_length,
    }
    return _Calibrated(held, pieces, by_length, icc)


@dataclass(frozen=True)
class ContrastFit:
    """What a contrast reference's likeness weights were learned from.

    ``learned`` holds every held-out fold, so more text derived from a contrast document
    (an edited copy, say) can be scored with the fold that left that document out, and
    judged against the same reference scores and calibration the report stores.
    """

    reference_held: Sequence[ZScores]
    reference_documents: list[str]
    contrast_chunks: list[Chunk]
    contrast_z: list[ZScores]
    contrast_documents: list[str]
    learned: CrossValidated


def _learn_contrast(
    report: ReferenceReport,
    reference: _Measured,
    calibrated: _Calibrated | None,
    contrast: Sequence[Chunk],
    label: str,
    parser: Parser | None,
    min_words: int,
    measurer: Measurer | None = None,
) -> tuple[Contrast, ContrastFit]:
    if calibrated is None:
        raise StyleProfileError(
            "a contrast set needs a reference drawn from at least two documents with enough "
            "chunks to measure the reference's own variation on held-out writing",
            code="contrast_needs_documents",
        )
    held = calibrated.held
    measured = _measure(
        contrast,
        parser,
        min_words,
        piece_lengths=sorted(calibrated.pieces),
        piece_words=CONTRAST_CALIBRATION_WORDS,
        # Only a score reads each chunk's pattern counts.
        keep_distributions=False,
        measurer=measurer,
        phase=Phase.MEASURE_CONTRAST,
    )
    if measurer is not None:
        measurer.report(Progress(Phase.CONTRAST))
    if measured.empty or measured.below:
        report["warnings"].append(
            NoteCode.SKIPPED_CONTRAST.message(
                "0", f"{measured.empty + measured.below}", f"{min_words}"
            )
        )
    floor = floors(report["summary"])
    contrast_z = [_z_against(metrics, report["summary"], floor) for metrics in measured.metrics]
    contrast_sources = _documents(measured.chunks)
    reference_sources = _documents(reference.chunks)
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
        # Measured chunks all have prose, so words is never None.
        _words(reference.metrics),
        # Measured chunks all have prose, so words is never None.
        _words(measured.metrics),
    )
    _calibrate_contrast_lengths(report["summary"], calibrated, fit, measured.pieces, floor)
    calibration = learned["calibration"]
    if not calibration["cross_validated"]:
        report["warnings"].append(NoteCode.SINGLE_CONTRAST.message("0"))
    if calibration["auc_ci"] is None:
        report["warnings"].append(NoteCode.NO_AUC_INTERVAL.message("0"))
    length = calibration["length_baseline"]
    if length and length["auc"] >= LENGTH_AUC_WARNING:
        report["warnings"].append(
            NoteCode.LENGTH_CONTRAST.message("0", f"{length['auc']:.2f}", f"{label}")
        )
    return {
        "label": label,
        "chunk_count": len(measured.chunks),
        "sources": len(set(contrast_sources)),
        **learned,
    }, fit


def _calibrate_contrast_lengths(
    summary: Summary,
    calibrated: _Calibrated,
    fit: ContrastFit,
    pieces: Sequence[_Piece],
    floor: dict[Key, float],
) -> None:
    """The likeness range at each calibrated length, from reference and contrast pieces."""
    for length, (piece_held, piece_documents) in calibrated.pieces.items():
        chosen = [piece for piece in pieces if piece.length == length]
        entry = calibrated.by_length[str(length)]
        entry["contrast_pieces"] = len(chosen)
        if len(chosen) < MIN_CONTRAST_PIECES:
            continue
        entry["likeness"] = likeness_range(
            fit.reference_held,
            fit.reference_documents,
            fit.contrast_z,
            piece_held,
            piece_documents,
            [_z_against(piece.metrics, summary, floor) for piece in chosen],
            [fit.contrast_documents[piece.chunk] for piece in chosen],
            fit.learned,
            icc=calibrated.icc,
        )


def build_reference(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None = None,
    top_k: int = 300,
    min_words: int = 1,
    settings: Mapping[str, Any] | None = None,
    contrast: Sequence[Chunk] | None = None,
    contrast_label: str = "LLM",
    keep_chunks: bool = False,
    passages: bool = False,
    measurer: Measurer | None = None,
) -> ReferenceReport:
    """Profile a writer's chunks as a reference: each metric's mean and spread, its held-out
    reliability when the chunks span two or more documents and, given ``contrast`` chunks
    (for example LLM drafts), the weights that score likeness to them.

    This is the lower-level step under ``styleprofile.build``: the chunks are measured as
    given, not cut into windows, and syntax metrics are left out unless ``parser`` (from
    ``load_parser``) is passed. Use ``styleprofile.build`` to get the same profile as
    ``styleprofile build``. Unless ``settings`` says otherwise, the profile records
    ``window_words`` 0, since these chunks were not windowed here.

    Scoring needs only the summary, so the per-chunk metrics are left out, which keeps the
    profile small however large the corpus; ``keep_chunks`` saves them too, for debugging.
    ``measurer`` sets how chunks are measured (a cache, worker processes, progress); by
    default everything is measured in this process, with nothing cached."""
    return _build_reference(
        chunks,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings={"window_words": 0, **(settings or {})},
        contrast=contrast,
        contrast_label=contrast_label,
        keep_chunks=keep_chunks,
        passages=passages,
        measurer=measurer,
    )[0]


def build_contrast_reference(
    chunks: Sequence[Chunk],
    contrast: Sequence[Chunk],
    *,
    parser: Parser | None = None,
    top_k: int = 300,
    min_words: int = 1,
    settings: Mapping[str, Any] | None = None,
    contrast_label: str = "LLM",
    calibrate_lengths: bool = True,
    measurer: Measurer | None = None,
) -> tuple[ReferenceReport, ContrastFit]:
    """``build_reference(chunks, contrast=contrast)``, plus what its weights were learned
    from, for scoring more text with the same folds. ``calibrate_lengths=False`` skips the
    calibration for shorter texts, for a caller that only scores window-sized chunks."""
    report, fit = _build_reference(
        chunks,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
        contrast=contrast,
        contrast_label=contrast_label,
        keep_chunks=False,
        calibrate_lengths=calibrate_lengths,
        measurer=measurer,
    )
    assert fit is not None  # a contrast always produces a fit or raises
    return report, fit


def _build_reference(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None,
    top_k: int,
    min_words: int,
    settings: Mapping[str, Any] | None,
    contrast: Sequence[Chunk] | None,
    contrast_label: str,
    keep_chunks: bool,
    calibrate_lengths: bool = True,
    passages: bool = False,
    measurer: Measurer | None = None,
) -> tuple[ReferenceReport, ContrastFit | None]:
    # The documents read for the paragraph null (``_calibrate_drift``), whose parses are kept.
    chosen = _drift_documents(chunks) if passages and calibrate_lengths else []
    measured = _measure(
        chunks,
        parser,
        min_words,
        piece_lengths=CALIBRATION_LENGTHS if calibrate_lengths else (),
        # A reference keeps only the pooled counts, never each chunk's.
        keep_distributions=False,
        measurer=measurer,
        keep_docs=set(chosen),
    )
    base = _base_report(
        measured,
        kind=REFERENCE,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
        rows=keep_chunks,
    )
    report: ReferenceReport = {
        "version": VERSION,
        "minor_version": MINOR_VERSION,
        "kind": REFERENCE,
        **base.described,
        "warnings": base.warnings,
    }
    if keep_chunks:
        report["chunks"] = base.rows
    calibrated = _calibrate(report, measured, measurer)
    fit: ContrastFit | None = None
    if contrast is not None:
        report["contrast"], fit = _learn_contrast(
            report, measured, calibrated, contrast, contrast_label, parser, min_words, measurer
        )
    if chosen and calibrated is not None:
        _calibrate_drift(report, measured, fit, parser, chosen, measurer)
    return report, fit


def _thin_reference(
    report: ReferenceReport, settings: Settings, windows: Sequence[Chunk] = ()
) -> list[Note]:
    """Why a reference may be too small to trust, each with its fix. ``windows``, the
    chunks measured, show when one document holds most of them, so that its held-out range
    rests on the few windows of the others."""
    documents = report["document_count"]
    thin: list[Note] = []

    def note(message: str, setting: str | None = None) -> None:
        thin.append(Note(message, NoteCode.THIN_REFERENCE, setting=setting))

    if documents < ENOUGH_DOCUMENTS and settings.group_field and len(windows) > 1:
        note(
            NoteCode.THIN_REFERENCE.message(
                "0", f"{_plural(documents, 'document')}", f"{settings.group_field}"
            ),
            "group_field",
        )
    elif documents < ENOUGH_DOCUMENTS:
        note(NoteCode.THIN_REFERENCE.message("3", f"{_plural(documents, 'document')}"))
    if report["chunk_count"] < ENOUGH_CHUNKS:
        smaller = (
            "a smaller window_words"
            if settings.window_words
            else f"window_words {DEFAULT_WINDOW_WORDS}"
        )
        note(
            NoteCode.THIN_REFERENCE.message(
                "1", f"{_plural(report['chunk_count'], 'chunk')}", f"{ENOUGH_CHUNKS}", f"{smaller}"
            ),
            "window_words",
        )
    # The windows' own held-out range needs only two documents, but a document holding
    # most windows is judged against the others' few; hold that to the rule each shorter
    # length is held to (``calibration``).
    sizes = Counter(map(chunk_document, windows))
    if len(sizes) >= ENOUGH_DOCUMENTS:
        largest = max(sizes.values())
        rest = len(windows) - largest
        few = rest < MIN_CALIBRATION_PIECES or len(sizes) < MIN_CALIBRATION_DOCUMENTS
        if largest > rest and few:
            [(document, _)] = sizes.most_common(1)
            # A whole text, unlike a group of records or a part already split off, may
            # divide at headings or rules the automatic split left alone.
            first = next(w for w in windows if chunk_document(w) == document)
            text = not is_record(first) and _PART not in document
            how = NoteCode.THIN_REFERENCE.message("split_advice")
            note(
                NoteCode.THIN_REFERENCE.message(
                    "dominant",
                    f"{largest:,}",
                    _plural(len(windows), "chunk"),
                    _plural(rest, "chunk"),
                    str(MIN_CALIBRATION_PIECES),
                    str(MIN_CALIBRATION_DOCUMENTS),
                    NoteCode.THIN_REFERENCE.message("split_suffix", how) if text else "",
                ),
                "split_on" if text else None,
            )
    if report["word_count"] < ENOUGH_WORDS:
        note(
            NoteCode.THIN_REFERENCE.message(
                "2", f"{_plural(report['word_count'], 'word')}", f"{ENOUGH_WORDS:,}"
            )
        )
    return thin
