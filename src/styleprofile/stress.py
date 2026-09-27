"""Rewording stress test: does contrast likeness survive editing of the contrast drafts?

A reference built with contrast drafts learns which metrics separate the writer from them.
The strongest tells (em dashes, sentence length) are exactly what light editing or a
"humanizer" pass removes, so this module scores edited copies of the same drafts without
leaking: each edited draft is matched to its original by name and scored with the weights
learned without that original, the same leave-one-document-out fold the original gets.
Reference chunks keep their own cross-validated scores.

The comparison is between the original drafts and each edited set, against the same
held-out reference scores: how far the AUC falls, how many drafts still read as contrast
text, and which of the strongest signals survive.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from styleprofile.display import likeness_level, likeness_words
from styleprofile.profile import (
    _WINDOW_SUFFIX,
    Chunk,
    StyleProfileError,
    _measure,
    _z_against,
    build_profile,
    document_of,
)
from styleprofile.surface import prose, words
from styleprofile.syntax import Parser
from styleprofile.weighting import (
    CrossValidated,
    Key,
    ZScores,
    _auc,
    _by_document,
    auc_interval,
    cross_validate,
    floors,
    fold_scores,
    held_out_z,
)

VERSION = 1
ORIGINAL = "original"
SIGNALS_TRACKED = 10
# Likeness levels at or above this read "leans <label>" or "like the <label> drafts".
FLAGGED_LEVEL = 2
# Below this original-vs-reference gap in mean z, a share of the gap is not meaningful.
MIN_GAP = 0.05
NGRAM = 13


def match_key(chunk: Chunk) -> str:
    """What pairs an edited chunk with its original: its file name (or JSONL id) with any
    window suffix removed. Folders are compared by relative path, so ``a/x.md`` and
    ``b/x.md`` pair with each other and with nothing else."""
    return _WINDOW_SUFFIX.sub("", chunk.id)


def ngram_changed(before: str, after: str, n: int = NGRAM) -> float | None:
    """Share of the original's n-word sequences (verbatim tokens) absent from the edit."""
    old, new = before.split(), after.split()
    grams = [tuple(old[index : index + n]) for index in range(len(old) - n + 1)]
    if not grams:
        return None
    kept = {tuple(new[index : index + n]) for index in range(len(new) - n + 1)}
    return sum(gram not in kept for gram in grams) / len(grams)


def _texts(chunks: Sequence[Chunk]) -> dict[str, str]:
    """Each draft's prose, its windows joined back together, keyed like ``match_key``."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        grouped[match_key(chunk)].append(chunk.text)
    return {key: prose("\n\n".join(texts)).text for key, texts in grouped.items()}


def edit_statistics(originals: Mapping[str, str], edited: Mapping[str, str]) -> dict[str, Any]:
    """How much an edited set changed its drafts' prose: the share of each original's
    13-word sequences that no longer appear verbatim, and the ratio of word counts."""
    changed: list[float] = []
    ratios: list[float] = []
    for key, after in edited.items():
        before = originals.get(key)
        if before is None:
            continue
        value = ngram_changed(before, after)
        if value is not None:
            changed.append(value)
        if count := len(words(before)):
            ratios.append(len(words(after)) / count)
    return {
        "ngram13_changed_median": statistics.median(changed) if changed else None,
        "ngram13_changed_mean": statistics.fmean(changed) if changed else None,
        "word_ratio_median": statistics.median(ratios) if ratios else None,
    }


def _set_summary(
    scores: Sequence[float],
    documents: Sequence[str],
    names: Mapping[str, str],
    learned: CrossValidated,
    reference_documents: Sequence[str],
    calibration: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """AUC against the reference's held-out chunks, and each draft's verdict."""
    by_document: dict[str, list[float]] = defaultdict(list)
    for score, document in zip(scores, documents, strict=True):
        by_document[document].append(score)
    drafts = []
    for document, values in sorted(by_document.items(), key=lambda item: names[item[0]]):
        mean = statistics.fmean(values)
        level = likeness_level(mean, dict(calibration), len(values))
        drafts.append(
            {
                "draft": names[document],
                "chunks": len(values),
                "likeness": mean,
                "level": level,
                "verdict": likeness_words(level, label),
            }
        )
    verdicts = Counter(draft["verdict"] for draft in drafts)
    return {
        "drafts": len(drafts),
        "chunks": len(scores),
        "auc": _auc(scores, learned.reference_scores),
        "auc_ci": auc_interval(
            _by_document(learned.reference_scores, reference_documents),
            list(by_document.values()),
        ),
        "likeness_median": statistics.median(scores),
        "flagged": sum(draft["level"] >= FLAGGED_LEVEL for draft in drafts),
        "verdicts": {
            likeness_words(level, label): verdicts[likeness_words(level, label)]
            for level in range(4)
        },
        "by_draft": drafts,
    }


def _mean_z(rows: Sequence[ZScores], key: Key) -> float | None:
    values = [row[key] for row in rows if key in row]
    return statistics.fmean(values) if values else None


def signal_survival(
    learned: CrossValidated,
    reference_held: Sequence[ZScores],
    original_z: Sequence[ZScores],
    edited_z: Mapping[str, Sequence[ZScores]],
    count: int = SIGNALS_TRACKED,
) -> list[dict[str, Any]]:
    """For the strongest contrast metrics, how much of the original drafts' gap from the
    reference each edited set keeps.

    The gap is the drafts' mean z minus the reference's held-out mean z. ``remaining`` is
    the edited set's gap over the original's: 1 means untouched, 0 means edited back to the
    reference's level, negative means pushed past it.
    """
    ranked = sorted(learned.effects.items(), key=lambda item: -abs(item[1]))[:count]
    rows: list[dict[str, Any]] = []
    for key, effect in ranked:
        reference = _mean_z(reference_held, key) or 0.0
        original = _mean_z(original_z, key)
        gap = None if original is None else original - reference
        edited: dict[str, Any] = {}
        for label, rows_z in edited_z.items():
            value = _mean_z(rows_z, key)
            remaining = (
                (value - reference) / gap
                if value is not None and gap is not None and abs(gap) >= MIN_GAP
                else None
            )
            edited[label] = {"z": value, "remaining": remaining}
        rows.append(
            {
                "metric": f"{key[0]}.{key[1]}",
                "effect": effect,
                "reference_z": reference,
                "original_z": original,
                "edited": edited,
            }
        )
    return rows


def evaluate_rewording(
    reference_chunks: Sequence[Chunk],
    contrast_chunks: Sequence[Chunk],
    edited: Mapping[str, Sequence[Chunk]],
    *,
    parser: Parser | None,
    min_words: int = 1,
    contrast_label: str = "LLM",
    retrain: bool = False,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score edited copies of the contrast drafts without leaking; see the module docstring.

    ``edited`` maps a label (``light``, ``humanize``) to that set's chunks, windowed the same
    way as the contrast drafts. With ``retrain``, also report the cross-validated AUC when
    the edited drafts join the contrast set, each lineage (an original and its edits) held
    out together.
    """
    if not edited:
        raise StyleProfileError("pass at least one --edited LABEL=DIR")
    if ORIGINAL in edited:
        raise StyleProfileError(f"{ORIGINAL!r} names the unedited drafts; use another label")
    profile = build_profile(
        reference_chunks,
        parser=parser,
        min_words=min_words,
        contrast=contrast_chunks,
        contrast_label=contrast_label,
    )
    summary = profile["summary"]
    floor = floors(summary)
    reference_documents = [document_of(row["source"], row["id"]) for row in profile["chunks"]]
    reference_held = held_out_z(
        [row["metrics"] for row in profile["chunks"]], reference_documents, floor
    )

    contrast = _measure(contrast_chunks, parser, min_words)
    contrast_z = [_z_against(metrics, summary, floor) for metrics in contrast.metrics]
    contrast_documents = [document_of(chunk.source, chunk.id) for chunk in contrast.chunks]
    documents_by_key: dict[str, str] = {}
    for chunk, document in zip(contrast.chunks, contrast_documents, strict=True):
        key = match_key(chunk)
        if documents_by_key.setdefault(key, document) != document:
            raise StyleProfileError(
                f"two contrast drafts are both named {key!r}, so edited copies cannot be "
                "matched to one of them; give them distinct names"
            )
    names = {document: key for key, document in documents_by_key.items()}
    original_texts = _texts(contrast_chunks)
    learned = cross_validate(reference_held, reference_documents, contrast_z, contrast_documents)
    calibration = profile["contrast"]["calibration"]
    label = contrast_label

    warnings: list[str] = list(profile["warnings"])
    sets = {
        ORIGINAL: _set_summary(
            learned.contrast_scores,
            contrast_documents,
            names,
            learned,
            reference_documents,
            calibration,
            label,
        )
    }
    edited_z: dict[str, list[ZScores]] = {}
    edited_documents: dict[str, list[str]] = {}
    for set_label, chunks in edited.items():
        measured = _measure(chunks, parser, min_words)
        unmatched = sorted({match_key(chunk) for chunk in measured.chunks} - set(documents_by_key))
        if unmatched:
            raise StyleProfileError(
                f"{set_label}: {len(unmatched)} edited file(s) have no original among the "
                f"contrast drafts (matched by name), e.g. {unmatched[0]!r}"
            )
        z_rows = [_z_against(metrics, summary, floor) for metrics in measured.metrics]
        documents = [documents_by_key[match_key(chunk)] for chunk in measured.chunks]
        scores = fold_scores(z_rows, documents, learned.contrast_folds)
        edited_z[set_label] = z_rows
        edited_documents[set_label] = documents
        result = _set_summary(
            scores, documents, names, learned, reference_documents, calibration, label
        )
        missing = sorted(names[document] for document in set(names) - set(documents))
        result["missing"] = missing
        if missing:
            warnings.append(
                f"{set_label}: {len(missing)} of {len(names)} drafts have no edited copy "
                f"(e.g. {missing[0]!r}), so its AUC is not over the same drafts as the original"
            )
            present = set(documents)
            kept = [index for index, doc in enumerate(contrast_documents) if doc in present]
            result["original_auc_same_drafts"] = _auc(
                [learned.contrast_scores[index] for index in kept], learned.reference_scores
            )
        result["edits"] = edit_statistics(original_texts, _texts(chunks))
        sets[set_label] = result

    report: dict[str, Any] = {
        "version": VERSION,
        "settings": {**(settings or {}), "min_words": min_words, "retrain": retrain},
        "label": label,
        "reference": {
            "chunks": profile["chunk_count"],
            "documents": len(set(reference_documents)),
            "likeness": calibration["reference"],
        },
        "contrast": {
            "chunks": len(contrast.chunks),
            "drafts": len(names),
            "cross_validated": learned.cross_validated,
        },
        "sets": sets,
        "signals": signal_survival(learned, reference_held, contrast_z, edited_z),
        "retrain": (
            _retrain(
                learned,
                reference_held,
                reference_documents,
                contrast_z,
                contrast_documents,
                edited_z,
                edited_documents,
            )
            if retrain
            else None
        ),
        "warnings": warnings,
    }
    return report


def _retrain(
    learned: CrossValidated,
    reference_held: Sequence[ZScores],
    reference_documents: Sequence[str],
    contrast_z: Sequence[ZScores],
    contrast_documents: Sequence[str],
    edited_z: Mapping[str, Sequence[ZScores]],
    edited_documents: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Cross-validated AUCs when the edited drafts join the contrast set.

    Edited chunks carry their original's document, so a lineage is held out as a whole: an
    edited draft is never scored with weights learned from its own original or vice versa.
    """
    z_rows = list(contrast_z)
    documents = list(contrast_documents)
    slices = {ORIGINAL: (0, len(z_rows))}
    for label, rows in edited_z.items():
        slices[label] = (len(z_rows), len(z_rows) + len(rows))
        z_rows += rows
        documents += edited_documents[label]
    retrained = cross_validate(reference_held, reference_documents, z_rows, documents)
    by_set: dict[str, Any] = {}
    for label, (start, end) in slices.items():
        scores = retrained.contrast_scores[start:end]
        by_set[label] = {
            "auc": _auc(scores, retrained.reference_scores),
            "auc_ci": auc_interval(
                _by_document(retrained.reference_scores, reference_documents),
                _by_document(scores, documents[start:end]),
            ),
            "before": _auc(
                fold_scores(z_rows[start:end], documents[start:end], learned.contrast_folds),
                learned.reference_scores,
            ),
        }
    return {
        "auc": _auc(retrained.contrast_scores, retrained.reference_scores),
        "auc_ci": auc_interval(
            _by_document(retrained.reference_scores, reference_documents),
            _by_document(retrained.contrast_scores, documents),
        ),
        "by_set": by_set,
    }
