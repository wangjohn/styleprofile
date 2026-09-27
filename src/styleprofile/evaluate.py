"""``styleprofile evaluate``: does contrast likeness survive editing of the contrast drafts?

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

from styleprofile.profile import (
    EVALUATION,
    EVALUATION_VERSION,
    Chunk,
    ContrastFit,
    StyleProfileError,
    base_id,
    build_contrast_reference,
    z_against_reference,
)
from styleprofile.surface import prose, words
from styleprofile.syntax import Parser
from styleprofile.weighting import (
    CrossValidated,
    Key,
    ZScores,
    auc,
    auc_interval,
    by_document,
    cross_validate,
    fold_scores,
    likeness_level,
    likeness_words,
)

VERSION = EVALUATION_VERSION
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
    return base_id(chunk.id)


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
    per_draft: dict[str, list[float]] = defaultdict(list)
    for score, document in zip(scores, documents, strict=True):
        per_draft[document].append(score)
    drafts = []
    for document, values in sorted(per_draft.items(), key=lambda item: names[item[0]]):
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
        "auc": auc(scores, learned.reference_scores),
        "auc_ci": auc_interval(
            by_document(learned.reference_scores, reference_documents),
            list(per_draft.values()),
        ),
        # Over chunks, like the reference's stored likeness range it is compared with.
        "likeness_median_chunks": statistics.median(scores) if scores else None,
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
    edited: Mapping[str, tuple[Sequence[ZScores], Sequence[ZScores]]],
    count: int = SIGNALS_TRACKED,
) -> list[dict[str, Any]]:
    """For the strongest contrast metrics, how much of the original drafts' gap from the
    reference each edited set keeps.

    ``edited`` maps each set to (its z-scores, the z-scores of the originals it covers): a
    set that edits only some drafts is compared with those drafts alone, so differences
    between drafts do not pass for an effect of the edit. The gap is the originals' mean z
    minus the reference's held-out mean z. ``remaining`` is the edited set's gap over the
    originals': 1 means untouched, 0 means edited back to the reference's level, negative
    means pushed past it, and above 1 means the edit strengthened the signal.
    """
    ranked = sorted(learned.effects.items(), key=lambda item: -abs(item[1]))[:count]
    rows: list[dict[str, Any]] = []
    for key, effect in ranked:
        reference = _mean_z(reference_held, key) or 0.0
        by_set: dict[str, Any] = {}
        for label, (rows_z, covered) in edited.items():
            value, original = _mean_z(rows_z, key), _mean_z(covered, key)
            gap = None if original is None else original - reference
            remaining = (
                (value - reference) / gap
                if value is not None and gap is not None and abs(gap) >= MIN_GAP
                else None
            )
            by_set[label] = {"z": value, "original_z": original, "remaining": remaining}
        rows.append(
            {
                "metric": f"{key[0]}.{key[1]}",
                "effect": effect,
                "reference_z": reference,
                "original_z": _mean_z(original_z, key),
                "edited": by_set,
            }
        )
    return rows


def _duplicates(chunks: Sequence[Chunk]) -> list[str]:
    """Names that more than one input maps to: two files (or a file and a JSONL record)
    with the same relative name, or one JSONL id used twice."""
    sources: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    repeated: set[str] = set()
    for chunk in chunks:
        sources[match_key(chunk)].add(chunk.source)
        if (chunk.source, chunk.id) in seen:
            repeated.add(match_key(chunk))
        seen.add((chunk.source, chunk.id))
    return sorted(repeated | {key for key, found in sources.items() if len(found) > 1})


def evaluate_rewording(
    reference_chunks: Sequence[Chunk],
    contrast_chunks: Sequence[Chunk],
    edited: Mapping[str, Sequence[Chunk]],
    *,
    parser: Parser | None = None,
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
        raise StyleProfileError("no edited sets to evaluate", code="no_edited_sets")
    if ORIGINAL in edited:
        raise StyleProfileError(
            f"{ORIGINAL!r} names the unedited drafts; use another label", code="reserved_label"
        )
    for set_label, chunks in {"contrast": contrast_chunks, **edited}.items():
        duplicated = _duplicates(chunks)
        if duplicated:
            raise StyleProfileError(
                f"{set_label}: {len(duplicated)} name(s) belong to more than one draft, e.g. "
                f"{duplicated[0]!r}, so edited copies cannot be matched to originals; give "
                "every draft a distinct name",
                code="duplicate_names",
            )
    profile, fit = build_contrast_reference(
        reference_chunks,
        contrast_chunks,
        parser=parser,
        min_words=min_words,
        contrast_label=contrast_label,
    )
    learned = fit.learned
    documents_by_key = {
        match_key(chunk): document
        for chunk, document in zip(fit.contrast_chunks, fit.contrast_documents, strict=True)
    }
    names = {document: key for key, document in documents_by_key.items()}
    # Originals dropped for having too little prose have no fold; their edits are skipped.
    dropped = {match_key(chunk) for chunk in contrast_chunks} - set(documents_by_key)
    original_texts = _texts(contrast_chunks)
    calibration = profile["contrast"]["calibration"]
    label = contrast_label

    warnings: list[str] = list(profile["warnings"])
    sets = {
        ORIGINAL: _set_summary(
            learned.contrast_scores,
            fit.contrast_documents,
            names,
            learned,
            fit.reference_documents,
            calibration,
            label,
        )
    }
    edited_z: dict[str, list[ZScores]] = {}
    edited_documents: dict[str, list[str]] = {}
    survival: dict[str, tuple[Sequence[ZScores], Sequence[ZScores]]] = {}
    for set_label, chunks in edited.items():
        skipped = sorted({match_key(chunk) for chunk in chunks} & dropped)
        if skipped:
            warnings.append(
                f"{set_label}: skipped {len(skipped)} edited draft(s) whose original has too "
                f"little prose to score (e.g. {skipped[0]!r})"
            )
            chunks = [chunk for chunk in chunks if match_key(chunk) not in dropped]
        loaded = {match_key(chunk) for chunk in chunks}
        unmatched = sorted(loaded - set(documents_by_key))
        if unmatched:
            raise StyleProfileError(
                f"{set_label}: {len(unmatched)} edited file(s) have no original among the "
                f"contrast drafts (matched by name), e.g. {unmatched[0]!r}",
                code="unmatched_edits",
            )
        kept_chunks, z_rows = z_against_reference(profile, chunks, parser, min_words)
        documents = [documents_by_key[match_key(chunk)] for chunk in kept_chunks]
        scores = fold_scores(z_rows, documents, learned.contrast_folds)
        edited_z[set_label] = z_rows
        edited_documents[set_label] = documents
        result = _set_summary(
            scores, documents, names, learned, fit.reference_documents, calibration, label
        )
        missing = sorted(set(documents_by_key) - loaded)
        too_short = sorted(loaded - {match_key(chunk) for chunk in kept_chunks})
        result["missing"] = missing
        result["too_short"] = too_short
        result["skipped"] = skipped
        if missing:
            warnings.append(
                f"{set_label}: {len(missing)} of {len(names)} drafts have no edited copy "
                f"(e.g. {missing[0]!r}); its AUC is over fewer drafts than the original's, "
                "and signal survival compares it with the drafts it covers"
            )
        if too_short:
            warnings.append(
                f"{set_label}: {len(too_short)} edited draft(s) have no chunk with at least "
                f"{min_words} prose words (e.g. {too_short[0]!r}), so they are not scored"
            )
        present = set(documents)
        covered = [index for index, doc in enumerate(fit.contrast_documents) if doc in present]
        result["partial"] = len(present) < len(names)
        if result["partial"]:
            result["original_auc_same_drafts"] = auc(
                [learned.contrast_scores[index] for index in covered], learned.reference_scores
            )
        survival[set_label] = (z_rows, [fit.contrast_z[index] for index in covered])
        result["edits"] = edit_statistics(original_texts, _texts(chunks))
        sets[set_label] = result

    return {
        "kind": EVALUATION,
        "version": VERSION,
        "settings": {
            **(settings or {}),
            "min_words": min_words,
            "retrain": retrain,
            "syntax_used": profile["settings"]["syntax_used"],
        },
        "label": label,
        "reference": {
            "chunks": profile["chunk_count"],
            "documents": len(set(fit.reference_documents)),
            "likeness": calibration["reference"],
        },
        "contrast": {
            "chunks": len(fit.contrast_chunks),
            "drafts": len(names),
            "cross_validated": learned.cross_validated,
        },
        "sets": sets,
        "signals": signal_survival(learned, fit.reference_held, fit.contrast_z, survival),
        "retrain": _retrain(fit, edited_z, edited_documents) if retrain else None,
        "warnings": warnings,
    }


def _retrain(
    fit: ContrastFit,
    edited_z: Mapping[str, Sequence[ZScores]],
    edited_documents: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Cross-validated AUCs when the edited drafts join the contrast set.

    Edited chunks carry their original's document, so a lineage is held out as a whole: an
    edited draft is never scored with weights learned from its own original or vice versa.
    """
    learned, reference_held, reference_documents = (
        fit.learned,
        fit.reference_held,
        fit.reference_documents,
    )
    contrast_z, contrast_documents = fit.contrast_z, fit.contrast_documents
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
            "auc": auc(scores, retrained.reference_scores),
            "auc_ci": auc_interval(
                by_document(retrained.reference_scores, reference_documents),
                by_document(scores, documents[start:end]),
            ),
            "before": auc(
                fold_scores(z_rows[start:end], documents[start:end], learned.contrast_folds),
                learned.reference_scores,
            ),
        }
    return {
        "auc": auc(retrained.contrast_scores, retrained.reference_scores),
        "auc_ci": auc_interval(
            by_document(retrained.reference_scores, reference_documents),
            by_document(retrained.contrast_scores, documents),
        ),
        "by_set": by_set,
    }
