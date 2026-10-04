"""Public build and evaluation pipelines over domain-owned modules."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Unpack

from styleprofile.core import Note, NoteCode, Phase, StyleProfileError
from styleprofile.corpus.preparation import (
    _chunked,
    _covered,
    _described,
    _edits_of_repeats,
    _items,
    _pooling,
    _read,
    _sources,
    _stdin_once,
    _typed,
)
from styleprofile.corpus.reading import Records, SourceNames
from styleprofile.corpus.types import Text as Text
from styleprofile.corpus.windows import pool, window
from styleprofile.evaluate import evaluate_rewording
from styleprofile.reference import _thin_reference, build_reference
from styleprofile.reports import dumps_report
from styleprofile.results import DocumentResult as DocumentResult
from styleprofile.results import Evaluation as Evaluation
from styleprofile.results import Profile
from styleprofile.results import ScoreResult as ScoreResult
from styleprofile.results import fails as fails
from styleprofile.runtime import (
    _finish,
    _measurer,
    _notes_on_error,
    _or_without_syntax,
    _parser,
    _progress,
)
from styleprofile.settings import AUTO_JOBS, DEFAULTS, _overridden, _strings
from styleprofile.settings import Settings as Settings
from styleprofile.settings import SettingsOverrides as SettingsOverrides
from styleprofile.surface import prose, words

if TYPE_CHECKING:
    from styleprofile.corpus.types import Chunk, Repeat
    from styleprofile.settings import Inputs, ProgressCallback, Settings, SettingsOverrides

load = Profile.load


def build_texts(
    texts: Iterable[str],
    *,
    contrast: Iterable[str] | None = None,
    contrast_label: str = "LLM",
    progress: ProgressCallback | None = None,
    keep_chunks: bool = False,
    passages: bool = False,
    jobs: int = AUTO_JOBS,
    cache: bool = False,
    **overrides: Unpack[SettingsOverrides],
) -> Profile:
    """Build from raw strings, with one document per string.

    Options are the same as for ``build``. Use ``build`` with ``Text`` objects to name
    documents, or with paths to read files and folders.
    """
    return build(
        _strings(texts),
        contrast=_strings(contrast) if contrast is not None else None,
        contrast_label=contrast_label,
        progress=progress,
        keep_chunks=keep_chunks,
        passages=passages,
        jobs=jobs,
        cache=cache,
        **overrides,
    )


def build(
    inputs: Inputs,
    settings: Settings = DEFAULTS,
    *,
    contrast: Inputs | None = None,
    contrast_label: str = "LLM",
    progress: ProgressCallback | None = None,
    keep_chunks: bool = False,
    passages: bool = False,
    jobs: int = AUTO_JOBS,
    cache: bool = False,
    **overrides: Unpack[SettingsOverrides],
) -> Profile:
    """Build a reference profile from a writer's texts, as ``styleprofile build`` does.

    Given ``contrast`` texts (LLM drafts of the same briefs, say), the profile also learns
    what separates the writer from them and scores likeness to them. A file or folder given
    twice, in ``inputs`` or ``contrast``, is read once, with a note. A reference too small
    to trust gets one ``NoteCode.THIN_REFERENCE`` note per reason.

    The profile keeps summaries only; ``keep_chunks`` also saves every chunk's metrics, for
    debugging. It is an option of this build rather than a ``Settings`` field: it changes
    what is saved, not how texts are read or cut, and scoring has nothing to inherit from it.

    ``passages=True`` also calibrates experimental paragraph checks; by default this
    calibration is omitted.

    ``jobs`` and ``cache`` are options of the run too, since they change how fast it goes and
    never a number in the profile. ``jobs`` is how many measurement processes run: 0
    (the default) picks one per CPU, up to 4, from 50,000 syntax words or 500,000 surface
    words. Explicit counts are capped by the memory allowance. Workers start only
    where worker processes can start (see ``measure.workers_can_start``: a script needs an
    ``if __name__ == "__main__":`` guard); 1 parses in this process. With ``cache=True``,
    chunks measured by an earlier run are read from the measurement cache and new ones are
    added to it (see ``styleprofile.cache``); the default,
    ``cache=False``, neither reads nor writes it. Keyword overrides change individual
    ``Settings`` fields, including when a settings object is given. ``progress`` is called
    as each phase starts and as each chunk is measured.
    """
    settings = _overridden(settings, overrides)
    notes: list[Note] = []
    with _notes_on_error(notes), _measurer(progress, jobs, cache) as measurer:
        pooling = _pooling(settings)
        step = _progress(progress)
        step(Phase.READ)
        items = _items(inputs)
        contrast_items = _items(contrast) if contrast is not None else None
        _stdin_once([*items, *(contrast_items or [])])
        seen: set[str] = set()
        names = SourceNames()
        texts: dict[bytes, str] = {}
        read = functools.partial(
            _read,
            text_field=settings.text_field,
            seen=seen,
            notes=notes,
            names=names,
            input_format=settings.input_format,
            known_texts=texts,
            group_field=settings.group_field,
        )
        records, contrast_records = Records(), Records()
        chunks = read(items, role="<text>", records=records)
        contrast_chunks = (
            read(contrast_items, role="<contrast>", records=contrast_records, require_groups=False)
            if contrast_items is not None
            else None
        )
        # Duplicates were dropped as the texts were read, before any pooling.
        cut = _chunked(
            chunks,
            settings,
            pooling,
            notes,
            calibrated=True,
            records=records,
            split="calibrate",
        )
        # The contrast set pools when the writer's texts do, so the two stay alike in length.
        # A contrast set in one file is split as the writer's texts are: its AUC resamples
        # and its likeness weights are learned by document.
        contrast_cut = (
            _chunked(
                contrast_chunks,
                settings,
                cut.pooled is not None,
                notes,
                role="contrast ",
                following=True,
                records=contrast_records,
                split="calibrate",
            )
            if contrast_chunks is not None
            else None
        )
        contrast_windows = contrast_cut.windows if contrast_cut is not None else None
        missing = NoteCode.NO_SYNTAX.forms["build_surface"].long
        parser = _parser(settings.syntax, notes, missing, step)
        step(Phase.BUILD)
        report = _or_without_syntax(
            settings.syntax,
            parser,
            notes,
            missing,
            lambda parser: build_reference(
                cut.windows,
                parser=parser,
                top_k=settings.top_k,
                min_words=settings.min_words,
                contrast=contrast_windows,
                contrast_label=contrast_label,
                settings={
                    "inputs": _described(items, names),
                    "contrast": (
                        _described(contrast_items, names) if contrast_items is not None else None
                    ),
                    **settings.to_report(),
                    "pool_used": cut.pooled is not None,
                    "split_used": list(cut.split),
                    "contrast_split_used": list(contrast_cut.split) if contrast_cut else [],
                },
                keep_chunks=keep_chunks,
                passages=passages,
                measurer=measurer,
            ),
        )
        if report["chunk_count"] < 2:

            def usable_windows(size: int) -> int:
                return sum(
                    len(words(" ".join(prose(chunk.text).blocks))) >= settings.min_words
                    for chunk in window(cut.windows, size)
                )

            suggested = max(1, report["word_count"] // 2)
            while suggested > 1 and usable_windows(suggested) < 2:
                suggested //= 2
            fix = "add documents"
            if usable_windows(suggested) >= 2:
                fix += f", or pass a smaller --window-words ({suggested})"
            elif settings.min_words == 1 and report["word_count"] >= 2:
                fix += ", or add paragraph breaks and pass a smaller --window-words (1)"
            else:
                fix += f" with at least {settings.min_words} prose words each"
            raise StyleProfileError(
                "a reference needs at least 2 chunks to compare metrics; " + fix,
                code="reference_needs_chunks",
            )
        # Saved, so ``show`` repeats them: what stand-in documents mean for this reference.
        report["warnings"] += [*cut.warnings, *(contrast_cut.warnings if contrast_cut else ())]
        notes += _thin_reference(report, settings, cut.windows)
        step(Phase.DONE)
        sources = _sources([*chunks, *(contrast_chunks or [])])
        # Keep the profile exactly as it is saved (floats rounded), so scoring it before or
        # after a save and load gives the same numbers.
        _finish(measurer, notes)
        return Profile(json.loads(dumps_report(report)), notes=notes, sources=sources)


def evaluate(
    inputs: Inputs,
    contrast: Inputs,
    edited: Mapping[str, Inputs],
    settings: Settings = DEFAULTS,
    *,
    contrast_label: str = "LLM",
    retrain: bool = False,
    progress: ProgressCallback | None = None,
    jobs: int = AUTO_JOBS,
    cache: bool = False,
) -> Evaluation:
    """Stress-test contrast likeness against edited drafts, as ``styleprofile evaluate``
    does: build a reference with ``contrast`` (the original drafts), then score each set in
    ``edited`` (label to edited copies, matched to their originals by name) with the weights
    learned without their original. ``settings.top_k`` does not apply here. ``jobs`` and
    ``cache`` are as for ``build``.
    """
    notes: list[Note] = []
    with _notes_on_error(notes), _measurer(progress, jobs, cache) as measurer:
        step = _progress(progress)
        step(Phase.READ)
        pooling = _pooling(settings)
        items, contrast_items = _items(inputs), _items(contrast)
        edited_items = {label: _items(value) for label, value in edited.items()}
        every = [*items, *contrast_items, *(item for v in edited_items.values() for item in v)]
        _stdin_once(every)
        seen: set[str] = set()
        names = SourceNames()
        texts: dict[bytes, str] = {}
        read = functools.partial(
            _read,
            text_field=settings.text_field,
            notes=notes,
            input_format=settings.input_format,
            group_field=settings.group_field,
        )
        records, contrast_records = Records(), Records()
        repeats: list[Repeat] = []
        reference_chunks = read(
            items, seen=seen, role="<text>", names=names, known_texts=texts, records=records
        )
        contrast_chunks = read(
            contrast_items,
            seen=seen,
            role="<contrast>",
            names=names,
            known_texts=texts,
            records=contrast_records,
            require_groups=False,
            repeats=repeats,
        )
        edited_chunks: dict[str, list[Chunk]] = {}
        # Each edited set is named on its own: its files carry the originals' names.
        edited_names = {label: SourceNames() for label in edited_items}
        for label, value in edited_items.items():
            # Edits are meant to resemble their originals, so they are not deduplicated.
            edited_chunks[label] = read(
                value,
                seen=set(),
                role=f"<{label}>",
                names=edited_names[label],
                require_groups=False,
            )
            edited_chunks[label] = _edits_of_repeats(
                label, edited_chunks[label], contrast_chunks, repeats, notes
            )
            overlap = {chunk.path for chunk in edited_chunks[label] if chunk.path} & seen
            if overlap:
                count = len(overlap)
                notes.append(
                    NoteCode.EDITED_OVERLAP.note(
                        "0",
                        f"{label}",
                        f"{count:,}",
                        f"{('' if count == 1 else 's')}",
                        f"{', '.join(_typed(value))}",
                    )
                )
        missing = NoteCode.NO_SYNTAX.forms["evaluate_surface"].long
        parser = _parser(settings.syntax, notes, missing, step)
        # The drafts pool as the writer's texts do, and each edited set as its originals did,
        # so an edited window pairs with its original window by name.
        # The originals and edited sets are not split: edits pair with originals by name.
        cut = _chunked(
            reference_chunks,
            settings,
            pooling,
            notes,
            calibrated=True,
            records=records,
            split="calibrate",
        )
        originals = _chunked(
            contrast_chunks,
            settings,
            cut.pooled is not None,
            notes,
            role="original ",
            following=True,
            records=contrast_records,
        )
        edited_windows = {
            label: _chunked(
                chunks, settings, originals.pooled is not None, [], like=originals.pooled
            ).windows
            for label, chunks in edited_chunks.items()
        }
        # An edited set that covers only some texts of a pooled window is compared with
        # that window rebuilt from just those texts.
        covered: dict[str, list[Chunk]] = {}
        if originals.pooled is not None:
            for label, chunks in edited_chunks.items():
                kept = _covered(contrast_chunks, chunks)
                if len(kept) < len(contrast_chunks):
                    covered[label] = pool(
                        kept, settings.window_words, like=originals.pooled
                    ).windows
        step(Phase.EVALUATE)
        report = _or_without_syntax(
            settings.syntax,
            parser,
            notes,
            missing,
            lambda parser: evaluate_rewording(
                cut.windows,
                originals.windows,
                edited_windows,
                covered=covered,
                parser=parser,
                min_words=settings.min_words,
                contrast_label=contrast_label,
                retrain=retrain,
                settings={
                    "inputs": _described(items, names),
                    "contrast": _described(contrast_items, names),
                    "edited": {
                        label: _described(value, edited_names[label])
                        for label, value in edited_items.items()
                    },
                    # top_k shapes only a reference's saved distributions, which this never saves.
                    **{
                        name: value
                        for name, value in settings.to_report().items()
                        if name != "top_k"
                    },
                    "pool_used": cut.pooled is not None,
                    "split_used": list(cut.split),
                },
                measurer=measurer,
            ),
        )
        report["warnings"] += cut.warnings
        step(Phase.DONE)
        loaded = [*reference_chunks, *contrast_chunks]
        loaded += [chunk for chunks in edited_chunks.values() for chunk in chunks]
        _finish(measurer, notes)
        return Evaluation(report, notes=notes, sources=_sources(loaded))
