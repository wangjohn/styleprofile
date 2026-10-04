"""Prepare library inputs: records, splitting, pooling and edited copies."""

from __future__ import annotations

import dataclasses
import os
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from styleprofile.calibration import MIN_CALIBRATION_DOCUMENTS
from styleprofile.core import Note, NoteCode, StyleProfileError
from styleprofile.corpus.duplicates import drop_duplicates
from styleprofile.corpus.ids import (
    bare_key,
    base_id,
    chunk_document,
    cover_key,
    document_label,
    group_id,
    is_grouped,
    is_record,
    paired_as,
)
from styleprofile.corpus.reading import Records, expand_path, literal_id, load_chunks
from styleprofile.corpus.types import Chunk, Repeat, Text
from styleprofile.corpus.windows import pool, window
from styleprofile.runtime import _plural
from styleprofile.settings import (
    AUTO,
    DEFAULT_WINDOW_WORDS,
    ENOUGH_CHUNKS,
    ENOUGH_DOCUMENTS,
    STDIN_SHOWN,
    _invalid,
)
from styleprofile.split import (
    ASKED_MIN_PARTS,
    HEADING,
    MIN_PARTS,
    NONE,
    STAND_IN,
    part_id,
    plan_split,
    stand_in_groups,
)
from styleprofile.surface import paragraph_metrics_missing, prose, unlikely_english, words
from styleprofile.weighting import FLOOR_DOCUMENTS

if TYPE_CHECKING:
    from styleprofile.corpus.reading import SourceNames
    from styleprofile.corpus.types import Pooled
    from styleprofile.settings import Input, Inputs, Settings

_K = TypeVar("_K")
_V = TypeVar("_V")


def _is_input(value: object) -> bool:
    return isinstance(value, str | os.PathLike | Text | Chunk)


def _items(inputs: Inputs) -> list[Input]:
    """One input or several, as a list; refuses anything that is not an input."""
    if isinstance(inputs, bytes | bytearray):
        raise TypeError(
            "bytes are not an input: pass a path as str or Path, or decoded text as Text(...)"
        )
    if isinstance(inputs, str | os.PathLike | Text | Chunk):
        return [inputs]
    items = list(inputs)
    for item in items:
        if not _is_input(item):
            raise TypeError(
                f"expected a path (str or Path), Text or Chunk, not {type(item).__name__}"
            )
    return items


def _described(items: Sequence[Input], names: SourceNames) -> list[str]:
    """How inputs are recorded in a report's settings: paths by the name their sources were
    saved under (``posts``, ``posts (2)``; see ``load_chunks``), never as paths; texts and
    chunks by name. A path that gave no sources of its own (all its files came from an
    earlier input, or it was given twice) is left out."""

    described: list[str] = []
    listed: set[str] = set()
    for item in items:
        if isinstance(item, Text):
            described.append(item.name or "<text>")
        elif isinstance(item, Chunk):
            described.append(item.id)
        else:
            value = os.fspath(item)
            # An input whose files all came from an earlier one, or the same path given
            # again, gave no sources of its own, so it is not listed.
            if value in listed or (value != "-" and value not in names.roots):
                continue
            listed.add(value)
            described.append(names.roots[value] if value != "-" else "stdin")
    return described


def _typed(items: Sequence[Input]) -> list[str]:
    """Inputs as the user gave them, for notes (which are shown, never saved)."""
    return [os.fspath(item) if isinstance(item, str | os.PathLike) else "<text>" for item in items]


def _sources(chunks: Sequence[Chunk]) -> list[str]:
    """Where each chunk was read from: its file when it has one, else its source."""
    return [chunk.path or chunk.source for chunk in chunks]


def _stdin_once(items: Sequence[Input]) -> None:
    if sum(isinstance(item, str | os.PathLike) and os.fspath(item) == "-" for item in items) > 1:
        raise StyleProfileError("- (stdin) can be given only once", code="stdin_twice")


def path_exists(value: str) -> bool:
    """Whether ``value`` names an existing path; False rather than an error for one that
    cannot (too long, a NUL character, or ``~user`` for an unknown user)."""
    try:
        return expand_path(value).exists()
    except (OSError, ValueError):  # StyleProfileError is a ValueError
        return False


def require_path(value: str, *, suggest_text: bool = True) -> None:
    """Refuse a path input that does not exist (``"-"``, stdin, always does). With
    ``suggest_text``, a value that reads like text points to ``Text``; a command line, whose
    arguments are paths by construction, passes False."""
    if value != "-" and not path_exists(value):
        raise _missing(value, suggest_text)


def _missing(value: str, suggest_text: bool) -> StyleProfileError:
    """An error for a path that does not exist, pointing to ``Text`` when the value reads
    like text: it has whitespace, or no folder separator and no file suffix."""
    path = Path(value)
    looks_like_text = any(char.isspace() for char in value) or (
        os.sep not in value and "/" not in value and not path.suffix
    )
    if not (suggest_text and looks_like_text):
        return StyleProfileError(f"{value} not found", code="input_not_found")
    shown = value.strip().split("\n")[0]
    shown = shown if len(shown) <= 40 else shown[:40] + "..."
    return StyleProfileError(
        f"{shown!r} not found; a str input is a path, so use build_texts(...) or "
        "Profile.score_text(...) for raw text, or Text(...)",
        code="input_not_found",
    )


def _text_chunks(texts: Sequence[Text], role: str) -> Iterator[Chunk]:
    """Chunks for ``Text`` inputs in one role, named as ``Text`` says."""
    taken: set[str] = set()
    for text in texts:
        if text.name is None:
            continue
        if text.name in taken:
            raise StyleProfileError(
                f"two texts are named {text.name!r}, so they would be read as one document; "
                "give each Text a distinct name",
                code="duplicate_names",
            )
        taken.add(text.name)
    number = 0
    for text in texts:
        name = text.name
        if name is None:
            number += 1
            while f"text{number}" in taken:
                number += 1
            name = f"text{number}"
        yield Chunk(literal_id(name), role, text.text)


def _read(
    items: Sequence[Input],
    text_field: str | Sequence[str] | None,
    seen: set[str],
    notes: list[Note],
    role: str,
    names: SourceNames,
    input_format: str = AUTO,
    known_texts: dict[bytes, str] | None = None,
    group_field: str | None = None,
    require_groups: bool = True,
    records: Records | None = None,
    repeats: list[Repeat] | None = None,
) -> list[Chunk]:
    """Chunks from each input, in order, skipping files an earlier path (tracked in
    ``seen`` by real path) already gave. ``Text`` inputs get ``role`` as their source, so
    texts in different roles (writer, contrast) are never the same document. ``names`` is
    shared across calls, so two inputs' files never get the same saved source.

    With ``known_texts`` (see ``drop_duplicates``), documents whose text repeats one read
    before, here or in an earlier role, are dropped with a note: twins on both sides of
    held-out calibration would make it look too tight. That is a separate check from a file
    given twice, which is recognized by its path. Grouped JSONL records are compared one by
    one, before they are pooled. ``repeats`` gathers the documents dropped, with the copies
    kept (``drop_duplicates``).

    With ``group_field``, JSONL records without a value in it are noted. An input where no
    record has one is an error naming the fields the records do have (the field is likely
    misspelled); with ``require_groups`` False (drafts, which rarely carry the writer's
    threads), it is read ungrouped instead, each record a document of its own, with a note.
    A group value found in several inputs typed one by one is noted too, since each input's
    records are separate documents. ``records`` gathers what the JSONL records held, for the
    pooling note."""
    texts = iter(_text_chunks([item for item in items if isinstance(item, Text)], role))
    chunks: list[Chunk] = []
    inputs_with: Counter[str] = Counter()  # how many inputs each group value is found in
    for item in items:
        if isinstance(item, Chunk):
            chunks.append(item)
            continue
        if isinstance(item, Text):
            chunks.append(next(texts))
            continue
        value = os.fspath(item)
        require_path(value)
        contributed = value in names.roots
        found = Records(group_field)
        loaded = load_chunks(
            [value],
            text_field,
            names=names,
            input_format=input_format,
            notes=notes,
            records=found,
        )
        if group_field is not None and _check_groups(
            value, found, group_field, notes, require_groups
        ):
            loaded = [
                found.ungrouped.get((chunk.path or chunk.source, chunk.id), chunk)
                for chunk in loaded
            ]
            found.missing = 0
            found.group_field = None
        elif group_field is not None:
            inputs_with.update(found.values.get(group_field, set()))
        if records is not None:
            records.add(found)
        files = {chunk.path for chunk in loaded if chunk.path is not None}
        repeated = files & seen
        if repeated and repeated == files:
            notes.append(NoteCode.REPEATED_INPUT.note("0", f"{value}"))
            if not contributed:
                # Every file came from an earlier input, so no source carries this name:
                # leave it out of the report's settings (``_described``).
                names.roots.pop(value, None)
        elif repeated:
            notes.append(
                NoteCode.REPEATED_INPUT.note("1", f"{_plural(len(repeated), 'file')}", f"{value}")
            )
        chunks += [chunk for chunk in loaded if chunk.path not in repeated]
        seen |= files
    shared = sorted(group for group, count in inputs_with.items() if count > 1)
    if shared:
        notes.append(NoteCode.GROUPING.note("0", f"{group_field}", f"{shared[0]!r}"))
    if known_texts is not None:
        chunks, note = drop_duplicates(chunks, known_texts, repeats)
        if note:
            notes.append(note)
    return chunks


def _locations(items: Sequence[Input], names: SourceNames) -> dict[str, str]:
    """Each saved source's path as its input was typed: ``drafts/2024/a.md`` for the saved
    ``2024/a.md`` inside ``drafts``, for the command line to name files the user can open.
    Nothing here is saved."""
    # Standard input is named apart from a file called ``stdin``.
    locations: dict[str, str] = {"stdin": STDIN_SHOWN}
    for item in items:
        if isinstance(item, Text | Chunk):
            continue
        value = os.fspath(item)
        root = names.roots.get(value)
        if value == "-" or root is None:
            continue
        for source, file in names.files.items():
            if source == root:
                locations[source] = value
            elif source.startswith(f"{root}/") or not root:
                inner = source[len(root) + 1 :] if root else source
                if Path(file).is_relative_to(expand_path(value).resolve()):
                    locations[source] = os.path.join(value, inner)
    return locations


def _check_groups(
    value: str, found: Records, group_field: str, notes: list[Note], required: bool
) -> bool:
    """Note the records of one input with no value in ``group_field``. When none has one,
    refuse the input if ``required``, naming the fields its records have; otherwise note it
    and return True: the caller reads it ungrouped."""
    if not found.missing:
        return False
    if found.missing == found.count:
        if required:
            fields = ", ".join(sorted(found.values)) or "none besides the text"
            raise StyleProfileError(
                f"{value}: no record has a value in the group field {group_field!r}; its "
                f"records have the fields {fields}",
                code="group_field",
            )
        notes.append(NoteCode.MISSING_GROUP.note("1", f"{value}", f"{group_field}"))
        return True
    notes.append(
        NoteCode.MISSING_GROUP.note(
            "0",
            f"{value}",
            f"{found.missing:,}",
            f"{_plural(found.count, 'record')}",
            f"{group_field}",
            f"{group_id(group_field, None)}",
        )
    )
    return False


def _pooling(settings: Settings) -> Literal["auto"] | bool:
    """Whether ``settings`` pool texts: ``"auto"`` leaves it to their median length. Called
    before any input is read, so an impossible setting fails first."""
    if not settings.window_words:
        if settings.pool is True:
            _invalid(
                "window_words", "window_words must be above 0 to pool texts into windows of it"
            )
        return False
    return settings.pool


@dataclass(frozen=True)
class _Cut:
    """Texts cut for measuring: the windows; how they were pooled (None when not); and
    whether the median text is under a quarter of a window."""

    windows: list[Chunk]
    pooled: Pooled | None
    short: bool
    # How texts were split into documents, as ``split_used`` records it.
    split: tuple[str, ...] = ()
    # What stand-in documents mean for the report, one warning per text cut into them.
    warnings: tuple[str, ...] = ()


def _noun(chunks: Sequence[Chunk], records: Records | None) -> str:
    """What the texts are, for notes: records, files of a folder, or texts."""
    if records is not None and chunks and records.count >= len(chunks):
        return "record"
    if chunks and all(chunk.folder for chunk in chunks):
        return "file"
    return "text"


def _chunked(
    chunks: list[Chunk],
    settings: Settings,
    pooled: Literal["auto"] | bool,
    notes: list[Note],
    *,
    role: str = "",
    calibrated: bool = False,
    following: bool = False,
    like: Pooled | None = None,
    records: Records | None = None,
    split: Literal["calibrate", "score"] | None = None,
) -> _Cut:
    """The chunks to measure: split into documents as ``split`` says (``_split``; None
    splits nothing), then windowed or pooled (``_pooled``), and last, texts left whole that
    ``_split`` marked are cut into stand-in documents (``_stand_ins``)."""
    for chunk in chunks:
        parsed = prose(chunk.text)
        if unlikely_english(parsed):
            notes.append(NoteCode.NON_ENGLISH.note("0", f"{_label(chunk)}"))
    stand_ins: set[str] = set()
    used: set[str] = set()
    if split is not None and settings.split_on != NONE:
        chunks, stand_ins, used = _split(chunks, settings, notes, role, split == "calibrate")
    cut = _pooled(
        chunks,
        settings,
        pooled,
        notes,
        role=role,
        calibrated=calibrated,
        following=following,
        like=like,
        records=records,
    )
    windows, warnings = cut.windows, ()
    for chunk in windows:
        parsed = prose(chunk.text)
        size = len(words(parsed.text))
        if paragraph_metrics_missing(parsed, size):
            notes.append(NoteCode.NO_PARAGRAPH_BREAKS.note("0", f"{chunk.id}"))
        if settings.window_words and size > 2 * settings.window_words:
            notes.append(
                NoteCode.OVERSIZE_CHUNK.note(
                    "0",
                    f"{chunk.id}",
                    f"{size:,}",
                    f"{settings.window_words:,}",
                    setting="window_words",
                )
            )
    if stand_ins:
        windows, warnings = _stand_ins(windows, stand_ins, notes, role)
        if warnings:
            used.add(STAND_IN)
    return dataclasses.replace(
        cut, windows=windows, split=tuple(sorted(used)), warnings=tuple(warnings)
    )


def _split(
    chunks: list[Chunk], settings: Settings, notes: list[Note], role: str, calibrating: bool
) -> tuple[list[Chunk], set[str], set[str]]:
    """``chunks`` with Markdown, text or HTML texts split into documents at their headings
    or rules as ``settings.split_on`` asks (``split.plan_split``; JSONL records are never
    split, and a ``.txt`` file's chapter lines are headings), each split noted and parts
    that repeat another word for word dropped; the documents left whole that may be cut
    into stand-ins; and the kinds of split made.

    ``"auto"`` splits only when ``calibrating`` (the writer's texts, or the contrast set);
    drafts never need a split to be judged. Then:

    - with fewer than ``MIN_CALIBRATION_DOCUMENTS`` documents, too few to calibrate at all,
      it splits every text it can into at least ``split.MIN_PARTS`` parts, and leaves the
      rest for stand-ins;
    - with fewer than ``FLOOR_DOCUMENTS`` documents (several manuscripts), it splits every
      text its markers divide into two parts or more of a median of a whole window (a
      manuscript's chapters, not a blog post's sections), never into stand-ins: below that many
      documents each is worth only a few independent calibration pieces (their similarity
      is floored), so chapters as documents calibrate more, and more tightly, than their
      books. This is for the writer's texts; the contrast set is left as it is;
    - with more, it splits nothing.

    ``"heading"``, ``"heading:N"`` or ``"rule"`` split any text their markers divide into
    two parts or more, and note those they found nothing to split at. Parts are at least half
    a window (of ``DEFAULT_WINDOW_WORDS`` when windowing is off)."""
    mode = settings.split_on
    automatic = mode == AUTO
    count = len(set(map(chunk_document, chunks)))
    few = count < MIN_CALIBRATION_DOCUMENTS
    if automatic and (not calibrating or count >= FLOOR_DOCUMENTS or (role and not few)):
        return chunks, set(), set()
    size = settings.window_words or DEFAULT_WINDOW_WORDS
    minimum = MIN_PARTS if automatic and few else ASKED_MIN_PARTS
    # Between the bands, a text's parts must hold a whole window each (a median): a
    # manuscript's chapters split, a blog post's sections do not.
    median = 1.0 if automatic and not few else 0.5
    plans = [
        None
        if is_record(chunk)
        else plan_split(chunk.text, size, mode, minimum, plain=_plain(chunk), median_windows=median)
        for chunk in chunks
    ]
    # With a few documents, each is worth only a few independent calibration pieces, so the
    # texts split at their structure whenever it gives more documents.
    reason = (
        NoteCode.SPLIT.message("reason", _plural(count, "document"))
        if automatic and not few
        else ""
    )
    out: list[Chunk] = []
    whole: list[Chunk] = []
    used: set[str] = set()
    done: list[str] = []
    parts = 0
    for chunk, plan in zip(chunks, plans, strict=True):
        if plan is None:
            out.append(chunk)
            whole += [] if is_record(chunk) else [chunk]
            continue
        document = chunk_document(chunk)
        for number, (part, text) in enumerate(zip(plan.parts, plan.texts, strict=True), 1):
            out.append(
                Chunk(
                    part_id(chunk.id, number, part.title),
                    chunk.source,
                    text,
                    chunk.path,
                    chunk.folder,
                    f"{document}{_PART}{number}",
                )
            )
        used.add(plan.kind)
        parts += len(plan.parts)
        done.append(
            NoteCode.SPLIT.message(
                "1",
                f"{role}",
                f"{_label(chunk)}",
                f"{_plural(len(plan.parts), 'document')}",
                f"{plan.describe()}",
            )
        )
    if len(done) > NOTED_SPLITS or (reason and len(done) > 1):
        where = "headings or rules" if len(used) > 1 else f"{next(iter(used))}s"
        done = [NoteCode.SPLIT.message("many", f"{len(done):,}", role, f"{parts:,}", where)]
    if reason and done:
        done = [done[0] + reason]
    notes += [Note(message, NoteCode.SPLIT, setting="split_on") for message in done]
    if done:
        # A newsletter issue pasted twice is two parts with one text.
        out, repeated = drop_duplicates(out)
        notes += [repeated] if repeated else []
    if not automatic and whole:
        level = mode.partition(":")[2]
        markers = f"level-{level} headings" if level else "headings" if mode == HEADING else "rules"
        if len(whole) == 1:
            message = NoteCode.SPLIT.message("2", f"{role}", f"{_label(whole[0])}", f"{markers}")
            kept = NoteCode.SPLIT.message("3")
        else:
            texts = len(chunks) - sum(map(is_record, chunks))
            message = NoteCode.SPLIT.message(
                "4", f"{len(whole):,}", f"{_plural(texts, role + 'text')}", f"{markers}"
            )
            message += NoteCode.SPLIT.message("5")
            kept = NoteCode.SPLIT.message("6")
        notes.append(NoteCode.SPLIT.note("0", f"{message}", f"{kept}", setting="split_on"))
    stand_ins = {chunk_document(chunk) for chunk in whole} if automatic and few else set()
    return out, stand_ins, used


# Joins a split text's document to its part's number (see ``Chunk.document``).
_PART = "\x1c"
# More texts split than this are noted together.
NOTED_SPLITS = 3


def _plain(chunk: Chunk) -> bool:
    """Whether ``chunk`` is a plain-text file, whose chapter lines are headings."""
    return chunk.path is not None and Path(chunk.path).suffix.lower() == ".txt"


def _label(chunk: Chunk) -> str:
    """A text as notes name it: its saved source (a file), or a ``Text``'s name."""
    return document_label(chunk.source, base_id(chunk.id))


def _stand_ins(
    windows: list[Chunk], documents: set[str], notes: list[Note], role: str
) -> tuple[list[Chunk], list[str]]:
    """``windows`` with those of each of ``documents`` that has ``split.MIN_PARTS`` windows
    or more grouped, in order, into stand-in documents (``split.stand_in_groups``), named
    ``book.md#3#w1``, each text with a note saying so; and, for the report, a warning on what
    that means for each (the same list and none when there are none)."""
    positions: dict[str, list[int]] = {}
    for index, chunk in enumerate(windows):
        document = chunk_document(chunk)
        if document in documents:
            positions.setdefault(document, []).append(index)
    cut = [(document, found) for document, found in positions.items() if len(found) >= MIN_PARTS]
    if not cut:
        return windows, []
    windows = list(windows)
    warnings: list[str] = []
    for document, found in cut:
        groups = stand_in_groups(len(found))
        first = windows[found[0]]
        for number, run in enumerate(groups, 1):
            for place, position in enumerate(run, 1):
                chunk = windows[found[position]]
                windows[found[position]] = Chunk(
                    f"{part_id(base_id(chunk.id), number)}#w{place}",
                    chunk.source,
                    chunk.text,
                    chunk.path,
                    chunk.folder,
                    f"{document}{_PART}{number}",
                )
        label = _label(first)
        grouped = (
            NoteCode.STAND_INS.message("windows", f"{len(found):,}")
            if len(groups) == len(found)
            else NoteCode.STAND_INS.message("groups", f"{len(found):,}", str(len(groups)))
        )
        notes.append(
            NoteCode.STAND_INS.note(
                "0", f"{role}", f"{label}", f"{MIN_PARTS}", f"{grouped}", setting="split_on"
            )
        )
        if role:
            warnings.append(
                NoteCode.CONTRAST_STAND_INS.message("0", f"{role}", f"{len(groups)}", f"{label}")
            )
        else:
            warnings.append(
                NoteCode.CALIBRATION_STAND_INS.message("0", f"{len(groups)}", f"{label}")
            )
    return windows, warnings


def _pooled(
    chunks: list[Chunk],
    settings: Settings,
    pooled: Literal["auto"] | bool,
    notes: list[Note],
    *,
    role: str = "",
    calibrated: bool = False,
    following: bool = False,
    like: Pooled | None = None,
    records: Records | None = None,
) -> _Cut:
    """The chunks to measure: windowed, or, when ``pooled`` and pooling joins any texts,
    pooled (see ``profile.pool``) with a note saying so.

    ``"auto"`` pools only when the median text is short and pooling still leaves
    ``ENOUGH_CHUNKS`` windows: a handful of short texts joined into one or two windows would
    lose held-out calibration altogether. A set that pools because another did
    (``following``: the contrast drafts) pools only if it keeps ``ENOUGH_DOCUMENTS``
    documents. ``role`` (``"contrast "``) names the set in the note; ``calibrated`` marks
    the writer's texts, whose notes say why they pooled or did not, what that means for
    held-out calibration, and what is wrong with a group field that cannot work. Given
    ``like``, the chunks pool as that set did, without a note."""
    if not settings.window_words:
        return _Cut(chunks, None, False)
    noun = _noun(chunks, records)
    if calibrated and settings.group_field:
        notes += _grouping_notes(chunks, settings, pooled)
    joined = pool(chunks, settings.window_words, like, auto=pooled == AUTO, join=bool(pooled))
    if like is not None:
        return _Cut(joined.windows, joined, joined.short)
    if not pooled or not joined.together:
        # Not pooled, or nothing to join (one text, or long ones): windows as ``window``
        # makes them, and no note.
        return _Cut(joined.windows, None, joined.short)
    if pooled == AUTO and len(joined.windows) < ENOUGH_CHUNKS:
        if calibrated:
            notes.append(
                NoteCode.POOLED.note(
                    "0", f"{noun}", f"{_plural(len(joined.windows), 'window')}", setting="pool"
                )
            )
        return _Cut(_windowed(chunks, settings.window_words), None, joined.short)
    if following and len(set(map(chunk_document, joined.windows))) < min(
        ENOUGH_DOCUMENTS, len(set(map(chunk_document, chunks)))
    ):
        return _Cut(_windowed(chunks, settings.window_words), None, joined.short)
    count = len(chunks)
    texts = (
        _plural(count, role + noun)
        if joined.together == count
        else f"{joined.together:,} of {_plural(count, role + noun)}"
    )
    message = NoteCode.POOLED.message(
        "1", f"{texts}", f"{_plural(len(joined.windows), 'window')}", f"{settings.window_words:,}"
    )
    setting = None
    if settings.pool == AUTO and calibrated:
        message += NoteCode.POOLED.message("2", f"{noun}")
    grouped = settings.group_field is not None and any(map(is_grouped, chunks))
    if grouped:
        message += NoteCode.POOLED.message("3", f"{settings.group_field}")
    elif calibrated:
        message += NoteCode.POOLED.message("4", f"{noun}")
        candidates = records.group_fields() if records is not None else []
        if candidates:
            shown = ", ".join(f"{name} ({values:,} values)" for name, values in candidates[:3])
            first = candidates[0][0]
            message += NoteCode.POOLED.message("5", f"{shown}", f"{first}")
        setting = "group_field"
    notes.append(Note(message, NoteCode.POOLED, setting=setting))
    return _Cut(joined.windows, joined, joined.short)


def _grouping_notes(
    chunks: Sequence[Chunk], settings: Settings, pooled: Literal["auto"] | bool
) -> list[Note]:
    """What is wrong with a group field that groups the writer's records into groups too
    small to pool. One group, which leaves nothing to hold out, is a thin reference
    (``_thin_reference``)."""
    field = settings.group_field
    sizes: Counter[str] = Counter()
    for chunk in chunks:
        sizes[chunk_document(chunk)] += len(words(chunk.text))
    median = statistics.median(sizes.values()) if sizes else 0
    if pooled and len(sizes) > 1 and median < settings.window_words / 4:
        return [NoteCode.GROUPING.note("1", f"{field}", f"{median:,.0f}", setting="group_field")]
    return []


def _covered(originals: Sequence[Chunk], edited: Sequence[Chunk]) -> list[Chunk]:
    """The original texts an edited set has copies of (``cover_key``: a record by its id, a
    grouped record by its group and own id), also when one set was read from a folder and
    the other from a file given directly (``bare_key``)."""
    keys: set[tuple[str, str | None]] = set()
    for chunk in edited:
        key, record = cover_key(chunk)
        keys.add((key, record))
        if (short := bare_key(key)) is not None:
            keys.add((short, record))

    def covers(chunk: Chunk) -> bool:
        key, record = cover_key(chunk)
        short = bare_key(key)
        return (key, record) in keys or (short is not None and (short, record) in keys)

    return [chunk for chunk in originals if covers(chunk)]


def _cover_keys(chunk: Chunk) -> list[tuple[str, str | None]]:
    """The ``cover_key`` of ``chunk``, then the same without its folder file (``bare_key``),
    so a record read from a folder and one read from its file directly find each other."""
    key, record = cover_key(chunk)
    short = bare_key(key)
    return [(key, record)] if short is None else [(key, record), (short, record)]


def _shown(chunk: Chunk) -> str:
    """A draft as edit notes name it: its id, and for a grouped record also its own id."""
    key, record = cover_key(chunk)
    return f"{record} in {key}" if record is not None else key


def _unique(pairs: Iterable[tuple[_K, _V]]) -> dict[_K, _V | None]:
    """``pairs`` as a mapping, with None for a key given different values."""
    found: dict[_K, _V | None] = {}
    for key, value in pairs:
        found[key] = value if found.get(key, value) == value else None
    return found


def _edits_of_repeats(
    label: str,
    edited: Sequence[Chunk],
    originals: Sequence[Chunk],
    repeats: Sequence[Repeat],
    notes: list[Note],
) -> list[Chunk]:
    """``edited`` with each edit of a draft dropped as a word-for-word repeat of another
    (``drop_duplicates``) renamed to pair with the copy kept: they had the same text, so
    the edit is an edit of that copy too. Each is noted.

    Only one edit of a text is scored. When the kept copy is edited as well, its own edit
    is used; otherwise the first edit in the set, and the others are left out with a note.
    An edit of a draft dropped for repeating one of the writer's texts has no draft to pair
    with, so it is left out with a note too.
    """
    if not repeats:
        return list(edited)
    # Cover keys (and their bare forms) of the drafts kept, and of those dropped. A bare key
    # that stands for several drafts pairs with none of them.
    kept = _unique((key, cover_key(chunk)) for chunk in originals for key in _cover_keys(chunk))
    dropped = _unique((key, repeat) for repeat in repeats for key in _cover_keys(repeat.dropped))

    def original(chunk: Chunk) -> tuple[str, str | None] | Repeat | None:
        """The kept draft ``chunk`` is an edit of (its cover key), or the repeat it edits."""
        for key in _cover_keys(chunk):
            if kept.get(key) is not None:
                return kept[key]
            if dropped.get(key) is not None:
                return dropped[key]
        return None

    used = {found for chunk in edited if isinstance(found := original(chunk), tuple)}
    decided: dict[tuple[str, str | None], Chunk | None] = {}  # by the edit's cover key
    paired: list[tuple[str, str]] = []
    doubled: list[tuple[str, str]] = []
    outside: list[tuple[str, str]] = []
    as_is: set[tuple[str, str | None]] = set()
    result: list[Chunk] = []
    for chunk in edited:
        found = original(chunk)
        if not isinstance(found, Repeat):
            result.append(chunk)
            continue
        own = cover_key(chunk)
        if own not in decided:
            if found.kept is None:
                decided[own] = None
                outside.append((_shown(chunk), found.label))
            elif paired_as(chunk, found.kept) is None:  # grouped and ungrouped never pair
                decided[own] = None
                as_is.add(own)
            elif cover_key(found.kept) in used:
                decided[own] = None
                doubled.append((_shown(chunk), _shown(found.kept)))
            else:
                decided[own] = found.kept
                used.add(cover_key(found.kept))
                paired.append((_shown(chunk), _shown(found.kept)))
        like = decided[own]
        if own in as_is:
            result.append(chunk)
        elif like is not None:
            result.append(paired_as(chunk, like) or chunk)

    def note(message: str) -> None:
        notes.append(NoteCode.DUPLICATES.note("0", f"{label}", f"{message}"))

    if len(paired) == 1:
        ((edit, copy),) = paired
        note(
            NoteCode.DUPLICATES.message(
                "1", f"{edit!r}", f"{copy!r}", f"{edit!r}", f"{copy!r}", f"{copy!r}"
            )
        )
    elif paired:
        edit, copy = paired[0]
        note(NoteCode.DUPLICATES.message("4", f"{len(paired):,}", f"{edit!r}", f"{copy!r}"))
    if len(doubled) == 1:
        ((edit, copy),) = doubled
        note(NoteCode.DUPLICATES.message("2", f"{edit!r}", f"{copy!r}", f"{copy!r}"))
    elif doubled:
        edit, copy = doubled[0]
        note(NoteCode.DUPLICATES.message("5", f"{len(doubled):,}", f"{edit!r}", f"{copy!r}"))
    if len(outside) == 1:
        ((edit, text),) = outside
        note(NoteCode.DUPLICATES.message("3", f"{edit!r}", f"{text}"))
    elif outside:
        edit, text = outside[0]
        note(NoteCode.DUPLICATES.message("6", f"{len(outside):,}", f"{edit!r}", f"{text}"))
    return result


def _pooling_mismatch(cut: _Cut, reference_pooled: bool) -> list[str]:
    """A warning for texts pooled into windows against a reference that was not pooled:
    they read closer to it than they are. The other way round needs none: each short text is
    judged at its own length against the reference's length calibration, or abstains."""
    if cut.pooled is not None and not reference_pooled:
        return [NoteCode.POOLING_MISMATCH.message("0")]
    return []


def _windowed(chunks: list[Chunk], window_words: int) -> list[Chunk]:
    return window(chunks, window_words) if window_words else chunks
