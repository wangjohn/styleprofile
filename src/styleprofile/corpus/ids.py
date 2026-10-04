"""Document identities, source names and readable labels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from styleprofile.corpus.types import Chunk


def _document_label(chunk: Chunk) -> str:
    """A document as reports name it: its saved source (never a path), and a record's id."""
    source = chunk.source
    if _grouped(chunk):
        # Grouped record ids are ``thread=t1#r12``: the group, then the line.
        group, _, line = chunk.id.rpartition("#r")
        return f"line {line} ({group}) in {source}"
    if source == "stdin" or source.endswith(base_id(chunk.id)):
        return source
    if not _record(chunk) and _PART_OF_SPLIT.search(base_id(chunk.id)):
        return document_label(source, base_id(chunk.id))  # a part of a split text
    return f"record {base_id(chunk.id)} in {source}"


def _duplicate_unit(chunk: Chunk) -> str:
    """What ``drop_duplicates`` compares: a whole document, except that a grouped JSONL
    record is compared on its own, so a comment posted twice is caught inside or across
    groups."""
    if _grouped(chunk):
        return f"{chunk.path or chunk.source}\x1f{chunk.id}"
    return chunk_document(chunk)


# What a document is. A document is the unit held-out calibration leaves out: every chunk
# of it is scored against a reference built without it. Which one a chunk belongs to is
# ``Chunk.document``, set when it is read and kept through windowing and pooling, and read
# by ``chunk_document``:
#
# - a Markdown, text or HTML file, or stdin, is one document;
# - a JSONL record is one document, unless a group field groups records: then the records
#   of one input (a file, or every file of a folder) with one value in it are one document,
#   and records with no value are one more;
# - an ungrouped window that ``pool`` joins from several short texts (the records of one
#   JSONL file, the files of one folder input, or ``Text`` inputs) is a document of its own;
# - a part of a split text is a document of its own (see ``split``). When the writer's texts
#   are too few documents to calibrate well (``api.Settings.split_on``), Markdown, text and
#   HTML texts among them are split at their headings or rules into parts of at least half
#   a window, ``book.md#3-mud-season``; with fewer than three documents, one with neither
#   is cut into stand-in documents of consecutive windows, ``book.md#3#w1``, whose
#   calibration is less certain. JSONL records are never split: each is already a document.
#
# Windows of a document stay in it. Ids name chunks for people (``post#w2``,
# ``thread=t1#r12``, ``c0012..c0019``) and are never read back to find a document, except
# for a ``Chunk`` made without one, which is placed by its file and its id without part
# suffixes (``document_of``). Part suffixes are ``#wN``, a window, and ``#rN``, a grouped
# record (its line); ``base_id`` strips them, so a report's rows gather by document, and
# ids from outside are escaped so they never end in one (``literal_id``).
_PART_SUFFIX = re.compile(r"(?:#[rw]\d+)+$")
# Document keys: a file's or a hand-made chunk's is ``document_of``; a record's joins its
# file and id with _RECORD; a group's starts with _GROUPED, then its input, field and value.
_RECORD = "\x1d"
_GROUPED = "\x1e"
# How a group of records with no value in the group field is shown in ids.
MISSING_GROUP = "(none)"


def document_of(source: str, chunk_id: str) -> str:
    """The document of a chunk made without one: its file plus its id, without the part
    suffixes (``base_id``). Windows ``post#w3`` share their document; files with the same
    name in different folders stay separate."""
    return f"{source}\x1f{base_id(chunk_id)}"


def base_id(chunk_id: str) -> str:
    """A chunk's id without the ``#wN`` (window) and ``#rN`` (grouped record) suffixes:
    ``thread=t1#r12#w1`` is ``thread=t1``, the group it is part of."""
    return _PART_SUFFIX.sub("", chunk_id)


def chunk_document(chunk: Chunk) -> str:
    """The document ``chunk`` belongs to: its ``document``, set when it was read, else its
    real file (or source) and id (``document_of``). Saved sources are short names, which two
    separately loaded inputs can share, so they never decide it alone."""
    if chunk.document is not None:
        return chunk.document
    return document_of(chunk.path or chunk.source, chunk.id)


def group_id(group_field: str, value: object) -> str:
    """How the group of records with ``value`` in ``group_field`` is shown, ``thread=t1``;
    with no value (missing, null or blank), ``thread=(none)``. Values compare as text, so
    ``1`` and ``"1"`` are one group."""
    shown = MISSING_GROUP if value is None or not str(value).strip() else value
    return f"{group_field}={shown}"


def _grouped(chunk: Chunk) -> bool:
    """Whether ``chunk`` is a JSONL record (or a window of records) grouped by a field."""
    return (chunk.document or "").startswith(_GROUPED)


def is_grouped(chunk: Chunk) -> bool:
    """Whether ``chunk`` is a JSONL record (or a window of records) grouped by a field."""
    return _grouped(chunk)


def cover_key(chunk: Chunk) -> tuple[str, str | None]:
    """What an edited record shares with its original, to compare an edited set that covers
    only some texts with just those originals: a grouped record's group and own id (or
    line), else its ``pair_key``."""
    return (pair_key(chunk), chunk.record if _grouped(chunk) else None)


def bare_key(key: str) -> str | None:
    """A folder record's ``pair_key`` without its file (``a.jsonl:17`` is ``17``), for
    pairing it with a copy read from the file directly; None for other keys."""
    inside, colon, rest = key.partition(":")
    return rest if colon and PurePosixPath(inside).suffix else None


def _record(chunk: Chunk) -> bool:
    """Whether ``chunk`` is a JSONL record, or a window of records."""
    return _grouped(chunk) or _RECORD in (chunk.document or "")


def is_record(chunk: Chunk) -> bool:
    """Whether ``chunk`` is a JSONL record, grouped or not, or a window of records."""
    return _record(chunk)


def pair_key(chunk: Chunk) -> str:
    """What pairs a text with its copy in another set (an edited draft with its original):
    its id without window suffixes. For a file that is its path inside its input; for a
    JSONL record, its id, with its file's path inside a folder input (``2024/a.jsonl:17``);
    for a group's window, the group (``thread=t1``), whichever files its records are in."""
    return base_id(chunk.id)


def paired_as(chunk: Chunk, like: Chunk) -> Chunk | None:
    """``chunk``, an edited copy of a text, renamed to pair with ``like`` instead (by
    ``pair_key``, and ``cover_key`` for a grouped record): the text its original repeats word
    for word. It stays in its own file, and keeps its window suffix and, grouped, its line.
    None when one is a grouped record and the other is not, which never pair."""
    if _grouped(chunk) != _grouped(like):
        return None
    suffix = chunk.id[len(pair_key(chunk)) :]
    new_id = pair_key(like) + suffix
    document = chunk.document
    if _grouped(chunk) and document is not None and like.document is not None:
        # ``<_GROUPED><scope><_GROUPED><field><_GROUPED><value>``: take ``like``'s value.
        scope = document.split(_GROUPED, 3)[:3]
        document = _GROUPED.join([*scope, like.document.split(_GROUPED, 3)[3]])
    elif document is not None and _RECORD in document:
        document = f"{document.partition(_RECORD)[0]}{_RECORD}{base_id(new_id)}"
    record = like.record if _grouped(chunk) else chunk.record
    return replace(chunk, id=new_id, document=document, record=record)


# ``literal_id``'s escaped window suffixes at the end of an id, before a repeated id's line.
_ESCAPED_SUFFIX = re.compile(r"(?:%23[rw]\d+)+(?=(?:@\d+)?$)")

# A number that keeps a saved source unique (``posts (2)/a.md``, ``notes (2).md``).
_SOURCE_NUMBER = re.compile(r" \(\d+\)(?=(?:\.[^/.]*)?(?:/|$))")
# The id suffix of a part of a split text (``split.part_id``): ``#3``, ``#3-mud-season``.
_PART_OF_SPLIT = re.compile(r"#\d+(?:-[\w-]*)?$")


def shown_id(value: str) -> str:
    """An id as it was given: ``literal_id``'s escape undone (``x%23w2`` is ``x#w2``), for
    reports to name documents by. The escape stays in chunk ids."""
    return _ESCAPED_SUFFIX.sub(lambda match: match.group(0).replace("%23", "#"), value)


def document_label(source: str, base: str) -> str:
    """How reports name a document, from its saved source and its id without window
    suffixes (``base_id``): a file or stdin by its source (``posts/2024/a.md``,
    ``notes (2).md``), a JSONL record by its file and id (``comments.jsonl:17``), and a
    ``Text`` by its name, each id as given (``shown_id``). Never an absolute path, since
    sources never are."""
    base = shown_id(base)
    if source.startswith("<"):  # Text inputs: <text>, <contrast>
        return base
    unnumbered = _SOURCE_NUMBER.sub("", source, count=1)
    if unnumbered == base or unnumbered.endswith("/" + base):
        return source
    file_name = PurePosixPath(source).name
    if base.startswith(file_name + ":"):  # a record without an id, named by its line
        return source[: len(source) - len(file_name)] + base
    part = _PART_OF_SPLIT.search(base)
    if part and document_label(source, base[: part.start()]) == source:
        return source + part.group(0)  # a part of a split file: ``book.md#3-mud-season``
    return f"{source}:{base}"


def _document_names(keys: Sequence[tuple[str, str]]) -> list[str]:
    """``document_label`` for each document (source, base id), in order; a label an earlier
    document already has (only possible for hand-made chunks) gets a number, ``x (2)``."""
    names: list[str] = []
    taken: set[str] = set()
    for source, base in keys:
        label = name = document_label(source, base)
        number = 1
        while name in taken:
            number += 1
            name = f"{label} ({number})"
        taken.add(name)
        names.append(name)
    return names
