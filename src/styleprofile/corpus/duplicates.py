"""Detect repeated corpus documents and preserve pairing evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from styleprofile.core import Note, NoteCode
from styleprofile.corpus.ids import _document_label, _duplicate_unit, _grouped
from styleprofile.corpus.types import Repeat
from styleprofile.surface import strip_front_matter, words

if TYPE_CHECKING:
    from styleprofile.corpus.types import Chunk

# Documents shorter than this are never dropped as duplicates: short records ("Thanks!")
# repeat legitimately.
DUPLICATE_MIN_WORDS = 20


def drop_duplicates(
    chunks: Sequence[Chunk],
    seen: dict[bytes, str] | None = None,
    repeats: list[Repeat] | None = None,
) -> tuple[list[Chunk], Note | None]:
    """Keep the first of documents whose text is word-for-word the same.

    Chunks are grouped into documents by ``chunk_document`` (their real file and record, so
    windows of one document stay together and two files that share a saved name stay
    apart), except that JSONL records grouped by a group field are compared one by one.
    Each document's words, lowercased, are compared with front matter, punctuation and
    Markdown markup left out, so a post and its generated HTML page (converted to Markdown)
    match; this reads each document once with one regular expression rather than
    parsing its Markdown, which measuring does later. A file given twice is a different
    check, made when inputs are read.

    ``seen`` maps digests of normalized documents read earlier (another input set) to their
    label,
    and is updated, so a contrast draft that repeats a reference document is dropped too.
    Twins would sit on both sides of held-out calibration and make it look too tight.
    Documents under ``DUPLICATE_MIN_WORDS`` words are always kept. The note names documents
    by their saved sources, never by path. ``repeats``, when given, gathers each dropped
    document with the copy kept, so an edit of a dropped draft can pair with that copy.
    """
    seen = {} if seen is None else seen
    firsts: dict[bytes, Chunk] = {}  # the first chunk of each document kept here, by text
    documents: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        documents.setdefault(_duplicate_unit(chunk), []).append(chunk)
    dropped_documents: set[str] = set()
    dropped: list[tuple[str, str]] = []
    for document, members in documents.items():
        tokens = words(strip_front_matter("\n\n".join(chunk.text for chunk in members)))
        if len(tokens) < DUPLICATE_MIN_WORDS:
            continue
        key = hashlib.blake2b(" ".join(tokens).lower().encode("utf-8")).digest()
        label = _document_label(members[0])
        if key in seen:
            dropped.append((label, seen[key]))
            dropped_documents.add(document)
            if repeats is not None:
                repeats.append(Repeat(members[0], firsts.get(key), seen[key]))
        else:
            seen[key] = label
            firsts[key] = members[0]
    kept = [chunk for chunk in chunks if _duplicate_unit(chunk) not in dropped_documents]
    if not dropped:
        return kept, None
    copy, original = dropped[0]
    count = len(dropped)
    kind = "document"
    if all(_grouped(documents[unit][0]) for unit in dropped_documents):
        kind = "record"
    repeating = f"1 {kind} that repeats" if count == 1 else f"{count:,} {kind}s that repeat"
    note = NoteCode.DUPLICATES.message("7", f"{repeating}", f"{copy}", f"{original}")
    return kept, Note(note, NoteCode.DUPLICATES)
