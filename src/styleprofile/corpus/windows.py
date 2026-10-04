"""Cut and pool prose into measurement windows."""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from styleprofile.corpus.ids import (
    _RECORD,
    _grouped,
    _record,
    bare_key,
    base_id,
    chunk_document,
    pair_key,
)
from styleprofile.corpus.types import Chunk, Pooled
from styleprofile.surface import block_word_count, classify, plain_sentences


def _pack(counts: Sequence[int], window_words: int, glued: Sequence[bool]) -> list[list[int]]:
    """Group consecutive items of ``counts`` prose words into windows of about
    ``window_words``, as indexes. ``glued[i]`` keeps item ``i`` with the item before it.

    A window closes once it reaches ``window_words``, or early rather than grow past one and
    a half windows. A remainder under half a window joins the previous window. No items
    give one empty window.
    """
    limit = window_words * 1.5
    groups: list[list[int]] = []
    current: list[int] = []
    count = 0
    for index, size in enumerate(counts):
        closes_early = count >= window_words / 2 and count + size > limit
        if current and closes_early and not glued[index]:
            groups.append(current)
            current, count = [], 0
        current.append(index)
        count += size
        following_glued = index + 1 < len(counts) and glued[index + 1]
        if count >= window_words and not following_glued:
            groups.append(current)
            current, count = [], 0
    if current and groups and count < window_words / 2:
        groups[-1] += current
    elif current or not groups:
        groups.append(current)
    return groups


def _pieces(text: str, window_words: int) -> list[tuple[str, int]]:
    """``text`` cut into windows (see ``window``), each with its prose word count."""
    blocks = classify(text)
    pieces: list[tuple[str, int, bool, int]] = []
    for index, block in enumerate(blocks):
        count = block_word_count(block)
        split = plain_sentences(block) if count > window_words * 1.5 else None
        if split and len(split) > 1:
            for sentence in split:
                size = block_word_count(replace(block, raw=sentence))
                pieces.append((sentence, size, False, index))
        else:
            pieces.append((block.raw, count, block.continues_list, index))
    counts = [piece[1] for piece in pieces]
    groups = _pack(counts, window_words, [piece[2] for piece in pieces])
    result: list[tuple[str, int]] = []
    for group in groups:
        text_parts: list[str] = []
        previous = None
        for index in group:
            raw, _, _, block_index = pieces[index]
            if text_parts:
                text_parts.append(" " if previous == block_index else "\n\n")
            text_parts.append(raw)
            previous = block_index
        result.append(("".join(text_parts), sum(counts[index] for index in group)))
    return result


def window(chunks: Sequence[Chunk], window_words: int) -> list[Chunk]:
    """Split chunks into roughly ``window_words``-word pieces at blocks or sentence boundaries.

    Plain prose blocks over one and a half windows split at sentence boundaries first.
    Sizes count prose words only (code, URLs and markup excluded). Fenced code stays whole
    and a list's indented continuation stays with its list. A window closes early rather
    than grow past one and a half windows. A remainder under half a window joins the
    previous piece, so no window is shorter than half a window unless its whole chunk is;
    that merge, or a single long block, can make a window longer than one and a half.
    A chunk with no prose stays as one window, so it is counted as skipped.
    """
    return _windows(chunks, [_pieces(chunk.text, window_words) for chunk in chunks])


def _windows(chunks: Sequence[Chunk], pieces: Sequence[list[tuple[str, int]]]) -> list[Chunk]:
    """Each chunk's ``_pieces`` as the windows ``window`` makes of it."""
    return [
        Chunk(
            f"{chunk.id}#w{index}",
            chunk.source,
            text,
            chunk.path,
            chunk.folder,
            chunk_document(chunk),
        )
        for chunk, cut in zip(chunks, pieces, strict=True)
        for index, (text, _) in enumerate(cut, start=1)
    ]


def _pool_key(chunk: Chunk) -> str:
    """What ungrouped texts must share to be joined: a record's file (the records of one
    JSONL file, or stdin), a file's folder input, else the source (``Text`` inputs)."""
    if _record(chunk):
        return chunk.path or chunk.source
    return chunk.folder or chunk.path or chunk.source


def pools_by_default(chunks: Sequence[Chunk], window_words: int) -> bool:
    """The ``"auto"`` pooling rule: pool when the median text has fewer prose words than a
    quarter of a window."""
    if not window_words:
        return False
    return _short([_pieces(chunk.text, window_words) for chunk in chunks], window_words)


def _short(pieces: Sequence[list[tuple[str, int]]], window_words: int) -> bool:
    """``pools_by_default`` for texts already cut into ``_pieces``."""
    sizes = [sum(size for _, size in cut) for cut in pieces]
    return bool(sizes) and statistics.median(sizes) < window_words / 4


@dataclass
class _Run:
    """Texts ``pool`` packs together: a group's records (``document``), consecutive short
    texts from one place, or (``target``) the texts joined into one window of another set."""

    members: list[int]
    document: str | None = None
    target: str | None = None


def pool(
    chunks: Sequence[Chunk],
    window_words: int,
    like: Pooled | None = None,
    *,
    auto: bool = False,
    join: bool = True,
) -> Pooled:
    """Window chunks as ``window`` does, and also join short texts into windows of about
    ``window_words`` prose words.

    - A group's records (see ``Chunk.document``) are joined in order, wherever they are in
      their input, into windows ``thread=t1#w1``, ``thread=t1#w2``, ... of that group.
    - Other texts are joined when they are consecutive, short enough not to need windowing
      on their own, and come from one place (``_pool_key``): the records of one JSONL file,
      the Markdown, text or HTML files of one folder input (in the walk's sorted order), or
      ``Text`` inputs. Each joined window is a document of its own, with the id
      ``first..last`` of its first and last texts.
    - Any other text (a long one, or one with nothing to join) is windowed as ``window``
      would window it.

    Given ``like``, another set's ``Pooled``, ungrouped texts whose ``pair_key`` it joined
    are joined into the same windows, under the same ids, whatever their lengths (texts
    from different files never share one), and other texts are not joined: an edited set
    pools as its originals did.

    With ``auto``, texts that ``pools_by_default`` would not pool are windowed as ``window``
    windows them; with ``join`` False, all texts are. A group's windows are always named by
    the group (``thread=t1#w1``), even a group of one record. ``short`` is always measured.
    """
    pieces = [_pieces(chunk.text, window_words) for chunk in chunks]
    short = _short(pieces, window_words)
    if not join or (auto and not short):
        return Pooled(_windows(chunks, pieces), 0, {}, short)
    runs: list[_Run] = []
    matched: dict[tuple[str, str], _Run] = {}  # by file and the window of ``like`` joined
    groups: dict[str, _Run] = {}
    open_run: _Run | None = None
    targets = _targets(like) if like is not None else {}
    for index, chunk in enumerate(chunks):
        grouped = _grouped(chunk)
        target = None if grouped else _target(targets, pair_key(chunk))
        if target is not None:
            key = (_pool_key(chunk), target)
            if key not in matched:
                matched[key] = _Run([], target=target)
                runs.append(matched[key])
            matched[key].members.append(index)
            open_run = None
        elif grouped:
            document = chunk_document(chunk)
            if document not in groups:
                groups[document] = _Run([], document=document)
                runs.append(groups[document])
            groups[document].members.append(index)
            open_run = None
        elif like is not None or len(pieces[index]) > 1:
            runs.append(_Run([index]))
            open_run = None
        else:
            if open_run is None or _pool_key(chunks[open_run.members[-1]]) != _pool_key(chunk):
                open_run = _Run([])
                runs.append(open_run)
            open_run.members.append(index)
    windows: list[Chunk] = []
    together: set[int] = set()
    joined: dict[str, str] = {}

    def add(window_id: str, first: Chunk, parts: Sequence[str], document: str) -> None:
        text = "\n\n".join(parts)
        joined_from = tuple(parts) if len(parts) > 1 else None
        windows.append(
            Chunk(window_id, first.source, text, first.path, first.folder, document, joined_from)
        )

    for run in runs:
        first = chunks[run.members[0]]
        if run.target is not None:
            parts = [chunks[index].text for index in run.members]
            add(run.target, first, parts, _joined_document(first, run.target))
            together.update(run.members if len(run.members) > 1 else ())
            continue
        units = [(index, text, size) for index in run.members for text, size in pieces[index]]
        packed = _pack([size for _, _, size in units], window_words, [False] * len(units))
        for number, group in enumerate(packed, start=1):
            members = sorted({units[position][0] for position in group})
            parts = [units[position][1] for position in group]
            together.update(members if len(members) > 1 else ())
            if run.document is not None:
                # Named by the group (``thread=t1#r12`` without its line), kept as its document.
                add(f"{base_id(first.id)}#w{number}", first, parts, run.document)
            elif len(members) == 1:
                member = chunks[members[0]]
                number = number if len(run.members) == 1 else 1
                add(f"{member.id}#w{number}", member, parts, chunk_document(member))
            else:
                start = chunks[members[0]]
                window_id = _span_id(start.id, chunks[members[-1]].id)
                add(window_id, start, parts, _joined_document(start, window_id))
                joined.update({pair_key(chunks[member]): window_id for member in members})
    return Pooled(windows, len(together), joined, short)


def _span_id(first: str, last: str) -> str:
    """The id of a window joined from texts ``first`` to ``last``: ``c0012..c0019``, or
    ``2024/a.jsonl:17..23`` when both share a ``file:`` prefix, which is shown once."""
    common = os.path.commonprefix([first, last])
    shared = common[: common.rfind(":") + 1]
    return f"{first}..{last[len(shared) :]}"


def _targets(like: Pooled) -> dict[str, str]:
    """``like.joined``, also by bare id (``bare_key``) both ways where that is unique, so
    records read from a file directly pair with their copies read from a folder."""
    targets = dict(like.joined)
    bare: dict[str, list[str]] = defaultdict(list)
    for key in like.joined:
        if (short := bare_key(key)) is not None:
            bare[short].append(key)
    for short, keys in bare.items():
        if len(keys) == 1 and short not in targets:
            targets[short] = like.joined[keys[0]]
    return targets


def _target(targets: Mapping[str, str], key: str) -> str | None:
    """The window of another set a text joins (see ``_targets``); an edited record read
    from a folder also finds an original read from a file directly, by bare id."""
    found = targets.get(key)
    if found is None and (short := bare_key(key)) is not None:
        found = targets.get(short)
    return found


def _joined_document(first: Chunk, window_id: str) -> str:
    """The document of an ungrouped window joined from several texts: its own, in the place
    its texts came from; a window of records stays a record, for ``pair_key``."""
    separator = _RECORD if _record(first) else "\x1f"
    return f"{_pool_key(first)}{separator}{window_id}"
