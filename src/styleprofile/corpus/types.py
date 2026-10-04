"""Text and chunk values shared by reading and measurement."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A piece of text to profile.

    ``source`` names where it came from as saved in reports: its input's name plus its path
    inside it (see ``load_chunks``), never an absolute path. ``path`` is the file it was read
    from, which identifies its document and recognizes a file given twice; it is never
    saved. ``folder`` is the input, as typed, of a file (or a JSONL file's records) found
    in a folder: ``pool`` joins short files only within one folder input. ``document`` is the
    document it belongs to (see ``chunk_document``), set when it is read and kept through
    windowing and pooling; a chunk made without one is placed by its file and id.
    ``parts`` are the texts a pooled window was joined from (whole records or files, or
    pieces of a long one), in order, so length calibration cuts its pieces where the texts
    people score begin and end (``calibration.plan_pieces``). ``record`` is a grouped JSONL
    record's own id (or line), which its id leaves out, so an edited copy of some of a
    group's records can be compared with just those originals.
    """

    id: str
    source: str
    text: str
    path: str | None = field(default=None, compare=False, repr=False)
    folder: str | None = field(default=None, compare=False, repr=False)
    document: str | None = field(default=None, compare=False, repr=False)
    parts: tuple[str, ...] | None = field(default=None, compare=False, repr=False)
    record: str | None = field(default=None, compare=False, repr=False)
    # Converted from HTML, so line numbers are the Markdown conversion's (``drift``).
    converted: bool = field(default=False, compare=False, repr=False)


@dataclass(frozen=True)
class Repeat:
    """A document ``drop_duplicates`` dropped: its first chunk, ``dropped``; the first chunk
    of the copy it kept, ``kept``, or None when that copy was read before (another input
    set, whose document ``label`` names)."""

    dropped: Chunk
    kept: Chunk | None
    label: str


@dataclass(frozen=True)
class Pooled:
    """What ``pool`` made: the windows; how many texts share a window with another text;
    for each ungrouped text joined with others, the id of its window (keyed by the text's
    ``pair_key``), so another set can be pooled alike; and whether the median text is
    short (``pools_by_default``)."""

    windows: list[Chunk]
    together: int
    joined: dict[str, str]
    short: bool


@dataclass(frozen=True)
class Text:
    """Raw text to profile or score, as opposed to a path to read it from.

    ``name`` identifies it in reports (as a file name does) and pairs an edited text with
    its original in ``evaluate``. Unnamed texts are ``text1``, ``text2``, ... in order,
    skipping names other texts already have; two texts with one name are an error.
    The texts of one role (the writer's, say) are like the records of one JSONL file: each
    is a document, and short ones are pooled in order (see ``Settings.pool``).
    """

    text: str
    name: str | None = None
