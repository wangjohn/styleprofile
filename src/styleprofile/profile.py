"""Profile arbitrary chunks of prose and optionally score them against a saved profile.

A profile is, for every metric, its mean and spread across chunks (a score report also
keeps each of its chunks' metrics and z-scores). The spread is what makes a reference
useful: a new chunk's z-score on each metric says how unusual it is *relative to how much
that writer normally varies*, and the mean absolute z-score over a group of metrics is
Burrows' Delta for that group. How metrics are weighted into Delta and into the optional
contrast-likeness score is in ``weighting``.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypedDict, cast

from styleprofile import drift
from styleprofile.calibration import (
    CALIBRATION_LENGTHS,
    CALIBRATION_WORDS,
    CONTRAST_CALIBRATION_WORDS,
    MIN_CONTRAST_PIECES,
    MIN_JUDGED_WORDS,
    AtLength,
    Lengths,
    calibrate_lengths,
    verdict,
)
from styleprofile.core import (
    LIKENESSES,
    LikenessVerdict,
    Note,
    NoteCode,
    Phase,
    Progress,
    StyleProfileError,
    Verdict,
)
from styleprofile.formats import (
    AUTO,
    HTML,
    HTML_SUFFIXES,
    INPUT_FORMATS,
    JSONL,
    html_to_markdown,
    looks_like_html,
    looks_like_jsonl,
)
from styleprofile.measure import JoinedParse as _Joined  # noqa: F401  (tests use it)
from styleprofile.measure import Measured as _Measured
from styleprofile.measure import (
    Measurer,
    measure,
    measure_in_parts,
    measure_spans,
    parsed_document,
    prose_text,
    span_markdown,
)
from styleprofile.measure import Piece as _Piece
from styleprofile.rows import MetricRows
from styleprofile.schema import (
    Baseline,
    BaselineCalibration,
    BaselineLength,
    Calibration,
    ChunkRow,
    ChunkScore,
    Contrast,
    DeltaRange,
    DocumentEntry,
    DocumentPassages,
    DriftCalibration,
    EvaluationReport,
    LengthCalibration,
    LikenessSignal,
    MetricStats,
    ParagraphScore,
    Problem,
    ReferenceReport,
    Report,
    ReportSettings,
    ScoredChunk,
    ScoreReport,
    SpanScore,
    Summary,
    Unseen,
    find_problem,
)
from styleprofile.surface import (
    Metrics,
    block_word_count,
    classify,
    jensen_shannon,
    strip_front_matter,
    words,
)
from styleprofile.syntax import Parser
from styleprofile.weighting import (
    LENGTH_AUC_WARNING,
    LIKENESS_MIN_CEILING,
    MIN_CEILING,
    UNSCORED_GROUPS,
    CrossValidated,
    Key,
    ZScores,
    calibrate_delta,
    cross_validate,
    delta,
    delta_weights,
    flatten,
    floors,
    held_out,
    held_out_effects,
    held_out_z,
    likeness,
    likeness_range,
    nest,
    reliability,
    summarize_contrast,
    z_score,
)

# 3: Delta weights areas equally and metrics by held-out reliability (no cap), and a
# reference built with --contrast scores contrast likeness.
# 4: percentage floors use their true denominators (paragraphs, apostrophes), and list
# continuations extend their item instead of counting as paragraphs.
# 5: the contrast AUC has a document-bootstrap 95% interval and a length-only baseline.
# Every report carries ``kind`` (see ``KINDS``), and score reports a ``baseline`` copy of what
# rendering needs from their reference.
# 6: settings are recorded verbatim, every field of ``api.Settings``: ``syntax`` is what was
# asked ("auto", true or false) and ``syntax_used`` the parser that ran; ``window_words`` 0
# is no windowing; ``inputs`` is always a list.
# Also calibration.by_length: the reference's held-out ranges for texts of about 75,
# 150 and 300 words, so each chunk is judged at its own length; score reports carry a
# ``verdict``, which is "too short to judge" below 75 words.
# Reports of any other version are refused (``check_version``): rebuild them.
# Also in 6, before any release: reference profiles leave out per-chunk rows (unless built
# with keep_chunks) and store ``document_count``; sources are saved by their input's name,
# never as paths; the contrast AUC's ``bootstrap`` records its method and resamples.
VERSION = 6
# The version of evaluation reports (``styleprofile evaluate``), counted separately.
EVALUATION_VERSION = 2
REFERENCE: Final = "reference"
SCORE: Final = "score"
# Written by ``styleprofile evaluate`` (see the evaluate module); shown, never scored against.
EVALUATION: Final = "evaluation"
KINDS = (REFERENCE, SCORE, EVALUATION)
# The shape of each kind of report, which ``load_report`` checks.
SCHEMAS: dict[str, type] = {
    REFERENCE: ReferenceReport,
    SCORE: ScoreReport,
    EVALUATION: EvaluationReport,
}
# The message ends with the remedy, since the kind that says which command wrote it is unknown.
UNREADABLE = (
    "not a style profile this version of styleprofile can read; make it again with "
    "`styleprofile build` (or `score` or `evaluate`, whichever wrote it)"
)
TEXT_FIELDS: tuple[str, ...] = ("text", "body_markdown", "output", "content", "body")
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
# What a directory walk reads; other files are reported as skipped.
INPUT_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | {".jsonl"}
_SKIPPED_SHOWN = 4
# Formats a directory walk names when it skips them: documents a user may have expected to
# be read. Other skipped files (images, code, backups) are only counted.
DOCUMENT_SUFFIXES = frozenset(
    {
        ".docx", ".doc", ".pdf", ".rtf", ".odt", ".epub", ".rst", ".org", ".tex", ".pages",
        ".adoc", ".asciidoc", ".textile", ".mdx", ".wiki", ".pptx", ".xhtml", ".mht", ".mhtml",
    }
)  # fmt: skip
# The documents pandoc can convert to Markdown.
PANDOC_SUFFIXES = frozenset(
    {".docx", ".odt", ".rtf", ".epub", ".rst", ".org", ".tex", ".textile", ".wiki", ".xhtml"}
)
# Static-site generators (Jekyll, Eleventy, Hugo). Their output folders hold built pages
# that repeat the posts beside them, so a walk skips the HTML there (a writer may keep
# Markdown in a folder named public/). Their template folders are not prose at all, so a
# walk skips them whole. Either way it says so, and a directory named directly is read.
SITE_OUTPUT_DIRS = frozenset({"_site", "public"})
TEMPLATE_DIRS = frozenset({"_layouts", "_includes", "layouts", "themes", "resources"})
# Documents shorter than this are never dropped as duplicates: short records ("Thanks!")
# repeat legitimately.
DUPLICATE_MIN_WORDS = 20
# Directory walks skip dot-directories (.git, .venv) and these vendored ones.
SKIPPED_DIRS = frozenset({"node_modules", "__pycache__", "site-packages"})
OTHER = "<other>"
# The saved name of an input whose own name says nothing (the home directory, "/").
UNNAMED_ROOT = "input"
DEVIATIONS_SHOWN = 8
# Per document, a score report keeps this many of the largest differences and signals.
DOCUMENT_TRAITS = 3
SHORT_CHUNK_WORDS = 150


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


def _decode(data: bytes, name: str | Path) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise StyleProfileError(f"{name} is not UTF-8 text ({error.reason})") from error
    # Universal newlines, as text-mode reading gives.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text(path: Path) -> str:
    return _decode(path.read_bytes(), path)


def _text_fields(text_field: str | Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(text_field, str):
        return (text_field,)
    return tuple(text_field) if text_field else TEXT_FIELDS


# A field value longer than this is text, never a group; ``Records.values`` skips it.
GROUP_VALUE_CHARS = 100


@dataclass
class Records:
    """How ``load_chunks`` reads JSONL records, and what it found in them, for notes.

    ``group_field`` groups records into documents (see ``group_id``); a group spans the
    files of one input, ``scope``, which ``load_chunks`` sets. ``count`` counts the records
    read and ``missing`` those with no value in the group field. ``values`` maps every
    other field (the text field aside) to its distinct short values, so a note can suggest
    a group field, or name the fields a misspelled one missed.
    """

    group_field: str | None = None
    scope: str = ""
    count: int = 0
    missing: int = 0
    values: dict[str, set[str]] = field(default_factory=dict)

    def add(self, other: Records) -> None:
        """Count ``other``'s records here too."""
        self.count += other.count
        self.missing += other.missing
        for name, found in other.values.items():
            self.values.setdefault(name, set()).update(found)

    def group_fields(self) -> list[tuple[str, int]]:
        """Fields that look like they name each record's source: a name like a thread's or
        conversation's (``_SOURCE_NAMES``), else a writer's (``_WRITER_NAMES``), with 2 or
        more distinct values and at most a third as many as records. The likeliest first,
        with each one's number of values. Other fields (a language, an app, a count) are
        never suggested, however few values they have."""

        def rank(name: str) -> int | None:
            tokens = [
                token.lower() for token in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)
            ]
            named = [token for token in tokens if token not in _NAME_FILLERS]
            for ranked, names in enumerate((_SOURCE_NAMES, _WRITER_NAMES)):
                # One source word, and nothing but fillers besides (``in_reply_to_status_id``
                # yes; ``reply_count`` or ``user_followers`` no).
                if len(named) == 1 and named[0] in names:
                    return ranked
            return None

        found = [
            (ranked, name, len(values))
            for name, values in self.values.items()
            if (ranked := rank(name)) is not None and 2 <= len(values) <= self.count / 3
        ]
        return [(name, count) for _, name, count in sorted(found, key=lambda f: (f[0], f[2]))]


# Words of field names (split at ``_`` and camelCase) that name where a record belongs
# (``thread_id``, ``conversationId``, ``in_reply_to_status_id``), and, as a fallback, who
# wrote it; and the words that may come with them.
_SOURCE_NAMES = ("thread", "conversation", "channel", "subject", "reply", "parent")
_WRITER_NAMES = ("author", "user", "sender")
_NAME_FILLERS = frozenset(
    {"id", "ids", "name", "key", "uuid", "screen", "handle", "in", "to", "status", "the"}
)


def _jsonl_chunks(
    text: str,
    name: str,
    source: str,
    path: str | None,
    text_field: str | Sequence[str] | None,
    records: Records | None = None,
    inside: str | None = None,
    folder: str | None = None,
    repeated: list[tuple[str, int]] | None = None,
) -> list[Chunk]:
    """Read JSONL ``text``: ``name`` labels errors and default ids, ``source`` is saved for
    each record, and ``path`` is the file it came from (None for stdin).

    Each record is its own document, unless ``records.group_field`` groups them: then its
    id names its group and its line (``thread=t1#r12``, see ``group_id``) instead of its
    own id. Otherwise records that share an id are told apart by line (``same@3``), and
    each such id and how many records have it is added to ``repeated``. An id (or group
    value) that ends like a part suffix is escaped (``literal_id``). A file found in a
    folder input (``folder``) names its records by its path inside the folder too
    (``inside``: ``2024/a.jsonl:17``), since ids often restart in each file; a file given
    directly names them by id alone."""
    records = Records() if records is None else records
    group_field = records.group_field
    fields = _text_fields(text_field)
    grouped: list[Chunk] = []
    ungrouped: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise StyleProfileError(f"{name}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise StyleProfileError(f"{name}:{line_number}: expected a JSON object")
        field = next((key for key in fields if isinstance(record.get(key), str)), None)
        if field is None:
            raise StyleProfileError(
                f"{name}:{line_number}: no string field among {', '.join(fields)}",
                code="text_field",
            )
        records.count += 1
        for key, value in record.items():
            if key == field:
                continue
            seen = records.values.setdefault(key, set())
            if isinstance(value, str | int | float) and len(str(value)) <= GROUP_VALUE_CHARS:
                seen.add(str(value))
        if group_field is not None:
            group = record.get(group_field)
            if isinstance(group, dict | list):
                raise StyleProfileError(
                    f"{name}:{line_number}: the group field {group_field} holds a "
                    f"{type(group).__name__}, not a name or number",
                    code="group_field",
                )
            missing = group is None or not str(group).strip()
            records.missing += missing
            shown = literal_id(group_id(group_field, None if missing else group))
            # Missing values get a key no value can have, so "(none)" is a group of its own.
            value = "\x00" if missing else str(group)
            document = f"{_GROUPED}{records.scope}\x1e{group_field}\x1e{value}"
            chunk_id = f"{shown}#r{line_number}"
            own = record.get("id")
            own_id = f"line {line_number}" if own in (None, "") else literal_id(str(own))
            grouped.append(
                Chunk(chunk_id, source, record[field], path, folder, document, record=own_id)
            )
            continue
        record_id = record.get("id")
        # An id of 0 is kept; a missing, null or empty id falls back to the line.
        if record_id in (None, ""):
            chunk_id = f"{inside or Path(name).name}:{line_number}"
        else:
            chunk_id = literal_id(str(record_id))
            chunk_id = f"{inside}:{chunk_id}" if inside else chunk_id
        ungrouped.append((line_number, chunk_id, record[field]))
    if grouped:
        return grouped
    counts = Counter(chunk_id for _, chunk_id, _ in ungrouped)
    shared = {chunk_id: count for chunk_id, count in counts.items() if count > 1}
    if repeated is not None:
        repeated += shared.items()
    chunks: list[Chunk] = []
    for line, chunk_id, body in ungrouped:
        unique = f"{chunk_id}@{line}" if chunk_id in shared else chunk_id
        document = f"{path or source}{_RECORD}{unique}"
        chunks.append(Chunk(unique, source, body, path, folder, document))
    return chunks


def literal_id(value: str) -> str:
    """An id from outside (a JSONL id, a group value, a ``Text`` name) that part suffixes
    cannot be read into: a trailing ``#w2`` or ``#r3`` becomes ``%23w2`` or ``%23r3``, so
    ``base_id`` strips only the suffixes ``window`` and grouping add, and a record ``x#w2``
    stays apart from a record ``x``."""
    return _PART_SUFFIX.sub(lambda match: match.group(0).replace("#", "%23"), value)


def _repeated_note(label: str, repeated: Sequence[tuple[str, int]]) -> Note:
    """The note for JSONL records in ``label`` that share ids (see ``_jsonl_chunks``)."""
    shared = ", ".join(f"{count} share {record_id!r}" for record_id, count in repeated[:3])
    more = f" and {len(repeated) - 3} more ids" if len(repeated) > 3 else ""
    return Note(
        f"records in {label} share ids ({shared}{more}); each record is still its own "
        "document, named by id and line (id@LINE)",
        NoteCode.REPEATED_ID,
    )


def expand_path(value: str) -> Path:
    """A path as typed with ``~`` expanded, as an error rather than a crash when it names an
    unknown user (``~other``)."""
    try:
        return Path(value).expanduser()
    except RuntimeError as error:
        raise StyleProfileError(f"{value}: {error}", code="not_found") from error


def root_name(value: str) -> str:
    """The name a report saves for an input: the final part of its normalized path, however
    it was typed (``posts``, ``./posts/``, ``../x/posts`` and ``/home/me/posts`` all give
    ``posts``), so nothing above it is saved and the same corpus gives the same names from
    any directory. The working directory itself (``.``) gives ``""``, so its files are saved
    by their paths inside it. A home directory (``$HOME``, or any folder directly in
    ``/home`` or ``/Users``, or ``/root``: another user's, or yours under sudo or CI), a
    filesystem root, or anything else without a telling final part gives ``input``, since
    its name is a user name. Standard input is ``stdin``."""
    if value in ("-", "stdin"):
        return "stdin"
    typed = expand_path(value)
    absolute = Path(os.path.abspath(typed))
    if absolute == Path(os.path.abspath(Path.cwd())) and not typed.is_absolute():
        return ""
    if _is_home(absolute) or absolute.name in ("", ".", ".."):
        return UNNAMED_ROOT
    return absolute.name


# Folders whose children are home directories, and home directories outside them.
HOME_PARENTS = (Path("/home"), Path("/Users"))
OTHER_HOMES = (Path("/root"),)


def _is_home(absolute: Path) -> bool:
    """Whether ``absolute`` is a home directory, whose name is a user name."""
    if absolute == Path(os.path.abspath(Path.home())) or absolute in OTHER_HOMES:
        return True
    return absolute.parent in HOME_PARENTS


@dataclass
class SourceNames:
    """How ``load_chunks`` names what it reads, shared across calls so names never collide.

    ``files`` maps each saved source to the file it names; ``roots`` maps each input, as
    typed, to the root name its sources were saved under (``posts``, or ``posts (2)`` when
    another input already took ``posts``; ``.`` for the working directory).
    """

    files: dict[str, str] = field(default_factory=lambda: {"stdin": "-"})
    roots: dict[str, str] = field(default_factory=lambda: {"-": "stdin"})


def _numbered(name: str, number: int, is_file: bool) -> str:
    """``name`` for the ``number``-th input to claim it: ``posts (2)``, ``notes (2).md``."""
    if number == 1:
        return name
    if not name:
        return f"{UNNAMED_ROOT} ({number})"
    if is_file:
        stem, suffix = os.path.splitext(name)
        return f"{stem} ({number}){suffix}"
    return f"{name} ({number})"


def _name_sources(value: str, path: Path, files: Sequence[Path], names: SourceNames) -> list[str]:
    """Each file's source: the input's ``root_name``, joined for a directory with the file's
    path inside it. A name another file already has (two folders both called ``posts``, say)
    gets a number, ``posts (2)``, so every source still names one file."""
    root = root_name(value)
    is_dir = path.is_dir()
    relative = [item.relative_to(path).as_posix() for item in files] if is_dir else [""]
    number = 1
    while True:
        prefix = PurePosixPath(_numbered(root, number, not is_dir))
        sources = [(prefix / part).as_posix() if part else prefix.as_posix() for part in relative]
        pairs = list(zip(sources, (str(item) for item in files), strict=True))
        if all(names.files.get(source, item) == item for source, item in pairs):
            names.files.update(pairs)
            names.roots[value] = prefix.as_posix()
            return sources
        number += 1


def _check_format(input_format: str) -> None:
    if input_format not in INPUT_FORMATS:
        raise StyleProfileError(
            f"unknown input format {input_format!r}; choose one of {', '.join(INPUT_FORMATS)}"
        )


def _format_of(path: Path, input_format: str) -> str:
    """The format a file is read in: the one asked for, else the one its suffix names."""
    if input_format != AUTO:
        return input_format
    suffix = path.suffix.lower()
    return JSONL if suffix == ".jsonl" else HTML if suffix in HTML_SUFFIXES else AUTO


def _forced_jsonl(
    text: str,
    name: str,
    source: str,
    path: str | None,
    text_field: str | Sequence[str] | None,
    records: Records,
    inside: str | None = None,
    folder: str | None = None,
    repeated: list[tuple[str, int]] | None = None,
) -> list[Chunk]:
    """Read JSONL that was asked for rather than detected, saying so when it fails."""
    try:
        return _jsonl_chunks(
            text, name, source, path, text_field, records, inside, folder, repeated
        )
    except StyleProfileError as error:
        if error.code:
            raise
        raise StyleProfileError(
            f"{error}; {Path(name).name} was read as JSONL because that format was asked for",
            code="forced_jsonl",
        ) from error


@dataclass
class _Detected:
    """What reading one input found, for its notes: files sniffed as HTML, and HTML files
    with no text once converted."""

    sniffed: list[str]
    empty: list[str]
    # JSONL files whose records share ids: (label, [(id, count), ...]).
    repeated: list[tuple[str, list[tuple[str, int]]]] = field(default_factory=list)


def _file_chunks(
    path: Path,
    label: str,
    chunk_id: str,
    source: str,
    input_format: str,
    text_field: str | Sequence[str] | None,
    detected: _Detected,
    records: Records,
    folder: str | None = None,
) -> list[Chunk]:
    text = _read_text(path)
    fmt = _format_of(path, input_format)
    if fmt == JSONL:
        repeated: list[tuple[str, int]] = []
        inside = chunk_id if folder is not None else None
        read = (
            _forced_jsonl
            if input_format == JSONL and path.suffix.lower() != ".jsonl"
            else _jsonl_chunks
        )
        chunks = read(
            text, str(path), source, str(path), text_field, records, inside, folder, repeated
        )
        if repeated:
            detected.repeated.append((label, repeated))
        return chunks
    if fmt == AUTO and looks_like_html(text):
        detected.sniffed.append(label)
        fmt = HTML
    if fmt == HTML:
        text = html_to_markdown(text)
        if not re.search(r"\w", text):
            detected.empty.append(label)
    document = document_of(str(path), chunk_id)
    return [Chunk(chunk_id, source, text, str(path), folder, document, converted=fmt == HTML)]


def _stdin_chunks(
    input_format: str,
    text_field: str | Sequence[str] | None,
    notes: list[Note],
    records: Records,
) -> list[Chunk]:
    text = _decode(sys.stdin.buffer.read(), "stdin")
    repeated: list[tuple[str, int]] = []
    if input_format == JSONL:
        chunks = _forced_jsonl(text, "stdin", "stdin", None, text_field, records, repeated=repeated)
        notes += [_repeated_note("stdin", repeated)] if repeated else []
        return chunks
    fmt = input_format
    if fmt == AUTO and looks_like_jsonl(text):
        notes.append(
            Note(
                "stdin is JSON objects, one per line, so it is read as JSONL",
                NoteCode.READ_AS_JSONL,
            )
        )
        chunks = _jsonl_chunks(text, "stdin", "stdin", None, text_field, records, repeated=repeated)
        notes += [_repeated_note("stdin", repeated)] if repeated else []
        return chunks
    if fmt == AUTO and looks_like_html(text):
        fmt = HTML
        notes.append(Note("stdin looks like HTML, so it is read as HTML", NoteCode.READ_AS_HTML))
    if fmt == HTML:
        text = html_to_markdown(text)
        if not re.search(r"\w", text):
            notes.append(Note("stdin has no readable text after conversion", NoteCode.EMPTY_HTML))
    return [
        Chunk("stdin", "stdin", text, document=document_of("stdin", "stdin"), converted=fmt == HTML)
    ]


def _listing(items: Sequence[str], shown: int = 3) -> str:
    """ "a, b and c", or "a, b, c and 4 more" past ``shown`` items."""
    if len(items) > shown:
        return f"{', '.join(items[:shown])} and {len(items) - shown:,} more"
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _skipped(files: Sequence[Path]) -> tuple[str, str] | None:
    """Describe skipped files and what to do about them: ("12 .docx and 40 other files", ...).

    Only ``DOCUMENT_SUFFIXES`` are named, most common first; everything else (images,
    code, backups, unknown extensions) counts toward the "other" tail. Pandoc is suggested
    only for documents it can convert. None when no document was skipped.
    """
    documents = Counter(
        suffix for item in files if (suffix := item.suffix.lower()) in DOCUMENT_SUFFIXES
    )
    if not documents:
        return None
    named = documents.most_common(_SKIPPED_SHOWN)
    shown = [f"{count:,} {suffix}" for suffix, count in named]
    others = len(files) - sum(count for _, count in named)
    if others:
        shown.append(f"{others:,} other")
    # "and 1 other file" reads better than "files" whatever came before it.
    listed = _listing(shown, len(shown)) + " file" + ("" if len(files) == 1 or others == 1 else "s")
    if len(documents) > len(named):
        which = "those documents"
    else:
        which = f"the {_listing([suffix for suffix, _ in named])} "
        which += "file" if sum(documents.values()) == 1 else "files"
    pandoc = " (for example with pandoc)" if set(documents) & PANDOC_SUFFIXES else ""
    return listed, f"convert {which} to Markdown, text or HTML first{pandoc}"


def _walk(path: Path) -> tuple[list[Path], list[Path], list[str]]:
    """The readable files under a directory, the other files it skips, and the static-site
    folders it left readable files out of: the HTML in ``SITE_OUTPUT_DIRS`` and everything
    in ``TEMPLATE_DIRS``.

    Dot-directories (.git, .venv) and vendored directories are left out of all three, and
    dotfiles (.DS_Store) are not reported as skipped.
    """
    readable: list[Path] = []
    skipped: list[Path] = []
    generated: list[str] = []
    for item in sorted(path.rglob("*")):
        parts = item.relative_to(path).parts
        if not item.is_file() or any(
            part.startswith(".") or part in SKIPPED_DIRS for part in parts[:-1]
        ):
            continue
        suffix = item.suffix.lower()
        site = next(
            (
                index
                for index, part in enumerate(parts[:-1])
                if part in TEMPLATE_DIRS or (part in SITE_OUTPUT_DIRS and suffix in HTML_SUFFIXES)
            ),
            None,
        )
        if site is not None:
            folder = "/".join(parts[: site + 1])
            if suffix in INPUT_SUFFIXES and folder not in generated:
                generated.append(folder)
        elif suffix in INPUT_SUFFIXES:
            readable.append(item)
        elif not parts[-1].startswith("."):
            skipped.append(item)
    return readable, skipped, generated


def load_chunks(
    inputs: Sequence[str],
    text_field: str | Sequence[str] | None = None,
    *,
    names: SourceNames | None = None,
    input_format: str = AUTO,
    notes: list[Note] | None = None,
    group_field: str | None = None,
    records: Records | None = None,
) -> list[Chunk]:
    """Read Markdown, text, HTML or JSONL files, directories of them, or ``-`` for stdin.

    ``text_field`` names the JSONL field that holds the text, or several to try in order;
    by default the first of ``TEXT_FIELDS`` that a record has. Each file is one document,
    and so is each JSONL record, unless ``group_field`` names a record field whose value
    groups the records of one input into documents (see ``group_id``). ``records``, when
    given, carries the group field instead, and gathers what the records held.

    Each chunk's ``source`` is its input's ``root_name`` (``posts``), joined for a file in a
    directory with its path inside it (``posts/2024/a.md``); nothing above the input is
    saved. ``names`` records the names given out; pass the same one to several calls to keep
    sources distinct across them, as one call does across its inputs. Chunks remember the
    file they came from (``Chunk.path``), so documents stay distinct either way.

    With ``input_format`` ``"auto"``, a file's suffix picks its format, a Markdown or text
    file that looks like HTML is read as HTML, and stdin is read as JSONL when every line is
    a JSON object. ``"markdown"``, ``"html"`` or ``"jsonl"`` reads every input that way.
    HTML is converted to Markdown (see ``formats.html_to_markdown``). A directory walk
    leaves out static-site output and templates (``SITE_OUTPUT_DIRS``, ``TEMPLATE_DIRS``).
    What was detected, and what a walk skipped, are appended to ``notes`` when it is given;
    notes name inputs as typed, since they are shown rather than saved.
    """
    _check_format(input_format)
    names = SourceNames() if names is None else names
    notes = [] if notes is None else notes
    if records is None:
        records = Records(group_field)
    chunks: list[Chunk] = []
    for value in inputs:
        if value == "-":
            records.scope = "stdin"
            chunks.extend(_stdin_chunks(input_format, text_field, notes, records))
            continue
        path = expand_path(value).resolve()
        records.scope = str(path)
        detected = _Detected([], [])
        if not path.is_dir():
            [source] = _name_sources(value, path, [path], names)
            chunks.extend(
                _file_chunks(
                    path,
                    value,
                    path.name,
                    source,
                    input_format,
                    text_field,
                    detected,
                    records,
                )
            )
        else:
            files, skipped, generated = _walk(path)
            described = _skipped(skipped)
            if generated:
                notes.append(
                    Note(
                        f"left out static-site output and template folders in {value}: "
                        f"{_listing(generated)}; name one directly to read it",
                        NoteCode.SKIPPED_DIRS,
                    )
                )
            if not files:
                if generated:
                    raise StyleProfileError(
                        f"{value} has readable files only in static-site output and template "
                        f"folders ({_listing(generated)}), which are left out; name one directly "
                        "to read it",
                        code="only_generated",
                    )
                if described:
                    has = f"; it has {described[0]}, so {described[1]}"
                elif skipped:
                    has = f"; it has only images, code and other files ({len(skipped):,})"
                else:
                    has = ""
                raise StyleProfileError(
                    f"{value} contains no .md, .markdown, .txt, .html, .htm, or .jsonl files{has}"
                )
            if described:
                notes.append(
                    Note(
                        f"skipped {described[0]} in {value}; {described[1]}", NoteCode.SKIPPED_FILES
                    )
                )
            sources = _name_sources(value, path, files, names)
            for item, source in zip(files, sources, strict=True):
                chunk_id = str(item.relative_to(path))
                label = str(Path(value) / chunk_id)
                chunks.extend(
                    _file_chunks(
                        item,
                        label,
                        chunk_id,
                        source,
                        input_format,
                        text_field,
                        detected,
                        records,
                        folder=value,
                    )
                )
        if detected.sniffed:
            they = "it is" if len(detected.sniffed) == 1 else "they are"
            looks = "looks" if len(detected.sniffed) == 1 else "look"
            notes.append(
                Note(
                    f"{_listing(detected.sniffed)} {looks} like HTML, so {they} read as HTML",
                    NoteCode.READ_AS_HTML_IN_FOLDER if path.is_dir() else NoteCode.READ_AS_HTML,
                )
            )
        notes += [_repeated_note(label, repeated) for label, repeated in detected.repeated]
        if detected.empty:
            has = "has" if len(detected.empty) == 1 else "have"
            notes.append(
                Note(
                    f"{_listing(detected.empty)} {has} no readable text after conversion from HTML",
                    NoteCode.EMPTY_HTML,
                )
            )
    return chunks


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


@dataclass(frozen=True)
class Repeat:
    """A document ``drop_duplicates`` dropped: its first chunk, ``dropped``; the first chunk
    of the copy it kept, ``kept``, or None when that copy was read before (another input
    set, whose document ``label`` names)."""

    dropped: Chunk
    kept: Chunk | None
    label: str


def drop_duplicates(
    chunks: Sequence[Chunk],
    seen: dict[str, str] | None = None,
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

    ``seen`` maps the text of documents read earlier (another input set) to their label,
    and is updated, so a contrast draft that repeats a reference document is dropped too.
    Twins would sit on both sides of held-out calibration and make it look too tight.
    Documents under ``DUPLICATE_MIN_WORDS`` words are always kept. The note names documents
    by their saved sources, never by path. ``repeats``, when given, gathers each dropped
    document with the copy kept, so an edit of a dropped draft can pair with that copy.
    """
    seen = {} if seen is None else seen
    firsts: dict[str, Chunk] = {}  # the first chunk of each document kept here, by text
    documents: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        documents.setdefault(_duplicate_unit(chunk), []).append(chunk)
    dropped_documents: set[str] = set()
    dropped: list[tuple[str, str]] = []
    for document, members in documents.items():
        tokens = words(strip_front_matter("\n\n".join(chunk.text for chunk in members)))
        if len(tokens) < DUPLICATE_MIN_WORDS:
            continue
        key = " ".join(tokens).lower()
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
    note = (
        f"dropped {repeating} another word for word, keeping the first copy (for example, "
        f"{copy} repeats {original})"
    )
    return kept, Note(note, NoteCode.DUPLICATES)


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
    counts = [block_word_count(block) for block in blocks]
    groups = _pack(counts, window_words, [block.continues_list for block in blocks])
    return [
        ("\n\n".join(blocks[index].raw for index in group), sum(counts[index] for index in group))
        for group in groups
    ]


def window(chunks: Sequence[Chunk], window_words: int) -> list[Chunk]:
    """Split chunks into roughly ``window_words``-word pieces at Markdown block boundaries.

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


def _pool_key(chunk: Chunk) -> str:
    """What ungrouped texts must share to be joined: a record's file (the records of one
    JSONL file, or stdin), a file's folder input, else the source (``Text`` inputs)."""
    if _record(chunk):
        return chunk.path or chunk.source
    return chunk.folder or chunk.path or chunk.source


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


def _z_against(metrics: Metrics, summary: Summary, floor: dict[Key, float]) -> ZScores:
    scored: ZScores = {}
    for group, values in metrics.items():
        if group in UNSCORED_GROUPS:
            continue
        for name, value in values.items():
            stats = summary.get(group, {}).get(name)
            mean = stats.get("mean") if stats else None
            if value is None or not stats or mean is None:
                continue
            z = z_score(
                value,
                mean,
                stats.get("sd"),
                stats.get("n", 0),
                floor.get((group, name), 0.0),
            )
            if z is not None:
                scored[(group, name)] = z
    return scored


@dataclass(frozen=True)
class _Prepared:
    """Everything scoring needs from a reference, computed once rather than per chunk."""

    floor: dict[Key, float]
    effects: dict[Key, float]
    lengths: Lengths


def _prepare(reference: ReferenceReport) -> _Prepared:
    return _Prepared(
        floor=floors(reference["summary"]),
        effects=flatten((reference.get("contrast") or {}).get("effects", {})),
        lengths=Lengths(reference),
    )


def _score(
    metrics: Metrics,
    distributions: dict[str, Counter[str]],
    reference: ReferenceReport,
    prepared: _Prepared,
    at: AtLength,
    cap: float | None = None,
) -> ChunkScore:
    """One chunk's scores, with Delta weights and likeness scaled by the reference's
    held-out rms at the chunk's own length (``at``). With ``cap`` (a span of ``drift``),
    Delta and likeness count each z up to that size."""
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    counted = _capped(z_scores, cap) if cap is not None else z_scores
    summary = reference["summary"]
    unseen: list[Unseen] = [
        {"metric": f"{group}.{name}", "value": metrics[group][name], "reference_value": mean}
        for (group, name), z in z_scores.items()
        if z and not summary[group][name].get("sd")
        for mean in [summary[group][name]["mean"]]
    ]
    overall, by_group = delta(counted, delta_weights(at.rms))
    deviations = sorted(z_scores.items(), key=lambda item: -abs(item[1]))[:DEVIATIONS_SHOWN]
    liked: _Liked = {}
    if prepared.effects:
        score, signals = likeness(counted, prepared.effects, at.rms)
        liked = {"likeness": score, "likeness_signals": signals}
    return {
        "delta": overall,
        "delta_by_group": by_group,
        "divergence": {
            name: jensen_shannon(
                _collapse(_distribution(counts), reference["distributions"][name]),
                reference["distributions"][name],
            )
            for name, counts in distributions.items()
            if name in reference.get("distributions", {})
        },
        "largest_deviations": [
            {
                "metric": f"{group}.{name}",
                "z": z,
                "value": metrics[group][name],
                "reference_mean": summary[group][name]["mean"],
            }
            for (group, name), z in deviations
        ],
        "unseen_in_reference": unseen,
        "z": nest(z_scores),
        **liked,
        "calibration": at.row(),
    }


class _Liked(TypedDict, total=False):
    """A chunk's likeness, against a reference with a contrast set."""

    likeness: float
    likeness_signals: list[LikenessSignal]


# What a score report keeps of each span (``drift.judge_span``).
SPAN_KEYS = ("by", "relative", "delta", "ceiling", "likeness", "likeness_ceiling", "level")


@dataclass(frozen=True)
class _Parts:
    """A document read in parts (``drift``): its paragraphs and spans, the metrics of each
    span, and of each paragraph on its own (for its traits)."""

    paragraphs: list[drift.Paragraph]
    layout: drift.Plan
    spans: list[Metrics]
    own: list[Metrics]


def _read_in_parts(
    text: str,
    windows: Sequence[tuple[str, Any]],
    parser: Parser | None,
    *,
    own: bool = True,
) -> _Parts:
    """Measure a document's spans, and with ``own`` its paragraphs, each once: from its
    Markdown, and with a parser from its part of the document's parse (``windows``: its
    windows' prose and parses, in order), so nothing is parsed twice."""
    found = drift.paragraphs(text)
    layout = drift.plan([paragraph.words for paragraph in found])
    if not layout.spans:
        return _Parts(found, layout, [], [])
    whole = parsed_document(prose_text(text), windows, parser) if parser else None
    spans = measure_spans(span_markdown(found, layout), whole)
    own_metrics = measure_spans([item.own for item in found], whole) if own else []
    return _Parts(found, layout, spans, own_metrics)


def _paragraph_alone(
    metrics: Metrics,
    reference: ReferenceReport,
    prepared: _Prepared,
    by: str,
) -> tuple[list[dict[str, Any]], float | None]:
    """A paragraph read on its own: its traits (its metrics' z at its own length, its
    strongest contrast signals when its spans are judged by likeness), and its score over
    the bound at its length (``drift.judge_span``'s ``relative``, parser metrics capped as a
    span's are; below 75 words the bound is widened, and it is not a verdict)."""
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
    signals = None
    if by == drift.LIKENESS and prepared.effects:
        signals = likeness(z_scores, prepared.effects, at.rms)[1]
    found = drift.traits(nest(z_scores), nest(at.scale), metrics, reference["summary"], signals)
    capped = _capped(z_scores, drift.SPAN_Z)
    relative = None
    if by == drift.LIKENESS and prepared.effects and at.likeness is not None:
        value = likeness(capped, prepared.effects, at.rms)[0]
        relative = value / max(at.likeness["p95"], LIKENESS_MIN_CEILING)
    elif by == drift.DELTA and at.delta is not None:
        value, _ = delta(capped, delta_weights(at.rms))
        relative = value / max(at.delta["p95"], MIN_CEILING) if value is not None else None
    return found, relative


# Why a pooled score is not read in parts (``score``).
POOLED_REASON = "paragraph checks don't apply to pooled records"


def _abstained(name: str, source: str, reason: str, *, pooled: bool = False) -> DocumentPassages:
    """A document not read in parts, and why."""
    return {
        "name": name,
        "source": source,
        "converted": False,
        "pooled": pooled,
        "words": 0,
        "span_words": drift.SPAN_WORDS,
        "judged": False,
        "reason": reason,
        "by": None,
        "sensitive": False,
        "own_level": None,
        "thresholds": None,
        "spans": [],
        "paragraphs": [],
    }


def _passages(
    document: Chunk,
    name: str,
    windows: Sequence[tuple[str, Any]],
    reference: ReferenceReport,
    prepared: _Prepared,
    parser: Parser | None,
) -> DocumentPassages:
    """Where one document drifts (``drift``): its paragraphs, the spans of at least
    ``SPAN_WORDS`` words read across them, and each paragraph's figures and flag, against
    the thresholds the reference's null sets for a document of its length."""
    parts = _read_in_parts(document.text, windows, parser)
    entry: DocumentPassages = {
        "name": name,
        "source": document.source,
        "converted": document.converted,
        "pooled": False,
        "words": sum(paragraph.words for paragraph in parts.paragraphs),
        "span_words": drift.SPAN_WORDS,
        "judged": False,
        "reason": None,
        "by": None,
        "sensitive": False,
        "own_level": None,
        "thresholds": None,
        "spans": [],
        "paragraphs": [],
    }
    if not parts.layout.spans:
        entry["reason"] = f"under {drift.SPAN_WORDS} words, too short to read in parts"
        return entry
    judged: list[dict[str, Any]] = []
    reason = None
    for metrics in parts.spans:
        at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
        reason = reason or at.reason
        scored = _score(metrics, {}, reference, prepared, at, cap=drift.SPAN_Z)
        judged.append(drift.judge_span(scored))
    entry["spans"] = [
        cast(
            SpanScore,
            {
                "paragraphs": [start, end],
                "words": sum(paragraph.words for paragraph in parts.paragraphs[start:end]),
                **{key: span[key] for key in SPAN_KEYS},
            },
        )
        for (start, end), span in zip(parts.layout.spans, judged, strict=True)
    ]
    scored = [span for span in judged if span["judged"]]
    if not scored:
        entry["reason"] = reason
        return entry
    by = scored[0]["by"]
    stats = drift.statistics(parts.layout, [span["relative"] for span in judged])
    limits = drift.thresholds(
        (reference.get("calibration") or {}).get("drift"),
        sum(stat is not None for stat in stats),
        by,
    )
    alone = [_paragraph_alone(metrics, reference, prepared, by) for metrics in parts.own]
    entry["judged"], entry["by"] = True, by
    entry["sensitive"], entry["thresholds"] = limits is not None, cast(Any, limits)
    entry["own_level"] = drift.level(stats)
    entry["paragraphs"] = cast(
        list[ParagraphScore],
        drift.judge(
            parts.paragraphs,
            parts.layout,
            judged,
            limits,
            [traits for traits, _ in alone],
            [relative for _, relative in alone],
        ),
    )
    return entry


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


def _held_out_relative(
    z_scores: ZScores,
    at: AtLength,
    by: str,
    effects: Callable[[str], dict[Key, float]] | None,
    document: str,
) -> float | None:
    """A held-out span's score over its 95% bound at its length, as ``drift.judge_span``
    reads a draft's, each z capped as a draft's span is (``drift.SPAN_Z``); None when its
    length is not judged or has no range for the score."""
    if not at.judged or at.delta is None:
        return None
    z_scores = _capped(z_scores, drift.SPAN_Z)
    if by == drift.LIKENESS:
        if at.likeness is None or effects is None:
            return None
        value = likeness(z_scores, effects(document), at.rms)[0]
        return value / max(at.likeness["p95"], LIKENESS_MIN_CEILING)
    value, _ = delta(z_scores, delta_weights(at.rms))
    return value / max(at.delta["p95"], MIN_CEILING) if value is not None else None


def _held_out_effects(fit: ContrastFit) -> Callable[[str], dict[Key, float]]:
    """The likeness effects learned without one reference document, as ``likeness_range``
    scores its pieces."""
    return held_out_effects(
        fit.reference_held, fit.reference_documents, fit.contrast_z, fit.learned
    )


def _capped(z_scores: ZScores, cap: float) -> ZScores:
    """Each z of the groups ``drift.SPAN_Z_GROUPS`` names (every group when None) limited
    to ``cap`` either way (``drift.SPAN_Z``)."""
    groups = drift.SPAN_Z_GROUPS
    return {
        key: max(-cap, min(cap, z)) if groups is None or key[0] in groups else z
        for key, z in z_scores.items()
    }


def _mean_of(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def report_kind(report: Mapping[str, Any]) -> str:
    """``"reference"``, ``"score"`` or ``"evaluation"``, from the report's ``kind``."""
    kind = report.get("kind")
    if kind not in KINDS:
        raise StyleProfileError(
            f"the report has no known kind, so it is {UNREADABLE}", code="outdated"
        )
    return kind


def load_report(path: Path, name: str | None = None) -> Report:
    """Read any styleprofile report: reference, score or evaluation.

    A report of another version, or one that lacks a key its kind needs (see ``schema``), is
    refused with code ``outdated``. Messages call the file ``name``, by default ``path``."""
    name = str(path) if name is None else name
    if path.is_dir():
        raise StyleProfileError(f"{name} is a directory, not a profile", code="directory")
    if not path.exists():
        raise StyleProfileError(f"{name} not found", code="not_found")
    try:
        report = json.loads(_read_text(path))
    except (json.JSONDecodeError, StyleProfileError):
        report = None
    if not isinstance(report, dict) or ("summary" not in report and "kind" not in report):
        raise StyleProfileError(f"{name} is not a style profile", code="not_a_profile")
    kind = report.get("kind")
    if kind not in KINDS:
        # A pre-release report without ``kind``, or a kind this version does not know. One
        # with a version says which.
        if kind is None and isinstance(report.get("version"), int):
            check_version(report, name)
        raise StyleProfileError(f"{name} is {UNREADABLE}", code="outdated")
    check_report(report, name)
    return cast(Report, report)


def check_report(report: Mapping[str, Any], name: str = "the report") -> None:
    """Refuse a report of a known kind (see ``report_kind``) that this styleprofile cannot
    read: one of another version (``check_version``), or one whose shape its kind does not
    have (``schema.find_problem``): a key missing, a part of the wrong kind, or a null where
    one is needed, naming the part. All get code ``outdated``."""
    check_version(report, name)
    kind = report_kind(report)
    problem = find_problem(report, SCHEMAS[kind]) or _unmatched(report, kind)
    if problem is None:
        return
    # The version does not change with every format change before a release, so a report of
    # this version can still lack a key that a later build added.
    what = (
        f"has no {problem.path}, which this version of styleprofile needs"
        if problem.reason == "missing"
        else f"has an unreadable {problem.path} ({problem.reason})"
    )
    raise StyleProfileError(
        f"{name} {what}: it was made by an earlier development version, or edited; {_again(kind)}",
        code="outdated",
    )


def _unmatched(report: Mapping[str, Any], kind: str) -> Problem | None:
    """The first part one section of a well-shaped report names but another lacks: a metric
    a score's chunks have z-scores for but its summaries lack, or an edited set an
    evaluation's signals leave out. Only an edited file has one."""
    if kind == SCORE:
        summaries = {
            "summary": report["summary"],
            "reference.baseline.summary": report["reference"]["baseline"]["summary"],
        }
        for row in report["chunks"]:
            for group, names in row["reference"]["z"].items():
                for where, summary in summaries.items():
                    metrics = summary.get(group)
                    if metrics is None:
                        return Problem(f"{where}.{group}", "missing")
                    if not names.keys() <= metrics.keys():
                        missing = next(name for name in names if name not in metrics)
                        return Problem(f"{where}.{group}.{missing}", "missing")
    if kind == EVALUATION:
        edited = [label for label in report["sets"] if label != "original"]
        for index, signal in enumerate(report["signals"]):
            for label in edited:
                if label not in signal["edited"]:
                    return Problem(f"signals[{index}].edited.{label}", "missing")
    return None


def _again(kind: str | None) -> str:
    """How to make a report of ``kind`` again."""
    return {
        SCORE: "score it again with `styleprofile score`",
        EVALUATION: "run `styleprofile evaluate` again",
    }.get(kind or "", "rebuild it with `styleprofile build`")


def _before_lengths(report: Mapping[str, Any]) -> bool:
    """Whether a version-6 report is from before length calibration, which version 6
    gained before any release: a reference whose calibration lacks ``by_length`` or
    ``chunk_words``, or a score report without a ``verdict``. Such a reference would judge
    short texts against its windows' range. A part of the wrong shape is left to
    ``check_report``."""
    kind = report.get("kind")
    calibration = report.get("calibration")
    if kind == REFERENCE and isinstance(calibration, Mapping) and calibration:
        return not {"by_length", "chunk_words"} <= calibration.keys()
    scored = report.get("reference")
    return kind == SCORE and isinstance(scored, Mapping) and "verdict" not in scored


def _before_means(report: Mapping[str, Any]) -> bool:
    """Whether a version-6 report is from before ranges stored their held-out mean and run
    similarity, which version 6 also gained before any release: a reference (or a score
    report's copy of it) whose Delta range lacks ``mean`` or ``similarity``. Its pooled
    verdicts would close in on the median and call the writer's own text different once
    enough of it is pooled. A part of the wrong shape is left to ``check_report``."""
    calibration: Any = report.get("calibration")
    if report.get("kind") == SCORE:
        scored = report.get("reference")
        baseline = scored.get("baseline") if isinstance(scored, Mapping) else None
        calibration = baseline.get("calibration") if isinstance(baseline, Mapping) else None
    delta = calibration.get("delta") if isinstance(calibration, Mapping) else None
    return isinstance(delta, Mapping) and not {"mean", "similarity"} <= delta.keys()


def check_version(report: Mapping[str, Any], name: str = "the report") -> None:
    """Refuse a report of another version than this styleprofile writes, saying how to
    make it again: its settings and metrics would be misread rather than migrated."""
    kind = report.get("kind")
    expected = EVALUATION_VERSION if kind == EVALUATION else VERSION
    version = report.get("version")
    if type(version) is not int:
        # Missing, or not a number ("6", say): no version this or any styleprofile writes.
        raise StyleProfileError(f"{name} is {UNREADABLE}", code="outdated")
    if version == expected:
        if _before_lengths(report):
            raise StyleProfileError(
                f"{name} was made by an older styleprofile, before length-aware verdicts; "
                f"{_again(kind)}",
                code="outdated",
            )
        if _before_means(report):
            raise StyleProfileError(
                f"{name} was made by an older styleprofile, before verdicts over several "
                f"chunks read the held-out mean; {_again(kind)}",
                code="outdated",
            )
        return
    age = "a newer" if version > expected else "an older"
    raise StyleProfileError(
        f"{name} was made by {age} styleprofile (report version {version}; this one reads "
        f"{expected}); {_again(kind)}",
        code="outdated",
    )


def load_reference(path: Path, name: str | None = None) -> ReferenceReport:
    """Read a reference profile, refusing a score report (a sample scored against one).
    Messages call the file ``name``, by default ``path``."""
    name = str(path) if name is None else name
    reference = load_report(path, name)
    if reference["kind"] == EVALUATION:
        raise StyleProfileError(
            f"{name} is an evaluation report, not a reference profile; build a reference "
            "from the writer's own texts",
            code="score_as_reference",
        )
    if reference["kind"] == SCORE:
        raise StyleProfileError(
            f"{name} is a score report (a sample scored against a reference), not a "
            "reference profile; build a reference from the writer's own texts",
            code="score_as_reference",
        )
    return reference


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
            f"skipped {measured.empty + measured.below} contrast chunk(s) with no prose or "
            f"fewer than {min_words} prose words"
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
        report["warnings"].append(
            "the contrast set is one document, so its likeness range is measured in-sample "
            "and is optimistic; add more contrast documents"
        )
    if calibration["auc_ci"] is None:
        report["warnings"].append(
            "the reference or contrast set has fewer than 2 documents, so the contrast AUC "
            "has no confidence interval"
        )
    length = calibration["length_baseline"]
    if length and length["auc"] >= LENGTH_AUC_WARNING:
        report["warnings"].append(
            f"the contrast set differs strongly in length (length alone separates it with "
            f"AUC {length['auc']:.2f}), so {label}-likeness may partly reflect length; match "
            "lengths or split both sets into windows of the same size"
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
        warnings.append(
            f"skipped {measured.empty} chunk(s) with no prose (only code, tables or markup)"
        )
    if measured.below:
        warnings.append(
            f"skipped {measured.below} chunk(s) with fewer than {min_words} prose words"
        )
    # A score judges each chunk at its own length instead (``calibration``).
    words = _words(chunk_metrics)
    short = sum(count < SHORT_CHUNK_WORDS for count in words)
    if short and kind == REFERENCE:
        warnings.append(
            f"{short} chunk(s) have fewer than {SHORT_CHUNK_WORDS} words; their rates are noisy"
        )
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


def z_against_reference(
    report: ReferenceReport,
    chunks: Sequence[Chunk],
    parser: Parser | None,
    min_words: int,
    measurer: Measurer | None = None,
) -> tuple[list[Chunk], list[ZScores]]:
    """Measure chunks and take their z-scores against a reference report, as the contrast
    drafts are; returns the chunks kept (enough prose), possibly none, and their z-scores."""
    measured = _measure(
        chunks, parser, min_words, allow_empty=True, keep_distributions=False, measurer=measurer
    )
    floor = floors(report["summary"])
    return measured.chunks, [
        _z_against(metrics, report["summary"], floor) for metrics in measured.metrics
    ]


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
    measurer: Measurer | None = None,
) -> tuple[ReferenceReport, ContrastFit | None]:
    # The documents read for the paragraph null (``_calibrate_drift``), whose parses are kept.
    chosen = _drift_documents(chunks) if calibrate_lengths else []
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


def _baseline(reference: ReferenceReport) -> Baseline:
    """What rendering a score needs from its reference, so a saved score can be shown without
    the reference file: each metric's mean and spread, the held-out Delta range, and the
    contrast set's name and likeness range."""
    contrast = reference.get("contrast")
    return {
        "chunk_count": reference["chunk_count"],
        "summary": {
            group: {
                name: {"mean": stats.get("mean"), "sd": stats.get("sd")}
                for name, stats in metrics.items()
            }
            for group, metrics in reference["summary"].items()
        },
        "calibration": _without_rms(reference.get("calibration")),
        "contrast": (
            {"label": contrast["label"], "calibration": contrast["calibration"]}
            if contrast
            else None
        ),
    }


def _left_out(rows: Sequence[ScoredChunk], shown: int = 5) -> str:
    """Which chunks a verdict leaves out as too short to judge, and why, for its warning.

    A chunk is named by its input (``notes.md``), or with its window (``draft.md#w3``) when
    its input was cut into several."""
    left_out = [row for row in rows if not row["reference"]["calibration"]["judged"]]
    windows = Counter(base_id(row["id"]) for row in rows)
    names = []
    for row in left_out[:shown]:
        words = int(row["metrics"]["size"]["words"] or 0)
        why = (
            f"under {MIN_JUDGED_WORDS}"
            if words < MIN_JUDGED_WORDS
            else "shorter than the reference is calibrated for"
        )
        name = row["id"] if windows[base_id(row["id"])] > 1 else base_id(row["id"])
        names.append(f"{name} ({words:,} words, {why})")
    more = f" and {len(left_out) - shown} more" if len(left_out) > shown else ""
    return "left out of the verdict and the means as too short to judge: " + ", ".join(names) + more


def _without_rms(calibration: Calibration | None) -> BaselineCalibration | None:
    """The calibration without each length's per-metric rms, which only scoring reads."""
    if calibration is None:
        return None
    by_length: dict[str, BaselineLength] = {}
    for length, entry in calibration["by_length"].items():
        copied = entry.copy()
        copied.pop("reliability", None)
        by_length[length] = copied
    return {
        "sources": calibration["sources"],
        "delta": calibration["delta"],
        "chunk_words": calibration["chunk_words"],
        "by_length": by_length,
    }


def score(
    chunks: Sequence[Chunk],
    reference: ReferenceReport,
    *,
    parser: Parser | None = None,
    top_k: int = 300,
    min_words: int = 1,
    settings: Mapping[str, Any] | None = None,
    reference_path: Path | None = None,
    read_in_parts: Sequence[Chunk] | None = None,
    pooled: bool = False,
    measurer: Measurer | None = None,
) -> ScoreReport:
    """Profile sample chunks and score each against ``reference``: z-scores, Delta, pattern
    divergence and, when the reference learned a contrast, likeness to the contrast set.

    With ``read_in_parts`` (the chunks before windowing, or some of them), each is also read
    in overlapping spans to show where it drifts (``drift``): ``report["passages"]`` holds one
    entry per document, and is None without ``read_in_parts``. When the chunks are
    ``pooled`` windows, every document abstains instead.

    This is the lower-level step under ``Profile.score``: nothing is inherited from the
    reference, so pass chunks cut into the reference's windows, its ``min_words`` and a
    ``parser`` when it has syntax metrics, or use ``Profile.score``, which does all that.
    A reference of another report version is refused (``check_version``).

    Each chunk is judged at its own length (``calibration``). The means and the
    ``verdict`` cover the chunks long enough to judge; when none is, they cover every
    chunk, and the verdict is "too short to judge"."""
    check_version(reference, "the reference")
    measured = _measure(
        chunks,
        parser,
        min_words,
        measurer=measurer,
        keep_docs=bool(read_in_parts),
        docs_required=True,
    )
    base = _base_report(
        measured, kind=SCORE, parser=parser, top_k=top_k, min_words=min_words, settings=settings
    )
    warnings, totals = base.warnings, base.totals
    prepared = _prepare(reference)
    rows: list[ScoredChunk] = []
    for row, metrics, distributions in zip(
        base.rows, measured.metrics, measured.distributions, strict=True
    ):
        at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
        rows.append({**row, "reference": _score(metrics, distributions, reference, prepared, at)})

    judged = [row for row in rows if row["reference"]["calibration"]["judged"]]
    scored = [row["reference"] for row in judged or rows]
    groups = sorted({group for scores in scored for group in scores["delta_by_group"]})
    reference_settings = reference.get("settings", {})
    if judged and len(judged) < len(rows):
        warnings.append(_left_out(rows))
    if not reference.get("reliability"):
        warnings.append(
            "the reference has no held-out reliability (it needs chunks from at least two "
            "documents and a current report version), so Delta caps each metric at 3 "
            "instead of weighting it by reliability"
        )
    if prepared.effects:
        present = {
            (group, name)
            for row in rows
            for group, names in row["reference"]["z"].items()
            for name in names
        }
        total_weight = sum(effect * effect for effect in prepared.effects.values())
        missing = sum(
            effect * effect for key, effect in prepared.effects.items() if key not in present
        )
        if total_weight and missing / total_weight > 0.1:
            warnings.append(
                f"this run lacks metrics that carry {100 * missing / total_weight:.0f}% of "
                "the likeness weight (for example syntax); likeness uses the rest"
            )
    if reference["chunk_count"] < 2:
        warnings.append("the reference has one chunk, so it has no spread; split it into windows")
    # 0 and None both mean no windowing.
    own_settings = base.described["settings"]
    own_window = own_settings.get("window_words") or None
    reference_window = reference_settings.get("window_words") or None
    if own_window != reference_window:
        warnings.append(
            "window sizes differ from the reference "
            f"({own_window or 'off'} vs {reference_window or 'off'}); "
            "z-scores assume equal-sized chunks"
        )
    own_syntax = own_settings["syntax_used"]
    reference_syntax = reference_settings.get("syntax_used")
    if own_syntax and not reference_syntax:
        warnings.append("the reference has no syntax metrics, so syntax is not scored")
    elif reference_syntax and not own_syntax:
        warnings.append(
            "the reference has syntax metrics but this run does not, so Delta leaves out "
            "syntax and sentence openers; its value is not comparable with syntax runs"
        )
    elif own_syntax and reference_syntax:
        keys = ("model", "model_version")
        if any(own_syntax.get(key) != reference_syntax.get(key) for key in keys):
            warnings.append(
                "the reference was parsed with a different spaCy model "
                f"({reference_syntax.get('model')} {reference_syntax.get('model_version')}); "
                "syntax metrics may not be comparable"
            )
    passages: list[DocumentPassages] | None = None
    if pooled:
        # Pooled windows join records: their paragraphs are records, which the paragraph
        # null was not calibrated for. Each document abstains, and each record's own
        # verdict is in ``documents``.
        keys = list(dict.fromkeys((row["source"], base_id(row["id"])) for row in rows))
        passages = [
            _abstained(name, source, POOLED_REASON, pooled=True)
            for (source, _), name in zip(keys, _document_names(keys), strict=True)
        ]
    elif read_in_parts is not None:
        parsed: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        if measured.docs is not None:
            for chunk, found in zip(measured.chunks, measured.docs, strict=True):
                if found is not None:
                    parsed[chunk_document(chunk)].append(found)
        # Named as ``documents`` names them, so the two can be read together.
        keys = list(dict.fromkeys((row["source"], base_id(row["id"])) for row in rows))
        names = dict(zip(keys, _document_names(keys), strict=True))
        passages = [
            _passages(
                document,
                names.get((document.source, base_id(document.id)))
                or document_label(document.source, base_id(document.id)),
                parsed[chunk_document(document)],
                reference,
                prepared,
                parser,
            )
            for document in read_in_parts
        ]
    return {
        "version": VERSION,
        "kind": SCORE,
        **base.described,
        "warnings": warnings,
        "chunks": rows,
        "documents": documents(rows, reference),
        "passages": passages,
        "reference": {
            # Only the profile's file name: a saved report never reveals where files live.
            "path": root_name(str(reference_path)) if reference_path else None,
            "chunk_count": reference["chunk_count"],
            "delta_mean": _mean_of([scores["delta"] for scores in scored]),
            "likeness_mean": _mean_of([scores.get("likeness") for scores in scored]),
            "delta_by_group_mean": {
                group: _mean_of([scores["delta_by_group"].get(group) for scores in scored])
                for group in groups
            },
            "divergence_mean": {
                name: _mean_of([scores["divergence"].get(name) for scores in scored])
                for name in reference.get("distributions", {})
            },
            "pooled_divergence": {
                name: jensen_shannon(
                    _collapse(_distribution(counts), reference["distributions"][name]),
                    reference["distributions"][name],
                )
                for name, counts in totals.items()
                if name in reference.get("distributions", {})
            },
            "verdict": verdict(rows, (reference.get("contrast") or {}).get("label")),
            "baseline": _baseline(reference),
        },
    }


def average_z(
    rows: Sequence[ScoredChunk], summary: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[tuple[str, str], float]:
    """Each metric's z against the reference, averaged over scored chunk rows. Each chunk's
    z is first divided by how many times more that metric swings at the chunk's length than
    in a reference window (its calibration's ``length_scale``), so a short chunk's z counts
    standard deviations of the writer's own text at its length.

    For a metric the reference never varies on (no ``sd`` in its ``summary``), every
    differing chunk scores the cap in one direction or the other; average the size instead,
    signed by which side of the reference's mean the chunks' mean value falls, so opposite
    differences cannot cancel.
    """
    zs: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        scale = row["reference"]["calibration"]["length_scale"]
        for group, values in row["reference"]["z"].items():
            for name, z in values.items():
                factor = scale.get(group, {}).get(name, 1.0)
                zs.setdefault((group, name), []).append(z / factor)
    averaged: dict[tuple[str, str], float] = {}
    for (group, name), values in zs.items():
        stats: Mapping[str, Any] = summary.get(group, {}).get(name, {})
        if stats.get("sd"):
            averaged[(group, name)] = sum(values) / len(values)
            continue
        size = sum(abs(z) for z in values) / len(values)
        here = _mean_of([(row["metrics"].get(group) or {}).get(name) for row in rows]) or 0.0
        there = stats.get("mean") or 0.0
        averaged[(group, name)] = -size if here < there else size
    return averaged


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


def _key(metric: str) -> tuple[str, str]:
    group, _, name = metric.partition(".")
    return group, name


def documents(rows: Sequence[ScoredChunk], reference: ReferenceReport) -> list[DocumentEntry]:
    """Scored chunk rows grouped by document (``document_of``), in input order, each judged
    by ``calibration.verdict`` on its own chunks, the function that makes the pooled
    headline, so each document is read at its own chunks' lengths: mean Delta and
    likeness, their verdicts (or "too short to judge", with ``reason``), and the metrics
    that differ most.

    Each is named by ``document_label`` and keeps its saved ``path``, its source.
    """
    grouped: dict[tuple[str, str], list[ScoredChunk]] = {}
    for row in rows:
        grouped.setdefault((row["source"], base_id(row["id"])), []).append(row)
    label = (reference.get("contrast") or {}).get("label")
    entries: list[DocumentEntry] = []
    for name, members in zip(_document_names(list(grouped)), grouped.values(), strict=True):
        judged = verdict(members, label)
        # The figures cover the chunks long enough to judge, as the verdict does.
        used = [row for row in members if row["reference"]["calibration"]["judged"]] or members
        mean_delta = judged["delta"]["value"]
        likeness = judged["likeness"]
        averaged = average_z(used, reference["summary"])
        largest = sorted(averaged.items(), key=lambda item: -abs(item[1]))[:DOCUMENT_TRAITS]
        likeness_verdict: str | None = None
        if likeness and mean_delta is not None:
            level = likeness["level"]
            likeness_verdict = str(
                LikenessVerdict.TOO_SHORT if level is None else LIKENESSES[level]
            )
        entry: DocumentEntry = {
            "name": name,
            "path": members[0]["source"],
            "chunks": len(members),
            "words": judged["words"],
            "judged": judged["judged"],
            "chunks_judged": judged["chunks_judged"],
            "reason": judged["reason"],
            "delta": mean_delta,
            "verdict": str(Verdict.NOT_COMPARABLE if mean_delta is None else judged["verdict"]),
            "likeness": likeness["value"] if likeness else None,
            "likeness_verdict": likeness_verdict,
            "flagged": judged["flagged"],
            "differences": [
                {"metric": f"{group}.{metric}", "z": z} for (group, metric), z in largest
            ],
        }
        if likeness:
            # Each metric's mean share of the likeness over the document's chunks.
            shares: dict[str, float] = {}
            for row in used:
                for signal in row["reference"].get("likeness_signals", []):
                    share = signal["contribution"] / len(used)
                    shares[signal["metric"]] = shares.get(signal["metric"], 0.0) + share
            ranked = sorted(shares.items(), key=lambda item: -item[1])[:DOCUMENT_TRAITS]
            entry["signals"] = [
                {"metric": metric, "z": averaged.get(_key(metric), 0.0), "contribution": share}
                for metric, share in ranked
            ]
        entries.append(entry)
    return entries


def _create_beside(path: Path) -> tuple[int, Path]:
    """Create a new, uniquely named hidden file next to ``path``, open for writing.

    Its mode is final before any bytes are written: that of the file it will replace
    (permission bits only, never setuid, setgid or sticky), or else 0o666 less the umask.
    Unlike ``tempfile.mkstemp``, which makes the file owner-only, the umask is applied by
    the OS, so no process-wide umask has to be read or changed.
    """
    try:
        existing: int | None = stat.S_IMODE(path.stat().st_mode) & 0o777
    except FileNotFoundError:
        existing = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        temporary = path.parent / f".{path.name}.{secrets.token_hex(4)}"
        try:
            # Created no wider than the file it replaces, then set to exactly its mode,
            # since the umask may have narrowed it.
            descriptor = os.open(temporary, flags, 0o666 if existing is None else existing)
        except FileExistsError:
            continue
        if existing is not None:
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(descriptor, existing)
                else:
                    os.chmod(temporary, existing)
            except BaseException:
                os.close(descriptor)
                temporary.unlink(missing_ok=True)
                raise
        return descriptor, temporary
    raise FileExistsError(f"could not create a temporary file beside {path}")


def dumps_report(report: Mapping[str, Any]) -> str:
    """A report as the JSON text ``write_report`` saves, with floats rounded."""
    return json.dumps(_round(report), ensure_ascii=False, indent=2) + "\n"


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Save a report atomically: write a new file beside ``path``, then rename it over.

    The file keeps the permission bits of the file it replaces, or gets read and write as
    the umask allows. If ``path`` is a symlink, the link itself is replaced by a regular
    file with its target's permissions; the target is left unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_report(report)
    descriptor, temporary = _create_beside(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
