"""Read files, folders and JSONL records into named chunks."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from styleprofile.core import Note, NoteCode, StyleProfileError
from styleprofile.corpus.ids import _GROUPED, _PART_SUFFIX, _RECORD, document_of, group_id
from styleprofile.corpus.types import Chunk
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
# Directory walks skip dot-directories (.git, .venv) and these vendored ones.
SKIPPED_DIRS = frozenset({"node_modules", "__pycache__", "site-packages"})
# The saved name of an input whose own name says nothing (the home directory, "/").
UNNAMED_ROOT = "input"


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
    ungrouped: dict[tuple[str, str], Chunk] = field(default_factory=dict)

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
        record_id = record.get("id")
        # An id of 0 is kept; a missing, null or empty id falls back to the line.
        if record_id in (None, ""):
            chunk_id = f"{inside or Path(name).name}:{line_number}"
        else:
            chunk_id = literal_id(str(record_id))
            chunk_id = f"{inside}:{chunk_id}" if inside else chunk_id
        ungrouped.append((line_number, chunk_id, record[field]))
    counts = Counter(chunk_id for _, chunk_id, _ in ungrouped)
    shared = {chunk_id: count for chunk_id, count in counts.items() if count > 1}
    if repeated is not None and not grouped:
        repeated += shared.items()
    chunks: list[Chunk] = []
    for line, chunk_id, body in ungrouped:
        unique = f"{chunk_id}@{line}" if chunk_id in shared else chunk_id
        document = f"{path or source}{_RECORD}{unique}"
        chunks.append(Chunk(unique, source, body, path, folder, document))
    if grouped:
        if records.missing != records.count:
            records.ungrouped.clear()
            return grouped
        records.ungrouped.update(
            ((group.path or group.source, group.id), chunk)
            for group, chunk in zip(grouped, chunks, strict=True)
        )
        return grouped
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
    return NoteCode.REPEATED_ID.note("0", f"{label}", f"{shared}", f"{more}")


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
        notes.append(NoteCode.READ_AS_JSONL.note("0"))
        chunks = _jsonl_chunks(text, "stdin", "stdin", None, text_field, records, repeated=repeated)
        notes += [_repeated_note("stdin", repeated)] if repeated else []
        return chunks
    if fmt == AUTO and looks_like_html(text):
        fmt = HTML
        notes.append(NoteCode.READ_AS_HTML.note("0"))
    if fmt == HTML:
        text = html_to_markdown(text)
        if not re.search(r"\w", text):
            notes.append(NoteCode.EMPTY_HTML.note("0"))
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
                notes.append(NoteCode.SKIPPED_DIRS.note("0", f"{value}", f"{_listing(generated)}"))
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
                    NoteCode.SKIPPED_FILES.note(
                        "0", f"{described[0]}", f"{value}", f"{described[1]}"
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
                (NoteCode.READ_AS_HTML_IN_FOLDER if path.is_dir() else NoteCode.READ_AS_HTML).note(
                    "files", _listing(detected.sniffed), looks, they
                )
            )
        notes += [_repeated_note(label, repeated) for label, repeated in detected.repeated]
        if detected.empty:
            has = "has" if len(detected.empty) == 1 else "have"
            notes.append(NoteCode.EMPTY_HTML.note("1", f"{_listing(detected.empty)}", f"{has}"))
    return chunks
