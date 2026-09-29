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
import os
import re
import secrets
import stat
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypedDict, cast

from styleprofile.calibration import (
    CALIBRATION_LENGTHS,
    CALIBRATION_WORDS,
    CONTRAST_CALIBRATION_WORDS,
    MIN_CONTRAST_PIECES,
    MIN_JUDGED_WORDS,
    AtLength,
    Lengths,
    calibrate_lengths,
    held_out_pieces,
    plan_pieces,
    verdict,
)
from styleprofile.core import (
    LIKENESSES,
    LikenessVerdict,
    Note,
    NoteCode,
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
    EvaluationReport,
    LengthCalibration,
    LikenessSignal,
    MetricStats,
    Problem,
    ReferenceReport,
    Report,
    ReportSettings,
    ScoredChunk,
    ScoreReport,
    Summary,
    Unseen,
    find_problem,
)
from styleprofile.surface import (
    Metrics,
    Prose,
    block_word_count,
    char_trigrams,
    classify,
    jensen_shannon,
    masked_bigrams,
    prose,
    strip_front_matter,
    surface_metrics,
    words,
)
from styleprofile.syntax import Parser, pos_trigrams, syntax_metrics
from styleprofile.weighting import (
    LENGTH_AUC_WARNING,
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
    saved.
    """

    id: str
    source: str
    text: str
    path: str | None = field(default=None, compare=False, repr=False)


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


def _jsonl_chunks(
    text: str,
    name: str,
    source: str,
    path: str | None,
    text_field: str | Sequence[str] | None,
    repeated: list[tuple[str, int]] | None = None,
) -> list[Chunk]:
    """Read JSONL ``text``: ``name`` labels errors and default ids, ``source`` is saved for
    each record, and ``path`` is the file it came from (None for stdin).

    Each record is its own document. Records that share an id are told apart by line
    (``same@3``), and each such id and how many records have it is added to ``repeated``;
    an id that ends like a window suffix is escaped (``literal_id``)."""
    fields = _text_fields(text_field)
    records: list[tuple[int, str, str]] = []
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
        record_id = record.get("id")
        # An id of 0 is kept; a missing, null or empty id falls back to the line.
        label = Path(name).name
        chunk_id = (
            f"{label}:{line_number}" if record_id in (None, "") else literal_id(str(record_id))
        )
        records.append((line_number, chunk_id, record[field]))
    counts = Counter(chunk_id for _, chunk_id, _ in records)
    shared = {chunk_id: count for chunk_id, count in counts.items() if count > 1}
    if repeated is not None:
        repeated += shared.items()
    return [
        Chunk(f"{chunk_id}@{line}" if chunk_id in shared else chunk_id, source, body, path)
        for line, chunk_id, body in records
    ]


def literal_id(value: str) -> str:
    """An id from outside (a JSONL id, a ``Text`` name) that window suffixes cannot be read
    into: a trailing ``#w2`` becomes ``%23w2``, so ``base_id`` strips only the suffixes
    ``window`` adds, and a record ``x#w2`` stays apart from a record ``x``."""
    return _WINDOW_SUFFIX.sub(lambda match: match.group(0).replace("#", "%23"), value)


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
    repeated: list[tuple[str, int]] | None = None,
) -> list[Chunk]:
    """Read JSONL that was asked for rather than detected, saying so when it fails."""
    try:
        return _jsonl_chunks(text, name, source, path, text_field, repeated)
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
) -> list[Chunk]:
    text = _read_text(path)
    fmt = _format_of(path, input_format)
    if fmt == JSONL:
        repeated: list[tuple[str, int]] = []
        if input_format == JSONL and path.suffix.lower() != ".jsonl":
            chunks = _forced_jsonl(text, str(path), source, str(path), text_field, repeated)
        else:
            chunks = _jsonl_chunks(text, str(path), source, str(path), text_field, repeated)
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
    return [Chunk(chunk_id, source, text, str(path))]


def _stdin_chunks(
    input_format: str, text_field: str | Sequence[str] | None, notes: list[Note]
) -> list[Chunk]:
    text = _decode(sys.stdin.buffer.read(), "stdin")
    repeated: list[tuple[str, int]] = []
    if input_format == JSONL:
        chunks = _forced_jsonl(text, "stdin", "stdin", None, text_field, repeated)
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
        chunks = _jsonl_chunks(text, "stdin", "stdin", None, text_field, repeated)
        notes += [_repeated_note("stdin", repeated)] if repeated else []
        return chunks
    if fmt == AUTO and looks_like_html(text):
        fmt = HTML
        notes.append(Note("stdin looks like HTML, so it is read as HTML", NoteCode.READ_AS_HTML))
    if fmt == HTML:
        text = html_to_markdown(text)
        if not re.search(r"\w", text):
            notes.append(Note("stdin has no readable text after conversion", NoteCode.EMPTY_HTML))
    return [Chunk("stdin", "stdin", text)]


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
) -> list[Chunk]:
    """Read Markdown, text, HTML or JSONL files, directories of them, or ``-`` for stdin.

    ``text_field`` names the JSONL field that holds the text, or several to try in order;
    by default the first of ``TEXT_FIELDS`` that a record has.

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
    chunks: list[Chunk] = []
    for value in inputs:
        if value == "-":
            chunks.extend(_stdin_chunks(input_format, text_field, notes))
            continue
        path = expand_path(value).resolve()
        detected = _Detected([], [])
        if not path.is_dir():
            [source] = _name_sources(value, path, [path], names)
            chunks.extend(
                _file_chunks(path, value, path.name, source, input_format, text_field, detected)
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
                    _file_chunks(item, label, chunk_id, source, input_format, text_field, detected)
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
    if source == "stdin" or source.endswith(base_id(chunk.id)):
        return source
    return f"record {base_id(chunk.id)} in {source}"


def drop_duplicates(
    chunks: Sequence[Chunk], seen: dict[str, str] | None = None
) -> tuple[list[Chunk], Note | None]:
    """Keep the first of documents whose text is word-for-word the same.

    Chunks are grouped into documents by ``chunk_document`` (their real file and record, so
    windows of one document stay together and two files that share a saved name stay
    apart). Each document's words, lowercased, are compared with front matter, punctuation
    and Markdown markup left out, so a post and its generated HTML page (converted to
    Markdown) match; this reads each document once with one regular expression rather than
    parsing its Markdown, which measuring does later. A file given twice is a different
    check, made when inputs are read.

    ``seen`` maps the text of documents read earlier (another input set) to their label,
    and is updated, so a contrast draft that repeats a reference document is dropped too.
    Twins would sit on both sides of held-out calibration and make it look too tight.
    Documents under ``DUPLICATE_MIN_WORDS`` words are always kept. The note names documents
    by their saved sources, never by path.
    """
    seen = {} if seen is None else seen
    documents: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        documents.setdefault(chunk_document(chunk), []).append(chunk)
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
        else:
            seen[key] = label
    kept = [chunk for chunk in chunks if chunk_document(chunk) not in dropped_documents]
    if not dropped:
        return kept, None
    copy, original = dropped[0]
    count = len(dropped)
    repeating = "1 document that repeats" if count == 1 else f"{count:,} documents that repeat"
    note = (
        f"dropped {repeating} another word for word, keeping the first copy (for example, "
        f"{copy} repeats {original})"
    )
    return kept, Note(note, NoteCode.DUPLICATES)


def window(chunks: Sequence[Chunk], window_words: int) -> list[Chunk]:
    """Split chunks into roughly ``window_words``-word pieces at Markdown block boundaries.

    Sizes count prose words only (code, URLs and markup excluded). Fenced code stays whole
    and a list's indented continuation stays with its list. A window closes early rather
    than grow past one and a half windows. A remainder under half a window joins the
    previous piece, so no window is shorter than half a window unless its whole chunk is;
    that merge, or a single long block, can make a window longer than one and a half.
    """
    limit = window_words * 1.5
    windows: list[Chunk] = []
    for chunk in chunks:
        blocks = classify(chunk.text)
        counts = [block_word_count(block) for block in blocks]
        pieces: list[tuple[list[str], int]] = []
        current: list[str] = []
        count = 0
        for index, (block, block_words) in enumerate(zip(blocks, counts, strict=True)):
            closes_early = count >= window_words / 2 and count + block_words > limit
            if current and closes_early and not block.continues_list:
                pieces.append((current, count))
                current, count = [], 0
            current.append(block.raw)
            count += block_words
            following = blocks[index + 1] if index + 1 < len(blocks) else None
            if count >= window_words and not (following and following.continues_list):
                pieces.append((current, count))
                current, count = [], 0
        if current and pieces and count < window_words / 2:
            pieces[-1] = (pieces[-1][0] + current, pieces[-1][1] + count)
        elif current or not pieces:
            # A chunk with no prose stays as one window, so it is counted as skipped.
            pieces.append((current, count))
        windows.extend(
            Chunk(f"{chunk.id}#w{index}", chunk.source, "\n\n".join(raw_blocks), chunk.path)
            for index, (raw_blocks, _) in enumerate(pieces, start=1)
        )
    return windows


def _merge(target: Metrics, extra: Metrics) -> Metrics:
    return {**target, **extra}


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


# Windowing already-windowed chunks stacks suffixes (``post#w1#w2``); all of them go.
_WINDOW_SUFFIX = re.compile(r"(?:#w\d+)+$")


def document_of(source: str, chunk_id: str) -> str:
    """The document a chunk came from: its file plus its id, with a window suffix removed.

    Windows ``post#w3`` share their document; files with the same name in different
    folders, and JSONL records with the same id in different files, stay separate.
    """
    return f"{source}\x1f{base_id(chunk_id)}"


def base_id(chunk_id: str) -> str:
    """A chunk's id without the ``#wN`` suffixes that windowing adds."""
    return _WINDOW_SUFFIX.sub("", chunk_id)


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
) -> ChunkScore:
    """One chunk's scores, with Delta weights and likeness scaled by the reference's
    held-out rms at the chunk's own length (``at``)."""
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    summary = reference["summary"]
    unseen: list[Unseen] = [
        {"metric": f"{group}.{name}", "value": metrics[group][name], "reference_value": mean}
        for (group, name), z in z_scores.items()
        if z and not summary[group][name].get("sd")
        for mean in [summary[group][name]["mean"]]
    ]
    overall, by_group = delta(z_scores, delta_weights(at.rms))
    deviations = sorted(z_scores.items(), key=lambda item: -abs(item[1]))[:DEVIATIONS_SHOWN]
    liked: _Liked = {}
    if prepared.effects:
        score, signals = likeness(z_scores, prepared.effects, at.rms)
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


@dataclass(frozen=True)
class _Piece:
    """A shorter piece cut from a measured chunk for length calibration (``calibration``)."""

    chunk: int  # index of the chunk it was cut from, among the measured chunks
    length: int  # the length it was cut for, in prose words
    metrics: Metrics


@dataclass(frozen=True)
class _Measured:
    chunks: list[Chunk]
    metrics: list[Metrics]
    distributions: list[dict[str, Counter[str]]]
    empty: int
    below: int
    pieces: list[_Piece]


def _measure(
    chunks: Sequence[Chunk],
    parser: Parser | None,
    min_words: int,
    *,
    allow_empty: bool = False,
    piece_lengths: Sequence[int] = (),
    piece_words: int = CALIBRATION_WORDS,
) -> _Measured:
    """Parse each chunk once, drop chunks without enough prose, and compute every metric.

    With ``piece_lengths``, also cut up to ``piece_words`` words of the chunks into pieces
    of those lengths and measure them (``calibration.plan_pieces``). A piece's syntax
    metrics come from its span of the chunk's parse, so nothing is parsed twice. With no
    chunk left this is an error, unless ``allow_empty``."""
    parsed_all = [prose(chunk.text) for chunk in chunks]
    sizes = [len(words(parsed.text)) for parsed in parsed_all]
    empty = sum(not size for size in sizes)
    below = sum(0 < size < min_words for size in sizes)
    kept = [
        (chunk, parsed, size)
        for chunk, parsed, size in zip(chunks, parsed_all, sizes, strict=True)
        if size and size >= min_words
    ]
    texts = [parsed.text for _, parsed, _ in kept]
    if not kept and allow_empty:
        return _Measured([], [], [], empty, below, [])
    if not kept:
        raise StyleProfileError(
            f"no chunks with at least {max(min_words, 1)} prose word(s) to profile "
            f"({empty} had no prose, {below} were shorter)",
            code="no_chunks",
        )
    chunk_metrics = [surface_metrics(chunk.text, parsed) for chunk, parsed, _ in kept]
    chunk_distributions: list[dict[str, Counter[str]]] = [
        {"masked_bigram": masked_bigrams(text), "char_trigram": char_trigrams(text)}
        for text in texts
    ]
    planned = plan_pieces(
        [chunk.text for chunk, _, _ in kept],
        [size for _, _, size in kept],
        piece_lengths,
        piece_words,
    )
    pieces: list[tuple[int, int, Prose, Metrics]] = []
    for index, length, markdown in planned:
        parsed = prose(markdown)
        if words(parsed.text):
            pieces.append((index, length, parsed, surface_metrics(markdown, parsed)))
    if parser is not None:
        by_chunk: dict[int, list[int]] = defaultdict(list)
        for position, (index, _, _, _) in enumerate(pieces):
            by_chunk[index].append(position)
        for index, doc in enumerate(parser.docs(texts)):
            chunk_metrics[index] = _merge(chunk_metrics[index], syntax_metrics(doc))
            chunk_distributions[index]["pos_trigram"] = pos_trigrams(doc)
            cursors: dict[int, int] = defaultdict(int)
            for position in by_chunk.get(index, ()):
                _, length, parsed, metrics = pieces[position]
                found = _span(doc, texts[index], parsed.text, cursors[length])
                if found is None:
                    continue  # the piece's prose is not verbatim in the chunk's; no syntax
                span, cursors[length] = found
                pieces[position] = (index, length, parsed, _merge(metrics, syntax_metrics(span)))
    return _Measured(
        [chunk for chunk, _, _ in kept],
        chunk_metrics,
        chunk_distributions,
        empty,
        below,
        [_Piece(index, length, metrics) for index, length, _, metrics in pieces],
    )


def chunk_document(chunk: Chunk) -> str:
    """The document ``chunk`` belongs to, keyed on the file it was read from when known:
    saved sources are short names, which two separately loaded inputs can share."""
    return document_of(chunk.path or chunk.source, chunk.id)


def _documents(chunks: Sequence[Chunk]) -> list[str]:
    return [chunk_document(chunk) for chunk in chunks]


def _span(doc: Any, text: str, part: str, start: int) -> tuple[Any, int] | None:
    """The span of a parsed chunk that holds ``part`` of its text, searching from ``start``,
    and where the part ends."""
    offset = text.find(part, start)
    if offset < 0:
        offset = text.find(part)
    if offset < 0:
        return None
    span = doc.char_span(offset, offset + len(part), alignment_mode="expand")
    return (span, offset + len(part)) if span is not None else None


@dataclass(frozen=True)
class _Calibrated:
    """The reference's held-out z-scores: of its chunks, and of the pieces of each
    calibrated length (``calibration``) with the documents they came from."""

    held: list[ZScores]
    pieces: dict[int, tuple[list[ZScores], list[str]]]
    # The report's ``calibration.by_length``, which a contrast set adds its ranges to.
    by_length: dict[str, LengthCalibration]
    # The intraclass correlation the pieces' bounds use (``calibration.similarity``).
    icc: float = 0.0


def _calibrate(report: ReferenceReport, measured: _Measured) -> _Calibrated | None:
    """Held-out reliability and Delta range, when the chunks span at least two documents,
    for the chunks and for shorter pieces cut from them (``calibration.by_length``).

    Without them the profile still works as a reference, but Delta falls back to capped z
    and --contrast is unavailable; scoring against such a reference warns about it.
    """
    documents = _documents(measured.chunks)
    if len(set(documents)) < 2:
        return None
    floor = floors(report["summary"])
    held = held_out_z(measured.metrics, documents, floor)
    calibration = calibrate_delta(held, documents)
    if calibration is None:
        return None
    report["reliability"] = nest(reliability(held, documents))
    by_length: dict[str, LengthCalibration] = {}
    pieces: dict[int, tuple[list[ZScores], list[str]]] = {}
    # Every length's pieces in one pass, so the windows' sums are taken once.
    all_held = held_out_pieces(
        measured.metrics,
        documents,
        [piece.metrics for piece in measured.pieces],
        [documents[piece.chunk] for piece in measured.pieces],
        floor,
    )
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
        "chunk_words": statistics.median(
            metrics["size"]["words"] or 0.0 for metrics in measured.metrics
        ),
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

    reference_held: list[ZScores]
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
    )
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
        [metrics["size"]["words"] or 0.0 for metrics in reference.metrics],
        # Measured chunks all have prose, so words is never None.
        [metrics["size"]["words"] or 0.0 for metrics in measured.metrics],
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
) -> _Base:
    """The part of every report of ``kind`` that describes its own chunks, plus a row of
    metrics per chunk: score reports need them to show which chunks stand out, while a
    reference stores only its summary unless asked to keep them."""
    chunk_metrics = measured.metrics
    totals: dict[str, Counter[str]] = {}
    for distributions in measured.distributions:
        for name, counts in distributions.items():
            totals.setdefault(name, Counter()).update(counts)

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
    short = sum((metrics["size"]["words"] or 0) < SHORT_CHUNK_WORDS for metrics in chunk_metrics)
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
            "syntax_used": (
                {
                    "model": parser.model,
                    "model_version": parser.model_version,
                    "spacy_version": parser.spacy_version,
                }
                if parser
                else None
            ),
        },
    )
    described: _Described = {
        "settings": recorded,
        "chunk_count": len(measured.chunks),
        "document_count": len(set(_documents(measured.chunks))),
        "word_count": sum(int(metrics["size"]["words"] or 0) for metrics in chunk_metrics),
        "summary": summarize(chunk_metrics),
        "distributions": {name: _distribution(counts, top_k) for name, counts in totals.items()},
    }
    rows: list[ChunkRow] = [
        {"id": chunk.id, "source": chunk.source, "metrics": metrics}
        for chunk, metrics in zip(measured.chunks, chunk_metrics, strict=True)
    ]
    return _Base(described, rows, warnings, totals)


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
    profile small however large the corpus; ``keep_chunks`` saves them too, for debugging."""
    return _build_reference(
        chunks,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings={"window_words": 0, **(settings or {})},
        contrast=contrast,
        contrast_label=contrast_label,
        keep_chunks=keep_chunks,
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
    )
    assert fit is not None  # a contrast always produces a fit or raises
    return report, fit


def z_against_reference(
    report: ReferenceReport, chunks: Sequence[Chunk], parser: Parser | None, min_words: int
) -> tuple[list[Chunk], list[ZScores]]:
    """Measure chunks and take their z-scores against a reference report, as the contrast
    drafts are; returns the chunks kept (enough prose), possibly none, and their z-scores."""
    measured = _measure(chunks, parser, min_words, allow_empty=True)
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
) -> tuple[ReferenceReport, ContrastFit | None]:
    measured = _measure(
        chunks,
        parser,
        min_words,
        piece_lengths=CALIBRATION_LENGTHS if calibrate_lengths else (),
    )
    base = _base_report(
        measured,
        kind=REFERENCE,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
    )
    report: ReferenceReport = {
        "version": VERSION,
        "kind": REFERENCE,
        **base.described,
        "warnings": base.warnings,
    }
    if keep_chunks:
        report["chunks"] = base.rows
    calibrated = _calibrate(report, measured)
    fit: ContrastFit | None = None
    if contrast is not None:
        report["contrast"], fit = _learn_contrast(
            report, measured, calibrated, contrast, contrast_label, parser, min_words
        )
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
) -> ScoreReport:
    """Profile sample chunks and score each against ``reference``: z-scores, Delta, pattern
    divergence and, when the reference learned a contrast, likeness to the contrast set.

    This is the lower-level step under ``Profile.score``: nothing is inherited from the
    reference, so pass chunks cut into the reference's windows, its ``min_words`` and a
    ``parser`` when it has syntax metrics, or use ``Profile.score``, which does all that.
    A reference of another report version is refused (``check_version``).

    Each chunk is judged at its own length (``calibration``). The means and the
    ``verdict`` cover the chunks long enough to judge; when none is, they cover every
    chunk, and the verdict is "too short to judge"."""
    check_version(reference, "the reference")
    measured = _measure(chunks, parser, min_words)
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
    return {
        "version": VERSION,
        "kind": SCORE,
        **base.described,
        "warnings": warnings,
        "chunks": rows,
        "documents": documents(rows, reference),
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
_ESCAPED_SUFFIX = re.compile(r"(?:%23w\d+)+(?=(?:@\d+)?$)")

# A number that keeps a saved source unique (``posts (2)/a.md``, ``notes (2).md``).
_SOURCE_NUMBER = re.compile(r" \(\d+\)(?=(?:\.[^/.]*)?(?:/|$))")


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
