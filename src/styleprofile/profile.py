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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from styleprofile.calibration import (
    CALIBRATION_LENGTHS,
    CALIBRATION_WORDS,
    CONTRAST_CALIBRATION_WORDS,
    MIN_CONTRAST_PIECES,
    AtLength,
    Lengths,
    calibrate_length,
    held_out_pieces,
    plan_pieces,
    verdict,
)
from styleprofile.core import Note, NoteCode, StyleProfileError
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
REFERENCE = "reference"
SCORE = "score"
# Written by ``styleprofile evaluate`` (see the evaluate module); shown, never scored against.
EVALUATION = "evaluation"
KINDS = (REFERENCE, SCORE, EVALUATION)
UNREADABLE = "not a style profile this version of styleprofile can read; rebuild it"
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
) -> list[Chunk]:
    """Read JSONL ``text``: ``name`` labels errors and default ids, ``source`` is saved for
    each record, and ``path`` is the file it came from (None for stdin)."""
    fields = _text_fields(text_field)
    chunks: list[Chunk] = []
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
        chunk_id = f"{label}:{line_number}" if record_id in (None, "") else str(record_id)
        chunks.append(Chunk(chunk_id, source, record[field], path))
    return chunks


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
) -> list[Chunk]:
    """Read JSONL that was asked for rather than detected, saying so when it fails."""
    try:
        return _jsonl_chunks(text, name, source, path, text_field)
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
        if input_format == JSONL and path.suffix.lower() != ".jsonl":
            return _forced_jsonl(text, str(path), source, str(path), text_field)
        return _jsonl_chunks(text, str(path), source, str(path), text_field)
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
    if input_format == JSONL:
        return _forced_jsonl(text, "stdin", "stdin", None, text_field)
    fmt = input_format
    if fmt == AUTO and looks_like_jsonl(text):
        notes.append(
            Note(
                "stdin is JSON objects, one per line, so it is read as JSONL",
                NoteCode.READ_AS_JSONL,
            )
        )
        return _jsonl_chunks(text, "stdin", "stdin", None, text_field)
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


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
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


def summarize(chunk_metrics: Sequence[Metrics]) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for metrics in chunk_metrics:
        for group, values in metrics.items():
            for name in values:
                summary.setdefault(group, {}).setdefault(name, {})
    for group, names in summary.items():
        for name in names:
            values = [
                value
                for metrics in chunk_metrics
                if (value := metrics.get(group, {}).get(name)) is not None
            ]
            names[name] = _stats(values)
    return summary


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


def _z_against(metrics: Metrics, summary: dict[str, Any], floor: dict[Key, float]) -> ZScores:
    scored: ZScores = {}
    for group, values in metrics.items():
        if group in UNSCORED_GROUPS:
            continue
        for name, value in values.items():
            stats = summary.get(group, {}).get(name)
            if value is None or not stats or stats.get("mean") is None:
                continue
            z = z_score(
                value,
                stats["mean"],
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


def _prepare(reference: dict[str, Any]) -> _Prepared:
    return _Prepared(
        floor=floors(reference["summary"]),
        effects=flatten((reference.get("contrast") or {}).get("effects", {})),
        lengths=Lengths(reference),
    )


def _score(
    metrics: Metrics,
    distributions: dict[str, Counter[str]],
    reference: dict[str, Any],
    prepared: _Prepared,
    at: AtLength,
) -> dict[str, Any]:
    """One chunk's scores, with Delta weights and likeness scaled by the reference's
    held-out rms at the chunk's own length (``at``)."""
    z_scores = _z_against(metrics, reference["summary"], prepared.floor)
    summary = reference["summary"]
    unseen = [
        {"metric": f"{group}.{name}", "value": metrics[group][name], "reference_value": mean}
        for (group, name), z in z_scores.items()
        if z and not summary[group][name].get("sd")
        for mean in [summary[group][name]["mean"]]
    ]
    overall, by_group = delta(z_scores, delta_weights(at.rms))
    deviations = sorted(z_scores.items(), key=lambda item: -abs(item[1]))[:DEVIATIONS_SHOWN]
    scored: dict[str, Any] = {
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
    }
    if prepared.effects:
        score, signals = likeness(z_scores, prepared.effects, at.rms)
        scored["likeness"] = score
        scored["likeness_signals"] = signals
    scored["calibration"] = at.row()
    return scored


def _mean_of(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def report_kind(report: dict[str, Any]) -> str:
    """``"reference"``, ``"score"`` or ``"evaluation"``, from the report's ``kind``."""
    kind = report.get("kind")
    if kind not in KINDS:
        raise StyleProfileError(
            f"the report has no known kind, so it is {UNREADABLE}", code="outdated"
        )
    return kind


def load_report(path: Path) -> dict[str, Any]:
    """Read any styleprofile report: reference, score or evaluation."""
    if path.is_dir():
        raise StyleProfileError(f"{path} is a directory, not a profile", code="directory")
    if not path.exists():
        raise StyleProfileError(f"{path} not found", code="not_found")
    try:
        report = json.loads(_read_text(path))
    except (json.JSONDecodeError, StyleProfileError):
        report = None
    if not isinstance(report, dict) or ("summary" not in report and "kind" not in report):
        raise StyleProfileError(f"{path} is not a style profile", code="not_a_profile")
    kind = report.get("kind")
    if kind not in KINDS or (kind != EVALUATION and "summary" not in report):
        # A pre-release report without ``kind``, or a kind this version does not know.
        raise StyleProfileError(f"{path} is {UNREADABLE}", code="outdated")
    check_version(report, str(path))
    return report


def check_version(report: dict[str, Any], name: str = "the report") -> None:
    """Refuse a report of another version than this styleprofile writes, saying how to
    make it again: its settings and metrics would be misread rather than migrated."""
    kind = report.get("kind")
    expected = EVALUATION_VERSION if kind == EVALUATION else VERSION
    version = report.get("version")
    if version == expected:
        return
    again = {
        SCORE: "score it again with `styleprofile score`",
        EVALUATION: "run `styleprofile evaluate` again",
    }.get(kind or "", "rebuild it with `styleprofile build`")
    age = "a newer" if isinstance(version, int) and version > expected else "an older"
    raise StyleProfileError(
        f"{name} was made by {age} styleprofile (report version {version}; this one reads "
        f"{expected}); {again}",
        code="outdated",
    )


def load_reference(path: Path) -> dict[str, Any]:
    """Read a reference profile, refusing a score report (a sample scored against one)."""
    reference = load_report(path)
    if report_kind(reference) == EVALUATION:
        raise StyleProfileError(
            f"{path} is an evaluation report, not a reference profile; build a reference "
            "from the writer's own texts",
            code="score_as_reference",
        )
    if report_kind(reference) == SCORE:
        raise StyleProfileError(
            f"{path} is a score report (a sample scored against a reference), not a "
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


def _calibrate(report: dict[str, Any], measured: _Measured) -> _Calibrated | None:
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
    by_length: dict[str, Any] = {}
    pieces: dict[int, tuple[list[ZScores], list[str]]] = {}
    # Every length's pieces in one pass, so the windows' sums are taken once.
    all_held = held_out_pieces(
        measured.metrics,
        documents,
        [piece.metrics for piece in measured.pieces],
        [documents[piece.chunk] for piece in measured.pieces],
        floor,
    )
    for length in sorted({piece.length for piece in measured.pieces}):
        chosen = [index for index, piece in enumerate(measured.pieces) if piece.length == length]
        piece_held = [all_held[index] for index in chosen]
        piece_documents = [documents[measured.pieces[index].chunk] for index in chosen]
        entry = calibrate_length(
            piece_held,
            piece_documents,
            [measured.pieces[index].metrics["size"]["words"] or 0.0 for index in chosen],
        )
        by_length[str(length)] = entry
        if "delta" in entry:
            pieces[length] = (piece_held, piece_documents)
    report["calibration"] = {
        "sources": len(set(documents)),
        "delta": calibration,
        # The windows' own length: the longest anchor of the length calibration.
        "chunk_words": statistics.median(
            metrics["size"]["words"] or 0.0 for metrics in measured.metrics
        ),
        "by_length": by_length,
    }
    return _Calibrated(held, pieces)


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
    report: dict[str, Any],
    reference: _Measured,
    calibrated: _Calibrated | None,
    contrast: Sequence[Chunk],
    label: str,
    parser: Parser | None,
    min_words: int,
) -> tuple[dict[str, Any], ContrastFit]:
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
    _calibrate_contrast_lengths(report, calibrated, fit, measured.pieces, floor)
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
    report: dict[str, Any],
    calibrated: _Calibrated,
    fit: ContrastFit,
    pieces: Sequence[_Piece],
    floor: dict[Key, float],
) -> None:
    """The likeness range at each calibrated length, from reference and contrast pieces."""
    for length, (piece_held, piece_documents) in calibrated.pieces.items():
        chosen = [piece for piece in pieces if piece.length == length]
        entry = report["calibration"]["by_length"][str(length)]
        entry["contrast_pieces"] = len(chosen)
        if len(chosen) < MIN_CONTRAST_PIECES:
            continue
        entry["likeness"] = likeness_range(
            fit.reference_held,
            fit.reference_documents,
            fit.contrast_z,
            piece_held,
            piece_documents,
            [_z_against(piece.metrics, report["summary"], floor) for piece in chosen],
            [fit.contrast_documents[piece.chunk] for piece in chosen],
            fit.learned,
        )


def _base_report(
    measured: _Measured,
    *,
    kind: str,
    parser: Parser | None,
    top_k: int,
    min_words: int,
    settings: dict[str, Any] | None,
    keep_chunks: bool = True,
) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    """The part of every report that describes its own chunks, plus the pooled counts.

    ``keep_chunks`` adds a row of metrics per chunk: score reports need them to show which
    chunks stand out, while a reference stores only its summary unless asked to keep them."""
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
    report: dict[str, Any] = {
        "version": VERSION,
        "kind": kind,
        "settings": {
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
        "chunk_count": len(measured.chunks),
        "document_count": len(set(_documents(measured.chunks))),
        "word_count": sum(int(metrics["size"]["words"] or 0) for metrics in chunk_metrics),
        "summary": summarize(chunk_metrics),
        "distributions": {name: _distribution(counts, top_k) for name, counts in totals.items()},
        "warnings": warnings,
    }
    if keep_chunks:
        report["chunks"] = [
            {"id": chunk.id, "source": chunk.source, "metrics": metrics}
            for chunk, metrics in zip(measured.chunks, chunk_metrics, strict=True)
        ]
    return report, totals


def build_reference(
    chunks: Sequence[Chunk],
    *,
    parser: Parser | None = None,
    top_k: int = 300,
    min_words: int = 1,
    settings: dict[str, Any] | None = None,
    contrast: Sequence[Chunk] | None = None,
    contrast_label: str = "LLM",
    keep_chunks: bool = False,
) -> dict[str, Any]:
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
    settings: dict[str, Any] | None = None,
    contrast_label: str = "LLM",
    calibrate_lengths: bool = True,
) -> tuple[dict[str, Any], ContrastFit]:
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
    report: dict[str, Any], chunks: Sequence[Chunk], parser: Parser | None, min_words: int
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
    settings: dict[str, Any] | None,
    contrast: Sequence[Chunk] | None,
    contrast_label: str,
    keep_chunks: bool,
    calibrate_lengths: bool = True,
) -> tuple[dict[str, Any], ContrastFit | None]:
    measured = _measure(
        chunks,
        parser,
        min_words,
        piece_lengths=CALIBRATION_LENGTHS if calibrate_lengths else (),
    )
    report, _ = _base_report(
        measured,
        kind=REFERENCE,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
        keep_chunks=keep_chunks,
    )
    calibrated = _calibrate(report, measured)
    fit: ContrastFit | None = None
    if contrast is not None:
        report["contrast"], fit = _learn_contrast(
            report, measured, calibrated, contrast, contrast_label, parser, min_words
        )
    return report, fit


def _baseline(reference: dict[str, Any]) -> dict[str, Any]:
    """What rendering a score needs from its reference, so a saved score can be shown without
    the reference file: each metric's mean and spread, the held-out Delta range, and the
    contrast set's name and likeness range."""
    contrast = reference.get("contrast")
    return {
        "chunk_count": reference.get("chunk_count"),
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


def _without_rms(calibration: dict[str, Any] | None) -> dict[str, Any] | None:
    """The calibration without each length's per-metric rms, which only scoring reads."""
    if not calibration:
        return calibration
    return {
        **calibration,
        "by_length": {
            length: {key: value for key, value in entry.items() if key != "reliability"}
            for length, entry in calibration.get("by_length", {}).items()
        },
    }


def score(
    chunks: Sequence[Chunk],
    reference: dict[str, Any],
    *,
    parser: Parser | None = None,
    top_k: int = 300,
    min_words: int = 1,
    settings: dict[str, Any] | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
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
    report, totals = _base_report(
        measured,
        kind=SCORE,
        parser=parser,
        top_k=top_k,
        min_words=min_words,
        settings=settings,
    )
    warnings: list[str] = report["warnings"]
    rows: list[dict[str, Any]] = report["chunks"]
    prepared = _prepare(reference)
    for row, metrics, distributions in zip(
        rows, measured.metrics, measured.distributions, strict=True
    ):
        at = prepared.lengths.at(int(metrics["size"]["words"] or 0))
        row["reference"] = _score(metrics, distributions, reference, prepared, at)

    judged = [row for row in rows if row["reference"]["calibration"]["judged"]]
    scored = [row["reference"] for row in judged or rows]
    groups = sorted({group for scores in scored for group in scores["delta_by_group"]})
    reference_settings = reference.get("settings", {})
    if judged and len(judged) < len(rows):
        left_out = len(rows) - len(judged)
        warnings.append(
            f"{left_out} chunk(s) are too short to judge and are left out of the verdict and "
            "the means"
        )
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
    if reference.get("chunk_count", 0) < 2:
        warnings.append("the reference has one chunk, so it has no spread; split it into windows")
    # 0 and None both mean no windowing.
    own_window = report["settings"].get("window_words") or None
    reference_window = reference_settings.get("window_words") or None
    if own_window != reference_window:
        warnings.append(
            "window sizes differ from the reference "
            f"({own_window or 'off'} vs {reference_window or 'off'}); "
            "z-scores assume equal-sized chunks"
        )
    own_syntax = report["settings"]["syntax_used"]
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
    report["reference"] = {
        # Only the profile's file name: a saved report never reveals where files live.
        "path": root_name(str(reference_path)) if reference_path else None,
        "chunk_count": reference.get("chunk_count"),
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
    }
    return report


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


def dumps_report(report: dict[str, Any]) -> str:
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


def write_report(report: dict[str, Any], path: Path) -> None:
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
