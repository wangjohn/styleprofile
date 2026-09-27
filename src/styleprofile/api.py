"""One pipeline for the library and the command line.

``build`` reads the writer's texts, cuts them into windows, loads the parser and profiles
them; ``Profile.score`` does the same to drafts with the settings the profile was built
with; ``evaluate`` runs the rewording stress test. The ``styleprofile`` command calls these
and only adds flags and output, so the two cannot give different numbers::

    import styleprofile as sp
    from pathlib import Path

    profile = sp.build([Path("posts/")], contrast=[Path("llm-drafts/")])
    result = profile.score(sp.Text("A draft to check against the writer."))
    print(result.verdict, result.delta)

Inputs are paths (``str`` or ``Path``: files, folders, or ``"-"`` for stdin), ``Text`` for
raw text, or ``Chunk`` objects, alone or in a list. A ``str`` is always a path, as on the
command line. Nothing here prints: what a front end should mention comes back as ``notes``,
and problems raise ``StyleProfileError`` with a ``code``.

The lower-level ``build_reference`` and ``score`` in ``styleprofile.profile`` take chunks
exactly as given; use them to control windowing and parsing yourself.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

from styleprofile.display import (
    describe_delta,
    format_evaluation,
    format_reference_summary,
    format_summary,
    mean_ceiling,
)
from styleprofile.evaluate import evaluate_rewording
from styleprofile.profile import (
    REFERENCE,
    TEXT_FIELDS,
    Chunk,
    Note,
    StyleProfileError,
    build_reference,
    dumps_report,
    load_chunks,
    load_reference,
    report_kind,
    score,
    window,
    write_report,
)
from styleprofile.syntax import Parser, SyntaxUnavailableError, load_parser
from styleprofile.weighting import likeness_level, likeness_words

AUTO = "auto"
DEFAULT_WINDOW_WORDS = 500
DEFAULT_TOP_K = 300
# How inputs are read. Only format detection by file extension exists so far.
INPUT_FORMATS = (AUTO,)
SYNTAX_INSTALL = "pip install 'styleprofile[syntax]'"

# Coarse phases passed to a ``progress`` callback, in the order a run reaches them.
READ = "read"
LOAD_PARSER = "load_parser"
BUILD = "build"
SCORE = "score"
EVALUATE = "evaluate"
DONE = "done"


@dataclass(frozen=True)
class Text:
    """Raw text to profile or score, as opposed to a path to read it from.

    ``name`` identifies it in reports (as a file name does) and pairs an edited text with
    its original in ``evaluate``; by default it is ``text1``, ``text2``, ... by position.
    """

    text: str
    name: str | None = None


Input = str | os.PathLike[str] | Text | Chunk
Inputs = Input | Iterable[Input]


@dataclass(frozen=True)
class Progress:
    """Where a run is: its ``phase`` (``READ``, ``LOAD_PARSER``, ``BUILD``, ``SCORE``,
    ``EVALUATE`` or ``DONE``) and, when a phase counts its work, how much of it is done."""

    phase: str
    done: int | None = None
    total: int | None = None


ProgressCallback = Callable[[Progress], None]


@dataclass(frozen=True)
class Settings:
    """How texts are read and cut into chunks. ``build`` records them in the profile, and
    ``Profile.score`` inherits them unless overridden.

    - ``window_words``: split texts into windows of about this many prose words at
      paragraph breaks; 0 profiles each text whole.
    - ``min_words``: drop chunks with fewer prose words than this.
    - ``text_field``: the JSONL field that holds the text; by default the first of
      ``TEXT_FIELDS`` a record has.
    - ``syntax``: ``True`` needs spaCy for the syntax metrics, ``False`` leaves them out, and
      ``"auto"`` uses spaCy when it is installed and adds a note when it is not.
    - ``top_k``: distribution entries kept in a profile.
    - ``input_format``: how inputs are read; ``"auto"`` goes by file extension.
    """

    window_words: int = DEFAULT_WINDOW_WORDS
    min_words: int = 1
    text_field: str | None = None
    syntax: Literal["auto"] | bool = AUTO
    top_k: int = DEFAULT_TOP_K
    input_format: str = AUTO

    def __post_init__(self) -> None:
        # Messages about a setting start with its name, so a front end can name its own
        # option instead (the CLI swaps in the flag).
        if self.window_words < 0:
            raise StyleProfileError(
                "window_words must be 0 (no windowing) or positive", code="window_words"
            )
        if self.min_words < 0:
            raise StyleProfileError("min_words must be 0 or more", code="min_words")
        if self.top_k < 1:
            raise StyleProfileError("top_k must be positive", code="top_k")
        if self.syntax not in (AUTO, True, False):
            raise StyleProfileError(
                f"syntax must be True, False or {AUTO!r}, not {self.syntax!r}", code="syntax"
            )
        if self.input_format not in INPUT_FORMATS:
            raise StyleProfileError(
                f"input_format {self.input_format!r} is not supported; use "
                + ", ".join(repr(name) for name in INPUT_FORMATS),
                code="input_format",
            )


DEFAULTS = Settings()


class _Result:
    """What every result wraps: its report dict (``report``, the saved JSON), and what the
    run noted and read, which are not saved."""

    def __init__(
        self, report: dict[str, Any], *, notes: Sequence[Note] = (), sources: Sequence[str] = ()
    ) -> None:
        self._report = report
        self._notes = tuple(notes)
        self._sources = tuple(sources)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the report as JSON, as the command's ``-o`` does."""
        write_report(self._report, _resolved(path))

    @property
    def report(self) -> dict[str, Any]:
        """The full report, as saved."""
        return self._report

    @property
    def notes(self) -> tuple[Note, ...]:
        """What the run wants the user to know, such as syntax left out or an input given
        twice. The CLI prints these as ``note:`` lines."""
        return self._notes

    @property
    def sources(self) -> tuple[str, ...]:
        """Where the chunks this run read came from: file paths, ``stdin``, or ``<text>``
        (``<contrast>``, ``<LABEL>``) for ``Text`` inputs."""
        return self._sources

    @property
    def warnings(self) -> list[str]:
        """Why the result may be unreliable, as saved in the report."""
        return self._report["warnings"]


class Profile(_Result):
    """A reference profile: what ``build`` makes and ``styleprofile build`` saves."""

    def __init__(
        self,
        report: dict[str, Any],
        *,
        path: str | os.PathLike[str] | None = None,
        notes: Sequence[Note] = (),
        sources: Sequence[str] = (),
    ) -> None:
        if report_kind(report) != REFERENCE:
            raise StyleProfileError(
                f"this is a {report_kind(report)} report, not a reference profile; build a "
                "reference from the writer's own texts",
                code="score_as_reference",
            )
        super().__init__(report, notes=notes, sources=sources)
        self._path = _resolved(path) if path is not None else None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Profile:
        """Read a reference profile saved by ``save`` or ``styleprofile build``."""
        return cls(load_reference(Path(path).expanduser()), path=path)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the profile as JSON; scores made after this record ``path`` as their
        reference."""
        self._path = _resolved(path)
        write_report(self._report, self._path)

    @property
    def path(self) -> Path | None:
        """Where the profile was loaded from or last saved, if anywhere."""
        return self._path

    @property
    def settings(self) -> Settings:
        """The settings it was built with, which ``score`` inherits."""
        recorded = self._report.get("settings") or {}
        return Settings(
            window_words=recorded.get("window_words") or 0,
            min_words=recorded.get("min_words") or 1,
            text_field=recorded.get("text_field"),
            syntax=recorded.get("syntax") is not None,
            top_k=recorded.get("top_k") or DEFAULT_TOP_K,
        )

    def to_text(self, *, full: bool = False, color: bool = False) -> str:
        """The summary ``styleprofile build`` prints; ``full`` adds every metric."""
        return format_reference_summary(self._report, color=color, full=full)

    def score(
        self,
        inputs: Inputs,
        *,
        window_words: int | None = None,
        min_words: int | None = None,
        text_field: str | None = None,
        syntax: Literal["auto"] | bool | None = None,
        input_format: str = AUTO,
        progress: ProgressCallback | None = None,
    ) -> ScoreResult:
        """Score drafts against this profile.

        Window size, minimum words, text field, syntax and ``top_k`` come from the profile;
        a keyword given here overrides it (``window_words=0`` turns windowing off). A
        different window size or syntax setting is warned about in the report, since
        z-scores assume chunks like the reference's. ``input_format`` is not inherited:
        drafts are often in another format than the writer's corpus.
        """
        notes: list[Note] = []
        with _notes_on_error(notes):
            inherited = self.settings
            if syntax is None:
                syntax = AUTO if inherited.syntax else False
            settings = Settings(
                window_words=inherited.window_words if window_words is None else window_words,
                min_words=inherited.min_words if min_words is None else min_words,
                text_field=text_field or inherited.text_field,
                syntax=syntax,
                top_k=inherited.top_k,
                input_format=input_format,
            )
            if settings.min_words != inherited.min_words:
                notes.append(
                    Note(
                        f"min_words {settings.min_words} overrides the reference's "
                        f"{inherited.min_words}",
                        "min_words",
                    )
                )
            # An explicit text field is the only one read; the reference's is tried first.
            fields: str | tuple[str, ...] | None = text_field
            if not text_field and inherited.text_field:
                fields = (
                    inherited.text_field,
                    *(name for name in TEXT_FIELDS if name != inherited.text_field),
                )
            step = _progress(progress)
            items = _items(inputs)
            _stdin_once(items)
            parser = _parser(
                settings.syntax,
                notes,
                (
                    "the reference has syntax metrics but spaCy is not installed, so syntax is "
                    f"left out of this score; {SYNTAX_INSTALL} to include it"
                )
                if inherited.syntax
                else (
                    "spaCy is not installed, so this score has surface metrics only; "
                    f"{SYNTAX_INSTALL} to include syntax"
                ),
                step,
            )
            step(READ)
            chunks = _read(items, fields, set(), notes, "<text>")
            step(SCORE)
            report = score(
                _windowed(chunks, settings.window_words),
                self._report,
                parser=parser,
                top_k=settings.top_k,
                min_words=settings.min_words,
                reference_path=self._path,
                settings={
                    "inputs": _described(inputs, items),
                    "text_field": settings.text_field,
                    "window_words": settings.window_words or None,
                    "min_words": settings.min_words,
                },
            )
            step(DONE)
            return ScoreResult(
                report, self._report, notes=notes, sources=[chunk.source for chunk in chunks]
            )

    def __repr__(self) -> str:
        where = f" from {self._path}" if self._path else ""
        return (
            f"<Profile{where}: {self._report['chunk_count']} chunks, "
            f"{self._report['word_count']:,} words>"
        )


class ScoreResult(_Result):
    """Drafts scored against a profile: what ``styleprofile score`` prints and saves."""

    def __init__(
        self,
        report: dict[str, Any],
        reference: dict[str, Any] | None = None,
        *,
        notes: Sequence[Note] = (),
        sources: Sequence[str] = (),
    ) -> None:
        super().__init__(report, notes=notes, sources=sources)
        # A score report carries a copy of what rendering needs from its reference.
        self._reference = reference or report["reference"]["baseline"]

    @property
    def chunk_count(self) -> int:
        return self._report["chunk_count"]

    @property
    def delta(self) -> float | None:
        """Mean Delta over the scored chunks: distance from the writer in the writer's own
        standard deviations. None when no metric could be compared."""
        return self._report["reference"]["delta_mean"]

    @property
    def verdict(self) -> str | None:
        """Delta in words against the writer's held-out range: ``close``, ``somewhat
        different``, ``clearly different`` or ``very different``."""
        if self.delta is None:
            return None
        held = (self._reference.get("calibration") or {}).get("delta") or {}
        return describe_delta(self.delta, mean_ceiling(held, self.chunk_count))

    @property
    def contrast_label(self) -> str | None:
        """The contrast set's name (``LLM``) when the profile was built with one."""
        contrast = self._reference.get("contrast")
        return contrast["label"] if contrast else None

    @property
    def likeness(self) -> float | None:
        """Mean likeness to the contrast set, when the profile was built with one."""
        return self._report["reference"].get("likeness_mean")

    @property
    def likeness_verdict(self) -> str | None:
        """Likeness in words, such as ``like the reference`` or ``leans LLM``."""
        contrast = self._reference.get("contrast")
        if not contrast or self.likeness is None:
            return None
        level = likeness_level(self.likeness, contrast["calibration"], self.chunk_count)
        return likeness_words(level, contrast["label"])

    def to_text(self, *, full: bool = False, color: bool = False) -> str:
        """The comparison ``styleprofile score`` prints; ``full`` shows every metric."""
        return format_summary(self._report, self._reference, color=color, full=full)

    def __repr__(self) -> str:
        if self.delta is None:
            return f"<ScoreResult: {self.chunk_count} chunks, nothing compared>"
        return f"<ScoreResult: {self.verdict} (Delta {self.delta:.2f})>"


class Evaluation(_Result):
    """The rewording stress test: what ``styleprofile evaluate`` prints and saves."""

    def to_text(self, *, color: bool = False) -> str:
        """The tables ``styleprofile evaluate`` prints."""
        return format_evaluation(self._report, color=color)


def build(
    inputs: Inputs,
    settings: Settings = DEFAULTS,
    *,
    contrast: Inputs | None = None,
    contrast_label: str = "LLM",
    progress: ProgressCallback | None = None,
) -> Profile:
    """Build a reference profile from a writer's texts, as ``styleprofile build`` does.

    Given ``contrast`` texts (LLM drafts of the same briefs, say), the profile also learns
    what separates the writer from them and scores likeness to them. A file or folder given
    twice, in ``inputs`` or ``contrast``, is read once, with a note.
    """
    notes: list[Note] = []
    with _notes_on_error(notes):
        step = _progress(progress)
        step(READ)
        items = _items(inputs)
        contrast_items = _items(contrast) if contrast is not None else None
        _stdin_once([*items, *(contrast_items or [])])
        seen: set[str] = set()
        chunks = _read(items, settings.text_field, seen, notes, "<text>")
        contrast_chunks = (
            _read(contrast_items, settings.text_field, seen, notes, "<contrast>")
            if contrast_items is not None
            else None
        )
        parser = _parser(
            settings.syntax,
            notes,
            "spaCy is not installed, so this profile has surface metrics only; for syntax "
            f"metrics, {SYNTAX_INSTALL} and build again",
            step,
        )
        step(BUILD)
        report = build_reference(
            _windowed(chunks, settings.window_words),
            parser=parser,
            top_k=settings.top_k,
            min_words=settings.min_words,
            contrast=(
                _windowed(contrast_chunks, settings.window_words)
                if contrast_chunks is not None
                else None
            ),
            contrast_label=contrast_label,
            settings={
                "inputs": _described(inputs, items),
                "text_field": settings.text_field,
                "window_words": settings.window_words or None,
                "min_words": settings.min_words,
                "contrast": (
                    _described(contrast, contrast_items)
                    if contrast is not None and contrast_items is not None
                    else None
                ),
            },
        )
        step(DONE)
        sources = [chunk.source for chunk in [*chunks, *(contrast_chunks or [])]]
        # Keep the profile exactly as it is saved (floats rounded), so scoring it before or
        # after a save and load gives the same numbers.
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
) -> Evaluation:
    """Stress-test contrast likeness against edited drafts, as ``styleprofile evaluate``
    does: build a reference with ``contrast`` (the original drafts), then score each set in
    ``edited`` (label to edited copies, matched to their originals by name) with the weights
    learned without their original. ``settings.top_k`` does not apply here.
    """
    notes: list[Note] = []
    with _notes_on_error(notes):
        step = _progress(progress)
        step(READ)
        items, contrast_items = _items(inputs), _items(contrast)
        edited_items = {label: _items(value) for label, value in edited.items()}
        _stdin_once([*items, *contrast_items, *(item for v in edited_items.values() for item in v)])
        seen: set[str] = set()
        reference_chunks = _read(items, settings.text_field, seen, notes, "<text>")
        contrast_chunks = _read(contrast_items, settings.text_field, seen, notes, "<contrast>")
        edited_chunks: dict[str, list[Chunk]] = {}
        for label, value in edited_items.items():
            edited_chunks[label] = _read(value, settings.text_field, set(), notes, f"<{label}>")
            overlap = {chunk.source for chunk in edited_chunks[label]} & seen
            if overlap:
                where = _described(edited[label], value)
                where = where if isinstance(where, str) else ", ".join(where)
                count = len(overlap)
                notes.append(
                    Note(
                        f"{label}: {count:,} file{'' if count == 1 else 's'} in {where} are also "
                        "given as the writer's texts or the original drafts, so that set is not "
                        "an edit of them",
                        "edited_overlap",
                    )
                )
        parser = _parser(
            settings.syntax,
            notes,
            "spaCy is not installed, so this uses surface metrics only; for syntax metrics, "
            f"{SYNTAX_INSTALL} and run again",
            step,
        )
        step(EVALUATE)
        report = evaluate_rewording(
            _windowed(reference_chunks, settings.window_words),
            _windowed(contrast_chunks, settings.window_words),
            {
                label: _windowed(chunks, settings.window_words)
                for label, chunks in edited_chunks.items()
            },
            parser=parser,
            min_words=settings.min_words,
            contrast_label=contrast_label,
            retrain=retrain,
            settings={
                "inputs": _described(inputs, items),
                "contrast": _described(contrast, contrast_items),
                "edited": {
                    label: _described(edited[label], value) for label, value in edited_items.items()
                },
                "text_field": settings.text_field,
                "window_words": settings.window_words or None,
            },
        )
        step(DONE)
        loaded = [
            *reference_chunks,
            *contrast_chunks,
            *(c for v in edited_chunks.values() for c in v),
        ]
        return Evaluation(report, notes=notes, sources=[chunk.source for chunk in loaded])


@contextmanager
def _notes_on_error(notes: list[Note]) -> Iterator[None]:
    """Attach the notes a run collected to a ``StyleProfileError`` it raises, so a front end
    can still show them (a skipped file often explains the error)."""
    try:
        yield
    except StyleProfileError as error:
        error.notes[:0] = notes
        raise


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _progress(callback: ProgressCallback | None) -> Callable[[str], None]:
    def step(phase: str) -> None:
        if callback is not None:
            callback(Progress(phase))

    return step


def _items(inputs: Inputs) -> list[Input]:
    """One input or several, as a list; refuses anything that is not an input."""
    if isinstance(inputs, str | os.PathLike | Text | Chunk):
        return [inputs]
    items = list(inputs)
    for item in items:
        if not isinstance(item, str | os.PathLike | Text | Chunk):
            raise TypeError(
                f"expected a path (str or Path), Text or Chunk, not {type(item).__name__}"
            )
    return items


def _described(inputs: Inputs, items: Sequence[Input]) -> str | list[str]:
    """How inputs are recorded in a report's settings: paths as given, texts and chunks by
    name; one input as a string, several as a list. ``items`` is ``_items(inputs)``, since
    an iterator of inputs can be read only once."""

    def name(item: Input) -> str:
        if isinstance(item, Text):
            return item.name or "<text>"
        if isinstance(item, Chunk):
            return item.id
        return os.fspath(item)

    if isinstance(inputs, str | os.PathLike | Text | Chunk):
        return name(inputs)
    return [name(item) for item in items]


def _stdin_once(items: Sequence[Input]) -> None:
    if sum(isinstance(item, str | os.PathLike) and os.fspath(item) == "-" for item in items) > 1:
        raise StyleProfileError("- (stdin) can be given only once")


def _exists(value: str) -> bool:
    try:
        return Path(value).expanduser().exists()
    except (OSError, ValueError):  # too long for a path, or a NUL character
        return False


def _missing(value: str) -> StyleProfileError:
    if "\n" in value or len(value) > 255:
        shown = value.strip().split("\n")[0][:40]
        return StyleProfileError(
            f"{shown!r}... is not a file or folder: a str input is a path, so pass raw text "
            "as Text(...)",
            code="input_not_found",
        )
    return StyleProfileError(f"{value} not found", code="input_not_found")


def _read(
    items: Sequence[Input],
    text_field: str | Sequence[str] | None,
    seen: set[str],
    notes: list[Note],
    role: str,
) -> list[Chunk]:
    """Chunks from each input, skipping files an earlier path (tracked in ``seen``) already
    gave. ``Text`` inputs get ``role`` as their source, so texts in different roles
    (writer, contrast) are never the same document."""
    chunks: list[Chunk] = []
    texts = 0
    for item in items:
        if isinstance(item, Chunk):
            chunks.append(item)
            continue
        if isinstance(item, Text):
            texts += 1
            chunks.append(Chunk(item.name or f"text{texts}", role, item.text))
            continue
        value = os.fspath(item)
        if value != "-" and not _exists(value):
            raise _missing(value)
        loaded = load_chunks([value], text_field)
        sources = {chunk.source for chunk in loaded if value != "-"}
        repeated = sources & seen
        if repeated and repeated == sources:
            notes.append(Note(f"{value} was already given; using it once", "repeated_input"))
        elif repeated:
            count = len(repeated)
            notes.append(
                Note(
                    f"skipping {count:,} file{'' if count == 1 else 's'} in {value} already given",
                    "repeated_input",
                )
            )
        chunks += [chunk for chunk in loaded if chunk.source not in repeated]
        seen |= sources
    return chunks


def _windowed(chunks: list[Chunk], window_words: int) -> list[Chunk]:
    return window(chunks, window_words) if window_words else chunks


@cache
def _default_parser() -> Parser:
    """The spaCy parser, loaded once per process: loading takes about a second."""
    return load_parser()


def _parser(
    syntax: Literal["auto"] | bool,
    notes: list[Note],
    missing: str,
    step: Callable[[str], None],
) -> Parser | None:
    """The parser ``syntax`` asks for; with ``"auto"`` and no spaCy, None and a note."""
    if syntax is False:
        return None
    step(LOAD_PARSER)
    try:
        return _default_parser()
    except SyntaxUnavailableError:
        if syntax is True:
            raise
        notes.append(Note(missing, "no_syntax"))
        return None
