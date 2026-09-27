"""One pipeline for the library and the command line.

``build`` reads the writer's texts, cuts them into windows, loads the parser and profiles
them; ``Profile.score`` does the same to drafts with the settings the profile was built
with; ``evaluate`` runs the rewording stress test. The ``styleprofile`` command calls these
and only adds flags and output, so the two cannot give different numbers::

    import styleprofile as sp
    from pathlib import Path

    profile = sp.build(Path("posts/"), contrast=Path("llm-drafts/"))
    result = profile.score(sp.Text("A draft to check against the writer."))
    print(result.verdict, result.delta)

Inputs are paths (``str`` or ``Path``: files, folders, or ``"-"`` for stdin), ``Text`` for
raw text, or ``Chunk`` objects, alone or in a list. A ``str`` is always a path, as on the
command line. Nothing here prints: what a front end should mention comes back as ``notes``,
and problems raise ``StyleProfileError`` with a ``code`` (and the notes collected so far).

The lower-level ``build_reference`` and ``score`` in ``styleprofile.profile`` take chunks
exactly as given; use them to control windowing and parsing yourself.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack

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
    SourceNames,
    build_reference,
    check_version,
    dumps_report,
    expand_path,
    load_chunks,
    load_reference,
    report_kind,
    score,
    window,
    write_report,
)
from styleprofile.syntax import Parser, SyntaxUnavailableError, load_parser
from styleprofile.weighting import likeness_level

AUTO = "auto"
DEFAULT_WINDOW_WORDS = 500
DEFAULT_TOP_K = 300
# How inputs are read. Only format detection by file extension exists so far.
INPUT_FORMATS = (AUTO,)
SYNTAX_INSTALL = "pip install 'styleprofile[syntax]'"
# A reference below these is usable but thin; ``build`` notes why and how to fix it.
ENOUGH_DOCUMENTS = 2
ENOUGH_CHUNKS = 15
ENOUGH_WORDS = 20_000


@dataclass(frozen=True)
class Text:
    """Raw text to profile or score, as opposed to a path to read it from.

    ``name`` identifies it in reports (as a file name does) and pairs an edited text with
    its original in ``evaluate``. Unnamed texts are ``text1``, ``text2``, ... in order,
    skipping names other texts already have; two texts with one name are an error.
    """

    text: str
    name: str | None = None


Input = str | os.PathLike[str] | Text | Chunk
Inputs = Input | Iterable[Input]
ProgressCallback = Callable[[Progress], None]


@dataclass(frozen=True)
class Settings:
    """How texts are read and cut into chunks. ``build`` records them in the profile
    verbatim (``to_report``), and ``Profile.score`` inherits them unless overridden.

    - ``window_words``: split texts into windows of about this many prose words at
      paragraph breaks; 0 profiles each text whole.
    - ``min_words``: drop chunks with fewer prose words than this.
    - ``text_field``: the JSONL field that holds the text; by default the first of
      ``TEXT_FIELDS`` a record has.
    - ``syntax``: ``True`` needs spaCy for the syntax metrics, ``False`` leaves them out, and
      ``"auto"`` uses spaCy when it is installed and adds a note when it is not. A report
      records this as asked, and what was used as ``syntax_used``.
    - ``top_k``: distribution entries kept in a report.
    - ``input_format``: how inputs are read; ``"auto"`` goes by file extension.

    A new setting is one field here: recording, reading back, inheriting and overriding
    all go through the fields.
    """

    window_words: int = DEFAULT_WINDOW_WORDS
    min_words: int = 1
    text_field: str | None = None
    syntax: Literal["auto"] | bool = AUTO
    top_k: int = DEFAULT_TOP_K
    input_format: str = AUTO

    def __post_init__(self) -> None:
        _count(self, "window_words", 0, "must be 0 (no windowing) or positive")
        _count(self, "min_words", 0, "must be 0 or more")
        _count(self, "top_k", 1, "must be positive")
        if not (self.text_field is None or (isinstance(self.text_field, str) and self.text_field)):
            _invalid(
                "text_field", f"text_field must be a field name or None, not {self.text_field!r}"
            )
        if not (type(self.syntax) is bool or self.syntax == AUTO):
            _invalid("syntax", f"syntax must be True, False or {AUTO!r}, not {self.syntax!r}")
        if self.input_format not in INPUT_FORMATS:
            choices = ", ".join(repr(name) for name in INPUT_FORMATS)
            _invalid(
                "input_format",
                f"input_format {self.input_format!r} is not supported; use {choices}",
            )

    def to_report(self) -> dict[str, Any]:
        """The settings as a report records them: every field, verbatim."""
        return dataclasses.asdict(self)

    @classmethod
    def from_report(cls, recorded: Mapping[str, Any]) -> Settings:
        """Settings read back from a report's ``settings``. A field the report lacks (it was
        made with the lower-level functions, say) takes its default, except that a missing
        ``window_words`` is 0: nothing says those chunks were windowed. Other keys
        (``inputs``, ``syntax_used``) are ignored."""
        names = {field.name for field in dataclasses.fields(cls)}
        known = {name: value for name, value in recorded.items() if name in names}
        return cls(**{"window_words": 0, **known})


class SettingsOverrides(TypedDict, total=False):
    """Keyword overrides for ``Profile.score``: any ``Settings`` field, by name. Leave a
    keyword out to inherit it; a value given (``None`` included, for ``text_field``) is used
    as is, and ``None`` for a number or ``syntax`` is refused like any invalid setting."""

    window_words: int
    min_words: int
    text_field: str | None
    syntax: Literal["auto"] | bool
    top_k: int
    input_format: str


def _invalid(name: str, message: str) -> None:
    raise StyleProfileError(message, code="invalid_setting", setting=name)


def _count(settings: Settings, name: str, minimum: int, rule: str) -> None:
    """Check an int setting (not a bool or a string of digits) is at least ``minimum``."""
    value = getattr(settings, name)
    if type(value) is not int:
        _invalid(name, f"{name} {rule}, not {value!r}")
    elif value < minimum:
        _invalid(name, f"{name} {rule}")


DEFAULTS = Settings()


class _Result:
    """What every result wraps: its report dict, plus what the run noted and read, which are
    not saved."""

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
        """The full report, as saved. This is the live dict, not a copy: changing it changes
        what the other properties return."""
        return self._report

    @property
    def notes(self) -> tuple[Note, ...]:
        """What the run wants the user to know, such as syntax left out or an input given
        twice. The CLI prints these as ``note:`` lines."""
        return self._notes

    @property
    def sources(self) -> tuple[str, ...]:
        """Where the chunks this run read came from: file paths, ``stdin``, or ``<text>``
        (``<contrast>``, ``<LABEL>``) for ``Text`` inputs. These are the real paths, which
        reports never save (they save ``Chunk.source`` names instead)."""
        return self._sources

    @property
    def warnings(self) -> tuple[str, ...]:
        """Why the result may be unreliable, as saved in the report."""
        return tuple(self._report["warnings"])


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
        check_version(report, "the profile")
        super().__init__(report, notes=notes, sources=sources)
        self._path = _resolved(path) if path is not None else None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Profile:
        """Read a reference profile saved by ``save`` or ``styleprofile build``."""
        return cls(load_reference(expand_path(os.fspath(path))), path=path)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the profile as JSON. This also sets ``path`` in place, so scores made
        afterwards record it as their reference."""
        self._path = _resolved(path)
        write_report(self._report, self._path)

    @property
    def path(self) -> Path | None:
        """Where the profile was loaded from or last saved, if anywhere."""
        return self._path

    @property
    def settings(self) -> Settings:
        """The settings it was built with, as recorded."""
        return Settings.from_report(self._report.get("settings") or {})

    @property
    def has_syntax(self) -> bool:
        """Whether the profile has syntax metrics (spaCy was used to build it)."""
        return (self._report.get("settings") or {}).get("syntax_used") is not None

    def to_text(self, *, full: bool = False, color: bool = False) -> str:
        """The summary ``styleprofile build`` prints; ``full`` adds every metric."""
        return format_reference_summary(self._report, color=color, full=full)

    def score(
        self,
        inputs: Inputs,
        settings: Settings | None = None,
        *,
        progress: ProgressCallback | None = None,
        **overrides: Unpack[SettingsOverrides],
    ) -> ScoreResult:
        """Score drafts against this profile.

        By default the settings are the profile's, except that syntax is used (``"auto"``)
        only when the profile has it, and ``input_format`` is ``"auto"``: drafts are often
        in another format than the writer's corpus. ``settings`` replaces them, and keyword
        overrides (``window_words=0``, ``min_words=5``; see ``SettingsOverrides``) change
        single fields: leave one out to inherit it.

        A window size or syntax setting unlike the profile's is warned about in the report,
        since z-scores assume chunks like the reference's; another ``min_words`` gets a
        note. A ``text_field`` given here is the only one read; the profile's is tried
        first, then the defaults.
        """
        notes: list[Note] = []
        with _notes_on_error(notes):
            base = settings
            if base is None:
                base = dataclasses.replace(
                    self.settings, syntax=AUTO if self.has_syntax else False, input_format=AUTO
                )
            unknown = sorted(set(overrides) - {f.name for f in dataclasses.fields(Settings)})
            if unknown:
                raise TypeError(f"unknown setting(s): {', '.join(unknown)}")
            chosen = dataclasses.replace(base, **overrides)
            recorded = self.settings
            if chosen.min_words != recorded.min_words:
                notes.append(
                    Note(
                        f"min_words {chosen.min_words} overrides the reference's "
                        f"{recorded.min_words}",
                        NoteCode.SETTING_OVERRIDDEN,
                        setting="min_words",
                    )
                )
            given = overrides.get("text_field") or (settings.text_field if settings else None)
            fields: str | tuple[str, ...] | None = given
            if not given and chosen.text_field:
                fields = (
                    chosen.text_field,
                    *(name for name in TEXT_FIELDS if name != chosen.text_field),
                )
            step = _progress(progress)
            step(Phase.READ)
            items = _items(inputs)
            _stdin_once(items)
            names = SourceNames()
            chunks = _read(items, fields, set(), notes, "<text>", names)
            parser = _parser(
                chosen.syntax,
                notes,
                (
                    "the reference has syntax metrics but spaCy is not installed, so syntax "
                    f"is left out of this score; {SYNTAX_INSTALL} to include it"
                )
                if self.has_syntax
                else (
                    "spaCy is not installed, so this score has surface metrics only; "
                    f"{SYNTAX_INSTALL} to include syntax"
                ),
                step,
            )
            step(Phase.SCORE)
            report = score(
                _windowed(chunks, chosen.window_words),
                self._report,
                parser=parser,
                top_k=chosen.top_k,
                min_words=chosen.min_words,
                reference_path=self._path,
                settings={"inputs": _described(items, names), **chosen.to_report()},
            )
            step(Phase.DONE)
            return ScoreResult(report, self._report, notes=notes, sources=_sources(chunks))

    def __repr__(self) -> str:
        where = f" from {self._path}" if self._path else ""
        return (
            f"<Profile{where}: {self._report['chunk_count']} chunks, "
            f"{self._report['word_count']:,} words>"
        )


class ScoreResult(_Result):
    """Drafts scored against a profile: what ``styleprofile score`` prints and saves.

    ``delta``, ``verdict`` and the likeness figures are pooled over every chunk of every
    input, so scoring several documents at once gives one figure for all of them;
    per-document results arrive with plan PR 7. ``report["chunks"]`` has each chunk's own.
    """

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
    def verdict(self) -> Verdict:
        """Delta in words against the writer's held-out range, as the CLI prints it, or
        ``Verdict.NOT_COMPARABLE`` when no metric could be compared."""
        if self.delta is None:
            return Verdict.NOT_COMPARABLE
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
    def likeness_verdict(self) -> LikenessVerdict | None:
        """Likeness in words, or None without a contrast set. ``.words(contrast_label)``
        gives the CLI's wording, such as ``leans LLM``."""
        contrast = self._reference.get("contrast")
        if not contrast or self.likeness is None:
            return None
        return LIKENESSES[likeness_level(self.likeness, contrast["calibration"], self.chunk_count)]

    def to_text(self, *, full: bool = False, color: bool = False) -> str:
        """The comparison ``styleprofile score`` prints; ``full`` shows every metric."""
        return format_summary(self._report, self._reference, color=color, full=full)

    def __repr__(self) -> str:
        if self.delta is None:
            return f"<ScoreResult: {self.verdict}>"
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
    keep_chunks: bool = False,
) -> Profile:
    """Build a reference profile from a writer's texts, as ``styleprofile build`` does.

    Given ``contrast`` texts (LLM drafts of the same briefs, say), the profile also learns
    what separates the writer from them and scores likeness to them. A file or folder given
    twice, in ``inputs`` or ``contrast``, is read once, with a note. A reference too small
    to trust gets one ``NoteCode.THIN_REFERENCE`` note per reason.

    The profile keeps summaries only; ``keep_chunks`` also saves every chunk's metrics, for
    debugging. It is an option of this build rather than a ``Settings`` field: it changes
    what is saved, not how texts are read or cut, and scoring has nothing to inherit from it.
    """
    notes: list[Note] = []
    with _notes_on_error(notes):
        step = _progress(progress)
        step(Phase.READ)
        items = _items(inputs)
        contrast_items = _items(contrast) if contrast is not None else None
        _stdin_once([*items, *(contrast_items or [])])
        seen: set[str] = set()
        names = SourceNames()
        chunks = _read(items, settings.text_field, seen, notes, "<text>", names)
        contrast_chunks = (
            _read(contrast_items, settings.text_field, seen, notes, "<contrast>", names)
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
        step(Phase.BUILD)
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
                "inputs": _described(items, names),
                "contrast": (
                    _described(contrast_items, names) if contrast_items is not None else None
                ),
                **settings.to_report(),
            },
            keep_chunks=keep_chunks,
        )
        notes += _thin_reference(report, settings)
        step(Phase.DONE)
        sources = _sources([*chunks, *(contrast_chunks or [])])
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
        step(Phase.READ)
        items, contrast_items = _items(inputs), _items(contrast)
        edited_items = {label: _items(value) for label, value in edited.items()}
        every = [*items, *contrast_items, *(item for v in edited_items.values() for item in v)]
        _stdin_once(every)
        seen: set[str] = set()
        names = SourceNames()
        reference_chunks = _read(items, settings.text_field, seen, notes, "<text>", names)
        contrast_chunks = _read(
            contrast_items, settings.text_field, seen, notes, "<contrast>", names
        )
        edited_chunks: dict[str, list[Chunk]] = {}
        # Each edited set is named on its own: its files carry the originals' names.
        edited_names = {label: SourceNames() for label in edited_items}
        for label, value in edited_items.items():
            edited_chunks[label] = _read(
                value, settings.text_field, set(), notes, f"<{label}>", edited_names[label]
            )
            overlap = {chunk.path for chunk in edited_chunks[label] if chunk.path} & seen
            if overlap:
                count = len(overlap)
                notes.append(
                    Note(
                        f"{label}: {count:,} file{'' if count == 1 else 's'} in "
                        f"{', '.join(_typed(value))} are also given as the writer's "
                        "texts or the original drafts, so that set is not an edit of them",
                        NoteCode.EDITED_OVERLAP,
                    )
                )
        parser = _parser(
            settings.syntax,
            notes,
            "spaCy is not installed, so this uses surface metrics only; for syntax metrics, "
            f"{SYNTAX_INSTALL} and run again",
            step,
        )
        step(Phase.EVALUATE)
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
                "inputs": _described(items, names),
                "contrast": _described(contrast_items, names),
                "edited": {
                    label: _described(value, edited_names[label])
                    for label, value in edited_items.items()
                },
                # top_k shapes only a reference's saved distributions, which this never saves.
                **{name: value for name, value in settings.to_report().items() if name != "top_k"},
            },
        )
        step(Phase.DONE)
        loaded = [*reference_chunks, *contrast_chunks]
        loaded += [chunk for chunks in edited_chunks.values() for chunk in chunks]
        return Evaluation(report, notes=notes, sources=_sources(loaded))


@contextmanager
def _notes_on_error(notes: list[Note]) -> Iterator[None]:
    """Give an error raised during a run the notes collected before it, so a front end can
    still show them (a skipped file often explains the error). An ``OSError`` becomes a
    ``StyleProfileError`` (code ``unreadable``) with the original as its cause."""
    try:
        yield
    except StyleProfileError as error:
        error.notes = (*notes, *error.notes)
        raise
    except OSError as error:
        wrapped = StyleProfileError(str(error), code="unreadable")
        wrapped.notes = tuple(notes)
        raise wrapped from error


def _thin_reference(report: dict[str, Any], settings: Settings) -> list[Note]:
    """Why a reference may be too small to trust, each with its fix."""
    documents = report["document_count"]
    thin: list[Note] = []

    def note(message: str, setting: str | None = None) -> None:
        thin.append(Note(message, NoteCode.THIN_REFERENCE, setting=setting))

    if documents < ENOUGH_DOCUMENTS:
        note(
            f"it comes from {_plural(documents, 'document')}, so it has no held-out "
            "calibration and cannot learn a contrast; add more of the writer's documents"
        )
    if report["chunk_count"] < ENOUGH_CHUNKS:
        smaller = (
            "a smaller window_words"
            if settings.window_words
            else f"window_words {DEFAULT_WINDOW_WORDS}"
        )
        note(
            f"it has {_plural(report['chunk_count'], 'chunk')}; aim for {ENOUGH_CHUNKS} or "
            f"more by adding documents or using {smaller}",
            "window_words",
        )
    if report["word_count"] < ENOUGH_WORDS:
        note(
            f"it has {_plural(report['word_count'], 'word')}; aim for {ENOUGH_WORDS:,} or "
            "more of the writer's text, in one genre"
        )
    return thin


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def _resolved(path: str | os.PathLike[str]) -> Path:
    return expand_path(os.fspath(path)).resolve()


def _progress(callback: ProgressCallback | None) -> Callable[[Phase], None]:
    def step(phase: Phase) -> None:
        if callback is not None:
            callback(Progress(phase))

    return step


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
        f"{shown!r} not found; a str input is a path, so pass raw text as Text(...)",
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
        yield Chunk(name, role, text.text)


def _read(
    items: Sequence[Input],
    text_field: str | Sequence[str] | None,
    seen: set[str],
    notes: list[Note],
    role: str,
    names: SourceNames,
) -> list[Chunk]:
    """Chunks from each input, in order, skipping files an earlier path (tracked in
    ``seen`` by real path) already gave. ``Text`` inputs get ``role`` as their source, so
    texts in different roles (writer, contrast) are never the same document. ``names`` is
    shared across calls, so two inputs' files never get the same saved source."""
    texts = iter(_text_chunks([item for item in items if isinstance(item, Text)], role))
    chunks: list[Chunk] = []
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
        loaded = load_chunks([value], text_field, names=names)
        files = {chunk.path for chunk in loaded if chunk.path is not None}
        repeated = files & seen
        if repeated and repeated == files:
            notes.append(Note(f"{value} was already given; using it once", NoteCode.REPEATED_INPUT))
            if not contributed:
                # Every file came from an earlier input, so no source carries this name:
                # leave it out of the report's settings (``_described``).
                names.roots.pop(value, None)
        elif repeated:
            notes.append(
                Note(
                    f"skipping {_plural(len(repeated), 'file')} in {value} already given",
                    NoteCode.REPEATED_INPUT,
                )
            )
        chunks += [chunk for chunk in loaded if chunk.path not in repeated]
        seen |= files
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
    step: Callable[[Phase], None],
) -> Parser | None:
    """The parser ``syntax`` asks for; with ``"auto"`` and no spaCy, None and a note."""
    if syntax is False:
        return None
    step(Phase.LOAD_PARSER)
    try:
        return _default_parser()
    except SyntaxUnavailableError:
        if syntax is True:
            raise
        notes.append(Note(missing, NoteCode.NO_SYNTAX))
        return None
