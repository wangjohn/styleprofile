"""Parser, progress and measurement lifetime for library pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING, Literal, TypeVar

from styleprofile import spacy_model
from styleprofile.cache import MeasurementCache, disabled_by_environment
from styleprofile.core import Note, NoteCode, Phase, Progress, StyleProfileError
from styleprofile.measure import Measurer, check_jobs
from styleprofile.settings import SETUP_COMMAND
from styleprofile.syntax import DEFAULT_MODEL, Parser, SyntaxUnavailableError, load_parser

if TYPE_CHECKING:
    from styleprofile.settings import ProgressCallback


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


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


@contextmanager
def _measurer(progress: ProgressCallback | None, jobs: int, cache: bool) -> Iterator[Measurer]:
    """The run's ``Measurer``, closed (workers stopped, cache written) when the run ends."""
    check_jobs(jobs)
    store = MeasurementCache() if cache and not disabled_by_environment() else None
    measurer = Measurer(cache=store, jobs=jobs, progress=progress)
    try:
        yield measurer
    finally:
        measurer.close()


def _finish(measurer: Measurer, notes: list[Note]) -> None:
    """Stop the run's workers and write its cache, noting when the cache could not be used:
    otherwise the only sign would be that rebuilding stays slow."""
    measurer.close()
    if measurer.requested_jobs > measurer.jobs:
        notes.append(
            NoteCode.WORKERS_LIMITED.note(
                "0", f"{measurer.requested_jobs}", f"{measurer.jobs}", setting="jobs"
            )
        )
    store = measurer.cache
    if store is not None and store.problem is not None:
        notes.append(NoteCode.CACHE_UNAVAILABLE.note("0", f"{store.path}", f"{store.problem}"))


def _progress(callback: ProgressCallback | None) -> Callable[[Phase], None]:
    def step(phase: Phase) -> None:
        if callback is not None:
            callback(Progress(phase))

    return step


@cache
def _default_parser() -> Parser:
    """The spaCy parser, made once per process. Its model loads (about a second, and 150 MB)
    only when this process parses: when worker processes do all the parsing, or every
    chunk comes from the measurement cache, it never does."""
    return load_parser(lazy=True)


_T = TypeVar("_T")


def _or_without_syntax(
    syntax: Literal["auto"] | bool,
    parser: Parser | None,
    notes: list[Note],
    missing: str,
    run: Callable[[Parser | None], _T],
) -> _T:
    """``run(parser)``; or, when the parser's model fails to load (it loads lazily, when
    first needed) and ``syntax`` is ``"auto"``, ``run(None)`` with the note ``"auto"`` gives
    when spaCy is missing. With ``syntax=True`` the failure is raised. The inputs are read
    once either way; the run itself starts again, which only the first parse can have
    begun."""
    try:
        return run(parser)
    except SyntaxUnavailableError as error:
        if syntax is True or parser is None or parser.loaded:
            raise
        notes.append(_no_syntax_note(missing, error))
        return run(None)


def _no_syntax_note(missing: str, error: SyntaxUnavailableError) -> Note:
    """The note that syntax is left out: ``missing`` with ``{missing}`` naming what is not
    installed and ``{fix}`` how to install it, the ``syntax`` extra without spaCy or
    `styleprofile setup` when only spaCy's English model is missing."""
    cause, fix = (
        (NoteCode.NO_SYNTAX.message("missing_model", DEFAULT_MODEL), SETUP_COMMAND)
        if error.model_missing
        else (NoteCode.NO_SYNTAX.message("missing_spacy"), spacy_model.syntax_install())
    )
    return Note(missing.format(missing=cause, fix=fix), NoteCode.NO_SYNTAX)


def _parser(
    syntax: Literal["auto"] | bool,
    notes: list[Note],
    missing: str,
    step: Callable[[Phase], None],
) -> Parser | None:
    """The parser ``syntax`` asks for; with ``"auto"`` and no spaCy, None and a note.

    ``missing`` is the note's template (see ``_no_syntax_note``)."""
    if syntax is False:
        return None
    step(Phase.LOAD_PARSER)
    try:
        return _default_parser()
    except SyntaxUnavailableError as error:
        if syntax is True:
            raise
        notes.append(_no_syntax_note(missing, error))
        return None
