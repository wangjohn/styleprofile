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
import functools
import json
import os
import statistics
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Generic, Literal, TypedDict, TypeVar, Unpack, cast

from styleprofile import spacy_model
from styleprofile.cache import MeasurementCache, disabled_by_environment
from styleprofile.calibration import (
    MIN_CALIBRATION_DOCUMENTS,
    MIN_CALIBRATION_PIECES,
    MIN_JUDGED_WORDS,
    too_short_text,
)
from styleprofile.core import (
    DISTANCES,
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
    format_evaluation,
    format_reference_summary,
    format_summary,
)
from styleprofile.drift import Passage
from styleprofile.evaluate import evaluate_rewording
from styleprofile.formats import INPUT_FORMATS
from styleprofile.measure import Measurer, check_jobs
from styleprofile.profile import (
    REFERENCE,
    TEXT_FIELDS,
    Chunk,
    Pooled,
    Records,
    Repeat,
    SourceNames,
    bare_key,
    base_id,
    build_reference,
    check_report,
    chunk_document,
    cover_key,
    document_label,
    drop_duplicates,
    dumps_report,
    expand_path,
    group_id,
    is_grouped,
    is_record,
    literal_id,
    load_chunks,
    load_reference,
    paired_as,
    pool,
    report_kind,
    score,
    version_notes,
    window,
    write_report,
)
from styleprofile.schema import (
    Baseline,
    DocumentEntry,
    EvaluationReport,
    InputSettings,
    ReferenceReport,
    ScoreReport,
    ScoreVerdict,
)
from styleprofile.split import (
    ASKED_MIN_PARTS,
    HEADING,
    MIN_PARTS,
    NONE,
    SPLIT_ON,
    STAND_IN,
    SplitOn,
    part_id,
    plan_split,
    stand_in_groups,
)
from styleprofile.surface import paragraph_metrics_missing, prose, unlikely_english, words
from styleprofile.syntax import DEFAULT_MODEL, Parser, SyntaxUnavailableError, load_parser
from styleprofile.weighting import FLOOR_DOCUMENTS

AUTO = "auto"
# How the command line names standard input, apart from a file called ``stdin``.
STDIN_SHOWN = "<stdin>"
# ``jobs``: pick the number of parser processes (see ``measure.Measurer``).
AUTO_JOBS = 0
DEFAULT_WINDOW_WORDS = 500
DEFAULT_TOP_K = 300
SETUP_COMMAND = "run `styleprofile setup`"
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
    The texts of one role (the writer's, say) are like the records of one JSONL file: each
    is a document, and short ones are pooled in order (see ``Settings.pool``).
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
    - ``input_format``: how inputs are read. ``"auto"`` goes by file extension, reads a
      Markdown or text file that looks like HTML as HTML, and reads stdin as JSONL when
      every line is a JSON object; ``"markdown"``, ``"html"`` or ``"jsonl"`` reads every
      input that way, folder contents included.
    - ``group_field``: the JSONL field (``thread``, ``conversation_id``) whose value groups
      the records of one input into documents; by default each record is its own document.
      Records with no value in it are one more group, ``thread=(none)``.
    - ``pool``: join short texts into windows of about ``window_words``, never across
      groups or inputs: ``True``, ``False``, or ``"auto"``, which pools when the median text is
      under a quarter of a window and pooling leaves ``ENOUGH_CHUNKS`` windows. Without
      ``group_field``, each joined window counts as a document. ``Profile.score`` does not
      inherit it: drafts are scored one by one unless it is asked for.
    - ``split_on``: split long Markdown, text or HTML texts into documents, so one big file
      or a few manuscripts get held-out calibration: ``"heading"`` at their headings
      (``"heading:2"``: at level-2 headings and above, such as a book's chapters under its
      parts), ``"rule"`` at their rules (``---``, ``***``, ``___``), ``"none"`` never. With
      ``"auto"`` (see ``_split`` and ``split``), fewer than ``MIN_CALIBRATION_DOCUMENTS``
      documents are split at their headings or rules, or, with neither, cut into 8
      (``split.STAND_INS``) stand-in documents of consecutive windows, whose caveat the
      report keeps as a warning; fewer than ``FLOOR_DOCUMENTS`` are split at their headings
      or rules only, into parts of about a window or more; more are left as they are.
      JSONL records are never split. Each split is noted. ``Profile.score`` does not inherit it, and
      ``"auto"`` never splits drafts: each is one document unless
      ``"heading"``, ``"heading:N"`` or ``"rule"`` is asked for, which splits a text into two
      parts or more and gives a manuscript a verdict per chapter.

    A new setting is one field here: recording, reading back, inheriting and overriding
    all go through the fields.
    """

    window_words: int = DEFAULT_WINDOW_WORDS
    min_words: int = 1
    text_field: str | None = None
    syntax: Literal["auto"] | bool = AUTO
    top_k: int = DEFAULT_TOP_K
    input_format: str = AUTO
    group_field: str | None = None
    pool: Literal["auto"] | bool = AUTO
    split_on: SplitOn = AUTO

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
        if not (
            self.group_field is None or (isinstance(self.group_field, str) and self.group_field)
        ):
            _invalid(
                "group_field",
                f"group_field must be a field name or None, not {self.group_field!r}",
            )
        if not (type(self.pool) is bool or self.pool == AUTO):
            _invalid("pool", f"pool must be True, False or {AUTO!r}, not {self.pool!r}")
        if self.split_on not in SPLIT_ON:
            choices = ", ".join(repr(name) for name in SPLIT_ON)
            _invalid("split_on", f"split_on must be one of {choices}, not {self.split_on!r}")
        if self.input_format not in INPUT_FORMATS:
            choices = ", ".join(repr(name) for name in INPUT_FORMATS)
            _invalid(
                "input_format",
                f"input_format {self.input_format!r} is not supported; use {choices}",
            )

    def to_report(self) -> InputSettings:
        """The settings as a report records them: every field, verbatim."""
        return cast(InputSettings, dataclasses.asdict(self))

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
    """Keyword overrides for ``build`` and ``Profile.score``: any ``Settings`` field,
    by name. Leave a keyword out to inherit it; a value given (``None`` included, for
    ``text_field``) is used as is, and ``None`` for a number or ``syntax`` is refused like
    any invalid setting."""

    window_words: int
    min_words: int
    text_field: str | None
    syntax: Literal["auto"] | bool
    top_k: int
    input_format: str
    group_field: str | None
    pool: Literal["auto"] | bool
    split_on: SplitOn


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


_R = TypeVar("_R", ReferenceReport, ScoreReport, EvaluationReport)
_K = TypeVar("_K")
_V = TypeVar("_V")


class _Result(Generic[_R]):
    """What every result wraps: its report dict, plus what the run noted and read, which are
    not saved."""

    def __init__(
        self, report: _R, *, notes: Sequence[Note] = (), sources: Sequence[str] = ()
    ) -> None:
        self._report = report
        self._notes = (*notes, *version_notes(report))
        self._sources = tuple(sources)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the report as JSON, as the command's ``-o`` does."""
        write_report(self._report, _resolved(path))

    @property
    def report(self) -> _R:
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


class Profile(_Result[ReferenceReport]):
    """A reference profile: what ``build`` makes and ``styleprofile build`` saves."""

    def __init__(
        self,
        report: ReferenceReport,
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
        check_report(report, "the profile")
        super().__init__(report, notes=notes, sources=sources)
        self._path = _resolved(path) if path is not None else None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Profile:
        """Read a reference profile saved by ``save`` or ``styleprofile build``. Errors name
        the file as ``path`` gives it."""
        return cls(load_reference(expand_path(os.fspath(path)), os.fspath(path)), path=path)

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
        return Settings.from_report(self._report["settings"])

    @property
    def has_syntax(self) -> bool:
        """Whether the profile has syntax metrics (spaCy was used to build it)."""
        return self._report["settings"]["syntax_used"] is not None

    def to_text(
        self,
        *,
        full: bool = False,
        color: bool = False,
        verbose: bool = False,
        warning_settings: Mapping[str, str] | None = None,
    ) -> str:
        """The summary ``styleprofile build`` prints; ``full`` adds every metric."""
        return format_reference_summary(
            self._report, color=color, full=full, verbose=verbose, warning_settings=warning_settings
        )

    def score_text(
        self,
        text: str | Iterable[str],
        *,
        progress: ProgressCallback | None = None,
        passages: bool = False,
        jobs: int = AUTO_JOBS,
        cache: bool = False,
        **overrides: Unpack[SettingsOverrides],
    ) -> ScoreResult:
        """Score raw text, or several strings with one document per string.

        Options and inherited settings are the same as for ``score``. Use ``score`` with
        ``Text`` objects when documents need names of their own.
        """
        return self.score(
            _strings(text),
            progress=progress,
            passages=passages,
            jobs=jobs,
            cache=cache,
            **overrides,
        )

    def score(
        self,
        inputs: Inputs,
        settings: Settings | None = None,
        *,
        progress: ProgressCallback | None = None,
        passages: bool = False,
        jobs: int = AUTO_JOBS,
        cache: bool = False,
        **overrides: Unpack[SettingsOverrides],
    ) -> ScoreResult:
        """Score drafts against this profile.

        By default the settings are the profile's, except that syntax is used (``"auto"``)
        only when the profile has it, and ``input_format`` is ``"auto"``: drafts are often
        in another format than the writer's corpus, and ``pool`` is ``False``: each draft is
        judged on its own (``pool=True`` judges short drafts as a batch, joined into
        windows, and a note suggests it when they are short), and ``split_on`` is
        ``"none"``: each draft is one document, however the reference's texts were split
        (``split_on="heading"`` gives a manuscript a verdict per chapter; see
        ``Settings.split_on``). ``group_field`` is inherited;
        drafts without it are read with a note, unless it is given here. The deprecated
        ``settings`` argument replaces them all. Keyword overrides (``window_words=0``,
        ``min_words=5``; see ``SettingsOverrides``) change single fields:
        leave one out to inherit it.

        A window size or syntax setting unlike the profile's is warned about in the report,
        since z-scores assume chunks like the reference's; another ``min_words`` gets a
        note. A ``text_field`` given here is the only one read; the profile's is tried
        first, then the defaults. ``jobs`` is as for ``build``. Scoring reads and writes the
        measurement cache only with ``cache=True``: a draft is quick to measure, and what it
        would leave there (pattern counts of its text) should not stay behind unasked.

        ``passages=True`` (experimental, off by default) also reads each document in
        overlapping spans of 100 words or more to show where it drifts
        (``ScoreResult.passages``). On writer text of topics the reference never saw, it
        found a paragraph drifting in up to about a third of the writer's own documents
        (docs/method.md, "Where a draft drifts"), so treat what it finds as a lead to read.
        """
        if settings is not None:
            warnings.warn(
                "Passing Settings to Profile.score is deprecated: it replaces all inherited "
                "settings. Use keyword overrides such as syntax=False instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        notes = list(version_notes(self._report))
        with _notes_on_error(notes), _measurer(progress, jobs, cache) as measurer:
            base = settings
            if base is None:
                base = dataclasses.replace(
                    self.settings,
                    syntax=AUTO if self.has_syntax else False,
                    input_format=AUTO,
                    pool=False,
                    split_on=NONE,
                )
            chosen = _overridden(base, overrides)
            recorded = self.settings
            if chosen.min_words != recorded.min_words:
                notes.append(
                    NoteCode.SETTING_OVERRIDDEN.note(
                        "0", f"{chosen.min_words}", f"{recorded.min_words}", setting="min_words"
                    )
                )
            given = overrides.get("text_field") or (settings.text_field if settings else None)
            fields: str | tuple[str, ...] | None = given
            if not given and chosen.text_field:
                fields = (
                    chosen.text_field,
                    *(name for name in TEXT_FIELDS if name != chosen.text_field),
                )
            pooling = _pooling(chosen)
            step = _progress(progress)
            step(Phase.READ)
            items = _items(inputs)
            _stdin_once(items)
            names = SourceNames()
            records = Records()
            chunks = _read(
                items,
                fields,
                set(),
                notes,
                "<text>",
                names,
                input_format=chosen.input_format,
                group_field=chosen.group_field,
                # An inherited group field is one the drafts may not have.
                require_groups="group_field" in overrides or settings is not None,
                records=records,
            )
            cut = _chunked(chunks, chosen, pooling, notes, records=records, split="score")
            reference_pooled = bool((self._report.get("settings") or {}).get("pool_used"))
            if cut.pooled is None and cut.short and len(chunks) > 1:
                notes.append(
                    NoteCode.SHORT_TEXTS.note(
                        "0", f"{len(chunks):,}", f"{MIN_JUDGED_WORDS}", setting="pool"
                    )
                )
            missing = (
                NoteCode.NO_SYNTAX.forms["score_reference"].long
                if self.has_syntax
                else NoteCode.NO_SYNTAX.forms["score_surface"].long
            )
            parser = _parser(chosen.syntax, notes, missing, step)
            step(Phase.SCORE)
            report = _or_without_syntax(
                chosen.syntax,
                parser,
                notes,
                missing,
                lambda parser: score(
                    cut.windows,
                    self._report,
                    parser=parser,
                    top_k=chosen.top_k,
                    min_words=chosen.min_words,
                    reference_path=self._path,
                    settings={
                        "inputs": _described(items, names),
                        **chosen.to_report(),
                        "pool_used": cut.pooled is not None,
                        "split_used": list(cut.split),
                    },
                    read_in_parts=chunks if passages and cut.pooled is None else None,
                    pooled=passages and cut.pooled is not None,
                    measurer=measurer,
                ),
            )
            report["warnings"] += _pooling_mismatch(cut, reference_pooled)
            step(Phase.DONE)
            _finish(measurer, notes)
            return ScoreResult(
                report,
                notes=notes,
                sources=_sources(chunks),
                locations=_locations(items, names),
            )

    def __repr__(self) -> str:
        where = f" from {self._path}" if self._path else ""
        return (
            f"<Profile{where}: {self._report['chunk_count']} chunks, "
            f"{self._report['word_count']:,} words>"
        )


@dataclass(frozen=True)
class DocumentResult:
    """One document of a score, judged on its own chunks the way a whole score is judged.

    A document is a file, a JSONL record or a ``Text``; windowing cuts it into ``chunks``.
    ``name`` is how reports list it: a file by its saved path (``posts/2024/a.md``), a
    JSONL record by its file and id (``comments.jsonl:17``), a ``Text`` by its name.
    ``path`` is the saved path of the file it came from (``comments.jsonl`` for a record,
    ``<text>`` for a ``Text``), and ``location`` where to open it as the inputs were typed
    (``drafts/2024/a.md``); ``location`` is None for a ``Text`` and for a report loaded from
    a file, since reports never save typed paths. Neither is ever an absolute path unless
    the input was typed as one.

    ``delta`` and ``likeness`` are means over the document's chunks long enough to judge
    (``chunks_judged`` of them), and ``verdict`` and ``likeness_verdict`` put them in words
    against the writer's range at each chunk's own length, by the same function as
    ``ScoreResult.verdict`` for a whole score. When none of its chunks is long enough,
    ``judged`` is False, ``reason`` says why, and the figures cover every chunk as an
    indication only; ``verdict`` is ``TOO_SHORT`` for short texts and ``NOT_COMPARABLE``
    when no metric could be compared. A document that is not judged, or not comparable, never
    fails a threshold (``ScoreResult.failing``).

    ``differences`` are the metrics (``"group.name"``) furthest from the reference, as
    (metric, mean z), largest first; ``signals`` the metrics that add most to its likeness,
    as (metric, mean z, share of the likeness). Both hold up to three entries, and
    ``signals`` is empty without a contrast set. The shares are approximate: each chunk
    records only its five strongest signals (``weighting.SIGNALS_SHOWN``), so a metric just
    outside some chunks' five counts as 0 there, and the shares are means over the chunks of
    those records.
    """

    name: str
    words: int
    chunks: int
    delta: float | None
    verdict: Verdict
    likeness: float | None = None
    likeness_verdict: LikenessVerdict | None = None
    differences: tuple[tuple[str, float], ...] = ()
    signals: tuple[tuple[str, float, float], ...] = ()
    judged: bool = True
    chunks_judged: int = 0
    reason: str | None = None
    path: str | None = None
    # Its judged chunks that are flagged on their own (``ScoreResult.flagged``).
    flagged: int = 0
    location: str | None = field(default=None, compare=False)

    @classmethod
    def from_report(cls, entry: DocumentEntry, location: str | None = None) -> DocumentResult:
        """A document from a score report's ``documents``; ``location`` is where its
        file was typed, if known."""
        likeness = entry["likeness_verdict"]
        return cls(
            name=entry["name"],
            words=entry["words"],
            chunks=entry["chunks"],
            delta=entry["delta"],
            verdict=Verdict(entry["verdict"]),
            likeness=entry["likeness"],
            likeness_verdict=LikenessVerdict(likeness) if likeness else None,
            differences=tuple((item["metric"], item["z"]) for item in entry["differences"]),
            signals=tuple(
                (item["metric"], item["z"], item["contribution"])
                for item in entry.get("signals", [])
            ),
            judged=entry["judged"] and entry["delta"] is not None,
            chunks_judged=entry["chunks_judged"],
            reason=(
                "no metrics could be compared with the reference"
                if entry["verdict"] == str(Verdict.NOT_COMPARABLE)
                else entry["reason"]
            ),
            path=entry["path"],
            flagged=entry["flagged"],
            location=location,
        )

    @property
    def shown(self) -> str:
        """How the command line names it: where it can be opened as typed when known
        (``location``, plus a JSONL record's id, ``exports/c.jsonl:17``, or a split file's
        part, ``drafts/book.md#3-mud-season``), else ``name``."""
        if self.location is None:
            return self.name
        inside = self.path and self.name.startswith(self.path)
        rest = self.name[len(self.path) :] if self.path and inside else ""
        return self.location + rest if rest[:1] in (":", "#") else self.location


class ScoreResult(_Result[ScoreReport]):
    """Drafts scored against a profile: what ``styleprofile score`` prints and saves.

    ``documents`` has a result for each document scored, each judged at its own chunks'
    lengths. ``delta``, ``verdict`` and the likeness figures are pooled over the chunks of
    every document that are long enough to judge, each read against the writer's range at
    its own length, so with several documents one very different draft can make the pooled
    verdict read "very different" while the others are close: read ``documents`` to judge
    each draft. With one document the two agree. ``report["chunks"]`` has each chunk's own
    scores. When no chunk is long enough, ``judged`` is False, both verdicts are
    ``TOO_SHORT``, ``reason`` says why, and the figures cover every chunk as an indication
    only.
    """

    def __init__(
        self,
        report: ScoreReport,
        *,
        notes: Sequence[Note] = (),
        sources: Sequence[str] = (),
        locations: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(report, notes=notes, sources=sources)
        # A score report carries a copy of what rendering needs from its reference.
        self._reference: Baseline = report["reference"]["baseline"]
        # Each saved source's path as the inputs were typed; never saved.
        self._locations = dict(locations or {})

    @property
    def chunk_count(self) -> int:
        return self._report["chunk_count"]

    @property
    def documents(self) -> tuple[DocumentResult, ...]:
        """Each scored document's own result, in the order the inputs gave them. The CLI
        lists them furthest from the reference first."""
        return tuple(
            DocumentResult.from_report(entry, self._locations.get(entry["path"]))
            for entry in self._report["documents"]
        )

    def failing(
        self,
        above: Verdict | None = None,
        likeness: LikenessVerdict | None = None,
        flagged: int | None = None,
    ) -> tuple[DocumentResult, ...]:
        """The documents at least ``above`` from the reference (``Verdict.CLEARLY_DIFFERENT``
        catches clearly and very different), or at least ``likeness`` like the contrast set,
        or with at least ``flagged`` chunks flagged on their own (``DocumentResult.flagged``),
        in input order: what ``styleprofile score --fail-above``, ``--fail-likeness`` and
        ``--fail-flagged`` check. ``above`` and ``likeness`` judge each document's verdict,
        over all of its chunks, which a few very different chunks among many move little;
        ``flagged`` catches those. A document without a verdict (too short to judge, or not
        comparable) never fails any check."""
        return tuple(
            document
            for document in self.documents
            if any(fails(document, above, likeness, flagged))
        )

    @property
    def delta(self) -> float | None:
        """Mean Delta over the judged chunks, pooled across documents (see ``documents`` for
        each one's): distance from the writer in the writer's own standard deviations. None
        when no metric could be compared."""
        return self._report["reference"]["delta_mean"]

    @property
    def judged(self) -> bool:
        """Whether any chunk could be compared and is long enough to judge;
        ``reason`` says why when none could be judged."""
        return self.delta is not None and self._verdict["judged"]

    @property
    def flagged(self) -> int:
        """How many of the judged chunks read clearly different or worse, or lean toward
        the contrast set or more, on their own (``calibration.chunk_flagged``). The verdict
        judges the chunks' mean, which answers whether the text as a whole is like the
        writer: a few very different chunks among many close ones move it little, so check
        this (and the chunk lists the CLI prints) for them."""
        return self._verdict["flagged"]

    @property
    def reason(self) -> str | None:
        """Why there is no verdict, such as "under 75 words, the writer's own text varies
        too much by chance to judge"; None when there is one."""
        if self.verdict is Verdict.NOT_COMPARABLE:
            return "no metrics could be compared with the reference"
        return self._verdict["reason"]

    @property
    def verdict(self) -> Verdict:
        """Delta in words against the writer's held-out range at the text's length, as the
        CLI prints it; ``Verdict.TOO_SHORT`` when the text is too short to judge, or
        ``Verdict.NOT_COMPARABLE`` when no metric could be compared."""
        if self.delta is None and (
            self._report["reference"].get("verdict", {}).get("verdict") != str(Verdict.TOO_SHORT)
        ):
            return Verdict.NOT_COMPARABLE
        return Verdict(self._verdict["verdict"])

    @property
    def _verdict(self) -> ScoreVerdict:
        return self._report["reference"]["verdict"]

    @property
    def passages(self) -> tuple[Passage, ...]:
        """Every paragraph of each document read in parts, in order, with how it reads
        against the writer and whether it ``drifts`` (see
        ``styleprofile.drift``). Empty unless the score was asked to read documents in parts
        (``passages=True``, experimental), or when every document was too short for a span
        of 100 words."""
        contrast = self.contrast_label is not None
        return tuple(
            Passage.from_report(document["name"], entry, contrast)
            for document in self._report.get("passages") or []
            for entry in document["paragraphs"]
        )

    @property
    def contrast_label(self) -> str | None:
        """The contrast set's name (``LLM``) when the profile was built with one."""
        contrast = self._reference["contrast"]
        return contrast["label"] if contrast else None

    @property
    def likeness(self) -> float | None:
        """Mean likeness to the contrast set over every scored chunk, pooled across
        documents, when the profile was built with one."""
        return self._report["reference"]["likeness_mean"]

    @property
    def likeness_verdict(self) -> LikenessVerdict | None:
        """Likeness in words, ``LikenessVerdict.TOO_SHORT`` when the text is too short to
        judge, or None without a contrast set. ``.words(contrast_label)`` gives the CLI's
        wording, such as ``leans LLM``."""
        if self.delta is None:
            return None
        entry = self._verdict.get("likeness")
        if not entry:
            return None
        if entry["level"] is None:
            return LikenessVerdict.TOO_SHORT
        return LIKENESSES[entry["level"]]

    def to_text(
        self,
        *,
        full: bool = False,
        color: bool = False,
        width: int = 80,
        by_paragraph: bool = False,
        verbose: bool = False,
        warning_settings: Mapping[str, str] | None = None,
    ) -> str:
        """The comparison ``styleprofile score`` prints; ``full`` shows every metric and
        every document, the document table fits ``width`` columns, and ``by_paragraph``
        lists every paragraph of the documents read in parts (``passages``)."""
        # The document table names each document as -q does: where it was typed, if known.
        shown = {document.name: document.shown for document in self.documents}
        return format_summary(
            self._report,
            self._reference,
            color=color,
            full=full,
            width=width,
            shown=shown,
            by_paragraph=by_paragraph,
            verbose=verbose,
            warning_settings=warning_settings,
        )

    def __repr__(self) -> str:
        if self.delta is None:
            return f"<ScoreResult: {self.verdict}>"
        if not self.judged:
            return f"<ScoreResult: {too_short_text(self._verdict)}>"
        return f"<ScoreResult: {self.verdict} (Delta {self.delta:.2f})>"


def fails(
    document: DocumentResult,
    above: Verdict | None,
    likeness: LikenessVerdict | None,
    flagged: int | None = None,
) -> tuple[bool, bool, bool]:
    """Whether ``document`` reaches ``above`` on Delta, ``likeness`` on likeness, and has
    at least ``flagged`` chunks flagged on their own, as ``ScoreResult.failing`` checks:
    never for a document without a Delta verdict (too short to judge, or not comparable)."""
    if not document.judged or document.verdict not in DISTANCES:
        return False, False, False
    return (
        _reaches(document.verdict, above, DISTANCES),
        _reaches(document.likeness_verdict, likeness, LIKENESSES),
        flagged is not None and document.flagged >= flagged,
    )


def _reaches(verdict: Any, limit: Any, levels: Sequence[Any]) -> bool:
    """Whether ``verdict`` is at or past ``limit`` on ``levels``; never for no limit, or a
    verdict that is not one of the levels."""
    return limit is not None and verdict in levels and levels.index(verdict) >= levels.index(limit)


class Evaluation(_Result[EvaluationReport]):
    """The rewording stress test: what ``styleprofile evaluate`` prints and saves."""

    def to_text(
        self,
        *,
        color: bool = False,
        verbose: bool = False,
        warning_settings: Mapping[str, str] | None = None,
    ) -> str:
        """The tables ``styleprofile evaluate`` prints."""
        return format_evaluation(
            self._report, color=color, verbose=verbose, warning_settings=warning_settings
        )


load = Profile.load


def _strings(texts: str | Iterable[str]) -> list[Text]:
    values = [texts] if isinstance(texts, str) else list(texts)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("expected raw text strings; use build or score for paths, Text or Chunk")
    return [Text(value) for value in values]


def build_texts(
    texts: Iterable[str],
    *,
    contrast: Iterable[str] | None = None,
    contrast_label: str = "LLM",
    progress: ProgressCallback | None = None,
    keep_chunks: bool = False,
    passages: bool = False,
    jobs: int = AUTO_JOBS,
    cache: bool = False,
    **overrides: Unpack[SettingsOverrides],
) -> Profile:
    """Build from raw strings, with one document per string.

    Options are the same as for ``build``. Use ``build`` with ``Text`` objects to name
    documents, or with paths to read files and folders.
    """
    return build(
        _strings(texts),
        contrast=_strings(contrast) if contrast is not None else None,
        contrast_label=contrast_label,
        progress=progress,
        keep_chunks=keep_chunks,
        passages=passages,
        jobs=jobs,
        cache=cache,
        **overrides,
    )


def _overridden(settings: Settings, overrides: SettingsOverrides) -> Settings:
    unknown = sorted(set(overrides) - {f.name for f in dataclasses.fields(Settings)})
    if unknown:
        raise TypeError(f"unknown setting(s): {', '.join(unknown)}")
    return dataclasses.replace(settings, **overrides)


def build(
    inputs: Inputs,
    settings: Settings = DEFAULTS,
    *,
    contrast: Inputs | None = None,
    contrast_label: str = "LLM",
    progress: ProgressCallback | None = None,
    keep_chunks: bool = False,
    passages: bool = False,
    jobs: int = AUTO_JOBS,
    cache: bool = False,
    **overrides: Unpack[SettingsOverrides],
) -> Profile:
    """Build a reference profile from a writer's texts, as ``styleprofile build`` does.

    Given ``contrast`` texts (LLM drafts of the same briefs, say), the profile also learns
    what separates the writer from them and scores likeness to them. A file or folder given
    twice, in ``inputs`` or ``contrast``, is read once, with a note. A reference too small
    to trust gets one ``NoteCode.THIN_REFERENCE`` note per reason.

    The profile keeps summaries only; ``keep_chunks`` also saves every chunk's metrics, for
    debugging. It is an option of this build rather than a ``Settings`` field: it changes
    what is saved, not how texts are read or cut, and scoring has nothing to inherit from it.

    ``passages=True`` also calibrates experimental paragraph checks; by default this
    calibration is omitted.

    ``jobs`` and ``cache`` are options of the run too, since they change how fast it goes and
    never a number in the profile. ``jobs`` is how many measurement processes run: 0
    (the default) picks one per CPU, up to 4, from 50,000 syntax words or 500,000 surface
    words. Explicit counts are capped by the memory allowance. Workers start only
    where worker processes can start (see ``measure.workers_can_start``: a script needs an
    ``if __name__ == "__main__":`` guard); 1 parses in this process. With ``cache=True``,
    chunks measured by an earlier run are read from the measurement cache and new ones are
    added to it (see ``styleprofile.cache``); the default,
    ``cache=False``, neither reads nor writes it. Keyword overrides change individual
    ``Settings`` fields, including when a settings object is given. ``progress`` is called
    as each phase starts and as each chunk is measured.
    """
    settings = _overridden(settings, overrides)
    notes: list[Note] = []
    with _notes_on_error(notes), _measurer(progress, jobs, cache) as measurer:
        pooling = _pooling(settings)
        step = _progress(progress)
        step(Phase.READ)
        items = _items(inputs)
        contrast_items = _items(contrast) if contrast is not None else None
        _stdin_once([*items, *(contrast_items or [])])
        seen: set[str] = set()
        names = SourceNames()
        texts: dict[bytes, str] = {}
        read = functools.partial(
            _read,
            text_field=settings.text_field,
            seen=seen,
            notes=notes,
            names=names,
            input_format=settings.input_format,
            known_texts=texts,
            group_field=settings.group_field,
        )
        records, contrast_records = Records(), Records()
        chunks = read(items, role="<text>", records=records)
        contrast_chunks = (
            read(contrast_items, role="<contrast>", records=contrast_records, require_groups=False)
            if contrast_items is not None
            else None
        )
        # Duplicates were dropped as the texts were read, before any pooling.
        cut = _chunked(
            chunks,
            settings,
            pooling,
            notes,
            calibrated=True,
            records=records,
            split="calibrate",
        )
        # The contrast set pools when the writer's texts do, so the two stay alike in length.
        # A contrast set in one file is split as the writer's texts are: its AUC resamples
        # and its likeness weights are learned by document.
        contrast_cut = (
            _chunked(
                contrast_chunks,
                settings,
                cut.pooled is not None,
                notes,
                role="contrast ",
                following=True,
                records=contrast_records,
                split="calibrate",
            )
            if contrast_chunks is not None
            else None
        )
        contrast_windows = contrast_cut.windows if contrast_cut is not None else None
        missing = NoteCode.NO_SYNTAX.forms["build_surface"].long
        parser = _parser(settings.syntax, notes, missing, step)
        step(Phase.BUILD)
        report = _or_without_syntax(
            settings.syntax,
            parser,
            notes,
            missing,
            lambda parser: build_reference(
                cut.windows,
                parser=parser,
                top_k=settings.top_k,
                min_words=settings.min_words,
                contrast=contrast_windows,
                contrast_label=contrast_label,
                settings={
                    "inputs": _described(items, names),
                    "contrast": (
                        _described(contrast_items, names) if contrast_items is not None else None
                    ),
                    **settings.to_report(),
                    "pool_used": cut.pooled is not None,
                    "split_used": list(cut.split),
                    "contrast_split_used": list(contrast_cut.split) if contrast_cut else [],
                },
                keep_chunks=keep_chunks,
                passages=passages,
                measurer=measurer,
            ),
        )
        if report["chunk_count"] < 2:

            def usable_windows(size: int) -> int:
                return sum(
                    len(words(" ".join(prose(chunk.text).blocks))) >= settings.min_words
                    for chunk in window(cut.windows, size)
                )

            suggested = max(1, report["word_count"] // 2)
            while suggested > 1 and usable_windows(suggested) < 2:
                suggested //= 2
            fix = "add documents"
            if usable_windows(suggested) >= 2:
                fix += f", or pass a smaller --window-words ({suggested})"
            elif settings.min_words == 1 and report["word_count"] >= 2:
                fix += ", or add paragraph breaks and pass a smaller --window-words (1)"
            else:
                fix += f" with at least {settings.min_words} prose words each"
            raise StyleProfileError(
                "a reference needs at least 2 chunks to compare metrics; " + fix,
                code="reference_needs_chunks",
            )
        # Saved, so ``show`` repeats them: what stand-in documents mean for this reference.
        report["warnings"] += [*cut.warnings, *(contrast_cut.warnings if contrast_cut else ())]
        notes += _thin_reference(report, settings, cut.windows)
        step(Phase.DONE)
        sources = _sources([*chunks, *(contrast_chunks or [])])
        # Keep the profile exactly as it is saved (floats rounded), so scoring it before or
        # after a save and load gives the same numbers.
        _finish(measurer, notes)
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
    jobs: int = AUTO_JOBS,
    cache: bool = False,
) -> Evaluation:
    """Stress-test contrast likeness against edited drafts, as ``styleprofile evaluate``
    does: build a reference with ``contrast`` (the original drafts), then score each set in
    ``edited`` (label to edited copies, matched to their originals by name) with the weights
    learned without their original. ``settings.top_k`` does not apply here. ``jobs`` and
    ``cache`` are as for ``build``.
    """
    notes: list[Note] = []
    with _notes_on_error(notes), _measurer(progress, jobs, cache) as measurer:
        step = _progress(progress)
        step(Phase.READ)
        pooling = _pooling(settings)
        items, contrast_items = _items(inputs), _items(contrast)
        edited_items = {label: _items(value) for label, value in edited.items()}
        every = [*items, *contrast_items, *(item for v in edited_items.values() for item in v)]
        _stdin_once(every)
        seen: set[str] = set()
        names = SourceNames()
        texts: dict[bytes, str] = {}
        read = functools.partial(
            _read,
            text_field=settings.text_field,
            notes=notes,
            input_format=settings.input_format,
            group_field=settings.group_field,
        )
        records, contrast_records = Records(), Records()
        repeats: list[Repeat] = []
        reference_chunks = read(
            items, seen=seen, role="<text>", names=names, known_texts=texts, records=records
        )
        contrast_chunks = read(
            contrast_items,
            seen=seen,
            role="<contrast>",
            names=names,
            known_texts=texts,
            records=contrast_records,
            require_groups=False,
            repeats=repeats,
        )
        edited_chunks: dict[str, list[Chunk]] = {}
        # Each edited set is named on its own: its files carry the originals' names.
        edited_names = {label: SourceNames() for label in edited_items}
        for label, value in edited_items.items():
            # Edits are meant to resemble their originals, so they are not deduplicated.
            edited_chunks[label] = read(
                value,
                seen=set(),
                role=f"<{label}>",
                names=edited_names[label],
                require_groups=False,
            )
            edited_chunks[label] = _edits_of_repeats(
                label, edited_chunks[label], contrast_chunks, repeats, notes
            )
            overlap = {chunk.path for chunk in edited_chunks[label] if chunk.path} & seen
            if overlap:
                count = len(overlap)
                notes.append(
                    NoteCode.EDITED_OVERLAP.note(
                        "0",
                        f"{label}",
                        f"{count:,}",
                        f"{('' if count == 1 else 's')}",
                        f"{', '.join(_typed(value))}",
                    )
                )
        missing = NoteCode.NO_SYNTAX.forms["evaluate_surface"].long
        parser = _parser(settings.syntax, notes, missing, step)
        # The drafts pool as the writer's texts do, and each edited set as its originals did,
        # so an edited window pairs with its original window by name.
        # The originals and edited sets are not split: edits pair with originals by name.
        cut = _chunked(
            reference_chunks,
            settings,
            pooling,
            notes,
            calibrated=True,
            records=records,
            split="calibrate",
        )
        originals = _chunked(
            contrast_chunks,
            settings,
            cut.pooled is not None,
            notes,
            role="original ",
            following=True,
            records=contrast_records,
        )
        edited_windows = {
            label: _chunked(
                chunks, settings, originals.pooled is not None, [], like=originals.pooled
            ).windows
            for label, chunks in edited_chunks.items()
        }
        # An edited set that covers only some texts of a pooled window is compared with
        # that window rebuilt from just those texts.
        covered: dict[str, list[Chunk]] = {}
        if originals.pooled is not None:
            for label, chunks in edited_chunks.items():
                kept = _covered(contrast_chunks, chunks)
                if len(kept) < len(contrast_chunks):
                    covered[label] = pool(
                        kept, settings.window_words, like=originals.pooled
                    ).windows
        step(Phase.EVALUATE)
        report = _or_without_syntax(
            settings.syntax,
            parser,
            notes,
            missing,
            lambda parser: evaluate_rewording(
                cut.windows,
                originals.windows,
                edited_windows,
                covered=covered,
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
                    **{
                        name: value
                        for name, value in settings.to_report().items()
                        if name != "top_k"
                    },
                    "pool_used": cut.pooled is not None,
                    "split_used": list(cut.split),
                },
                measurer=measurer,
            ),
        )
        report["warnings"] += cut.warnings
        step(Phase.DONE)
        loaded = [*reference_chunks, *contrast_chunks]
        loaded += [chunk for chunks in edited_chunks.values() for chunk in chunks]
        _finish(measurer, notes)
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


def _thin_reference(
    report: ReferenceReport, settings: Settings, windows: Sequence[Chunk] = ()
) -> list[Note]:
    """Why a reference may be too small to trust, each with its fix. ``windows``, the
    chunks measured, show when one document holds most of them, so that its held-out range
    rests on the few windows of the others."""
    documents = report["document_count"]
    thin: list[Note] = []

    def note(message: str, setting: str | None = None) -> None:
        thin.append(Note(message, NoteCode.THIN_REFERENCE, setting=setting))

    if documents < ENOUGH_DOCUMENTS and settings.group_field and len(windows) > 1:
        note(
            NoteCode.THIN_REFERENCE.message(
                "0", f"{_plural(documents, 'document')}", f"{settings.group_field}"
            ),
            "group_field",
        )
    elif documents < ENOUGH_DOCUMENTS:
        note(NoteCode.THIN_REFERENCE.message("3", f"{_plural(documents, 'document')}"))
    if report["chunk_count"] < ENOUGH_CHUNKS:
        smaller = (
            "a smaller window_words"
            if settings.window_words
            else f"window_words {DEFAULT_WINDOW_WORDS}"
        )
        note(
            NoteCode.THIN_REFERENCE.message(
                "1", f"{_plural(report['chunk_count'], 'chunk')}", f"{ENOUGH_CHUNKS}", f"{smaller}"
            ),
            "window_words",
        )
    # The windows' own held-out range needs only two documents, but a document holding
    # most windows is judged against the others' few; hold that to the rule each shorter
    # length is held to (``calibration``).
    sizes = Counter(map(chunk_document, windows))
    if len(sizes) >= ENOUGH_DOCUMENTS:
        largest = max(sizes.values())
        rest = len(windows) - largest
        few = rest < MIN_CALIBRATION_PIECES or len(sizes) < MIN_CALIBRATION_DOCUMENTS
        if largest > rest and few:
            [(document, _)] = sizes.most_common(1)
            # A whole text, unlike a group of records or a part already split off, may
            # divide at headings or rules the automatic split left alone.
            first = next(w for w in windows if chunk_document(w) == document)
            text = not is_record(first) and _PART not in document
            how = NoteCode.THIN_REFERENCE.message("split_advice")
            note(
                NoteCode.THIN_REFERENCE.message(
                    "dominant",
                    f"{largest:,}",
                    _plural(len(windows), "chunk"),
                    _plural(rest, "chunk"),
                    str(MIN_CALIBRATION_PIECES),
                    str(MIN_CALIBRATION_DOCUMENTS),
                    NoteCode.THIN_REFERENCE.message("split_suffix", how) if text else "",
                ),
                "split_on" if text else None,
            )
    if report["word_count"] < ENOUGH_WORDS:
        note(
            NoteCode.THIN_REFERENCE.message(
                "2", f"{_plural(report['word_count'], 'word')}", f"{ENOUGH_WORDS:,}"
            )
        )
    return thin


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def _resolved(path: str | os.PathLike[str]) -> Path:
    return expand_path(os.fspath(path)).resolve()


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
        f"{shown!r} not found; a str input is a path, so use build_texts(...) or "
        "Profile.score_text(...) for raw text, or Text(...)",
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
        yield Chunk(literal_id(name), role, text.text)


def _read(
    items: Sequence[Input],
    text_field: str | Sequence[str] | None,
    seen: set[str],
    notes: list[Note],
    role: str,
    names: SourceNames,
    input_format: str = AUTO,
    known_texts: dict[bytes, str] | None = None,
    group_field: str | None = None,
    require_groups: bool = True,
    records: Records | None = None,
    repeats: list[Repeat] | None = None,
) -> list[Chunk]:
    """Chunks from each input, in order, skipping files an earlier path (tracked in
    ``seen`` by real path) already gave. ``Text`` inputs get ``role`` as their source, so
    texts in different roles (writer, contrast) are never the same document. ``names`` is
    shared across calls, so two inputs' files never get the same saved source.

    With ``known_texts`` (see ``drop_duplicates``), documents whose text repeats one read
    before, here or in an earlier role, are dropped with a note: twins on both sides of
    held-out calibration would make it look too tight. That is a separate check from a file
    given twice, which is recognized by its path. Grouped JSONL records are compared one by
    one, before they are pooled. ``repeats`` gathers the documents dropped, with the copies
    kept (``drop_duplicates``).

    With ``group_field``, JSONL records without a value in it are noted. An input where no
    record has one is an error naming the fields the records do have (the field is likely
    misspelled); with ``require_groups`` False (drafts, which rarely carry the writer's
    threads), it is read ungrouped instead, each record a document of its own, with a note.
    A group value found in several inputs typed one by one is noted too, since each input's
    records are separate documents. ``records`` gathers what the JSONL records held, for the
    pooling note."""
    texts = iter(_text_chunks([item for item in items if isinstance(item, Text)], role))
    chunks: list[Chunk] = []
    inputs_with: Counter[str] = Counter()  # how many inputs each group value is found in
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
        found = Records(group_field)
        loaded = load_chunks(
            [value],
            text_field,
            names=names,
            input_format=input_format,
            notes=notes,
            records=found,
        )
        if group_field is not None and _check_groups(
            value, found, group_field, notes, require_groups
        ):
            loaded = [
                found.ungrouped.get((chunk.path or chunk.source, chunk.id), chunk)
                for chunk in loaded
            ]
            found.missing = 0
            found.group_field = None
        elif group_field is not None:
            inputs_with.update(found.values.get(group_field, set()))
        if records is not None:
            records.add(found)
        files = {chunk.path for chunk in loaded if chunk.path is not None}
        repeated = files & seen
        if repeated and repeated == files:
            notes.append(NoteCode.REPEATED_INPUT.note("0", f"{value}"))
            if not contributed:
                # Every file came from an earlier input, so no source carries this name:
                # leave it out of the report's settings (``_described``).
                names.roots.pop(value, None)
        elif repeated:
            notes.append(
                NoteCode.REPEATED_INPUT.note("1", f"{_plural(len(repeated), 'file')}", f"{value}")
            )
        chunks += [chunk for chunk in loaded if chunk.path not in repeated]
        seen |= files
    shared = sorted(group for group, count in inputs_with.items() if count > 1)
    if shared:
        notes.append(NoteCode.GROUPING.note("0", f"{group_field}", f"{shared[0]!r}"))
    if known_texts is not None:
        chunks, note = drop_duplicates(chunks, known_texts, repeats)
        if note:
            notes.append(note)
    return chunks


def _locations(items: Sequence[Input], names: SourceNames) -> dict[str, str]:
    """Each saved source's path as its input was typed: ``drafts/2024/a.md`` for the saved
    ``2024/a.md`` inside ``drafts``, for the command line to name files the user can open.
    Nothing here is saved."""
    # Standard input is named apart from a file called ``stdin``.
    locations: dict[str, str] = {"stdin": STDIN_SHOWN}
    for item in items:
        if isinstance(item, Text | Chunk):
            continue
        value = os.fspath(item)
        root = names.roots.get(value)
        if value == "-" or root is None:
            continue
        for source, file in names.files.items():
            if source == root:
                locations[source] = value
            elif source.startswith(f"{root}/") or not root:
                inner = source[len(root) + 1 :] if root else source
                if Path(file).is_relative_to(expand_path(value).resolve()):
                    locations[source] = os.path.join(value, inner)
    return locations


def _check_groups(
    value: str, found: Records, group_field: str, notes: list[Note], required: bool
) -> bool:
    """Note the records of one input with no value in ``group_field``. When none has one,
    refuse the input if ``required``, naming the fields its records have; otherwise note it
    and return True: the caller reads it ungrouped."""
    if not found.missing:
        return False
    if found.missing == found.count:
        if required:
            fields = ", ".join(sorted(found.values)) or "none besides the text"
            raise StyleProfileError(
                f"{value}: no record has a value in the group field {group_field!r}; its "
                f"records have the fields {fields}",
                code="group_field",
            )
        notes.append(NoteCode.MISSING_GROUP.note("1", f"{value}", f"{group_field}"))
        return True
    notes.append(
        NoteCode.MISSING_GROUP.note(
            "0",
            f"{value}",
            f"{found.missing:,}",
            f"{_plural(found.count, 'record')}",
            f"{group_field}",
            f"{group_id(group_field, None)}",
        )
    )
    return False


def _pooling(settings: Settings) -> Literal["auto"] | bool:
    """Whether ``settings`` pool texts: ``"auto"`` leaves it to their median length. Called
    before any input is read, so an impossible setting fails first."""
    if not settings.window_words:
        if settings.pool is True:
            _invalid(
                "window_words", "window_words must be above 0 to pool texts into windows of it"
            )
        return False
    return settings.pool


@dataclass(frozen=True)
class _Cut:
    """Texts cut for measuring: the windows; how they were pooled (None when not); and
    whether the median text is under a quarter of a window."""

    windows: list[Chunk]
    pooled: Pooled | None
    short: bool
    # How texts were split into documents, as ``split_used`` records it.
    split: tuple[str, ...] = ()
    # What stand-in documents mean for the report, one warning per text cut into them.
    warnings: tuple[str, ...] = ()


def _noun(chunks: Sequence[Chunk], records: Records | None) -> str:
    """What the texts are, for notes: records, files of a folder, or texts."""
    if records is not None and chunks and records.count >= len(chunks):
        return "record"
    if chunks and all(chunk.folder for chunk in chunks):
        return "file"
    return "text"


def _chunked(
    chunks: list[Chunk],
    settings: Settings,
    pooled: Literal["auto"] | bool,
    notes: list[Note],
    *,
    role: str = "",
    calibrated: bool = False,
    following: bool = False,
    like: Pooled | None = None,
    records: Records | None = None,
    split: Literal["calibrate", "score"] | None = None,
) -> _Cut:
    """The chunks to measure: split into documents as ``split`` says (``_split``; None
    splits nothing), then windowed or pooled (``_pooled``), and last, texts left whole that
    ``_split`` marked are cut into stand-in documents (``_stand_ins``)."""
    for chunk in chunks:
        parsed = prose(chunk.text)
        if unlikely_english(parsed):
            notes.append(NoteCode.NON_ENGLISH.note("0", f"{_label(chunk)}"))
    stand_ins: set[str] = set()
    used: set[str] = set()
    if split is not None and settings.split_on != NONE:
        chunks, stand_ins, used = _split(chunks, settings, notes, role, split == "calibrate")
    cut = _pooled(
        chunks,
        settings,
        pooled,
        notes,
        role=role,
        calibrated=calibrated,
        following=following,
        like=like,
        records=records,
    )
    windows, warnings = cut.windows, ()
    for chunk in windows:
        parsed = prose(chunk.text)
        size = len(words(parsed.text))
        if paragraph_metrics_missing(parsed, size):
            notes.append(NoteCode.NO_PARAGRAPH_BREAKS.note("0", f"{chunk.id}"))
        if settings.window_words and size > 2 * settings.window_words:
            notes.append(
                NoteCode.OVERSIZE_CHUNK.note(
                    "0",
                    f"{chunk.id}",
                    f"{size:,}",
                    f"{settings.window_words:,}",
                    setting="window_words",
                )
            )
    if stand_ins:
        windows, warnings = _stand_ins(windows, stand_ins, notes, role)
        if warnings:
            used.add(STAND_IN)
    return dataclasses.replace(
        cut, windows=windows, split=tuple(sorted(used)), warnings=tuple(warnings)
    )


def _split(
    chunks: list[Chunk], settings: Settings, notes: list[Note], role: str, calibrating: bool
) -> tuple[list[Chunk], set[str], set[str]]:
    """``chunks`` with Markdown, text or HTML texts split into documents at their headings
    or rules as ``settings.split_on`` asks (``split.plan_split``; JSONL records are never
    split, and a ``.txt`` file's chapter lines are headings), each split noted and parts
    that repeat another word for word dropped; the documents left whole that may be cut
    into stand-ins; and the kinds of split made.

    ``"auto"`` splits only when ``calibrating`` (the writer's texts, or the contrast set);
    drafts never need a split to be judged. Then:

    - with fewer than ``MIN_CALIBRATION_DOCUMENTS`` documents, too few to calibrate at all,
      it splits every text it can into at least ``split.MIN_PARTS`` parts, and leaves the
      rest for stand-ins;
    - with fewer than ``FLOOR_DOCUMENTS`` documents (several manuscripts), it splits every
      text its markers divide into two parts or more of a median of a whole window (a
      manuscript's chapters, not a blog post's sections), never into stand-ins: below that many
      documents each is worth only a few independent calibration pieces (their similarity
      is floored), so chapters as documents calibrate more, and more tightly, than their
      books. This is for the writer's texts; the contrast set is left as it is;
    - with more, it splits nothing.

    ``"heading"``, ``"heading:N"`` or ``"rule"`` split any text their markers divide into
    two parts or more, and note those they found nothing to split at. Parts are at least half
    a window (of ``DEFAULT_WINDOW_WORDS`` when windowing is off)."""
    mode = settings.split_on
    automatic = mode == AUTO
    count = len(set(map(chunk_document, chunks)))
    few = count < MIN_CALIBRATION_DOCUMENTS
    if automatic and (not calibrating or count >= FLOOR_DOCUMENTS or (role and not few)):
        return chunks, set(), set()
    size = settings.window_words or DEFAULT_WINDOW_WORDS
    minimum = MIN_PARTS if automatic and few else ASKED_MIN_PARTS
    # Between the bands, a text's parts must hold a whole window each (a median): a
    # manuscript's chapters split, a blog post's sections do not.
    median = 1.0 if automatic and not few else 0.5
    plans = [
        None
        if is_record(chunk)
        else plan_split(chunk.text, size, mode, minimum, plain=_plain(chunk), median_windows=median)
        for chunk in chunks
    ]
    # With a few documents, each is worth only a few independent calibration pieces, so the
    # texts split at their structure whenever it gives more documents.
    reason = (
        NoteCode.SPLIT.message("reason", _plural(count, "document"))
        if automatic and not few
        else ""
    )
    out: list[Chunk] = []
    whole: list[Chunk] = []
    used: set[str] = set()
    done: list[str] = []
    parts = 0
    for chunk, plan in zip(chunks, plans, strict=True):
        if plan is None:
            out.append(chunk)
            whole += [] if is_record(chunk) else [chunk]
            continue
        document = chunk_document(chunk)
        for number, (part, text) in enumerate(zip(plan.parts, plan.texts, strict=True), 1):
            out.append(
                Chunk(
                    part_id(chunk.id, number, part.title),
                    chunk.source,
                    text,
                    chunk.path,
                    chunk.folder,
                    f"{document}{_PART}{number}",
                )
            )
        used.add(plan.kind)
        parts += len(plan.parts)
        done.append(
            NoteCode.SPLIT.message(
                "1",
                f"{role}",
                f"{_label(chunk)}",
                f"{_plural(len(plan.parts), 'document')}",
                f"{plan.describe()}",
            )
        )
    if len(done) > NOTED_SPLITS or (reason and len(done) > 1):
        where = "headings or rules" if len(used) > 1 else f"{next(iter(used))}s"
        done = [NoteCode.SPLIT.message("many", f"{len(done):,}", role, f"{parts:,}", where)]
    if reason and done:
        done = [done[0] + reason]
    notes += [Note(message, NoteCode.SPLIT, setting="split_on") for message in done]
    if done:
        # A newsletter issue pasted twice is two parts with one text.
        out, repeated = drop_duplicates(out)
        notes += [repeated] if repeated else []
    if not automatic and whole:
        level = mode.partition(":")[2]
        markers = f"level-{level} headings" if level else "headings" if mode == HEADING else "rules"
        if len(whole) == 1:
            message = NoteCode.SPLIT.message("2", f"{role}", f"{_label(whole[0])}", f"{markers}")
            kept = NoteCode.SPLIT.message("3")
        else:
            texts = len(chunks) - sum(map(is_record, chunks))
            message = NoteCode.SPLIT.message(
                "4", f"{len(whole):,}", f"{_plural(texts, role + 'text')}", f"{markers}"
            )
            message += NoteCode.SPLIT.message("5")
            kept = NoteCode.SPLIT.message("6")
        notes.append(NoteCode.SPLIT.note("0", f"{message}", f"{kept}", setting="split_on"))
    stand_ins = {chunk_document(chunk) for chunk in whole} if automatic and few else set()
    return out, stand_ins, used


# Joins a split text's document to its part's number (see ``Chunk.document``).
_PART = "\x1c"
# More texts split than this are noted together.
NOTED_SPLITS = 3


def _plain(chunk: Chunk) -> bool:
    """Whether ``chunk`` is a plain-text file, whose chapter lines are headings."""
    return chunk.path is not None and Path(chunk.path).suffix.lower() == ".txt"


def _label(chunk: Chunk) -> str:
    """A text as notes name it: its saved source (a file), or a ``Text``'s name."""
    return document_label(chunk.source, base_id(chunk.id))


def _stand_ins(
    windows: list[Chunk], documents: set[str], notes: list[Note], role: str
) -> tuple[list[Chunk], list[str]]:
    """``windows`` with those of each of ``documents`` that has ``split.MIN_PARTS`` windows
    or more grouped, in order, into stand-in documents (``split.stand_in_groups``), named
    ``book.md#3#w1``, each text with a note saying so; and, for the report, a warning on what
    that means for each (the same list and none when there are none)."""
    positions: dict[str, list[int]] = {}
    for index, chunk in enumerate(windows):
        document = chunk_document(chunk)
        if document in documents:
            positions.setdefault(document, []).append(index)
    cut = [(document, found) for document, found in positions.items() if len(found) >= MIN_PARTS]
    if not cut:
        return windows, []
    windows = list(windows)
    warnings: list[str] = []
    for document, found in cut:
        groups = stand_in_groups(len(found))
        first = windows[found[0]]
        for number, run in enumerate(groups, 1):
            for place, position in enumerate(run, 1):
                chunk = windows[found[position]]
                windows[found[position]] = Chunk(
                    f"{part_id(base_id(chunk.id), number)}#w{place}",
                    chunk.source,
                    chunk.text,
                    chunk.path,
                    chunk.folder,
                    f"{document}{_PART}{number}",
                )
        label = _label(first)
        grouped = (
            NoteCode.STAND_INS.message("windows", f"{len(found):,}")
            if len(groups) == len(found)
            else NoteCode.STAND_INS.message("groups", f"{len(found):,}", str(len(groups)))
        )
        notes.append(
            NoteCode.STAND_INS.note(
                "0", f"{role}", f"{label}", f"{MIN_PARTS}", f"{grouped}", setting="split_on"
            )
        )
        if role:
            warnings.append(
                NoteCode.CONTRAST_STAND_INS.message("0", f"{role}", f"{len(groups)}", f"{label}")
            )
        else:
            warnings.append(
                NoteCode.CALIBRATION_STAND_INS.message("0", f"{len(groups)}", f"{label}")
            )
    return windows, warnings


def _pooled(
    chunks: list[Chunk],
    settings: Settings,
    pooled: Literal["auto"] | bool,
    notes: list[Note],
    *,
    role: str = "",
    calibrated: bool = False,
    following: bool = False,
    like: Pooled | None = None,
    records: Records | None = None,
) -> _Cut:
    """The chunks to measure: windowed, or, when ``pooled`` and pooling joins any texts,
    pooled (see ``profile.pool``) with a note saying so.

    ``"auto"`` pools only when the median text is short and pooling still leaves
    ``ENOUGH_CHUNKS`` windows: a handful of short texts joined into one or two windows would
    lose held-out calibration altogether. A set that pools because another did
    (``following``: the contrast drafts) pools only if it keeps ``ENOUGH_DOCUMENTS``
    documents. ``role`` (``"contrast "``) names the set in the note; ``calibrated`` marks
    the writer's texts, whose notes say why they pooled or did not, what that means for
    held-out calibration, and what is wrong with a group field that cannot work. Given
    ``like``, the chunks pool as that set did, without a note."""
    if not settings.window_words:
        return _Cut(chunks, None, False)
    noun = _noun(chunks, records)
    if calibrated and settings.group_field:
        notes += _grouping_notes(chunks, settings, pooled)
    joined = pool(chunks, settings.window_words, like, auto=pooled == AUTO, join=bool(pooled))
    if like is not None:
        return _Cut(joined.windows, joined, joined.short)
    if not pooled or not joined.together:
        # Not pooled, or nothing to join (one text, or long ones): windows as ``window``
        # makes them, and no note.
        return _Cut(joined.windows, None, joined.short)
    if pooled == AUTO and len(joined.windows) < ENOUGH_CHUNKS:
        if calibrated:
            notes.append(
                NoteCode.POOLED.note(
                    "0", f"{noun}", f"{_plural(len(joined.windows), 'window')}", setting="pool"
                )
            )
        return _Cut(_windowed(chunks, settings.window_words), None, joined.short)
    if following and len(set(map(chunk_document, joined.windows))) < min(
        ENOUGH_DOCUMENTS, len(set(map(chunk_document, chunks)))
    ):
        return _Cut(_windowed(chunks, settings.window_words), None, joined.short)
    count = len(chunks)
    texts = (
        _plural(count, role + noun)
        if joined.together == count
        else f"{joined.together:,} of {_plural(count, role + noun)}"
    )
    message = NoteCode.POOLED.message(
        "1", f"{texts}", f"{_plural(len(joined.windows), 'window')}", f"{settings.window_words:,}"
    )
    setting = None
    if settings.pool == AUTO and calibrated:
        message += NoteCode.POOLED.message("2", f"{noun}")
    grouped = settings.group_field is not None and any(map(is_grouped, chunks))
    if grouped:
        message += NoteCode.POOLED.message("3", f"{settings.group_field}")
    elif calibrated:
        message += NoteCode.POOLED.message("4", f"{noun}")
        candidates = records.group_fields() if records is not None else []
        if candidates:
            shown = ", ".join(f"{name} ({values:,} values)" for name, values in candidates[:3])
            first = candidates[0][0]
            message += NoteCode.POOLED.message("5", f"{shown}", f"{first}")
        setting = "group_field"
    notes.append(Note(message, NoteCode.POOLED, setting=setting))
    return _Cut(joined.windows, joined, joined.short)


def _grouping_notes(
    chunks: Sequence[Chunk], settings: Settings, pooled: Literal["auto"] | bool
) -> list[Note]:
    """What is wrong with a group field that groups the writer's records into groups too
    small to pool. One group, which leaves nothing to hold out, is a thin reference
    (``_thin_reference``)."""
    field = settings.group_field
    sizes: Counter[str] = Counter()
    for chunk in chunks:
        sizes[chunk_document(chunk)] += len(words(chunk.text))
    median = statistics.median(sizes.values()) if sizes else 0
    if pooled and len(sizes) > 1 and median < settings.window_words / 4:
        return [NoteCode.GROUPING.note("1", f"{field}", f"{median:,.0f}", setting="group_field")]
    return []


def _covered(originals: Sequence[Chunk], edited: Sequence[Chunk]) -> list[Chunk]:
    """The original texts an edited set has copies of (``cover_key``: a record by its id, a
    grouped record by its group and own id), also when one set was read from a folder and
    the other from a file given directly (``bare_key``)."""
    keys: set[tuple[str, str | None]] = set()
    for chunk in edited:
        key, record = cover_key(chunk)
        keys.add((key, record))
        if (short := bare_key(key)) is not None:
            keys.add((short, record))

    def covers(chunk: Chunk) -> bool:
        key, record = cover_key(chunk)
        short = bare_key(key)
        return (key, record) in keys or (short is not None and (short, record) in keys)

    return [chunk for chunk in originals if covers(chunk)]


def _cover_keys(chunk: Chunk) -> list[tuple[str, str | None]]:
    """The ``cover_key`` of ``chunk``, then the same without its folder file (``bare_key``),
    so a record read from a folder and one read from its file directly find each other."""
    key, record = cover_key(chunk)
    short = bare_key(key)
    return [(key, record)] if short is None else [(key, record), (short, record)]


def _shown(chunk: Chunk) -> str:
    """A draft as edit notes name it: its id, and for a grouped record also its own id."""
    key, record = cover_key(chunk)
    return f"{record} in {key}" if record is not None else key


def _unique(pairs: Iterable[tuple[_K, _V]]) -> dict[_K, _V | None]:
    """``pairs`` as a mapping, with None for a key given different values."""
    found: dict[_K, _V | None] = {}
    for key, value in pairs:
        found[key] = value if found.get(key, value) == value else None
    return found


def _edits_of_repeats(
    label: str,
    edited: Sequence[Chunk],
    originals: Sequence[Chunk],
    repeats: Sequence[Repeat],
    notes: list[Note],
) -> list[Chunk]:
    """``edited`` with each edit of a draft dropped as a word-for-word repeat of another
    (``drop_duplicates``) renamed to pair with the copy kept: they had the same text, so
    the edit is an edit of that copy too. Each is noted.

    Only one edit of a text is scored. When the kept copy is edited as well, its own edit
    is used; otherwise the first edit in the set, and the others are left out with a note.
    An edit of a draft dropped for repeating one of the writer's texts has no draft to pair
    with, so it is left out with a note too.
    """
    if not repeats:
        return list(edited)
    # Cover keys (and their bare forms) of the drafts kept, and of those dropped. A bare key
    # that stands for several drafts pairs with none of them.
    kept = _unique((key, cover_key(chunk)) for chunk in originals for key in _cover_keys(chunk))
    dropped = _unique((key, repeat) for repeat in repeats for key in _cover_keys(repeat.dropped))

    def original(chunk: Chunk) -> tuple[str, str | None] | Repeat | None:
        """The kept draft ``chunk`` is an edit of (its cover key), or the repeat it edits."""
        for key in _cover_keys(chunk):
            if kept.get(key) is not None:
                return kept[key]
            if dropped.get(key) is not None:
                return dropped[key]
        return None

    used = {found for chunk in edited if isinstance(found := original(chunk), tuple)}
    decided: dict[tuple[str, str | None], Chunk | None] = {}  # by the edit's cover key
    paired: list[tuple[str, str]] = []
    doubled: list[tuple[str, str]] = []
    outside: list[tuple[str, str]] = []
    as_is: set[tuple[str, str | None]] = set()
    result: list[Chunk] = []
    for chunk in edited:
        found = original(chunk)
        if not isinstance(found, Repeat):
            result.append(chunk)
            continue
        own = cover_key(chunk)
        if own not in decided:
            if found.kept is None:
                decided[own] = None
                outside.append((_shown(chunk), found.label))
            elif paired_as(chunk, found.kept) is None:  # grouped and ungrouped never pair
                decided[own] = None
                as_is.add(own)
            elif cover_key(found.kept) in used:
                decided[own] = None
                doubled.append((_shown(chunk), _shown(found.kept)))
            else:
                decided[own] = found.kept
                used.add(cover_key(found.kept))
                paired.append((_shown(chunk), _shown(found.kept)))
        like = decided[own]
        if own in as_is:
            result.append(chunk)
        elif like is not None:
            result.append(paired_as(chunk, like) or chunk)

    def note(message: str) -> None:
        notes.append(NoteCode.DUPLICATES.note("0", f"{label}", f"{message}"))

    if len(paired) == 1:
        ((edit, copy),) = paired
        note(
            NoteCode.DUPLICATES.message(
                "1", f"{edit!r}", f"{copy!r}", f"{edit!r}", f"{copy!r}", f"{copy!r}"
            )
        )
    elif paired:
        edit, copy = paired[0]
        note(NoteCode.DUPLICATES.message("4", f"{len(paired):,}", f"{edit!r}", f"{copy!r}"))
    if len(doubled) == 1:
        ((edit, copy),) = doubled
        note(NoteCode.DUPLICATES.message("2", f"{edit!r}", f"{copy!r}", f"{copy!r}"))
    elif doubled:
        edit, copy = doubled[0]
        note(NoteCode.DUPLICATES.message("5", f"{len(doubled):,}", f"{edit!r}", f"{copy!r}"))
    if len(outside) == 1:
        ((edit, text),) = outside
        note(NoteCode.DUPLICATES.message("3", f"{edit!r}", f"{text}"))
    elif outside:
        edit, text = outside[0]
        note(NoteCode.DUPLICATES.message("6", f"{len(outside):,}", f"{edit!r}", f"{text}"))
    return result


def _pooling_mismatch(cut: _Cut, reference_pooled: bool) -> list[str]:
    """A warning for texts pooled into windows against a reference that was not pooled:
    they read closer to it than they are. The other way round needs none: each short text is
    judged at its own length against the reference's length calibration, or abstains."""
    if cut.pooled is not None and not reference_pooled:
        return [NoteCode.POOLING_MISMATCH.message("0")]
    return []


def _windowed(chunks: list[Chunk], window_words: int) -> list[Chunk]:
    return window(chunks, window_words) if window_words else chunks


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
