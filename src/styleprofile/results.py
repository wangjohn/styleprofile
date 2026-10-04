"""Library result objects and their report views."""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Unpack

from styleprofile.calibration import MIN_JUDGED_WORDS, too_short_text
from styleprofile.core import (
    DISTANCES,
    LIKENESSES,
    LikenessVerdict,
    Note,
    NoteCode,
    Phase,
    StyleProfileError,
    Verdict,
)
from styleprofile.corpus.preparation import (
    _chunked,
    _described,
    _items,
    _locations,
    _pooling,
    _pooling_mismatch,
    _read,
    _sources,
    _stdin_once,
)
from styleprofile.corpus.reading import TEXT_FIELDS, Records, SourceNames, expand_path
from styleprofile.display import format_evaluation, format_reference_summary, format_summary
from styleprofile.drift import Passage
from styleprofile.reports import (
    REFERENCE,
    _resolved,
    check_report,
    load_reference,
    report_kind,
    version_notes,
    write_report,
)
from styleprofile.runtime import (
    _finish,
    _measurer,
    _notes_on_error,
    _or_without_syntax,
    _parser,
    _progress,
)
from styleprofile.schema import (
    Baseline,
    DocumentEntry,
    EvaluationReport,
    ReferenceReport,
    ScoreReport,
    ScoreVerdict,
)
from styleprofile.scoring import score
from styleprofile.settings import AUTO, AUTO_JOBS, Settings, _overridden, _strings
from styleprofile.split import NONE

if TYPE_CHECKING:
    from styleprofile.settings import Inputs, ProgressCallback, SettingsOverrides

_R = TypeVar("_R", ReferenceReport, ScoreReport, EvaluationReport)


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
