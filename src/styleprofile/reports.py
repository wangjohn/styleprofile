"""Saved report versions, validation and JSON I/O."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

from styleprofile.core import Note, NoteCode, StyleProfileError
from styleprofile.corpus.reading import _read_text, expand_path
from styleprofile.schema import (
    EvaluationReport,
    Problem,
    ReferenceReport,
    Report,
    ScoreReport,
    find_problem,
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
# Reports of another major version are refused: their measurements may not match.
# Also in 6, before any release: reference profiles leave out per-chunk rows (unless built
# with keep_chunks) and store ``document_count``; sources are saved by their input's name,
# never as paths; the contrast AUC's ``bootstrap`` records its method and resamples.
# 7 (0.2.0): nominalizations_per_1k no longer counts words such as fence, city or sentence
# (see ``syntax.is_nominalization``), so a version 6 profile's values would not match.
# 8 (0.2.0): long plain blocks split at sentences; long single paragraphs omit structure.
VERSION = 8
# Additive report fields increment this; legacy integer versions have minor zero.
MINOR_VERSION = 1
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


def report_kind(report: Mapping[str, Any]) -> str:
    """``"reference"``, ``"score"`` or ``"evaluation"``, from the report's ``kind``."""
    kind = report.get("kind")
    if kind not in KINDS:
        raise StyleProfileError(
            f"the report has no known kind, so it is {UNREADABLE}", code="outdated"
        )
    return kind


def load_report(path: Path, name: str | None = None, *, notes: list[Note] | None = None) -> Report:
    """Read any styleprofile report: reference, score or evaluation.

    A report of another major version, or one that lacks a key its kind needs, is
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
    if notes is not None:
        notes.extend(version_notes(report))
    return cast(Report, report)


def check_report(report: Mapping[str, Any], name: str = "the report") -> None:
    """Refuse a report of a known kind (see ``report_kind``) that this styleprofile cannot
    read: one of another major version (``check_version``), or one whose shape its kind does not
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
    """Refuse incompatible majors and malformed numbers; matching majors share semantics.

    The missing minor of a legacy integer report is zero. Additive minors do not change
    existing measurements, required fields or their meaning.
    """
    kind = report.get("kind")
    expected = EVALUATION_VERSION if kind == EVALUATION else VERSION
    version = report.get("version")
    minor = report.get("minor_version", 0)
    if type(version) is not int or version < 1 or type(minor) is not int or minor < 0:
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


def version_notes(report: Mapping[str, Any]) -> tuple[Note, ...]:
    """Transient compatibility advice for a valid report, also used by library results."""
    expected = EVALUATION_VERSION if report.get("kind") == EVALUATION else VERSION
    minor = report.get("minor_version", 0)
    if report.get("version") == expected and type(minor) is int and minor > MINOR_VERSION:
        return (NoteCode.NEWER_REPORT_VERSION.note(),)
    return ()


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


def _resolved(path: str | os.PathLike[str]) -> Path:
    return expand_path(os.fspath(path)).resolve()
