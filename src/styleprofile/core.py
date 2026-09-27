"""The vocabulary every module shares, with no imports of its own: errors, notes, progress
and verdicts. Anything in the package can import it without a cycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StyleProfileError(ValueError):
    """Style profile inputs, settings or a reference profile are unusable.

    ``code`` names the kind of problem so a front end can add its own advice (such as the
    command-line flag that fixes it); the messages themselves never mention flags. When the
    problem is a setting, ``setting`` names it (``"min_words"``), and the message spells it
    the same way, as a whole word, so a front end can substitute its own name for it.
    ``notes`` holds what the run noted before it failed, for a front end to show before the
    error: a skipped file often explains it.
    """

    def __init__(
        self, message: str, *, code: str | None = None, setting: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.setting = setting
        self.notes: tuple[Note, ...] = ()


class NoteCode(StrEnum):
    """The kinds of ``Note``. Front ends dispatch on these, so every note has one."""

    NO_SYNTAX = "no_syntax"
    """spaCy is not installed, so syntax metrics are left out."""
    REPEATED_INPUT = "repeated_input"
    """A file was given more than once and is read once."""
    EDITED_OVERLAP = "edited_overlap"
    """An edited set shares files with the writer's texts or the original drafts."""
    SETTING_OVERRIDDEN = "setting_overridden"
    """A score overrides a setting the profile was built with (``Note.setting`` names it)."""
    THIN_REFERENCE = "thin_reference"
    """The reference is too small to trust, for the reason in the message."""
    READ_AS_HTML = "read_as_html"
    """A Markdown or text file, or stdin, looked like HTML and was read as HTML."""
    READ_AS_HTML_IN_FOLDER = "read_as_html_in_folder"
    """Markdown or text files in a folder looked like HTML and were read as HTML."""
    READ_AS_JSONL = "read_as_jsonl"
    """Stdin was JSON objects, one per line, and was read as JSONL."""
    EMPTY_HTML = "empty_html"
    """HTML had no readable text once converted (all chrome, or an empty page)."""
    SKIPPED_FILES = "skipped_files"
    """A folder walk skipped documents it cannot read (Word, PDF, ...)."""
    SKIPPED_DIRS = "skipped_dirs"
    """A folder walk left out static-site output and template folders."""
    DUPLICATES = "duplicates"
    """Documents repeating another's text word for word were dropped."""


@dataclass(frozen=True)
class Note:
    """Something a front end should tell the user about a run that is not an error: an input
    given twice, or syntax metrics left out because spaCy is missing.

    ``message`` never mentions flags; when it names a setting, ``setting`` says which, as
    for ``StyleProfileError``. Notes describe the run, not the text, so they are not saved
    in reports; warnings about the text are (``report["warnings"]``).
    """

    message: str
    code: NoteCode
    setting: str | None = None


class Phase(StrEnum):
    """The phases a run reports to its ``progress`` callback. Every run goes ``READ``,
    ``LOAD_PARSER`` (only when it uses spaCy), then its own work (``BUILD``, ``SCORE`` or
    ``EVALUATE``), then ``DONE``."""

    READ = "read"
    LOAD_PARSER = "load_parser"
    BUILD = "build"
    SCORE = "score"
    EVALUATE = "evaluate"
    DONE = "done"


@dataclass(frozen=True)
class Progress:
    """Where a run is. ``done`` and ``total`` count a phase's work (chunks, say) and are
    None for a phase that does not count it; every phase is reported once at its start."""

    phase: Phase
    done: int | None = None
    total: int | None = None


class Verdict(StrEnum):
    """Delta in words, against the writer's own held-out range."""

    CLOSE = "close"
    SOMEWHAT_DIFFERENT = "somewhat different"
    CLEARLY_DIFFERENT = "clearly different"
    VERY_DIFFERENT = "very different"
    NOT_COMPARABLE = "not comparable"
    """No metric could be compared with the reference."""
    TOO_SHORT = "too short to judge"
    """Too little text for any verdict: under 75 words, or shorter than any length the
    reference is calibrated for (see ``calibration``)."""


# The Delta verdicts from closest to furthest, indexed by level.
DISTANCES = (
    Verdict.CLOSE,
    Verdict.SOMEWHAT_DIFFERENT,
    Verdict.CLEARLY_DIFFERENT,
    Verdict.VERY_DIFFERENT,
)


class LikenessVerdict(StrEnum):
    """Likeness to the contrast set in words, from the writer's own range up to the contrast
    drafts' typical score. ``words(label)`` names the contrast set, as reports print it."""

    LIKE_REFERENCE = "like the reference"
    FEW_TRAITS = "a few traits"
    LEANS = "leans"
    LIKE_DRAFTS = "like the drafts"
    TOO_SHORT = "too short to judge"
    """Too little text for any verdict, as ``Verdict.TOO_SHORT``."""

    def words(self, label: str) -> str:
        """``"leans LLM"`` for ``LEANS`` and the label ``LLM``."""
        return {
            LikenessVerdict.LIKE_REFERENCE: "like the reference",
            LikenessVerdict.FEW_TRAITS: f"a few {label} traits",
            LikenessVerdict.LEANS: f"leans {label}",
            LikenessVerdict.LIKE_DRAFTS: f"like the {label} drafts",
            LikenessVerdict.TOO_SHORT: "too short to judge",
        }[self]


# The likeness verdicts by level, 0 to 3.
LIKENESSES = (
    LikenessVerdict.LIKE_REFERENCE,
    LikenessVerdict.FEW_TRAITS,
    LikenessVerdict.LEANS,
    LikenessVerdict.LIKE_DRAFTS,
)
