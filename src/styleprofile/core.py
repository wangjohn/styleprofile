"""The vocabulary every module shares, with no imports of its own: errors, notes, progress
and verdicts. Anything in the package can import it without a cycle."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from string import Formatter


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

    NEWER_REPORT_VERSION = "newer_report_version"
    """A saved report uses a newer additive minor; unknown fields may be ignored."""
    NO_PARAGRAPH_BREAKS = "no_paragraph_breaks"
    """Paragraph structure is omitted for a long single paragraph."""
    OVERSIZE_CHUNK = "oversize_chunk"
    """A chunk is still longer than twice the requested window."""
    NON_ENGLISH = "non_english"
    """Text has little evidence of English; English-based metrics may be unreliable."""
    WORKERS_LIMITED = "workers_limited"
    """An explicit worker count was reduced to fit the memory allowance."""
    NO_SYNTAX = "no_syntax"
    """spaCy or its English model is not installed, so syntax metrics are left out."""
    REPEATED_INPUT = "repeated_input"
    """A file was given more than once and is read once."""
    REPEATED_ID = "repeated_id"
    """Records in one JSONL file share an id; each is still its own document."""
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
    CACHE_UNAVAILABLE = "cache_unavailable"
    """The measurement cache could not be opened, read or written, so the run did without."""
    DUPLICATES = "duplicates"
    """Documents repeating another's text word for word were dropped."""
    POOLED = "pooled"
    """Short texts were joined into windows (the message says how many, and why), or were
    short but too few to join."""
    MISSING_GROUP = "missing_group"
    """Some records have no value in the group field, so they are read as one group."""
    SHORT_TEXTS = "short_texts"
    """Texts scored one by one are short, so each is judged on little text."""
    GROUPING = "grouping"
    """The group field groups records into groups too short to pool."""
    SPLIT = "split"
    """A text was split into documents at its headings or rules (the message says where),
    or a split that was asked for found none to split at."""
    STAND_INS = "stand_ins"
    """A text with no headings or rules was cut into stand-in documents of consecutive
    windows, whose calibration may be optimistic (if topics run on between parts)."""

    CONTRAST_STAND_INS = "contrast_stand_ins"
    CALIBRATION_STAND_INS = "calibration_stand_ins"
    SKIPPED_CONTRAST = "skipped_contrast"
    SINGLE_CONTRAST = "single_contrast"
    NO_AUC_INTERVAL = "no_auc_interval"
    LENGTH_CONTRAST = "length_contrast"
    EMPTY_CHUNKS = "empty_chunks"
    BELOW_MIN_WORDS = "below_min_words"
    NOISY_CHUNKS = "noisy_chunks"
    NO_RELIABILITY = "no_reliability"
    SINGLE_CHUNK = "single_chunk"
    WINDOW_MISMATCH = "window_mismatch"
    REFERENCE_NO_SYNTAX = "reference_no_syntax"
    MISSING_LIKENESS_METRICS = "missing_likeness_metrics"
    SCORE_NO_SYNTAX = "score_no_syntax"
    SYNTAX_MODEL_MISMATCH = "syntax_model_mismatch"
    SKIPPED_EDITS = "skipped_edits"
    MISSING_EDITS = "missing_edits"
    SHORT_EDITS = "short_edits"
    POOLING_MISMATCH = "pooling_mismatch"
    LEFT_OUT = "left_out"
    PARAGRAPH_PROFILE = "paragraph_profile"
    WARNING_COUNT = "warning_count"

    @property
    def hint(self) -> str | None:
        """Optional frontend advice; render its setting names with frontend flags."""
        return _NOTE_HINTS.get(self)

    @property
    def forms(self) -> dict[str, NoteForm]:
        """Central short and long forms; variants share the same public note code."""
        return _NOTE_FORMS[self]

    def message(self, variant: str = "default", *values: str) -> str:
        """Render the long form saved in reports and exposed by the library."""
        return self.forms[variant].long.format(*values)

    def note(self, variant: str = "default", *values: str, setting: str | None = None) -> Note:
        """Make a run note without changing its long ``message`` contract."""
        return Note(self.message(variant, *values), self, setting)

    def text(
        self,
        message: str,
        *,
        verbose: bool = False,
        settings: Mapping[str, str] | None = None,
    ) -> str:
        """Choose human prose; verbose output keeps the original long message."""
        if verbose:
            return message
        if self == NoteCode.NO_SYNTAX:
            for name in ("missing_spacy", "missing_model"):
                form = self.forms[name]
                if _note_pattern(form.long).search(message):
                    return form.short
        for name, form in self.forms.items():
            if name == "fallback":
                continue
            short = form.shorten(message, settings=settings)
            if short is not None:
                return short
        # Some notes combine central fragments (splits and pooling). Their code still
        # gives one actionable short form; their complete explanation remains available.
        return _setting_text(self.forms["fallback"].short, settings)


@dataclass(frozen=True)
class Note:
    """Something a front end should tell the user about a run that is not an error: an input
    given twice, or syntax metrics left out because spaCy is missing.

    ``message`` never mentions flags; when it names a setting, ``setting`` says which, as
    for ``StyleProfileError``. Notes describe the run, not the text, so they are not saved
    in reports; warnings about the text are (``report["warnings"]``). Frontends use
    ``text()`` for the short form or ``text(verbose=True)`` for the original message.
    """

    message: str
    code: NoteCode
    setting: str | None = None

    def text(self, *, verbose: bool = False, settings: Mapping[str, str] | None = None) -> str:
        """The short human form, or the original message with ``verbose=True``."""
        return self.code.text(self.message, verbose=verbose, settings=settings)


class Phase(StrEnum):
    """The phases a run reports to its ``progress`` callback. Every run goes ``READ``,
    ``LOAD_PARSER`` (only when it uses spaCy), then its own work (``BUILD``, ``SCORE`` or
    ``EVALUATE``), then ``DONE``. Within its work it reports ``MEASURE`` for its chunks (the
    writer's, or the drafts'), then for a reference ``CALIBRATE`` and, with a contrast set,
    ``MEASURE_CONTRAST`` and ``CONTRAST``."""

    READ = "read"
    LOAD_PARSER = "load_parser"
    BUILD = "build"
    SCORE = "score"
    EVALUATE = "evaluate"
    MEASURE = "measure"
    """Measuring chunks, reported as each one is done (``done`` of ``total``)."""
    CALIBRATE = "calibrate"
    """The reference's held-out calibration, overall and for shorter texts."""
    MEASURE_CONTRAST = "measure_contrast"
    """Measuring the contrast set's chunks, reported as ``MEASURE`` is."""
    CONTRAST = "contrast"
    """Learning what separates the writer from the contrast set."""
    DONE = "done"


@dataclass(frozen=True)
class Progress:
    """Where a run is. Every phase is reported once at its start. ``MEASURE`` and
    ``MEASURE_CONTRAST`` are also reported after each chunk: ``done`` of ``total`` chunks, and
    ``words``, the prose words in the chunks done so far. ``parsing`` says whether those chunks
    go through the spaCy parser, which makes measuring about ten times slower. Other phases
    leave the counts None."""

    phase: Phase
    done: int | None = None
    total: int | None = None
    words: int | None = None
    parsing: bool = False


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


@dataclass(frozen=True)
class NoteForm:
    """Both forms live together. Slots contain already formatted context, never flags."""

    short: str
    long: str
    continuation: bool = False

    def shorten(self, message: str, *, settings: Mapping[str, str] | None = None) -> str | None:
        pattern = _note_pattern(self.long)
        found = pattern.match(message) if self.continuation else pattern.fullmatch(message)
        if found is None:
            return None
        values = found.groups()
        # Bound context such as an arbitrarily long filename, preserving the action.
        template = _setting_text(self.short, settings)
        slots = sum(field is not None for _, field, _, _ in Formatter().parse(template))
        literal = sum(len(text) for text, _, _, _ in Formatter().parse(template))
        budget = max(1, (100 - literal) // slots) if slots else 100
        values = tuple(" ".join(value.split()) for value in values)
        compact = tuple(
            value if len(value) <= budget else value[: budget - 1] + "…" for value in values
        )
        return template.format(*compact)


def _setting_text(template: str, settings: Mapping[str, str] | None) -> str:
    """Translate advice before inserting context, so filenames and labels stay intact."""
    for name, replacement in (settings or {}).items():
        template = re.sub(
            rf"\b{re.escape(name)}\b", lambda _, replacement=replacement: replacement, template
        )
    return template


@cache
def _note_pattern(template: str) -> re.Pattern[str]:
    parts = []
    for text, field, _, _ in Formatter().parse(template):
        parts.append(re.escape(text))
        if field is not None:
            parts.append("(.*?)")
    return re.compile("".join(parts), re.DOTALL)


def warning_text(
    message: str, *, verbose: bool = False, settings: Mapping[str, str] | None = None
) -> str:
    """Read saved warning strings without changing JSON, including older reports.

    Unrecognized warnings are kept intact, so a future or user-supplied warning cannot
    disappear merely because this version does not know its short form.
    """
    if verbose:
        return message
    for code in _WARNING_CODES:
        for name, form in code.forms.items():
            if name == "fallback":
                continue
            short = form.shorten(message, settings=settings)
            if short is not None:
                return short
    return message


_NOTE_FORMS: dict[NoteCode, dict[str, NoteForm]] = {
    NoteCode.NEWER_REPORT_VERSION: {
        "default": NoteForm(
            "Newer minor report version; some fields may be ignored. Upgrade styleprofile.",
            "This report uses a newer minor version; some fields may be ignored.",
        ),
        "fallback": NoteForm(
            "Newer minor report version; some fields may be ignored. Upgrade styleprofile.",
            "{0}",
        ),
    },
    NoteCode.WARNING_COUNT: {
        "default": NoteForm("{0}; run without -q to see them", "{0}; run without -q to see them"),
        "fallback": NoteForm("Warnings present; run without -q to see them.", "{0}"),
    },
    NoteCode.PARAGRAPH_PROFILE: {
        "fallback": NoteForm("Rebuild with --by-paragraph to enable paragraph checks.", "{0}"),
        "default": NoteForm(
            (
                "Paragraph checks need --by-paragraph at build time; rebuild with "
                "your original inputs."
            ),
            (
                "Paragraph checks need a profile built with --by-paragraph; use "
                "your original inputs in: {0}"
            ),
        ),
    },
    NoteCode.MISSING_GROUP: {
        "fallback": NoteForm(
            ("Records lack the group field; supply it to keep independent sources separate."), "{0}"
        ),
        "0": NoteForm(
            ("{1} of {2} lack {3}; read as one document. Supply the group field."),
            ("{0}: {1} of {2} have no {3}, so they are read as one document, {4}"),
        ),
        "1": NoteForm(
            ("No record has {1}; records read separately. Supply the group field."),
            ("{0}: no record has a {1}, so each record is read as a document of its own"),
        ),
    },
    NoteCode.WORKERS_LIMITED: {
        "fallback": NoteForm(
            ("Worker count reduced to fit memory; use fewer jobs to avoid the limit."), "{0}"
        ),
        "0": NoteForm(
            "jobs reduced from {0} to {1} to fit available memory",
            "jobs reduced from {0} to {1} to fit available memory",
        ),
    },
    NoteCode.CACHE_UNAVAILABLE: {
        "fallback": NoteForm(
            (
                "Measurement cache unavailable; run continued without it. Check "
                "cache permissions or disable cache."
            ),
            "{0}",
        ),
        "0": NoteForm(
            "Cache unavailable ({1}); run continued without it. Check storage or disable cache.",
            (
                "the measurement cache at {0} could not be used ({1}), so this "
                "run went on without it: what it measured from then on is not "
                "saved for the next"
            ),
        ),
    },
    NoteCode.GROUPING: {
        "fallback": NoteForm(
            (
                "Groups cannot pool reliably; choose a coarser group_field or "
                "give their common folder."
            ),
            "{0}",
        ),
        "0": NoteForm(
            ("Groups span inputs; give their common folder to keep each group together."),
            (
                "records with the same {0} (such as {1}) are in more than one "
                "input, and each input's records are separate documents; to keep "
                "a group together, give the folder that holds them"
            ),
        ),
        "1": NoteForm(
            (
                "{0} groups are short (median {1} words); choose a coarser field "
                "or omit group_field."
            ),
            (
                "most {0} groups are short (a median of {1} words), so pooling "
                "has little to join and their records are measured nearly alone; "
                "group by a coarser field, or leave out group_field"
            ),
        ),
    },
    NoteCode.SPLIT: {
        "fallback": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "{0}",
        ),
        "reason": NoteForm(
            "Few documents; give more independent texts for calibration.",
            ", since {0} are too few to calibrate well",
        ),
        "many": NoteForm(
            ("Texts split into {2} documents at {3}; use split_on to choose boundaries."),
            "split {0} {1}texts into {2} documents at their {3}",
        ),
        "0": NoteForm(
            "Text kept whole: no usable split; supply document boundaries or choose split_on.",
            "{0} into parts of at least half a window, so {1}",
        ),
        "1": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "split {0}{1} into {2} at {3}",
        ),
        "2": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "{0}{1} has no {2} that split it",
        ),
        "3": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "it is kept whole",
        ),
        "4": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "{0} of {1} have no {2} ",
        ),
        "5": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "that split them",
        ),
        "6": NoteForm(
            ("Text split into documents where possible; use split_on to choose headings or rules."),
            "they are kept whole",
        ),
    },
    NoteCode.STAND_INS: {
        "fallback": NoteForm(
            (
                "Consecutive parts stand in for documents; mark document "
                "boundaries for sounder calibration."
            ),
            "{0}",
        ),
        "groups": NoteForm(
            ("Window groups stand in for documents; mark real document boundaries."),
            ("its {0} windows are grouped into {1} stand-in documents of consecutive text"),
        ),
        "windows": NoteForm(
            "Windows stand in for documents; mark real document boundaries.",
            "each of its {0} windows stands in for a document",
        ),
        "0": NoteForm(
            (
                "Consecutive parts stand in for documents; mark document "
                "boundaries for sounder calibration."
            ),
            (
                "{0}{1} has no headings or rules that divide it into {2} or more "
                "parts of at least half a window, so {3}"
            ),
        ),
    },
    NoteCode.CONTRAST_STAND_INS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            (
                "Contrast uses {1} stand-ins from one file; give separate drafts "
                "or mark their boundaries."
            ),
            (
                "the {0}set's documents are {1} stand-ins, consecutive parts of "
                "one file ({2}), so its AUC interval and cross-validated weights "
                "treat neighbouring parts as separate drafts and are less certain "
                "than they look; give the drafts as separate files, or mark where "
                "each begins with a heading or rule"
            ),
        ),
    },
    NoteCode.CALIBRATION_STAND_INS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            (
                "Calibration uses {0} stand-ins; short drift is harder to catch. "
                "Mark document boundaries."
            ),
            (
                "held-out calibration comes from {0} stand-in documents, "
                "consecutive parts of one file ({1}): calibration from them is "
                "less sensitive, so short off-voice passages are caught less "
                "often; mark where pieces begin with headings or rules, or give "
                "them as separate files"
            ),
        ),
    },
    NoteCode.DUPLICATES: {
        "fallback": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."), "{0}"
        ),
        "0": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            "{0}: {1}",
        ),
        "1": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "the edit of {0} is paired with {1}: the original of {2} repeats "
                "{3} word for word, so only {4} was kept"
            ),
        ),
        "2": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "left out the edit of {0}: its original repeats {1} word for "
                "word, and another edit of that text is already paired with {2}"
            ),
        ),
        "3": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "left out the edit of {0}: its original repeats the writer's {1} "
                "word for word, so it was dropped from the drafts and has no "
                "draft to pair with"
            ),
        ),
        "4": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "{0} edits of drafts that repeat another word for word are paired "
                "with the copy kept (for example, {1} with {2})"
            ),
        ),
        "5": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "left out {0} edits of drafts that repeat another word for word, "
                "whose text another edit already pairs with (for example, {1}, a "
                "copy of {2})"
            ),
        ),
        "6": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "left out {0} edits of drafts dropped for repeating the writer's "
                "texts word for word, which leaves them no draft to pair with "
                "(for example, {1}, a copy of the writer's {2})"
            ),
        ),
        "7": NoteForm(
            ("Repeated text dropped or paired with its kept copy; give distinct documents."),
            (
                "dropped {0} another word for word, keeping the first copy (for "
                "example, {1} repeats {2})"
            ),
        ),
    },
    NoteCode.REPEATED_INPUT: {
        "fallback": NoteForm("Input was already given; using it once.", "{0}"),
        "0": NoteForm(
            "{0} was already given; using it once", "{0} was already given; using it once"
        ),
        "1": NoteForm("skipping {0} in {1} already given", "skipping {0} in {1} already given"),
    },
    NoteCode.NON_ENGLISH: {
        "fallback": NoteForm(
            ("Text may not be English; use English texts for reliable measurements."), "{0}"
        ),
        "0": NoteForm(
            ("{0} may not be English; use English texts for dependable measurements."),
            (
                "{0} may not be English; English-based measurements may be "
                "unreliable. Use English texts for a dependable comparison."
            ),
        ),
    },
    NoteCode.NO_PARAGRAPH_BREAKS: {
        "fallback": NoteForm(
            "Paragraph metrics omitted; keep the original paragraph breaks.", "{0}"
        ),
        "0": NoteForm(
            ("{0}: paragraph metrics omitted; keep the original paragraph breaks."),
            (
                "{0}: paragraph metrics were left out because the text has no "
                "paragraph breaks. Keep the original paragraph breaks when "
                "available."
            ),
        ),
    },
    NoteCode.OVERSIZE_CHUNK: {
        "fallback": NoteForm(
            ("Oversize chunk could not split safely; add sentence breaks or raise window_words."),
            "{0}",
        ),
        "0": NoteForm(
            ("{0}: {1} words exceed twice window_words ({2}); add sentence breaks."),
            (
                "{0}: {1} prose words exceed twice window_words ({2}); no safe "
                "sentence boundary could split this block further. Add sentence "
                "or paragraph breaks, or use a larger window_words."
            ),
        ),
    },
    NoteCode.POOLED: {
        "fallback": NoteForm(
            (
                "Short texts pooled into windows; use group_field to keep "
                "independent sources separate."
            ),
            "{0}",
        ),
        "0": NoteForm(
            (
                "Pooling makes only {1}, too few to calibrate; texts kept "
                "separate. Use pool to join anyway."
            ),
            (
                "the median {0} is under a quarter of a window, but joining them "
                "would make only {1}, too few to calibrate, so each is kept on "
                "its own; use pool to join them anyway"
            ),
        ),
        "1": NoteForm(
            "joined {0} into {1}; use group_field to keep sources separate.",
            "joined {0} into {1} of about {2} words",
            continuation=True,
        ),
        "2": NoteForm(
            (
                "Short texts pooled into windows; use group_field to keep "
                "independent sources separate."
            ),
            ", since the median {0} is under a quarter of a window",
        ),
        "3": NoteForm(
            (
                "Short texts pooled into windows; use group_field to keep "
                "independent sources separate."
            ),
            ", never across {0} groups",
        ),
        "4": NoteForm(
            (
                "Short texts pooled into windows; use group_field to keep "
                "independent sources separate."
            ),
            (
                ". With no group_field, each window counts as one document for "
                "held-out calibration: if the {0}s come from different threads or "
                "authors, and especially if those are interleaved, its ranges are "
                "too narrow and verdicts on new text can be far too harsh"
            ),
        ),
        "5": NoteForm(
            (
                "Short texts pooled into windows; use group_field to keep "
                "independent sources separate."
            ),
            (". These records have {0}: pass group_field {1} if each value is a separate source"),
        ),
    },
    NoteCode.SETTING_OVERRIDDEN: {
        "fallback": NoteForm(
            ("min_words overrides the reference; omit the override to inherit it."), "{0}"
        ),
        "0": NoteForm(
            "min_words {0} overrides the reference's {1}",
            "min_words {0} overrides the reference's {1}",
        ),
    },
    NoteCode.SHORT_TEXTS: {
        "minimum": NoteForm(
            "styleprofile judges {0} words or more; score several short texts together with pool",
            "styleprofile judges {0} words or more; score several short texts together with pool",
        ),
        "fallback": NoteForm(
            "Short texts judged separately; use pool to judge them as a batch.", "{0}"
        ),
        "0": NoteForm(
            (
                "{0} short texts judged separately; under {1} words gets no "
                "verdict. Use pool to judge a batch."
            ),
            (
                "these {0} texts are short (the median is under a quarter of a "
                "window), so each is judged at its own length, and those under "
                "{1} words get no verdict; use pool to judge them as one batch"
            ),
        ),
    },
    NoteCode.EDITED_OVERLAP: {
        "fallback": NoteForm(
            ("Edited inputs overlap writer or draft files; give a separate edited set."), "{0}"
        ),
        "0": NoteForm(
            "{0}: {1} files overlap writer/drafts; give a separate edited set.",
            (
                "{0}: {1} file{2} in {3} are also given as the writer's texts or "
                "the original drafts, so that set is not an edit of them"
            ),
        ),
    },
    NoteCode.REPEATED_ID: {
        "fallback": NoteForm(
            ("Records share ids; each remains a separate document, named id@LINE."), "{0}"
        ),
        "0": NoteForm(
            ("Records share ids; each remains a separate document, named id@LINE."),
            (
                "records in {0} share ids ({1}{2}); each record is still its own "
                "document, named by id and line (id@LINE)"
            ),
        ),
    },
    NoteCode.SKIPPED_CONTRAST: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("Skipped {0} contrast chunks below {1} prose words or empty; give longer prose."),
            ("skipped {0} contrast chunk(s) with no prose or fewer than {1} prose words"),
        ),
    },
    NoteCode.SINGLE_CONTRAST: {
        "fallback": NoteForm(
            (
                "One contrast document gives an optimistic likeness range; add "
                "more contrast documents."
            ),
            "{0}",
        ),
        "0": NoteForm(
            (
                "One contrast document gives an optimistic likeness range; add "
                "more contrast documents."
            ),
            (
                "the contrast set is one document, so its likeness range is "
                "measured in-sample and is optimistic; add more contrast documents"
            ),
        ),
    },
    NoteCode.NO_AUC_INTERVAL: {
        "fallback": NoteForm(
            ("Fewer than 2 documents in a set; AUC has no confidence interval. Add documents."),
            "{0}",
        ),
        "0": NoteForm(
            ("Fewer than 2 documents in a set; AUC has no confidence interval. Add documents."),
            (
                "the reference or contrast set has fewer than 2 documents, so the "
                "contrast AUC has no confidence interval"
            ),
        ),
    },
    NoteCode.LENGTH_CONTRAST: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("Length alone separates drafts (AUC {0}); match lengths or use equal-sized windows."),
            (
                "the contrast set differs strongly in length (length alone "
                "separates it with AUC {0}), so {1}-likeness may partly reflect "
                "length; match lengths or split both sets into windows of the "
                "same size"
            ),
        ),
    },
    NoteCode.EMPTY_CHUNKS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("Skipped {0} chunks with no prose; give prose, rather than code, tables or markup."),
            "skipped {0} chunk(s) with no prose (only code, tables or markup)",
        ),
    },
    NoteCode.BELOW_MIN_WORDS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("Skipped {0} chunks below {1} prose words; add prose or lower min_words."),
            "skipped {0} chunk(s) with fewer than {1} prose words",
        ),
    },
    NoteCode.NOISY_CHUNKS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("{0} chunks have fewer than {1} words; add prose for steadier rates."),
            "{0} chunk(s) have fewer than {1} words; their rates are noisy",
        ),
    },
    NoteCode.NO_RELIABILITY: {
        "fallback": NoteForm(
            (
                "No held-out reliability; Delta caps metrics at 3. Rebuild from "
                "at least two documents."
            ),
            "{0}",
        ),
        "0": NoteForm(
            (
                "No held-out reliability; Delta caps metrics at 3. Rebuild from "
                "at least two documents."
            ),
            (
                "the reference has no held-out reliability (it needs chunks from "
                "at least two documents and a current report version), so Delta "
                "caps each metric at 3 instead of weighting it by reliability"
            ),
        ),
    },
    NoteCode.SINGLE_CHUNK: {
        "fallback": NoteForm(
            "The reference has one chunk and no spread; split it into windows.", "{0}"
        ),
        "0": NoteForm(
            "The reference has one chunk and no spread; split it into windows.",
            ("the reference has one chunk, so it has no spread; split it into windows"),
        ),
    },
    NoteCode.WINDOW_MISMATCH: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("window sizes differ from the reference ({0} vs {1}); use equal-sized chunks."),
            (
                "window sizes differ from the reference ({0} vs {1}); z-scores "
                "assume equal-sized chunks"
            ),
        ),
    },
    NoteCode.REFERENCE_NO_SYNTAX: {
        "fallback": NoteForm(
            ("The reference has no syntax metrics; rebuild with syntax to score them."), "{0}"
        ),
        "0": NoteForm(
            ("The reference has no syntax metrics; rebuild with syntax to score them."),
            "the reference has no syntax metrics, so syntax is not scored",
        ),
    },
    NoteCode.READ_AS_JSONL: {
        "fallback": NoteForm(
            (
                "stdin is JSON objects, one per line; read as JSONL. Use "
                "input_format markdown for prose."
            ),
            "{0}",
        ),
        "0": NoteForm(
            (
                "stdin is JSON objects, one per line; read as JSONL. Use "
                "input_format markdown for prose."
            ),
            "stdin is JSON objects, one per line, so it is read as JSONL",
        ),
    },
    NoteCode.READ_AS_HTML: {
        "fallback": NoteForm(
            (
                "Input looks like HTML; read as HTML. Use input_format markdown "
                "to read it as written."
            ),
            "{0}",
        ),
        "files": NoteForm(
            (
                "Input looks like HTML; read as HTML. Use input_format markdown "
                "to read it as written."
            ),
            "{0} {1} like HTML, so {2} read as HTML",
        ),
        "0": NoteForm(
            (
                "Input looks like HTML; read as HTML. Use input_format markdown "
                "to read it as written."
            ),
            "stdin looks like HTML, so it is read as HTML",
        ),
    },
    NoteCode.MISSING_LIKENESS_METRICS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            (
                "Missing metrics carry {0}% of likeness weight; likeness uses the "
                "rest. Include syntax."
            ),
            (
                "this run lacks metrics that carry {0}% of the likeness weight "
                "(for example syntax); likeness uses the rest"
            ),
        ),
    },
    NoteCode.SCORE_NO_SYNTAX: {
        "fallback": NoteForm(
            (
                "Syntax omitted; Delta differs from syntax runs. Install syntax "
                "and run `styleprofile setup`."
            ),
            "{0}",
        ),
        "0": NoteForm(
            (
                "Syntax omitted; Delta differs from syntax runs. Install syntax "
                "and run `styleprofile setup`."
            ),
            (
                "the reference has syntax metrics but this run does not, so Delta "
                "leaves out syntax and sentence openers; its value is not "
                "comparable with syntax runs"
            ),
        ),
    },
    NoteCode.EMPTY_HTML: {
        "fallback": NoteForm(
            ("HTML has no readable text after conversion; give a page with prose."), "{0}"
        ),
        "0": NoteForm(
            ("HTML has no readable text after conversion; give a page with prose."),
            "stdin has no readable text after conversion",
        ),
        "1": NoteForm(
            ("HTML has no readable text after conversion; give a page with prose."),
            "{0} {1} no readable text after conversion from HTML",
        ),
    },
    NoteCode.SKIPPED_DIRS: {
        "fallback": NoteForm(
            ("Static-site output and templates skipped; name a folder directly to read it."), "{0}"
        ),
        "0": NoteForm(
            ("Static-site output and templates skipped; name a folder directly to read it."),
            (
                "left out static-site output and template folders in {0}: {1}; "
                "name one directly to read it"
            ),
        ),
    },
    NoteCode.SKIPPED_FILES: {
        "fallback": NoteForm(
            ("Unsupported files skipped; convert Word/PDF to text or Markdown, then retry."), "{0}"
        ),
        "0": NoteForm(
            ("Unsupported files skipped; convert Word/PDF to text or Markdown, then retry."),
            "skipped {0} in {1}; {2}",
        ),
    },
    NoteCode.SYNTAX_MODEL_MISMATCH: {
        "fallback": NoteForm(
            ("spaCy models differ; rebuild and score with the same model for comparable syntax."),
            "{0}",
        ),
        "0": NoteForm(
            ("spaCy models differ; rebuild and score with the same model for comparable syntax."),
            (
                "the reference was parsed with a different spaCy model ({0} {1}); "
                "syntax metrics may not be comparable"
            ),
        ),
    },
    NoteCode.SKIPPED_EDITS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            (
                "{0}: skipped {1} edits whose originals have too little prose; "
                "supply longer originals."
            ),
            (
                "{0}: skipped {1} edited draft(s) whose original has too little "
                "prose to score (e.g. {2})"
            ),
        ),
    },
    NoteCode.MISSING_EDITS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            (
                "{0}: {1} of {2} drafts lack edits; AUC covers fewer drafts. "
                "Supply the missing edits."
            ),
            (
                "{0}: {1} of {2} drafts have no edited copy (e.g. {3}); its AUC "
                "is over fewer drafts than the original's, and signal survival "
                "compares it with the drafts it covers"
            ),
        ),
    },
    NoteCode.SHORT_EDITS: {
        "fallback": NoteForm(
            "See the verbose note for context; supply more comparable prose.", "{0}"
        ),
        "0": NoteForm(
            ("{0}: {1} edited drafts lack {2} prose words; add prose to score them."),
            (
                "{0}: {1} edited draft(s) have no chunk with at least {2} prose "
                "words (e.g. {3}), so they are not scored"
            ),
        ),
    },
    NoteCode.THIN_REFERENCE: {
        "fallback": NoteForm("Thin reference; add more of the writer's documents.", "{0}"),
        "split_suffix": NoteForm(
            "Mark boundaries or add separate documents.", ", or split that one{0}"
        ),
        "split_advice": NoteForm(
            "Mark boundaries or add separate documents.",
            (
                " (split_on heading or split_on rule splits each text at its "
                "headings or rules, where it has them)"
            ),
        ),
        "dominant": NoteForm(
            ("One document dominates held-out calibration; add documents of a similar size."),
            (
                "one document holds {0} of its {1}, so its held-out range rests "
                "on the other {2}, where a range needs {3} from {4} or more "
                "documents; add documents of a similar size{5}"
            ),
        ),
        "summary": NoteForm(
            "Thin reference; add more documents.",
            (
                "Thin reference: {0} chunks, {1} words (aim for 15+ chunks and "
                "20,000+ words); add more of the writer's documents"
            ),
        ),
        "summary_group": NoteForm(
            "Thin reference: {0} chunks, {1} words; no calibration or contrast. "
            "Use finer group_field.",
            "Thin reference: {0} chunks, {1} words; no calibration or contrast. "
            "Use finer group_field.",
        ),
        "summary_document": NoteForm(
            "Thin reference: {0} chunks, {1} words; no held-out calibration or contrast. "
            "Add documents.",
            "Thin reference: {0} chunks, {1} words; no held-out calibration or contrast. "
            "Add documents.",
        ),
        "summary_dominant": NoteForm(
            "Thin reference: {0} chunks, {1} words; one document dominates. "
            "Add similar-sized documents.",
            "Thin reference: {0} chunks, {1} words; one document dominates. "
            "Add similar-sized documents.",
        ),
        "0": NoteForm(
            "One group prevents calibration and contrast; omit group_field or use finer groups.",
            (
                "it comes from {0}, so it has no held-out calibration and cannot "
                "learn a contrast: every record has the same {1}; leave out "
                "group_field, or group by a finer field"
            ),
        ),
        "1": NoteForm(
            "Thin reference; add more of the writer's documents.",
            "it has {0}; aim for {1} or more by adding documents or using {2}",
        ),
        "2": NoteForm(
            "Thin reference; add more of the writer's documents.",
            ("it has {0}; aim for {1} or more of the writer's text, in one genre"),
        ),
        "3": NoteForm(
            "One document cannot calibrate held-out ranges or contrast; add independent documents.",
            (
                "it comes from {0}, so it has no held-out calibration and cannot "
                "learn a contrast; add more of the writer's documents"
            ),
        ),
    },
    NoteCode.POOLING_MISMATCH: {
        "fallback": NoteForm(
            ("Pooling makes these texts read closer than they are; score without pooling."), "{0}"
        ),
        "0": NoteForm(
            ("Pooling makes these texts read closer than they are; score without pooling."),
            (
                "these texts were joined into windows, but the reference's were "
                "not, so they read closer to it than they are; score them without "
                "pooling"
            ),
        ),
    },
    NoteCode.NO_SYNTAX: {
        "missing_spacy": NoteForm(
            "spaCy is not installed; surface metrics only. Install the syntax extra.",
            "spaCy is not installed",
        ),
        "missing_model": NoteForm(
            "English model missing; surface metrics only; run `styleprofile setup`.",
            "spaCy's English model ({0}) is not installed",
        ),
        "fallback": NoteForm(
            (
                "syntax is left out; surface metrics only. Install syntax and run "
                "`styleprofile setup`."
            ),
            "{0}",
        ),
        "evaluate_surface": NoteForm(
            (
                "syntax is left out; surface metrics only. Install syntax and run "
                "`styleprofile setup`."
            ),
            (
                "{missing}, so this uses surface metrics only; for syntax "
                "metrics, {fix} and run again"
            ),
        ),
        "build_surface": NoteForm(
            (
                "syntax is left out; surface metrics only. Install syntax and run "
                "`styleprofile setup`."
            ),
            (
                "{missing}, so this profile has surface metrics only; for syntax "
                "metrics, {fix} and build again"
            ),
        ),
        "score_surface": NoteForm(
            (
                "syntax is left out; surface metrics only. Install syntax and run "
                "`styleprofile setup`."
            ),
            ("{missing}, so this score has surface metrics only; {fix} to include syntax"),
        ),
        "score_reference": NoteForm(
            (
                "syntax is left out; surface metrics only. Install syntax and run "
                "`styleprofile setup`."
            ),
            (
                "the reference has syntax metrics but {missing}, so syntax is "
                "left out of this score; {fix} to include it"
            ),
        ),
        "default": NoteForm(
            (
                "syntax is left out; surface metrics only. Install syntax and run "
                "`styleprofile setup`."
            ),
            "{0}",
        ),
    },
    NoteCode.READ_AS_HTML_IN_FOLDER: {
        "fallback": NoteForm(
            (
                "Files look like HTML; read as HTML. Give Markdown files "
                "separately with input_format markdown."
            ),
            "{0}",
        ),
        "files": NoteForm(
            ("Files read as HTML; give Markdown files separately with input_format markdown."),
            "{0} {1} like HTML, so {2} read as HTML",
        ),
        "default": NoteForm(
            (
                "Files look like HTML; read as HTML. Give Markdown files "
                "separately with input_format markdown."
            ),
            "{0}",
        ),
    },
    NoteCode.LEFT_OUT: {
        "fallback": NoteForm(
            (
                "Short chunks omitted from verdict and means; add prose or use "
                "pool to judge a batch."
            ),
            "{0}",
        ),
        "details": NoteForm(
            (
                "Short chunks omitted from verdict and means; add prose or use "
                "pool to judge a batch."
            ),
            ("left out of the verdict and the means as too short to judge: {0}{1}"),
        ),
    },
}


_WARNING_CODES = (
    NoteCode.CONTRAST_STAND_INS,
    NoteCode.CALIBRATION_STAND_INS,
    NoteCode.SKIPPED_CONTRAST,
    NoteCode.SINGLE_CONTRAST,
    NoteCode.NO_AUC_INTERVAL,
    NoteCode.LENGTH_CONTRAST,
    NoteCode.EMPTY_CHUNKS,
    NoteCode.BELOW_MIN_WORDS,
    NoteCode.NOISY_CHUNKS,
    NoteCode.NO_RELIABILITY,
    NoteCode.SINGLE_CHUNK,
    NoteCode.WINDOW_MISMATCH,
    NoteCode.REFERENCE_NO_SYNTAX,
    NoteCode.MISSING_LIKENESS_METRICS,
    NoteCode.SCORE_NO_SYNTAX,
    NoteCode.SYNTAX_MODEL_MISMATCH,
    NoteCode.SKIPPED_EDITS,
    NoteCode.MISSING_EDITS,
    NoteCode.SHORT_EDITS,
    NoteCode.POOLING_MISMATCH,
    NoteCode.LEFT_OUT,
)


_NOTE_HINTS = {
    NoteCode.READ_AS_HTML: "pass input_format markdown to read it as written",
    NoteCode.READ_AS_HTML_IN_FOLDER: (
        "if any are really Markdown, give them separately with "
        "input_format markdown, which would also apply to .html files in "
        "the folder"
    ),
    NoteCode.READ_AS_JSONL: "pass input_format markdown to read it as prose",
}
