"""Library input types, validated settings and defaults."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from styleprofile.core import Progress, StyleProfileError
from styleprofile.corpus.types import Chunk, Text
from styleprofile.formats import INPUT_FORMATS
from styleprofile.schema import InputSettings
from styleprofile.split import SPLIT_ON, SplitOn

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


def _strings(texts: str | Iterable[str]) -> list[Text]:
    values = [texts] if isinstance(texts, str) else list(texts)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("expected raw text strings; use build or score for paths, Text or Chunk")
    return [Text(value) for value in values]


def _overridden(settings: Settings, overrides: SettingsOverrides) -> Settings:
    unknown = sorted(set(overrides) - {f.name for f in dataclasses.fields(Settings)})
    if unknown:
        raise TypeError(f"unknown setting(s): {', '.join(unknown)}")
    return dataclasses.replace(settings, **overrides)
