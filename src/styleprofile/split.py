"""Splitting one long text into documents (see "What a document is" in ``profile``).

Held-out calibration leaves out one document at a time, so a writer's texts given as one
file (a manuscript, or a newsletter archive in one Markdown file) have nothing to hold out.
This module finds where such a text divides:

- on structure: its headings (``# Title``, or ``Title`` underlined with ``===`` or ``---``)
  at one level, or its rules (``---``, ``***`` or ``___`` alone on a line, spaced or not),
  never inside fenced code or front matter. HTML is read as the Markdown ``formats`` makes of
  it, so its ``<h1>`` to ``<h6>`` are headings, and separate ``<article>``s are rules;
- failing that, by length: consecutive windows grouped into ``STAND_INS`` stand-in
  documents (``stand_ins``), for calibration only.

``plan_split`` picks the split: among heading levels and rules, each giving parts of at
least half a window (smaller ones join the next), the one with the fewest parts that still
gives ``min_parts``. The fewest parts, because parts of one text share their author, time
and often their topics, and the coarser a split, the less a held-out part has in common
with the parts left in; a chapter's sections held out against each other would make the
writer's range look tighter than it is.
"""

from __future__ import annotations

import itertools
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from styleprofile.surface import block_word_count, classify, strip_front_matter

SplitOn = Literal["auto", "heading", "rule", "none"]
AUTO: Final = "auto"
HEADING: Final = "heading"
RULE: Final = "rule"
NONE: Final = "none"
SPLIT_ON: tuple[SplitOn, ...] = (AUTO, HEADING, RULE, NONE)
# What ``Settings.split_used`` records for windows grouped into stand-in documents.
STAND_IN = "stand-in"
# How many stand-in documents a text without structure is cut into (fewer when it has fewer
# windows). Stand-ins of one text are consecutive and share topics, which the within-document
# similarity of calibration pieces cannot see; below ``weighting.FLOOR_DOCUMENTS`` (10) that
# similarity is floored, which keeps their ranges from being read as more independent than
# they are. Eight is enough documents for held-out calibration, under that floor.
STAND_INS = 8
# The fewest parts a split of one text must give to be used when splitting is automatic:
# ``calibration.MIN_CALIBRATION_DOCUMENTS``, the documents a calibrated length needs. A
# split that is asked for (``heading`` or ``rule``) needs two.
MIN_PARTS = 3
ASKED_MIN_PARTS = 2
# Heading titles make readable ids (``book.md#3-mud-season``) up to this many characters.
_SLUG_CHARS = 40

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+$")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
# Lines a setext underline cannot turn into a heading: list items, quotes, fences, tables.
_NOT_PARAGRAPH = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|>|\||```|~~~)")
_FENCE_OPEN = re.compile(r"^\s*(?:(`{3,})[^`]*|(~{3,}).*)$")
_FENCE_CLOSE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")
_SLUG = re.compile(r"[^\w]+")


@dataclass(frozen=True)
class Marker:
    """Where a text can be split: a heading (``level`` 1 to 6, with its title) or a rule
    (``level`` 0), starting at ``line`` (0-based, in the text with CRLF made LF)."""

    line: int
    kind: str
    level: int = 0
    title: str = ""


@dataclass(frozen=True)
class Part:
    """One part of a split text: its lines ``[start, end)``, the title of the heading it
    starts at (empty for a rule or the text's start), and its prose words."""

    start: int
    end: int
    title: str
    words: int


@dataclass(frozen=True)
class Plan:
    """How ``plan_split`` splits a text: at ``kind`` markers (``level`` for headings) into
    ``parts``, whose texts are ``texts``."""

    kind: str
    level: int
    parts: tuple[Part, ...]
    texts: tuple[str, ...]

    def describe(self) -> str:
        """Where it splits, for notes: "its level-1 headings", "its rules"."""
        return f"its level-{self.level} headings" if self.kind == HEADING else "its rules"


def _lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _front_matter_lines(lines: Sequence[str]) -> int:
    """How many lines at the start are front matter (``surface.strip_front_matter``)."""
    text = "\n".join(lines)
    stripped = strip_front_matter(text)
    if stripped == text:
        return 0
    return len(lines) - len(stripped.split("\n"))


def markers(text: str) -> list[Marker]:
    """Every heading and rule in ``text`` that could start a part, in order.

    Front matter and fenced code are skipped, as ``surface.markdown_blocks`` skips them, and
    so is indented code (a marker may be indented by at most three spaces). A line of ``-``
    under a paragraph line underlines it as a level-2 heading (``=`` as level 1), as in
    Markdown, rather than being a rule; the heading starts where the paragraph does.
    """
    lines = _lines(text)
    found: list[Marker] = []
    index = _front_matter_lines(lines)
    paragraph: int | None = None  # where the current paragraph began, if in one
    fence: str | None = None
    fence_line = 0
    while index < len(lines):
        line = lines[index]
        if fence is not None:
            closing = _FENCE_CLOSE.match(line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence):
                fence = None
            index += 1
            if fence is not None and index == len(lines):
                opener = lines[fence_line].strip().lstrip("`~").strip()
                if not opener:
                    # A bare opener that never closes is dropped, and what follows is read
                    # as usual, as ``markdown_blocks`` reads it.
                    fence, index, paragraph = None, fence_line + 1, None
            continue
        opening = _FENCE_OPEN.match(line)
        if opening:
            fence = opening.group(1) or opening.group(2)
            fence_line, paragraph = index, None
            index += 1
            continue
        if not line.strip():
            paragraph = None
        elif atx := _ATX.match(line):
            title = _CLOSING_HASHES.sub("", atx.group(2) or "").strip()
            found.append(Marker(index, HEADING, len(atx.group(1)), title))
            paragraph = None
        elif paragraph is not None and (setext := _SETEXT.match(line)):
            head = lines[paragraph:index]
            if _NOT_PARAGRAPH.match(head[0]):
                if _RULE.match(line):
                    found.append(Marker(index, RULE))
                paragraph = None
            else:
                level = 1 if setext.group(1)[0] == "=" else 2
                title = " ".join(part.strip() for part in head)
                found.append(Marker(paragraph, HEADING, level, title))
                paragraph = None
        elif _RULE.match(line):
            found.append(Marker(index, RULE))
            paragraph = None
        elif paragraph is None:
            paragraph = index
        index += 1
    return found


def _words(lines: Sequence[str]) -> int:
    return sum(block_word_count(block) for block in classify("\n".join(lines)))


def _merge(parts: Sequence[Part], smallest: float) -> list[Part]:
    """Parts under ``smallest`` words joined to the next part (the last to the one before),
    so no part is tiny: a preamble or a short interlude joins its neighbour."""
    merged: list[Part] = []
    pending: Part | None = None
    for part in parts:
        if pending is not None:
            part = Part(
                pending.start, part.end, pending.title or part.title, pending.words + part.words
            )
            pending = None
        if part.words < smallest:
            pending = part
            continue
        merged.append(part)
    if pending is not None:
        if merged:
            last = merged[-1]
            merged[-1] = Part(last.start, pending.end, last.title, last.words + pending.words)
        else:
            merged.append(pending)
    return merged


def plan_split(
    text: str, window_words: int, split_on: str = AUTO, min_parts: int = MIN_PARTS
) -> Plan | None:
    """The split of ``text`` into parts of about half a window of ``window_words`` or more,
    at the markers ``split_on`` allows (``auto``: headings or rules), or None when no
    level gives ``min_parts`` parts.

    A level is usable when at least half of the parts its markers start hold half a window
    of prose words (so its markers divide pieces of writing rather than paragraphs), and at
    least ``min_parts`` remain once parts under half a window join their neighbours. Of the
    usable levels, the one with the fewest parts is chosen (see the module docstring); on a
    tie, headings over rules, and higher headings over lower ones.
    """
    if split_on == NONE:
        return None
    lines = _lines(text)
    found = markers(text)
    if split_on == HEADING:
        found = [marker for marker in found if marker.kind == HEADING]
    elif split_on == RULE:
        found = [marker for marker in found if marker.kind == RULE]
    if not found:
        return None
    # Prose words between consecutive markers of any kind, counted once; each level's parts
    # are runs of these segments.
    starts = sorted({marker.line for marker in found})
    bounds = [0, *starts, len(lines)] if starts[0] else [*starts, len(lines)]
    # Words before each bound, so a run of segments is a difference.
    before = {bounds[0]: 0}
    for start, end in itertools.pairwise(bounds):
        before[end] = before[start] + _words(lines[start:end])
    candidates: list[tuple[str, int]] = [
        (HEADING, level) for level in sorted({m.level for m in found if m.kind == HEADING})
    ]
    candidates += [(RULE, 0)] if any(marker.kind == RULE for marker in found) else []
    half = window_words / 2
    best: Plan | None = None
    for kind, level in candidates:
        cuts = [
            marker
            for marker in found
            if marker.kind == kind and (kind == RULE or marker.level <= level)
        ]
        titles = {marker.line: marker.title for marker in cuts}
        edges = sorted(titles)
        spans = ([(0, edges[0])] if edges[0] else []) + list(
            zip(edges, [*edges[1:], len(lines)], strict=True)
        )
        parts = [
            Part(
                start,
                end,
                titles.get(start, ""),
                before[end] - before[start],
            )
            for start, end in spans
        ]
        marked = [part for part in parts if part.start in titles]
        if len(marked) < min_parts or statistics.median(p.words for p in marked) < half:
            continue
        merged = _merge(parts, half)
        if len(merged) < min_parts:
            continue
        if best is None or len(merged) < len(best.parts):
            texts = tuple("\n".join(lines[part.start : part.end]) for part in merged)
            best = Plan(kind, level, tuple(merged), texts)
    return best


def part_id(chunk_id: str, number: int, title: str = "") -> str:
    """The id of part ``number`` (from 1) of the text ``chunk_id``: ``book.md#3``, or with a
    heading's title, ``book.md#3-mud-season``. It starts with a digit, so it is never read as
    a window or record suffix (``profile.base_id``)."""
    slug = _SLUG.sub("-", title.lower()).strip("-")[:_SLUG_CHARS].strip("-")
    return f"{chunk_id}#{number}" + (f"-{slug}" if slug else "")


def stand_in_groups(count: int, groups: int = STAND_INS) -> list[range]:
    """``count`` consecutive windows as ``groups`` runs of nearly equal size (as many as
    there are windows, when fewer)."""
    groups = min(count, groups)
    edges = [round(index * count / groups) for index in range(groups + 1)]
    return [range(start, end) for start, end in itertools.pairwise(edges)]
