"""Parser-free stylometric metrics: sentence shape, rhythm, vocabulary, punctuation, voice.

Every rate is per 1,000 words and every share is a percentage, so chunks of different
lengths are comparable. Metrics that are undefined for a chunk (too short, no apostrophes)
are ``None`` rather than a misleading zero.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from styleprofile.metrics import (
    AND_BUT_SO_OPENER,
    BOOSTER,
    CONTRACTION_SUFFIX,
    FIRST_PLURAL,
    FIRST_SINGULAR,
    FUNCTION_WORDS,
    HEDGE,
    LLM_MARKER,
    LONG_SENTENCE_WORDS,
    LONG_WORD_CHARS,
    NEGATIONS,
    NOT_JUST_BUT,
    S_CONTRACTIONS,
    SECOND,
    SHORT_SENTENCE_WORDS,
    TRANSITION_OPENER,
    grouped,
)

Metrics = dict[str, dict[str, float | None]]

WORD = re.compile(r"[\w]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][\w]+)*", re.UNICODE)

# Front matter: lines that look like YAML or TOML (keys, list items, comments, tables).
_FRONT_MATTER_LINE = re.compile(r"^(?:[\w-]+[ \t]*[:=]|\s|-\s|#|\[)")
_FRONT_MATTER_LINES = 100
# Indented blocks count as code only with code-like signals and when they do not read like
# sentences, so plain text that indents its paragraphs with a tab is still read as prose.
_CODE_SIGNAL = re.compile(
    r"[{};=<>]|\(\)|^\s*(?:def|class|import|from|return|if|for|while|const|let|var|"
    r"function|fn|pub|func|package|#include|\$)\b"
)
# A backtick fence's info string cannot contain backticks (CommonMark), so a line such as
# "```npm i``` is all you need" is inline code, not a fence.
_FENCE_OPEN = re.compile(r"^\s*(?:(`{3,})[^`]*|(~{3,}).*)$")
_FENCE_CLOSE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")
_INDENTED_CODE = re.compile(r"^(?: {4}|\t)")
# A prose token is a word with optional punctuation around it ("late;", "(usually"), not an
# operator or a call such as "compute(a,".
_PROSE_TOKEN = re.compile(
    r"^[^\w\s]*[^\W\d_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W\d_]+)*[^\w\s]*$"
)
_PROSE_TOKEN_SHARE = 0.8
# A line comment ("# note", "// note"), which reads like prose inside code.
_LINE_COMMENT = re.compile(r"(?:^|\s)(?:#|//)(?:\s.*)?$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<!\w)([*_])(?=\S)(.+?)(?<=\S)\1(?!\w)")
_INLINE_CODE = re.compile(r"(`+)(?!`)[^\n]+?(?<!`)\1(?!`)")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"(?<![\w*])[*_]{1,3}(?=\S)|(?<=\S)[*_]{1,3}(?![\w*])")
_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_CLOSERS = "\"'\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK})\\]"
_SENTENCE_BREAK = re.compile(
    rf"(?:(?<=[.!?])|(?<=[.!?][{_CLOSERS}]))"
    r"\s+(?=[\"\N{LEFT DOUBLE QUOTATION MARK}'\N{LEFT SINGLE QUOTATION MARK}(\[]?[A-Z0-9])"
)
_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|vs|mr|mrs|ms|dr|st|jr|sr|fig|approx)\.$", re.I)
# Abbreviations that are also ordinary words, only before a number ("No. 5", not "said no.").
_NUMBER_ABBREVIATION = re.compile(r"\b(?:No|Nos|[Vv]ol|pp|[Cc]h)\.$")
_EM_DASH = re.compile(r"\N{EM DASH}|(?<=\w)--(?=\w)|(?<=\w) -{1,2} (?=\w)")
_ELLIPSIS = re.compile(r"\.\.\.|\N{HORIZONTAL ELLIPSIS}")
_FUNCTION_SET = frozenset(FUNCTION_WORDS)
PARAGRAPH_METRICS = (
    "paragraph_sentences_mean",
    "paragraph_words_mean",
    "one_sentence_paragraphs_pct",
)
# Fifty common English function words; a low share is only a warning, not a language label.
_ENGLISH_WORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "a",
        "in",
        "is",
        "it",
        "you",
        "that",
        "he",
        "was",
        "for",
        "on",
        "are",
        "with",
        "as",
        "i",
        "his",
        "they",
        "be",
        "at",
        "been",
        "have",
        "this",
        "from",
        "or",
        "had",
        "by",
        "not",
        "but",
        "what",
        "all",
        "were",
        "we",
        "when",
        "your",
        "can",
        "there",
        "would",
        "an",
        "each",
        "which",
        "she",
        "do",
        "how",
        "their",
        "if",
        "will",
        "up",
    ]
)


def paragraph_metrics_missing(parsed: Prose, word_count: int) -> bool:
    """A long single paragraph provides no useful paragraph-structure evidence."""
    return word_count >= 300 and len(parsed.paragraphs) == 1


def unlikely_english(parsed: Prose) -> bool:
    """Flag only strong evidence: few English function words or mostly non-ASCII letters.

    Very short texts have too little evidence for the function-word check.
    """
    tokens = words(parsed.text)
    letters = [character for character in parsed.text if character.isalpha()]
    if not tokens or not letters:
        return False
    ascii_share = sum(character.isascii() for character in letters) / len(letters)
    function_share = sum(token in _ENGLISH_WORDS for token in tokens) / len(tokens)
    return ascii_share < 0.5 or (len(tokens) >= 50 and function_share < 0.05)


MASK = "\N{MIDDLE DOT}"


@dataclass(frozen=True)
class Prose:
    """Markdown reduced to prose blocks, with the markup counts measured before stripping."""

    blocks: list[str]
    paragraphs: list[str]
    headings: int
    list_items: int
    links: int
    bold: int
    code_spans: int

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


def _strip_inline(line: str) -> str:
    line = _IMAGE.sub(r"\1", line)
    line = _LINK.sub(r"\1", line)
    line = _INLINE_CODE.sub("", line)
    line = _HTML_TAG.sub("", line)
    line = _URL.sub("", line)
    line = _EMPHASIS.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def strip_front_matter(text: str) -> str:
    """Remove a leading ---/+++ block whose lines all look like YAML or TOML."""
    lines = text.split("\n")
    delimiter = lines[0].strip()
    if delimiter not in ("---", "+++"):
        return text
    for index in range(1, min(len(lines), _FRONT_MATTER_LINES + 1)):
        if lines[index].strip() == delimiter:
            body = [line for line in lines[1:index] if line.strip()]
            if body and all(_FRONT_MATTER_LINE.match(line) for line in body):
                return "\n".join(lines[index + 1 :])
            return text
    return text


def _fence(line: str) -> str | None:
    opening = _FENCE_OPEN.match(line)
    return (opening.group(1) or opening.group(2)) if opening else None


def markdown_blocks(markdown: str) -> list[str]:
    """Split Markdown at blank lines, keeping each fenced code block whole.

    Front matter is dropped. Blockquote markers do not hide a blank line ("> " alone ends
    a quoted paragraph). A fence may be indented (inside a list item). An unclosed fence
    with a language (model output cut off mid-block) runs to the end of the text; a bare
    unclosed ``` or ~~~ line (often a section break) is dropped on its own.
    """
    return [raw for raw, _ in numbered_blocks(markdown)]


def numbered_blocks(markdown: str) -> list[tuple[str, int]]:
    """``markdown_blocks`` with the line each block starts on in ``markdown`` (from 1)."""
    markdown = markdown.replace("\r\n", "\n")
    text = strip_front_matter(markdown)
    return _numbered(text, markdown.count("\n") - text.count("\n") + 1)


def _numbered(text: str, first_line: int) -> list[tuple[str, int]]:
    blocks: list[tuple[str, int]] = []
    current: list[str] = []
    start = first_line
    fence: str | None = None
    for number, line in enumerate(text.split("\n"), start=first_line):
        opening = _fence(line) if fence is None else None
        if opening:
            if current:
                blocks.append(("\n".join(current), start))
            current, start, fence = [line], number, opening
        elif fence is not None:
            current.append(line)
            closing = _FENCE_CLOSE.match(line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence):
                blocks.append(("\n".join(current), start))
                current, fence = [], None
        elif _BLOCKQUOTE.sub("", line).strip():
            if not current:
                start = number
            current.append(line)
        elif current:
            blocks.append(("\n".join(current), start))
            current = []
    if fence is not None and not current[0].strip().lstrip("`~").strip():
        # A bare opener that never closes: drop that line and read the rest as usual.
        return blocks + _numbered("\n".join(current[1:]), start + 1)
    if current:
        blocks.append(("\n".join(current), start))
    return blocks


def _reads_like_prose(lines: Sequence[str]) -> bool:
    """Mostly plain words and ending in sentence punctuation, as prose does and code rarely.

    Line comments are left out, so code ending in a sentence-like comment stays code.
    """
    text = " ".join(_LINE_COMMENT.sub("", line).strip() for line in lines)
    tokens = text.split()
    prose_tokens = sum(bool(_PROSE_TOKEN.match(token)) for token in tokens)
    ends_sentence = text.rstrip(_CLOSERS).endswith((".", "!", "?"))
    return ends_sentence and prose_tokens >= _PROSE_TOKEN_SHARE * len(tokens)


@dataclass(frozen=True)
class Block:
    """One Markdown block and how it reads in context."""

    raw: str
    code: bool
    continues_list: bool
    # The line it starts on in the Markdown it was read from, counting from 1.
    line: int = 1

    @property
    def end_line(self) -> int:
        """The line it ends on."""
        return self.line + self.raw.count("\n")


def classify(markdown: str) -> list[Block]:
    """Mark code blocks, using list context to tell indented code from list continuations.
    Each block records the line it starts on."""
    classified: list[Block] = []
    after_list = False
    for raw, number in numbered_blocks(markdown):
        lines = [line for line in raw.split("\n") if line.strip()]
        indented = all(_INDENTED_CODE.match(line) for line in lines)
        code_like = (
            indented
            and any(_CODE_SIGNAL.search(line) for line in lines)
            and not _reads_like_prose(lines)
        )
        if _fence(lines[0]):
            classified.append(Block(raw, code=True, continues_list=False, line=number))
            continue
        if code_like and not after_list:
            classified.append(Block(raw, code=True, continues_list=False, line=number))
            after_list = False
            continue
        continues = indented and after_list
        has_items = any(_LIST_ITEM.match(_BLOCKQUOTE.sub("", line)) for line in lines)
        after_list = has_items or continues
        classified.append(Block(raw, code=False, continues_list=continues, line=number))
    return classified


@dataclass
class _Tally:
    blocks: list[str]
    paragraphs: list[str]
    headings: int = 0
    list_items: int = 0
    links: int = 0
    bold: int = 0
    code_spans: int = 0


def _read_block(raw: str, tally: _Tally, continues_list: bool = False) -> None:
    """Add one block's prose to the tally, in reading order.

    A block that continues a list item (indented under it after a blank line) extends that
    item rather than starting a paragraph.
    """
    source = _BLOCKQUOTE.sub("", raw)
    tally.links += len(_LINK.findall(source))
    tally.bold += len(_BOLD.findall(source))
    tally.code_spans += len(_INLINE_CODE.findall(source))
    paragraph_lines: list[str] = []
    in_list = False

    def flush() -> None:
        paragraph = _strip_inline(" ".join(paragraph_lines))
        paragraph_lines.clear()
        if not WORD.search(paragraph):
            return
        if continues_list and tally.blocks:
            tally.blocks[-1] = f"{tally.blocks[-1]} {paragraph}"
            return
        tally.blocks.append(paragraph)
        if not continues_list:
            tally.paragraphs.append(paragraph)

    for line in (line for line in source.split("\n") if line.strip()):
        if _HEADING.match(line):
            tally.headings += 1
            in_list = False
        elif line.lstrip().startswith("|"):
            continue
        elif _LIST_ITEM.match(line):
            flush()  # an intro line directly above a list comes before its items
            tally.list_items += 1
            in_list = True
            tally.blocks.append(_strip_inline(_LIST_ITEM.sub("", line)))
        elif in_list:
            tally.blocks[-1] = f"{tally.blocks[-1]} {_strip_inline(line)}".strip()
        else:
            paragraph_lines.append(line)
    flush()


def prose(markdown: str) -> Prose:
    """Drop front matter, code, headings, tables, and markup; keep paragraphs and list items."""
    tally = _Tally(blocks=[], paragraphs=[])
    for block in classify(markdown):
        if not block.code:
            _read_block(block.raw, tally, block.continues_list)
    return Prose(
        blocks=[text for text in tally.blocks if WORD.search(text)],
        paragraphs=tally.paragraphs,
        headings=tally.headings,
        list_items=tally.list_items,
        links=tally.links,
        bold=tally.bold,
        code_spans=tally.code_spans,
    )


def block_word_count(block: Block) -> int:
    """Prose words in one classified block (0 for code), as prose() would count them."""
    if block.code:
        return 0
    tally = _Tally(blocks=[], paragraphs=[])
    _read_block(block.raw, tally, block.continues_list)
    return sum(len(WORD.findall(text)) for text in tally.blocks)


def plain_sentences(block: Block) -> list[str] | None:
    """A plain paragraph's sentences, as raw Markdown, or None for any other block.

    Only running text can be cut between sentences without changing what it is: code,
    headings, lists, list continuations, quotes and tables stay whole. Inline markup
    also stays whole, so a cut never turns its contents into different prose.
    """
    if block.code or block.continues_list:
        return None
    lines = [line for line in block.raw.split("\n") if line.strip()]
    for line in lines:
        if _HEADING.match(line) or _LIST_ITEM.match(line) or line.lstrip().startswith(("|", ">")):
            return None
    text = " ".join(line.strip() for line in lines)
    protected = sorted(
        match.span()
        for pattern in (_INLINE_CODE, _LINK, _IMAGE, _BOLD, _ITALIC, _HTML_TAG)
        for match in pattern.finditer(text)
    )
    parts: list[str] = []
    start = protected_index = 0
    for boundary in _SENTENCE_BREAK.finditer(text):
        position = boundary.start()
        while protected_index < len(protected) and protected[protected_index][1] <= position:
            protected_index += 1
        if protected_index < len(protected) and protected[protected_index][0] <= position:
            continue
        piece = text[start:position]
        following = text[boundary.end() : boundary.end() + 1]
        if _ABBREVIATION.search(piece) or (
            _NUMBER_ABBREVIATION.search(piece) and following[:1].isdigit()
        ):
            continue
        parts.append(piece)
        start = boundary.end()
    parts.append(text[start:])
    return [part for part in parts if WORD.search(part)]


def words(text: str) -> list[str]:
    return [
        token.casefold().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
        for token in WORD.findall(text)
    ]


def sentences(block: str) -> list[str]:
    parts: list[str] = []
    for piece in _SENTENCE_BREAK.split(block.strip()):
        if parts and (
            _ABBREVIATION.search(parts[-1])
            or (_NUMBER_ABBREVIATION.search(parts[-1]) and piece[:1].isdigit())
        ):
            parts[-1] = f"{parts[-1]} {piece}"
        else:
            parts.append(piece)
    return [part for part in parts if WORD.search(part)]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pct(part: int, whole: int) -> float | None:
    return 100 * part / whole if whole else None


def mattr(tokens: Sequence[str], window: int = 100) -> float | None:
    """Moving-average type/token ratio, which does not shrink as texts get longer."""
    if len(tokens) < window:
        return None
    counts: Counter[str] = Counter(tokens[:window])
    total = len(counts)
    for index in range(window, len(tokens)):
        leaving = tokens[index - window]
        counts[leaving] -= 1
        if not counts[leaving]:
            del counts[leaving]
        counts[tokens[index]] += 1
        total += len(counts)
    return total / (len(tokens) - window + 1) / window


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> float | None:
    """Measure of textual lexical diversity (McCarthy & Jarvis 2010), averaged both ways."""
    if len(tokens) < 50:
        return None

    def one_pass(sequence: Sequence[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        count = 0
        for token in sequence:
            count += 1
            types.add(token)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - threshold)
        return len(sequence) / factors if factors else float(len(sequence))

    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def hapax_share(tokens: Sequence[str], block: int = 100) -> float | None:
    """Percent of word types used once, averaged over consecutive fixed-size blocks."""
    shares = []
    for start in range(0, len(tokens) - block + 1, block):
        counts = Counter(tokens[start : start + block])
        shares.append(100 * sum(value == 1 for value in counts.values()) / len(counts))
    return _mean(shares)


def surface_metrics(
    markdown: str, parsed: Prose | None = None, tokens: list[str] | None = None
) -> Metrics:
    """All parser-free metrics for one chunk, grouped by stylistic level, as the registry in
    ``styleprofile.metrics`` defines them.

    Pass ``parsed`` when the chunk's ``prose()`` is already at hand, to avoid re-parsing.
    """
    parsed = parsed or prose(markdown)
    text = parsed.text
    tokens = words(text) if tokens is None else tokens
    word_count = len(tokens)
    per_1k = 1000 / word_count if word_count else None
    # Phrase patterns are written with straight apostrophes; curly ones must match too.
    plain = text.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")

    sentence_texts = [sentence for block in parsed.blocks for sentence in sentences(block)]
    # Counting words needs no folding: as many as ``words`` gives.
    lengths = [len(WORD.findall(sentence)) for sentence in sentence_texts]
    paragraph_sentence_counts = [len(sentences(paragraph)) for paragraph in parsed.paragraphs]
    mean_length = _mean(lengths)
    length_sd = statistics.pstdev(lengths) if lengths else None

    apostrophes = text.count("'") + text.count("\N{RIGHT SINGLE QUOTATION MARK}")
    straight_double = text.count('"')
    function_counts = Counter(token for token in tokens if token in _FUNCTION_SET)

    def rate(count: int) -> float | None:
        return count * per_1k if per_1k is not None else None

    values: dict[str, float | None] = {
        "words": float(word_count),
        "sentences": float(len(lengths)),
        "paragraphs": float(len(parsed.paragraphs)),
        "apostrophes": float(apostrophes),
        "sentence_words_mean": mean_length,
        "sentence_words_median": float(statistics.median(lengths)) if lengths else None,
        "sentence_words_sd": length_sd,
        "short_sentences_pct": _pct(sum(n <= SHORT_SENTENCE_WORDS for n in lengths), len(lengths)),
        "long_sentences_pct": _pct(sum(n >= LONG_SENTENCE_WORDS for n in lengths), len(lengths)),
        "paragraph_sentences_mean": _mean(paragraph_sentence_counts),
        "paragraph_words_mean": _mean([len(WORD.findall(p)) for p in parsed.paragraphs]),
        "one_sentence_paragraphs_pct": _pct(
            sum(count == 1 for count in paragraph_sentence_counts),
            len(paragraph_sentence_counts),
        ),
        "sentence_length_cv": (
            length_sd / mean_length if mean_length and length_sd is not None else None
        ),
        "sentence_length_change_mean": _mean(
            [abs(left - right) for left, right in pairwise(lengths)]
        ),
        "word_chars_mean": _mean([len(token) for token in tokens]),
        "long_words_pct": _pct(sum(len(token) >= LONG_WORD_CHARS for token in tokens), word_count),
        "mattr_100": mattr(tokens),
        "mtld": mtld(tokens),
        "hapax_share_100": hapax_share(tokens),
        "commas_per_1k": rate(text.count(",")),
        "semicolons_per_1k": rate(text.count(";")),
        "colons_per_1k": rate(text.count(":")),
        "em_dashes_per_1k": rate(len(_EM_DASH.findall(text))),
        "parentheses_per_1k": rate(text.count("(")),
        "questions_per_1k": rate(text.count("?")),
        "exclamations_per_1k": rate(text.count("!")),
        "ellipses_per_1k": rate(len(_ELLIPSIS.findall(text))),
        "quotations_per_1k": rate(
            text.count("\N{LEFT DOUBLE QUOTATION MARK}") + straight_double // 2
        ),
        "curly_apostrophe_pct": _pct(text.count("\N{RIGHT SINGLE QUOTATION MARK}"), apostrophes),
        "contractions_per_1k": rate(
            sum(
                token in S_CONTRACTIONS or bool(CONTRACTION_SUFFIX.search(token))
                for token in tokens
            )
        ),
        "first_singular_per_1k": rate(sum(token in FIRST_SINGULAR for token in tokens)),
        "first_plural_per_1k": rate(sum(token in FIRST_PLURAL for token in tokens)),
        "second_person_per_1k": rate(sum(token in SECOND for token in tokens)),
        "negations_per_1k": rate(
            sum(token in NEGATIONS or token.endswith("n't") for token in tokens)
        ),
        "hedges_per_1k": rate(len(HEDGE.findall(plain))),
        "boosters_per_1k": rate(len(BOOSTER.findall(plain))),
        "and_but_so_openers_pct": _pct(
            sum(bool(AND_BUT_SO_OPENER.match(s)) for s in sentence_texts),
            len(sentence_texts),
        ),
        "transition_openers_pct": _pct(
            sum(bool(TRANSITION_OPENER.match(s)) for s in sentence_texts),
            len(sentence_texts),
        ),
        "llm_markers_per_1k": rate(len(LLM_MARKER.findall(plain))),
        "not_just_but_per_1k": rate(len(NOT_JUST_BUT.findall(plain))),
        "headings_per_1k": rate(parsed.headings),
        "list_items_per_1k": rate(parsed.list_items),
        "links_per_1k": rate(parsed.links),
        "bold_per_1k": rate(parsed.bold),
        "code_spans_per_1k": rate(parsed.code_spans),
        **{f"fw_{word}_per_1k": rate(function_counts[word]) for word in FUNCTION_WORDS},
    }
    if paragraph_metrics_missing(parsed, word_count):
        values.update(dict.fromkeys(PARAGRAPH_METRICS))
    return grouped(values, syntax=False)


def masked_bigrams(text: str, tokens: list[str] | None = None) -> Counter[str]:
    """Transitions between function words, with every content word masked to one symbol.

    ``text`` is prose, as ``prose(markdown).text`` returns it.
    """
    tokens = [
        token if token in _FUNCTION_SET else MASK
        for token in (words(text) if tokens is None else tokens)
    ]
    return Counter(
        f"{left} {right}" for left, right in pairwise(tokens) if (left, right) != (MASK, MASK)
    )


def char_trigrams(text: str) -> Counter[str]:
    """Character trigrams of prose text (``prose(markdown).text``), case-folded."""
    text = re.sub(r"\s+", " ", text.casefold())
    return Counter(map("".join, zip(text, text[1:], text[2:], strict=False)))


def jensen_shannon(sample: dict[str, float], reference: dict[str, float]) -> float | None:
    """Jensen-Shannon divergence in bits (0 identical, 1 disjoint) between two distributions."""
    if not sample or not reference:
        return None
    total = 0.0
    for key in sorted(sample.keys() | reference.keys()):
        p = sample.get(key, 0.0)
        q = reference.get(key, 0.0)
        m = (p + q) / 2
        if p:
            total += p * math.log2(p / m)
        if q:
            total += q * math.log2(q / m)
    return max(0.0, total / 2)
