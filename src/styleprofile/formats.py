"""Read HTML as Markdown, and tell HTML, JSONL and plain text apart.

Every metric is measured on Markdown, so HTML is converted rather than stripped: stripping
tags inline loses the paragraph breaks that ``<p>`` and ``<li>`` carry, and a whole essay
then reads as one paragraph. The conversion keeps paragraphs, list items, headings, quotes,
code and inline bold, links and code spans, and drops page chrome (navigation, site
headers and footers, scripts, comment sections).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser

HTML_SUFFIXES = frozenset({".html", ".htm"})
AUTO = "auto"
MARKDOWN = "markdown"
HTML = "html"
JSONL = "jsonl"
INPUT_FORMATS = (AUTO, MARKDOWN, HTML, JSONL)

# Elements whose content is never the writer's prose. A <form> itself is kept, since
# ASP.NET WebForms pages wrap the whole body in one; its controls and labels are not.
_DROPPED = frozenset(
    {
        "script", "style", "noscript", "template", "title", "nav", "aside", "svg", "iframe",
        "input", "textarea", "select", "option", "button", "label",
    }
)  # fmt: skip
# Site chrome. Inside an <article> or <main> a <header> holds the post's title, so only
# its headings are kept there (not the byline or date); a footer (tags, share buttons) is
# dropped everywhere.
_CHROME = frozenset({"header", "footer"})
# What may appear in <head>; any other start tag closes a <head> whose end tag was left out.
_HEAD_CONTENT = frozenset(
    {"title", "meta", "link", "style", "script", "base", "noscript", "template"}
)
# Comment sections and subscribe widgets, by an id or class token: WordPress (comments,
# comment-list, comment-body, respond), Disqus, Substack and most themes.
_WIDGETS = re.compile(
    r"^(?:comments?|comments-area|commentlist|comment-(?!open|closed|status)[\w-]+|respond|"
    r"disqus_thread|subscribe(?:-[\w-]+)?|subscription-widget[\w-]*|newsletter-signup)$",
    re.IGNORECASE,
)
# An inline element left open (``<p><b>text`` again and again) is reopened in each later
# block, as browsers do; like browsers, keep at most this many of one tag open.
_MAX_OPEN_PER_TAG = 3
_VOID = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
        "source", "track", "wbr",
    }
)  # fmt: skip
# Elements that break the text into blocks. Headings, lists, quotes and pre are handled
# on their own below.
_BLOCKS = frozenset(
    {
        "p", "div", "section", "article", "main", "body", "html", "figure", "figcaption",
        "table", "thead", "tbody", "tfoot", "tr", "dl", "dt", "dd", "details", "summary",
        "address", "center", "fieldset", "form", "header", "footer", "hgroup",
    }
)  # fmt: skip
_CONTAINERS = ("article", "main")
_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_LISTS = frozenset({"ul", "ol", "menu"})
_INLINE_MARKERS = {"strong": "**", "b": "**", "em": "*", "i": "*", "code": "`"}
# Paragraph text that Markdown would read as a heading, list item, quote or table row.
_MARKDOWN_START = re.compile(r"^(?:#{1,6}(?:\s|$)|[-+*]\s|>|\|)")
_NUMBERED_START = re.compile(r"^(\d+)([.)])(?=\s|$)")
_WORDS = re.compile(r"\w+")

# Sniffing. A block-level tag, opening or closing, and one that starts a text block.
_BLOCK_NAMES = (
    r"p|div|li|ul|ol|dl|dt|dd|h[1-6]|blockquote|pre|section|article|main|header|footer|nav|"
    r"aside|figure|table|thead|tbody|tr|td|th|body|html|head|hr|br"
)
_BLOCK_TAG = re.compile(rf"</?(?:{_BLOCK_NAMES})\b[^>]*>", re.IGNORECASE)
_STARTS_WITH_TAG = re.compile(rf"<(?:/?(?:{_BLOCK_NAMES})\b|!--)", re.IGNORECASE)
_OPENS = re.compile(
    r"<(?:div|section|article|main|body|html|blockquote|ul|ol|dl|table|p|li|pre|header|"
    r"footer|nav|aside|figure)\b[^>]*>",
    re.IGNORECASE,
)
_CLOSES = re.compile(
    r"</(?:div|section|article|main|body|html|blockquote|ul|ol|dl|table|p|li|pre|header|"
    r"footer|nav|aside|figure)\s*>",
    re.IGNORECASE,
)
_PREAMBLE = re.compile(r"^\s*(?:<\?xml[^>]*\?>\s*|<!--.*?-->\s*)*", re.DOTALL)
_DOCTYPE = re.compile(r"<!doctype\s+html|<html\b", re.IGNORECASE)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_MIN_BLOCK_TAGS = 3


def looks_like_html(text: str) -> bool:
    """Whether a Markdown or text file is really an HTML document or fragment.

    A doctype (after any XML declaration or comments) settles it. Otherwise the text is
    split at blank lines, and it is HTML when it has at least three block-level tags and
    every block either starts with one or sits inside an element a previous block opened.
    Continuation lines may be plain text, as in HTML hard-wrapped at 72 columns (pandoc's
    output). Any Markdown outside an element (a heading, list, fence or plain paragraph)
    marks the text as Markdown with some HTML in it, as READMEs with a centred logo are.
    """
    preamble = _PREAMBLE.match(text)
    if _DOCTYPE.match(text, preamble.end() if preamble else 0):
        return True
    tags = 0
    depth = 0
    for block in _BLANK_LINE.split(text):
        stripped = block.strip()
        if not stripped:
            continue
        if depth <= 0 and not _STARTS_WITH_TAG.match(stripped):
            return False
        tags += len(_BLOCK_TAG.findall(block))
        depth += len(_OPENS.findall(block)) - len(_CLOSES.findall(block))
    return tags >= _MIN_BLOCK_TAGS


def looks_like_jsonl(text: str) -> bool:
    """Whether every non-blank line is a JSON object (and there is at least one)."""
    found = False
    for line in text.split("\n"):
        if not line.strip():
            continue
        if not line.lstrip().startswith("{"):
            return False
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(record, dict):
            return False
        found = True
    return found


@dataclass
class _Marker:
    """An open inline element. Its Markdown opener is written only when text arrives inside
    it (``index`` is then its place in the buffer), so an empty ``<b></b>`` writes nothing."""

    tag: str
    opener: str
    closer: str
    index: int | None = None
    broken: bool = False  # a link that contained a block: its text is kept, unlinked


@dataclass
class _Candidate:
    """An ``<article>`` or ``<main>`` outside dropped chrome, as a range of output lines."""

    tag: str
    start: int
    end: int
    nested: bool  # an article inside another article is part of it, not a candidate


def _widget(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    if tag in ("html", "body", "main"):
        return False
    values = dict(attrs)
    tokens = [*(values.get("class") or "").split(), values.get("id") or ""]
    return any(_WIDGETS.match(token) for token in tokens if token)


def _escape(line: str) -> str:
    """Escape paragraph text that would otherwise read as Markdown block syntax."""
    numbered = _NUMBERED_START.match(line)
    if numbered:
        return f"{numbered.group(1)}\\{line[len(numbered.group(1)) :]}"
    return "\\" + line if _MARKDOWN_START.match(line) else line


class _Converter(HTMLParser):
    """Turn HTML into Markdown lines, one block element at a time."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.inline: list[str] = []
        self.ends_with_newline = False  # the inline buffer ends with a line break
        self.skipping: list[str] = []  # open tags of the dropped subtree being skipped
        self.in_head = False
        self.markers: list[_Marker] = []
        self.lists: list[list[int]] = []  # per open list: [ordered, next number]
        self.item_open = False  # the current list item has not written its marker yet
        self.quote = 0
        self.heading = 0
        self.post_header = 0  # depth inside an article's or main's <header>
        self.pre: list[str] | None = None
        self.open_containers: list[tuple[str, int, bool]] = []  # (tag, start line, nested)
        self.candidates: list[_Candidate] = []
        # Paragraph lines to escape if the page turns out to use HTML headings or lists.
        self.plain: list[tuple[int, int]] = []  # (line, length of its quote prefix)
        self.structured = False

    # Parsing events.

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skipping:
            if tag not in _VOID:
                self.skipping.append(tag)
            return
        if self.in_head and tag not in _HEAD_CONTENT:
            self.in_head = False  # </head> was left out, as HTML5 allows
        if tag == "head":
            self.in_head = True
            return
        dropped = (
            tag in _DROPPED
            or (tag in _CHROME and not (tag == "header" and self.open_containers))
            or _widget(tag, attrs)
        )
        if dropped:
            if tag not in _VOID:
                self.skipping.append(tag)
            return
        if self.in_head:
            return
        if self.pre is not None:
            if tag == "br":
                self.pre.append("\n")
            return
        if tag in _HEADINGS or tag in _LISTS or tag == "li":
            self.structured = True
        if tag == "header":
            self.post_header += 1
        if tag in _CONTAINERS:
            self._flush()
            nested = tag == "article" and any(t == "article" for t, _, _ in self.open_containers)
            self.open_containers.append((tag, len(self.lines), nested))
        if tag in _BLOCKS or tag in _HEADINGS or tag in ("blockquote", "hr"):
            self._flush()
            self.heading = _HEADINGS.get(tag, self.heading)
            if tag == "blockquote":
                self.quote += 1
            elif tag == "hr" and self.lines:
                self.lines += ["", "---", ""]
        elif tag in _LISTS:
            self._flush()
            self.lists.append([tag == "ol", _start(attrs)])
        elif tag == "li":
            self._flush()
            self.item_open = True
        elif tag == "pre":
            self._flush()
            self.pre = []
        elif tag == "br":
            if self.ends_with_newline:
                self._flush()  # <br><br> separates paragraphs
            else:
                self.inline.append("\n")
                self.ends_with_newline = True
        elif tag in ("td", "th"):
            self._text(" ")
        elif tag in _INLINE_MARKERS:
            marker = _INLINE_MARKERS[tag]
            self._push(_Marker(tag, marker, marker))
        elif tag == "a":
            href = dict(attrs).get("href")
            self._push(_Marker("a", "[", f"]({href})" if href else "", broken=not href))

    def handle_endtag(self, tag: str) -> None:
        if self.skipping:
            if tag in self.skipping:
                while self.skipping.pop() != tag:
                    pass
            return
        if tag == "head":
            self.in_head = False
            return
        if self.in_head:
            return
        if self.pre is not None:
            if tag == "pre":
                code = "".join(self.pre).strip("\n")
                self.pre = None
                self.lines += ["```", *code.split("\n"), "```", ""]
            return
        if tag in _BLOCKS or tag in _HEADINGS or tag in ("blockquote", "li"):
            self._flush()
            if tag in _HEADINGS:
                self.heading = 0
            elif tag == "blockquote":
                self.quote = max(self.quote - 1, 0)
            elif tag == "li":
                self.item_open = False
        elif tag in _LISTS:
            self._flush()
            if self.lists:
                self.lists.pop()
            if not self.lists:
                self.lines.append("")
        elif tag in _INLINE_MARKERS or tag == "a":
            self._close_marker(tag)
        if tag == "header":
            self.post_header = max(self.post_header - 1, 0)
        if tag in _CONTAINERS:
            self._close_container(tag)

    def handle_data(self, data: str) -> None:
        if self.skipping or self.in_head or (self.post_header and not self.heading):
            return
        if self.pre is not None:
            self.pre.append(data)
            return
        # A blank line in the text is a paragraph break, so Markdown that was misread as
        # HTML keeps its paragraphs.
        for index, part in enumerate(_BLANK_LINE.split(data)):
            if index:
                self._flush()
            self._text(re.sub(r"\s+", " ", part))

    def close(self) -> None:
        super().close()
        if self.pre is not None:
            self.lines += ["```", *"".join(self.pre).strip("\n").split("\n"), "```"]
            self.pre = None
        self._flush()
        while self.open_containers:
            self._close_container(self.open_containers[-1][0])

    # Inline text and markers.

    def _text(self, text: str) -> None:
        if not text:
            return
        content = text.lstrip(" ")
        if not content:
            if not self.ends_with_newline:
                self.inline.append(text)
            return
        if len(content) < len(text):
            self.inline.append(" ")  # leading space goes outside markers opened here
        for marker in self.markers:
            if marker.index is None and not (marker.broken and marker.tag == "a"):
                marker.index = len(self.inline)
                self.inline.append(marker.opener)
        self.inline.append(content)
        self.ends_with_newline = False

    def _push(self, marker: _Marker) -> None:
        """Open an inline marker, forgetting the oldest open one of its tag past
        ``_MAX_OPEN_PER_TAG``: unclosed tags would otherwise pile up, and every later text
        would walk the whole pile."""
        same = [open_marker for open_marker in self.markers if open_marker.tag == marker.tag]
        if len(same) >= _MAX_OPEN_PER_TAG:
            oldest = same[0]
            if oldest.index is not None:
                self.inline[oldest.index] = ""  # never closed, so never written
            self.markers.remove(oldest)
        self.markers.append(marker)

    def _close_marker(self, tag: str) -> None:
        """Close the innermost open ``tag`` against its text: ``**bold** ``, not ``**bold **``."""
        position = next(
            (i for i in range(len(self.markers) - 1, -1, -1) if self.markers[i].tag == tag), None
        )
        if position is None:
            return
        marker = self.markers.pop(position)
        if marker.index is not None:  # else nothing inside it was written
            self._write_closer(marker)

    def _write_closer(self, marker: _Marker) -> None:
        """Write an opened marker's closer before any trailing space or line break."""
        index = marker.index
        if index is None:
            return
        trailing = ""
        while self.inline and not self.inline[-1].strip():
            trailing = self.inline.pop() + trailing
        if self.inline and self.inline[-1] != self.inline[-1].rstrip():
            last = self.inline.pop()
            self.inline.append(last.rstrip())
            trailing = last[len(last.rstrip()) :] + trailing
        closer = marker.closer
        # A heading's permalink or a footnote mark ("#", "¶", "1") is not prose.
        if (
            marker.tag == "a"
            and closer.startswith("](#")
            and len("".join(self.inline[index + 1 :]).strip()) <= 1
        ):
            del self.inline[index:]
            closer = ""
        if "\n" in trailing:
            trailing = "\n"  # the element ended with a line break: keep it after the closer
        self.inline += [closer, trailing]
        self.ends_with_newline = trailing == "\n"

    # Blocks.

    def _close_container(self, tag: str) -> None:
        position = next(
            (
                i
                for i in range(len(self.open_containers) - 1, -1, -1)
                if self.open_containers[i][0] == tag
            ),
            None,
        )
        if position is None:
            return
        self._flush()
        _, start, nested = self.open_containers.pop(position)
        self.candidates.append(_Candidate(tag, start, len(self.lines), nested))

    def _flush(self) -> None:
        """Write the text gathered so far as one block, with its heading, list or quote."""
        for marker in reversed(self.markers):
            if marker.index is None:
                continue
            if marker.tag == "a":
                self.inline[marker.index] = ""  # a link around blocks is not a link
                marker.broken = True
            else:
                self._write_closer(marker)  # bold and the like reopen in the next block
            marker.index = None
        text = "".join(self.inline)
        self.inline = []
        self.ends_with_newline = False
        lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            return
        prefix = "> " * self.quote
        if self.heading:
            self.lines += [prefix + "#" * self.heading + " " + " ".join(lines), ""]
            return
        if self.lists:
            indent = "  " * (len(self.lists) - 1)
            if self.item_open:
                ordered, number = self.lists[-1]
                marker = f"{number}. " if ordered else "- "
                self.lists[-1][1] += 1
                self.item_open = False
            else:
                marker = "  "  # a second paragraph in an item continues it
            self.lines += [prefix + indent + marker + lines[0]]
            self.lines += [prefix + indent + "  " + line for line in lines[1:]]
            return
        self.plain += [(len(self.lines) + i, len(prefix)) for i in range(len(lines))]
        self.lines += [prefix + line for line in lines] + [""]

    def selected(self) -> list[str]:
        """The lines of the page's main content; see ``html_to_markdown``."""
        if self.structured:
            # The page marks headings and lists with tags, so a paragraph starting "# of
            # users" or "1. Not a list" is text. A page without them may be Markdown wrapped
            # in <p> tags line by line, whose "# Title" is a heading.
            for index, quoted in self.plain:
                line = self.lines[index]
                self.lines[index] = line[:quoted] + _escape(line[quoted:])

        def words(ranges: list[_Candidate]) -> int:
            return sum(len(_WORDS.findall("\n".join(self.lines[c.start : c.end]))) for c in ranges)

        articles = [c for c in self.candidates if c.tag == "article" and not c.nested]
        articles.sort(key=lambda c: c.start)
        mains = [c for c in self.candidates if c.tag == "main"]
        choice: list[_Candidate] = []
        if articles:
            largest = max(articles, key=lambda c: words([c]))
            total = words(articles)
            # One article with twice the text of all the others together is the post,
            # and the rest are related-post cards; otherwise the page lists posts (the
            # newest may be in full above shorter excerpts), and all are read.
            largest_words = words([largest])
            choice = [largest] if largest_words >= 2 * (total - largest_words) else articles
        if mains:
            main = max(mains, key=lambda c: words([c]))
            if not choice or words(choice) < words([main]) / 2:
                choice = [main]
        if not choice or not words(choice):
            return self.lines
        selected: list[str] = []
        for number, candidate in enumerate(choice):
            # Separate articles are separate pieces: a rule between them, which has no
            # words to measure, lets ``split`` cut a page of posts into its posts.
            selected += ["---", ""] if number else []
            selected += [*self.lines[candidate.start : candidate.end], ""]
        return selected


def _start(attrs: list[tuple[str, str | None]]) -> int:
    value = dict(attrs).get("start") or "1"
    return int(value) if value.strip().isdigit() else 1


def html_to_markdown(html: str) -> str:
    """Convert an HTML page or fragment to the Markdown its metrics are measured on.

    Only the page's main content is read, so blog-export chrome around a post is not
    measured. The candidates are its ``<main>`` elements and its top-level ``<article>``
    elements, ignoring any inside dropped chrome (an aside's "related posts" cards):

    - among the articles, the one with the most text when it has at least twice the text
      of all the others together (a post beside smaller cards), else all of them (an index
      page of post excerpts, the newest perhaps in full), separated by rules (``---``), so
      ``split`` can cut a page of posts into its posts;
    - a ``<main>`` instead when that choice holds under half of the main's text (the post
      body sits in a ``<div>`` next to an article card);
    - the whole ``<body>`` when there are no candidates, or the chosen ones have no text.

    Scripts, styles, form controls, navigation, asides, footers, comment sections and
    subscribe widgets are dropped everywhere. A header is dropped outside an article or
    main, where it is the site's; inside one only its headings (the post's title) are kept.
    Entities are decoded, and a ``<head>`` whose end tag was left out ends at the first tag
    that cannot be in it.

    On a page that marks headings and lists with tags, paragraph text that Markdown would
    read as block syntax ("# of users", "1. Not a list") is escaped. A page without them may
    be Markdown wrapped in ``<p>`` tags line by line, and its "# Title" stays a heading.
    """
    converter = _Converter()
    converter.feed(html)
    converter.close()
    markdown = "\n".join(converter.selected())
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
