"""Typography invariance: no way of typing or converting a text may change its verdict.

    python bench/invariance.py [--syntax auto|off|on] [--no-medium]

Each transform in ``TRANSFORMS`` changes how a text is typed, not what it says: curly or
straight quotes, line endings, hard wraps, dash and ellipsis glyphs, spacing, paragraph
separators, an HTML round trip. The same checks run as ``tests/test_invariance.py``; this
script prints the table for a pull request.

- Writer side: each essay in ``examples/writer`` against a profile built without it, and the
  held-out documents of the synthetic ``medium`` corpus (every fifth, sorted) against a
  profile built without them. Both verdicts must not change, and Delta may move by at most
  ``MAX_DELTA_SHIFT``.
- LLM side: each draft in ``examples/llm-drafts`` against the ``examples/writer`` profile.
  Its verdicts must not drop a band, so that normalizing typography can't erase real signal.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from styleprofile import Profile, Text, build  # noqa: E402
from styleprofile.core import DISTANCES, LikenessVerdict  # noqa: E402

MAX_DELTA_SHIFT = 0.25
SYNTAX: dict[str, bool | Literal["auto"]] = {"auto": "auto", "off": False, "on": True}
LIKENESS_LEVELS = (
    LikenessVerdict.LIKE_REFERENCE,
    LikenessVerdict.FEW_TRAITS,
    LikenessVerdict.LEANS,
    LikenessVerdict.LIKE_DRAFTS,
)


def _curly(text: str) -> str:
    text = re.sub(r"(\w)'(\w)", "\\1\u2019\\2", text)
    text = re.sub(r"(^|\s)'", "\\1\u2018", text)
    text = text.replace("'", "\u2019")
    text = re.sub(r'(^|\s)"', "\\1\u201c", text)
    return text.replace('"', "\u201d")


def _straight(text: str) -> str:
    for curly, straight in (
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
    ):
        text = text.replace(curly, straight)
    return text


def _paragraphs(text: str, change: Callable[[str], str]) -> str:
    """Apply ``change`` to each plain paragraph, leaving headings, lists and code alone."""
    parts = re.split(r"(\n\s*\n)", text)
    return "".join(
        part if not part.strip() or re.match(r"\s*([#>*+-]|\d+\.|```|\|)", part) else change(part)
        for part in parts
    )


def _wrap(text: str) -> str:
    return _paragraphs(text, lambda p: textwrap.fill(p, 72, break_long_words=False))


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"<strong>\2</strong>", text)
    return re.sub(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])", r"<em>\2</em>", text)


def _html_block(block: str) -> list[str]:
    out: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []
    ordered = False

    def flush() -> None:
        if paragraph:
            out.append("<p>" + _inline("\n".join(paragraph)) + "</p>")
            paragraph.clear()
        if items:
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + f"</{tag}>")
            items.clear()

    for line in block.split("\n"):
        heading = re.match(r"(#{1,6})\s+(.*)", line)
        item = re.match(r"\s*(?:([-*+])|(\d+\.))\s+(.*)", line)
        if heading:
            flush()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif item:
            if paragraph:
                flush()
            ordered = item.group(2) is not None
            items.append(item.group(3))
        else:
            if items:
                flush()
            paragraph.append(line)
    flush()
    return out


def _html(text: str) -> str:
    """Markdown rendered to simple HTML: paragraphs, headings, lists and emphasis."""
    blocks = [part for block in re.split(r"\n\s*\n", text.strip()) for part in _html_block(block)]
    return "<html><body>\n" + "\n".join(blocks) + "\n</body></html>\n"


def _sentence_end(spacing: str) -> Callable[[str], str]:
    return lambda text: re.sub(
        r"([.!?][\"'\u201d\u2019)]?) (?=[A-Z\u201c\"'])", r"\1" + spacing, text
    )


@dataclass(frozen=True)
class Transform:
    name: str
    change: Callable[[str], str]
    input_format: str = "auto"
    # Removing paragraph breaks leaves the paragraph metrics out (they would be one huge
    # paragraph), which can only make a text look closer; its Delta may fall, but not rise.
    drops_metrics: bool = False


TRANSFORMS: tuple[Transform, ...] = (
    Transform("straight to typographic quotes", _curly),
    Transform("typographic to straight quotes", _straight),
    Transform("CRLF line endings", lambda t: t.replace("\n", "\r\n")),
    Transform("hard wrap at 72 columns", _wrap),
    Transform("em dash to --", lambda t: t.replace("\u2014", "--")),
    Transform("-- to em dash", lambda t: re.sub(r"(?<=\w)--(?=\w)", "\u2014", t)),
    Transform("... to ellipsis glyph", lambda t: t.replace("...", "\u2026")),
    Transform("ellipsis glyph to ...", lambda t: t.replace("\u2026", "...")),
    Transform("two spaces after sentences", _sentence_end("  ")),
    Transform("non-breaking space after sentences", _sentence_end("\u00a0")),
    Transform(
        "single newline between paragraphs",
        lambda t: re.sub(r"\n\s*\n", "\n", t),
        drops_metrics=True,
    ),
    Transform(
        "blank lines removed",
        lambda t: "\n".join(x for x in t.split("\n") if x.strip()),
        drops_metrics=True,
    ),
    Transform("Markdown to HTML and back", _html, input_format="html"),
    Transform("trailing spaces and tabs", lambda t: re.sub(r"(?m)$", " \t", t)),
)


@dataclass(frozen=True)
class Case:
    side: str  # "writer" or "llm"
    name: str
    text: str
    profile: Profile


@dataclass(frozen=True)
class Outcome:
    transform: str
    side: str
    cases: int
    failures: list[str]
    worst_shift: float


def _read(folder: Path) -> list[Text]:
    return [
        Text(path.read_text(encoding="utf-8"), path.name) for path in sorted(folder.glob("*.md"))
    ]


def _score(profile: Profile, text: str, syntax: str, input_format: str = "auto"):
    """Score ``text`` as the command line would: from a file, so line endings and HTML are
    read the way they are for a user's own files."""
    flag = SYNTAX[syntax]
    suffix = ".html" if input_format == "html" else ".md"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"doc{suffix}"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return profile.score(path, syntax=flag, input_format=input_format)


def _levels(result) -> tuple[int, int, float | None]:
    verdict = result.verdict
    likeness = result.likeness_verdict
    distance = DISTANCES.index(verdict) if verdict in DISTANCES else -1
    like = LIKENESS_LEVELS.index(likeness) if likeness in LIKENESS_LEVELS else -1
    return distance, like, result.delta


def cases(syntax: str, medium: bool = True) -> list[Case]:
    flag = SYNTAX[syntax]
    writer = _read(ROOT / "examples" / "writer")
    contrast = _read(ROOT / "examples" / "llm-drafts")
    out: list[Case] = []
    for held in writer:
        rest = [text for text in writer if text is not held]
        profile = build(rest, contrast=contrast, syntax=flag)
        out.append(Case("writer", held.name or "", held.text, profile))
    whole = build(writer, contrast=contrast, syntax=flag)
    out += [Case("llm", text.name or "", text.text, whole) for text in contrast]
    if medium:
        out += _medium(syntax)
    return out


def _medium(syntax: str) -> list[Case]:
    import gen

    flag = SYNTAX[syntax]
    folder = gen.generate("medium")
    documents = sorted((folder / "writer").glob("*.md"))
    held = documents[::5]
    kept = [path for path in documents if path not in held]
    contrast = sorted((folder / "contrast").glob("*.md"))
    profile = build(
        [Text(path.read_text(encoding="utf-8"), path.name) for path in kept],
        contrast=[Text(path.read_text(encoding="utf-8"), path.name) for path in contrast],
        syntax=flag,
    )
    return [Case("writer", path.name, path.read_text(encoding="utf-8"), profile) for path in held]


def run(syntax: str = "auto", medium: bool = True) -> list[Outcome]:
    all_cases = cases(syntax, medium)
    base = {
        (case.side, case.name): _levels(_score(case.profile, case.text, syntax))
        for case in all_cases
    }
    outcomes = []
    for transform in TRANSFORMS:
        for side in ("writer", "llm"):
            failures: list[str] = []
            worst = 0.0
            chosen = [case for case in all_cases if case.side == side]
            for case in chosen:
                changed = transform.change(case.text)
                after = _levels(_score(case.profile, changed, syntax, transform.input_format))
                distance, like, delta = base[(case.side, case.name)]
                new_distance, new_like, new_delta = after
                moved = (new_delta or 0.0) - (delta or 0.0)
                shift = abs(moved)
                counted = max(moved, 0.0) if transform.drops_metrics else shift
                if side == "writer":
                    worst = max(worst, shift)
                    if transform.drops_metrics:
                        bad = (
                            new_distance > distance or new_like > like or counted > MAX_DELTA_SHIFT
                        )
                    else:
                        bad = (new_distance, new_like) != (
                            distance,
                            like,
                        ) or counted > MAX_DELTA_SHIFT
                else:
                    bad = new_distance < distance or new_like < like
                if bad:
                    failures.append(
                        f"{case.name}: {DISTANCES[distance] if distance >= 0 else '-'}"
                        f"/{like} to {new_distance}/{new_like}, Delta "
                        f"{delta if delta is None else round(delta, 2)} to "
                        f"{new_delta if new_delta is None else round(new_delta, 2)}"
                    )
            outcomes.append(Outcome(transform.name, side, len(chosen), failures, worst))
    return outcomes


def table(outcomes: list[Outcome]) -> str:
    lines = [
        "| Transform | Side | Cases | Failing | Largest Delta shift |",
        "|---|---|---|---|---|",
    ]
    for item in outcomes:
        shift = f"{item.worst_shift:.2f}" if item.side == "writer" else "-"
        lines.append(
            f"| {item.transform} | {item.side} | {item.cases} | {len(item.failures)} | {shift} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--syntax", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--no-medium", action="store_true", help="skip the synthetic corpus")
    parser.add_argument("--details", action="store_true", help="list each failing case")
    args = parser.parse_args()
    outcomes = run(args.syntax, not args.no_medium)
    print(table(outcomes))
    if args.details:
        for item in outcomes:
            for failure in item.failures:
                print(f"{item.transform} [{item.side}] {failure}")
    sys.exit(1 if any(item.failures for item in outcomes) else 0)


if __name__ == "__main__":
    main()
