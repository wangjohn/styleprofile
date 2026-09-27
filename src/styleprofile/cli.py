"""Command-line entry point: build a writer's reference profile once, then score drafts.

    styleprofile build posts/ --contrast llm-drafts/ -o writer.json
    styleprofile score draft.md writer.json
    styleprofile show writer.json
    styleprofile metrics

The flat form of earlier releases (``styleprofile INPUT... --output X [--reference R]``)
still runs for one release, with a deprecation note naming the equivalent new command.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from styleprofile import __version__
from styleprofile.display import (
    describe_delta,
    format_summary,
    likeness_level,
    likeness_words,
    mean_ceiling,
)
from styleprofile.metrics import describe
from styleprofile.profile import (
    REFERENCE,
    VERSION,
    Chunk,
    StyleProfileError,
    build_reference,
    document_of,
    dumps_report,
    load_chunks,
    load_reference,
    load_report,
    report_kind,
    score,
    window,
    write_report,
)
from styleprofile.syntax import Parser, SyntaxUnavailableError, load_parser

PROG = "styleprofile"
COMMANDS = ("build", "score", "show", "metrics")
DEFAULT_WINDOW_WORDS = 500
DEFAULT_TOP_K = 300
# A reference below these is usable but thin; `build` says so and how to fix it.
ENOUGH_DOCUMENTS = 2
ENOUGH_CHUNKS = 15
ENOUGH_WORDS = 20_000
# Advice for library errors, which name the problem but never a flag.
HINTS = {
    "text_field": "pass --text-field with the JSONL field that holds the text",
    "contrast_needs_documents": "add more of the writer's documents, or build without --contrast",
    "score_as_reference": "score against the reference profile made by `styleprofile build`",
}
SYNTAX_INSTALL = "pip install 'styleprofile[syntax]'"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _color() -> bool:
    """NO_COLOR turns color off and FORCE_COLOR on (NO_COLOR wins); else color a terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR", "0") not in ("", "0"):
        return True
    return sys.stdout.isatty()


def _note(message: str) -> None:
    print(f"note: {message}", file=sys.stderr)


def _version_text() -> str:
    return f"{PROG} {__version__} (report schema {VERSION})"


def _help_parser(description: str, example: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "help": description.split(". ")[0].rstrip(".") + ".",
        "description": textwrap.fill(description, 88),
        "epilog": f"example:\n  {example}",
        "formatter_class": argparse.RawDescriptionHelpFormatter,
        **kwargs,
    }


def _add_input_flags(parser: argparse.ArgumentParser, *, inherited: bool) -> None:
    """Flags that decide how text is read and cut into chunks; ``score`` inherits them."""
    suffix = " (default: the reference's)" if inherited else ""
    parser.add_argument(
        "--window-words",
        type=int,
        default=None if inherited else DEFAULT_WINDOW_WORDS,
        metavar="N",
        help="split texts into ~N-word windows at paragraph breaks; 0 turns windowing off"
        + (suffix or f" (default: {DEFAULT_WINDOW_WORDS})"),
    )
    parser.add_argument(
        "--no-window",
        dest="window_words",
        action="store_const",
        const=0,
        default=argparse.SUPPRESS,
        help="profile each text whole (same as --window-words 0)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=None if inherited else 1,
        metavar="N",
        help="drop chunks with fewer prose words than this" + (suffix or " (default: 1)"),
    )
    parser.add_argument(
        "--text-field",
        metavar="FIELD",
        help="JSONL field holding the text"
        + (suffix or " (default: text, body_markdown, output, content, body)"),
    )
    parser.add_argument(
        "--no-syntax",
        action="store_true",
        help="skip the spaCy parser and its metrics",
    )


def _subparsers() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Measure a writer's style, then see how far a draft drifts from it.",
        epilog=(
            "Build a reference from the writer's texts once, then score drafts against it:\n"
            f"  {PROG} build posts/ --contrast llm-drafts/ -o writer.json\n"
            f"  {PROG} score draft.md writer.json\n\n"
            f"Run `{PROG} COMMAND --help` for a command's options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_version_text(),
        help="print the version and report schema, then exit",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    build = commands.add_parser(
        "build",
        **_help_parser(
            "Build a reference profile from a writer's texts.",
            f"{PROG} build posts/ --contrast llm-drafts/ -o writer.json",
            usage=f"{PROG} build [options] INPUT [INPUT ...] -o PROFILE.json",
        ),
    )
    build.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="the writer's Markdown, text or JSONL files, directories of them, or - for stdin",
    )
    build.add_argument(
        "-o", "--output", required=True, metavar="PROFILE.json", help="where to save the profile"
    )
    build.add_argument(
        "--contrast",
        action="append",
        metavar="PATH",
        help="text to contrast the writer with, such as LLM drafts of the same briefs; "
        "repeat for more paths",
    )
    build.add_argument(
        "--contrast-label",
        default="LLM",
        metavar="NAME",
        help="name of the contrast set in reports (default: LLM)",
    )
    _add_input_flags(build, inherited=False)
    build.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"distribution entries kept in the profile (default: {DEFAULT_TOP_K})",
    )
    build.add_argument("--all", action="store_true", help="show every metric, not just key ones")

    score_parser = commands.add_parser(
        "score",
        **_help_parser(
            "Score drafts against a reference profile. Window size, minimum words, text field "
            "and syntax come from the reference unless overridden.",
            f"{PROG} score draft.md writer.json",
            usage=f"{PROG} score [options] SAMPLE [SAMPLE ...] REFERENCE.json",
        ),
    )
    score_parser.add_argument(
        "paths",
        nargs="+",
        metavar="SAMPLE ... REFERENCE.json",
        help="the texts to score (files, directories, or - for stdin), then the reference "
        "profile last",
    )
    score_parser.add_argument(
        "-o", "--output", metavar="REPORT.json", help="also save the full report as JSON"
    )
    output = score_parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json", action="store_true", help="print the report as JSON on stdout, and nothing else"
    )
    output.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="print one verdict line and no notes",
    )
    score_parser.add_argument(
        "--all", action="store_true", help="show every metric, not just key ones"
    )
    _add_input_flags(score_parser, inherited=True)
    # The flat form took the reference as --reference; point anyone who reaches for it here.
    score_parser.add_argument("-r", "--reference", help=argparse.SUPPRESS)
    score_parser.set_defaults(top_k=None)

    show = commands.add_parser(
        "show",
        **_help_parser(
            "Show a saved reference profile or score report without recomputing it.",
            f"{PROG} show writer.json --all",
        ),
    )
    show.add_argument("report", metavar="REPORT.json", help="a report written by build or score")
    show.add_argument("--all", action="store_true", help="show every metric, not just key ones")

    metrics = commands.add_parser(
        "metrics",
        **_help_parser(
            "List every metric styleprofile measures, grouped by area.",
            f"{PROG} metrics --no-syntax",
        ),
    )
    metrics.add_argument(
        "--syntax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="only the spaCy syntax metrics (--syntax) or only the others (--no-syntax)",
    )
    return parser, dict(commands.choices)


def build_parser() -> argparse.ArgumentParser:
    return _subparsers()[0]


def _window_words(value: int) -> int | None:
    if value < 0:
        raise StyleProfileError("--window-words must be 0 (no windowing) or positive")
    return value or None


def _stdin_once(paths: Sequence[str]) -> None:
    if list(paths).count("-") > 1:
        raise StyleProfileError("- (stdin) can be given only once")


def _load(paths: Sequence[str], text_field: str | None, window_words: int | None) -> list[Chunk]:
    chunks = load_chunks(paths, text_field)
    return window(chunks, window_words) if window_words else chunks


def _thin_reference(report: dict[str, Any], window_words: int | None) -> list[str]:
    """Why a reference may be too small to trust, each with its fix."""
    documents = len({document_of(row["source"], row["id"]) for row in report["chunks"]})
    thin = []
    if documents < ENOUGH_DOCUMENTS:
        thin.append(
            f"it comes from {documents} document, so it has no held-out calibration and "
            "cannot learn a contrast; add more of the writer's documents"
        )
    if report["chunk_count"] < ENOUGH_CHUNKS:
        smaller = "a smaller --window-words" if window_words else "--window-words 500"
        thin.append(
            f"it has {report['chunk_count']} chunks; aim for {ENOUGH_CHUNKS} or more by adding "
            f"documents or using {smaller}"
        )
    if report["word_count"] < ENOUGH_WORDS:
        thin.append(
            f"it has {report['word_count']:,} words; aim for {ENOUGH_WORDS:,} or more of the "
            "writer's text, in one genre"
        )
    return thin


def _warn(text: str, color: bool) -> str:
    return f"\033[33m{text}\033[0m" if color else text


def _run_build(args: argparse.Namespace) -> int:
    window_words = _window_words(args.window_words)
    if args.top_k < 1:
        raise StyleProfileError("--top-k must be positive")
    _stdin_once([*args.inputs, *(args.contrast or [])])
    parser: Parser | None = None
    if not args.no_syntax:
        try:
            parser = load_parser()
        except SyntaxUnavailableError:
            _note(
                f"spaCy is not installed, so this profile has surface metrics only; for syntax "
                f"metrics, {SYNTAX_INSTALL} and build again"
            )
    chunks = _load(args.inputs, args.text_field, window_words)
    contrast = _load(args.contrast, args.text_field, window_words) if args.contrast else None
    report = build_reference(
        chunks,
        parser=parser,
        top_k=args.top_k,
        min_words=args.min_words,
        contrast=contrast,
        contrast_label=args.contrast_label,
        settings={
            "inputs": args.inputs,
            "text_field": args.text_field,
            "window_words": window_words,
            "min_words": args.min_words,
            "contrast": args.contrast,
        },
    )
    write_report(report, _path(args.output))
    color = _color()
    print(format_summary(report, color=color, full=args.all))
    thin = _thin_reference(report, window_words)
    if thin:
        print()
        for reason in thin:
            print(_warn(f"Thin reference: {reason}.", color))
    print(f"\nwrote {args.output}")
    print(f"Next, score a draft against it:\n  {PROG} score <draft> {shlex.quote(args.output)}")
    return 0


def _split_score_paths(args: argparse.Namespace) -> tuple[list[str], str]:
    usage = f"{PROG} score DRAFT [DRAFT ...] REFERENCE.json"
    if args.reference:
        raise StyleProfileError(
            f"score takes the reference profile as its last argument, not --reference: {usage}"
        )
    if len(args.paths) < 2:
        raise StyleProfileError(f"score needs a sample and then a reference profile: {usage}")
    *samples, reference = args.paths
    if reference == "-":
        raise StyleProfileError(
            f"the reference profile (the last argument) must be a file: {usage}"
        )
    if not _path(reference).is_file():
        raise StyleProfileError(f"reference profile {reference} not found; it goes last: {usage}")
    return samples, reference


def _load_score_reference(samples: Sequence[str], reference_arg: str) -> dict[str, Any]:
    try:
        return load_reference(Path(reference_arg).expanduser())
    except StyleProfileError as error:
        if error.code is None and any(Path(path).suffix.lower() == ".json" for path in samples):
            raise StyleProfileError(
                f"{reference_arg} is not a style profile; the reference profile goes last: "
                f"{PROG} score DRAFT [DRAFT ...] REFERENCE.json"
            ) from error
        raise


def _headline(report: dict[str, Any], reference: dict[str, Any], samples: Sequence[str]) -> str:
    """One line with the same verdict words as the full comparison view."""
    name = samples[0] if len(samples) == 1 else f"{len(samples)} inputs"
    scored = report["reference"]
    delta = scored["delta_mean"]
    if delta is None:
        return f"{name}: no metrics could be compared with the reference"
    held = (reference.get("calibration") or {}).get("delta") or {}
    ceiling = mean_ceiling(held, report["chunk_count"])
    parts = [f"{name}: {describe_delta(delta, ceiling)} (Delta {delta:.2f})"]
    contrast = reference.get("contrast")
    likeness = scored.get("likeness_mean")
    if contrast and likeness is not None:
        level = likeness_level(likeness, contrast["calibration"], report["chunk_count"])
        words = likeness_words(level, contrast["label"])
        parts.append(f"{contrast['label']}-likeness {words} ({likeness:.2f})")
    return "; ".join(parts)


def _run_score(args: argparse.Namespace) -> int:
    samples, reference_arg = _split_score_paths(args)
    reference_path = _path(reference_arg)
    if args.output and _path(args.output) == reference_path:
        raise StyleProfileError("--output is the reference file; choose another output path")
    _stdin_once(samples)
    note: Callable[[str], None] = (lambda _: None) if args.quiet else _note
    reference = _load_score_reference(samples, reference_arg)

    inherited = reference.get("settings") or {}
    reference_window = inherited.get("window_words")
    reference_min_words = inherited.get("min_words") or 1
    reference_syntax = inherited.get("syntax") is not None
    window_words = (
        reference_window if args.window_words is None else _window_words(args.window_words)
    )
    min_words = reference_min_words if args.min_words is None else args.min_words
    text_field = inherited.get("text_field") if args.text_field is None else args.text_field
    changed = []
    if window_words != reference_window:
        changed.append(f"window words {reference_window or 'off'} -> {window_words or 'off'}")
    if min_words != reference_min_words:
        changed.append(f"min words {reference_min_words} -> {min_words}")
    if args.no_syntax and reference_syntax:
        changed.append("syntax on -> off")
    if changed:
        note(
            f"overriding the reference's settings ({'; '.join(changed)}); results may not be "
            "comparable with the reference"
        )
    parser: Parser | None = None
    if reference_syntax and not args.no_syntax:
        try:
            parser = load_parser()
        except SyntaxUnavailableError:
            note(
                "the reference has syntax metrics but spaCy is not installed, so syntax is left "
                f"out of this score; {SYNTAX_INSTALL} to include it"
            )

    report = score(
        _load(samples, text_field, window_words),
        reference,
        parser=parser,
        top_k=args.top_k or inherited.get("top_k") or DEFAULT_TOP_K,
        min_words=min_words,
        reference_path=reference_path,
        settings={
            "inputs": samples,
            "text_field": text_field,
            "window_words": window_words,
            "min_words": min_words,
        },
    )
    if args.output:
        write_report(report, _path(args.output))
    if args.json:
        sys.stdout.write(dumps_report(report))
        if args.output:
            note(f"wrote {args.output}")
    elif args.quiet:
        print(_headline(report, reference, samples))
    else:
        print(format_summary(report, reference, color=_color(), full=args.all))
        if args.output:
            print(f"\nwrote {args.output}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    report = load_report(Path(args.report).expanduser())
    color = _color()
    if report_kind(report) == REFERENCE:
        print(format_summary(report, color=color, full=args.all))
        return 0
    baseline = report["reference"].get("baseline")
    if baseline is None:
        # Score reports from before baselines were saved: fall back to the reference file.
        saved = report["reference"].get("path")
        if not saved or not Path(saved).is_file():
            raise StyleProfileError(
                f"{args.report} was scored before reports kept a copy of their reference, and "
                f"its reference ({saved or 'unknown'}) is not available; score the sample again"
            )
        _note(f"this report has no saved copy of its reference; showing it against {saved}")
        baseline = load_reference(Path(saved))
    print(format_summary(report, baseline, color=color, full=args.all))
    return 0


def _run_metrics(args: argparse.Namespace) -> int:
    rows = describe(syntax=args.syntax is not False)
    if args.syntax:
        surface = set(describe(syntax=False))
        rows = [row for row in rows if row not in surface]
    color = _color()
    width = max(len(row.label) for row in rows) + 2
    unit_width = max(len(row.unit) for row in rows) + 2
    indent = " " * (2 + width + unit_width)
    columns = shutil.get_terminal_size().columns if sys.stdout.isatty() else 0
    group = None
    print(f"{len(rows)} metrics. Syntax and sentence-opening metrics need spaCy.")
    for row in rows:
        if row.group != group:
            group = row.group
            print(f"\n\033[1m{group}\033[0m" if color else f"\n{group}")
        line = f"  {row.label:{width}}{row.unit:{unit_width}}{row.about}"
        if columns > len(indent) + 20:
            line = textwrap.fill(line, columns, subsequent_indent=indent)
        print(line)
    return 0


RUNNERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build": _run_build,
    "score": _run_score,
    "show": _run_show,
    "metrics": _run_metrics,
}


def _legacy_parser() -> argparse.ArgumentParser:
    """The flat form of earlier releases. Defaults are None so explicit flags can be told
    apart from omitted ones when writing the equivalent new command."""
    parser = argparse.ArgumentParser(
        prog=PROG, description="Deprecated flat form; see `styleprofile --help`."
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field")
    parser.add_argument("--window-words", type=int)
    parser.add_argument("--min-words", type=int)
    parser.add_argument("--reference")
    parser.add_argument("--no-syntax", action="store_true")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--contrast", nargs="+")
    parser.add_argument("--contrast-label")
    parser.add_argument("--all", action="store_true")
    return parser


def _legacy(argv: Sequence[str]) -> int:
    old = _legacy_parser().parse_args(argv)
    if old.window_words is not None and old.window_words < 1:
        raise StyleProfileError("--window-words must be positive")
    if old.reference and old.contrast:
        raise StyleProfileError(
            "--contrast builds a reference; score samples against it in a separate run"
        )
    flags: list[str] = []
    for flag, given in (
        ("--min-words", old.min_words),
        ("--text-field", old.text_field),
    ):
        if given is not None:
            flags += [flag, str(given)]
    if old.no_syntax:
        flags.append("--no-syntax")
    if old.all:
        flags.append("--all")
    shared = {
        "output": old.output,
        "text_field": old.text_field,
        "no_syntax": old.no_syntax,
        "all": old.all,
    }
    if old.reference:
        if old.window_words is not None:
            flags += ["--window-words", str(old.window_words)]
        command = [PROG, "score", *old.inputs, old.reference, "-o", old.output, *flags]
        args = argparse.Namespace(
            **shared,
            paths=[*old.inputs, old.reference],
            reference=None,
            json=False,
            quiet=False,
            window_words=old.window_words,
            min_words=old.min_words,
            top_k=old.top_k,
        )
        runner = _run_score
    else:
        # The flat form did not window unless asked; build does by default.
        flags += ["--window-words", str(old.window_words)] if old.window_words else ["--no-window"]
        for path in old.contrast or []:
            flags += ["--contrast", path]
        if old.contrast_label is not None:
            flags += ["--contrast-label", old.contrast_label]
        if old.top_k is not None:
            flags += ["--top-k", str(old.top_k)]
        command = [PROG, "build", *old.inputs, "-o", old.output, *flags]
        args = argparse.Namespace(
            **shared,
            inputs=old.inputs,
            contrast=old.contrast,
            contrast_label=old.contrast_label or "LLM",
            window_words=old.window_words or 0,
            min_words=1 if old.min_words is None else old.min_words,
            top_k=DEFAULT_TOP_K if old.top_k is None else old.top_k,
        )
        runner = _run_build
    print(
        "note: this form of the command is deprecated and will be removed in the next release; "
        f"use: {shlex.join(command)}",
        file=sys.stderr,
    )
    return runner(args)


def _is_output_flag(arg: str) -> bool:
    """--output, or an abbreviation argparse accepted for it in the flat form (--out)."""
    name = arg.split("=", 1)[0]
    return len(name) >= len("--o") and "--output".startswith(name)


def _is_legacy(argv: Sequence[str]) -> bool:
    """The flat form always had --output and never starts with a command name."""
    return bool(argv) and argv[0] not in COMMANDS and any(_is_output_flag(arg) for arg in argv)


def _dispatch(argv: Sequence[str]) -> int:
    if _is_legacy(argv):
        return _legacy(argv)
    parser, commands = _subparsers()
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    if argv[0] not in commands:
        parser.parse_args(argv)  # --help, --version, or an error naming the valid commands
        return 2
    # Intermixed parsing lets options sit anywhere, e.g. `score draft.md -o out.json ref.json`.
    args = commands[argv[0]].parse_intermixed_args(argv[1:])
    return RUNNERS[argv[0]](args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _dispatch(sys.argv[1:] if argv is None else list(argv))
    except (StyleProfileError, SyntaxUnavailableError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        hint = HINTS.get(getattr(error, "code", None) or "")
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
