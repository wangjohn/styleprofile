"""Command-line entry point: build a writer's reference profile once, then score drafts.

    styleprofile build posts/ --contrast llm-drafts/ -o writer.json
    styleprofile score draft.md writer.json
    styleprofile show writer.json
    styleprofile metrics
    styleprofile evaluate --reference-inputs posts/ --contrast llm-drafts/ --edited light=edits/

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
from typing import Any, TextIO

from styleprofile import __version__
from styleprofile.display import (
    describe_delta,
    format_evaluation,
    format_reference_summary,
    format_summary,
    likeness_level,
    likeness_words,
    mean_ceiling,
)
from styleprofile.metrics import describe
from styleprofile.profile import (
    REFERENCE,
    TEXT_FIELDS,
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
from styleprofile.stress import evaluate_rewording
from styleprofile.syntax import Parser, SyntaxUnavailableError, load_parser

PROG = "styleprofile"
COMMANDS = ("build", "score", "show", "metrics", "evaluate")
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
    "unmatched_edits": "give each edited file its original's name and relative path",
    "duplicate_names": "give each draft a distinct file name or JSONL id",
}
SYNTAX_INSTALL = "pip install 'styleprofile[syntax]'"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _color(stream: TextIO | None = None) -> bool:
    """NO_COLOR turns color off and FORCE_COLOR on (NO_COLOR wins); else color a terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR", "0") not in ("", "0"):
        return True
    return (stream or sys.stdout).isatty()


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
    evaluate = commands.add_parser(
        "evaluate",
        **_help_parser(
            "Stress-test LLM-likeness against edited drafts. Builds a reference with the "
            "original drafts as contrast, then scores edited copies of those drafts (matched "
            "to their originals by file name) with the weights learned without their original.",
            f"{PROG} evaluate --reference-inputs posts/ --contrast llm-drafts/ "
            "--edited light=edits/light humanize=edits/humanize",
        ),
    )
    evaluate.add_argument(
        "--reference-inputs",
        nargs="+",
        required=True,
        metavar="INPUT",
        help="the writer's Markdown, text or JSONL files, directories of them, or - for stdin",
    )
    evaluate.add_argument(
        "--contrast",
        action="append",
        required=True,
        metavar="PATH",
        help="the original, unedited drafts; repeat for more paths",
    )
    evaluate.add_argument(
        "--edited",
        type=_edited,
        action="extend",
        nargs="+",
        required=True,
        metavar="LABEL=DIR",
        help="folders of edited drafts with the originals' file names, e.g. light=edits/light",
    )
    evaluate.add_argument(
        "--retrain",
        action="store_true",
        help="also report the AUC with the edited drafts added to the contrast set",
    )
    evaluate.add_argument(
        "--contrast-label",
        default="LLM",
        metavar="NAME",
        help="name of the contrast set in reports (default: LLM)",
    )
    evaluate.add_argument(
        "-o", "--output", metavar="REPORT.json", help="also save the full report as JSON"
    )
    evaluate.add_argument(
        "--json", action="store_true", help="print the report as JSON on stdout, and nothing else"
    )
    _add_input_flags(evaluate, inherited=False)
    return parser, dict(commands.choices)


def _edited(value: str) -> tuple[str, str]:
    label, separator, folder = value.partition("=")
    if not separator or not label or not folder:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {value!r}")
    return label, folder


def build_parser() -> argparse.ArgumentParser:
    return _subparsers()[0]


def _window_words(value: int) -> int | None:
    if value < 0:
        raise StyleProfileError("--window-words must be 0 (no windowing) or positive")
    return value or None


def _min_words(value: int) -> int:
    if value < 0:
        raise StyleProfileError("--min-words must be 0 or more")
    return value


def _stdin_once(paths: Sequence[str]) -> None:
    if list(paths).count("-") > 1:
        raise StyleProfileError("- (stdin) can be given only once")


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def _read(
    paths: Sequence[str], text_field: str | Sequence[str] | None, seen: set[str]
) -> list[Chunk]:
    """Load each path, skipping files an earlier path (tracked in ``seen``) already gave."""
    chunks: list[Chunk] = []
    for value in paths:
        loaded = load_chunks([value], text_field)
        sources = {chunk.source for chunk in loaded if value != "-"}
        repeated = sources & seen
        if repeated and repeated == sources:
            _note(f"{value} was already given; using it once")
        elif repeated:
            _note(f"skipping {_plural(len(repeated), 'file')} in {value} already given")
        chunks += [chunk for chunk in loaded if chunk.source not in repeated]
        seen |= sources
    return chunks


def _windowed(chunks: list[Chunk], window_words: int | None) -> list[Chunk]:
    return window(chunks, window_words) if window_words else chunks


def _refuse_overwrite(output: str, typed: Sequence[str], chunks: Sequence[Chunk]) -> None:
    """Refuse an output path that is one of the input files, typed or inside a directory."""
    target = _path(output)
    for value in typed:
        if value != "-" and _path(value) == target:
            raise StyleProfileError(f"--output would overwrite input {value}; choose another path")
    for chunk in chunks:
        if chunk.source != "stdin" and Path(chunk.source) == target:
            raise StyleProfileError(
                f"--output would overwrite input {chunk.source}; choose another path"
            )


def _thin_reference(report: dict[str, Any], window_words: int | None) -> list[str]:
    """Why a reference may be too small to trust, each with its fix."""
    documents = len({document_of(row["source"], row["id"]) for row in report["chunks"]})
    thin = []
    if documents < ENOUGH_DOCUMENTS:
        thin.append(
            f"it comes from {_plural(documents, 'document')}, so it has no held-out calibration "
            "and cannot learn a contrast; add more of the writer's documents"
        )
    if report["chunk_count"] < ENOUGH_CHUNKS:
        smaller = "a smaller --window-words" if window_words else "--window-words 500"
        thin.append(
            f"it has {_plural(report['chunk_count'], 'chunk')}; aim for {ENOUGH_CHUNKS} or more "
            f"by adding documents or using {smaller}"
        )
    if report["word_count"] < ENOUGH_WORDS:
        thin.append(
            f"it has {_plural(report['word_count'], 'word')}; aim for {ENOUGH_WORDS:,} or more "
            "of the writer's text, in one genre"
        )
    return thin


def _warn(text: str, color: bool) -> str:
    return f"\033[33m{text}\033[0m" if color else text


def _run_build(args: argparse.Namespace) -> int:
    window_words = _window_words(args.window_words)
    min_words = _min_words(args.min_words)
    if args.top_k < 1:
        raise StyleProfileError("--top-k must be positive")
    typed = [*args.inputs, *(args.contrast or [])]
    _stdin_once(typed)
    seen: set[str] = set()
    chunks = _read(args.inputs, args.text_field, seen)
    contrast = _read(args.contrast, args.text_field, seen) if args.contrast else None
    _refuse_overwrite(args.output, typed, [*chunks, *(contrast or [])])
    parser: Parser | None = None
    if not args.no_syntax:
        try:
            parser = load_parser()
        except SyntaxUnavailableError:
            _note(
                f"spaCy is not installed, so this profile has surface metrics only; for syntax "
                f"metrics, {SYNTAX_INSTALL} and build again"
            )
    report = build_reference(
        _windowed(chunks, window_words),
        parser=parser,
        top_k=args.top_k,
        min_words=min_words,
        contrast=_windowed(contrast, window_words) if contrast is not None else None,
        contrast_label=args.contrast_label,
        settings={
            "inputs": args.inputs,
            "text_field": args.text_field,
            "window_words": window_words,
            "min_words": min_words,
            "contrast": args.contrast,
        },
    )
    write_report(report, _path(args.output))
    print(format_reference_summary(report, color=_color(), full=args.all))
    sys.stdout.flush()  # keep the warnings after the summary when both go to one pipe
    color_err = _color(sys.stderr)
    for reason in _thin_reference(report, window_words):
        print(_warn(f"Thin reference: {reason}.", color_err), file=sys.stderr)
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
    if _path(reference).is_dir():
        raise StyleProfileError(
            f"{reference} is a directory, not a profile; the reference profile goes last: {usage}"
        )
    if not _path(reference).is_file():
        raise StyleProfileError(f"reference profile {reference} not found; it goes last: {usage}")
    return samples, reference


def _load_score_reference(reference_arg: str) -> dict[str, Any]:
    try:
        return load_reference(Path(reference_arg).expanduser())
    except StyleProfileError as error:
        if error.code == "not_a_profile":
            raise StyleProfileError(
                f"{reference_arg} is not a style profile; the reference profile goes last: "
                f"{PROG} score DRAFT [DRAFT ...] REFERENCE.json"
            ) from error
        raise


def _headline(report: dict[str, Any], reference: dict[str, Any], samples: Sequence[str]) -> str:
    """One line with the same verdict words as the full comparison view."""
    if len(samples) == 1:
        name = "stdin" if samples[0] == "-" else samples[0]
    else:
        name = f"{len(samples)} inputs"
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


def _score_fields(explicit: str | None, inherited: str | None) -> str | tuple[str, ...] | None:
    """An explicit --text-field is the only field read; the reference's is tried first."""
    if explicit:
        return explicit
    if inherited:
        return (inherited, *(name for name in TEXT_FIELDS if name != inherited))
    return None


def _run_score(args: argparse.Namespace) -> int:
    samples, reference_arg = _split_score_paths(args)
    reference_path = _path(reference_arg)
    if args.output and _path(args.output) == reference_path:
        raise StyleProfileError("--output is the reference file; choose another output path")
    _stdin_once(samples)
    reference = _load_score_reference(reference_arg)

    inherited = reference.get("settings") or {}
    reference_window = inherited.get("window_words")
    reference_min_words = inherited.get("min_words") or 1
    reference_syntax = inherited.get("syntax") is not None
    window_words = (
        reference_window if args.window_words is None else _window_words(args.window_words)
    )
    min_words = reference_min_words if args.min_words is None else _min_words(args.min_words)
    text_field = args.text_field or inherited.get("text_field")
    # Window and syntax overrides are warned about in the report itself.
    if min_words != reference_min_words:
        _note(f"--min-words {min_words} overrides the reference's {reference_min_words}")
    parser: Parser | None = None
    if reference_syntax and not args.no_syntax:
        try:
            parser = load_parser()
        except SyntaxUnavailableError:
            _note(
                "the reference has syntax metrics but spaCy is not installed, so syntax is left "
                f"out of this score; {SYNTAX_INSTALL} to include it"
            )

    chunks = _read(samples, _score_fields(args.text_field, inherited.get("text_field")), set())
    if args.output:
        _refuse_overwrite(args.output, samples, chunks)
    report = score(
        _windowed(chunks, window_words),
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
            _note(f"wrote {args.output}")
    elif args.quiet:
        print(_headline(report, reference, samples), flush=True)
        if report["warnings"]:
            _note(f"{_plural(len(report['warnings']), 'warning')}; run without -q to see them")
    else:
        print(format_summary(report, reference, color=_color(), full=args.all))
        if args.output:
            print(f"\nwrote {args.output}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    try:
        report = load_report(Path(args.report).expanduser())
    except StyleProfileError as error:
        # Name the path as typed: Path() drops a trailing slash or a leading ./
        problems = {
            "not_found": "not found",
            "directory": "is a directory, not a profile",
            "not_a_profile": "is not a style profile",
        }
        if error.code in problems:
            raise StyleProfileError(f"{args.report} {problems[error.code]}") from error
        raise
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
    spacy = "" if args.syntax is False else " Syntax and sentence-opening metrics need spaCy."
    print(f"{len(rows)} metrics.{spacy}")
    for row in rows:
        if row.group != group:
            group = row.group
            print(f"\n\033[1m{group}\033[0m" if color else f"\n{group}")
        line = f"  {row.label:{width}}{row.unit:{unit_width}}{row.about}"
        if columns > len(indent) + 20:
            line = textwrap.fill(line, columns, subsequent_indent=indent)
        print(line)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    window_words = _window_words(args.window_words)
    min_words = _min_words(args.min_words)
    labels = [label for label, _ in args.edited]
    if len(set(labels)) != len(labels):
        raise StyleProfileError("each --edited LABEL must be distinct")
    folders = [folder for _, folder in args.edited]
    if "-" in folders:
        raise StyleProfileError("--edited takes folders of files, not - (stdin)")
    typed = [*args.reference_inputs, *args.contrast]
    _stdin_once(typed)
    seen: set[str] = set()
    reference = _read(args.reference_inputs, args.text_field, seen)
    contrast = _read(args.contrast, args.text_field, seen)
    edited = {label: load_chunks([folder], args.text_field) for label, folder in args.edited}
    if args.output:
        loaded = [*reference, *contrast, *(chunk for chunks in edited.values() for chunk in chunks)]
        _refuse_overwrite(args.output, [*typed, *folders], loaded)
    parser: Parser | None = None
    if not args.no_syntax:
        try:
            parser = load_parser()
        except SyntaxUnavailableError:
            _note(
                f"spaCy is not installed, so this uses surface metrics only; for syntax "
                f"metrics, {SYNTAX_INSTALL} and run again"
            )
    result = evaluate_rewording(
        _windowed(reference, window_words),
        _windowed(contrast, window_words),
        {label: _windowed(chunks, window_words) for label, chunks in edited.items()},
        parser=parser,
        min_words=min_words,
        contrast_label=args.contrast_label,
        retrain=args.retrain,
        settings={
            "reference_inputs": args.reference_inputs,
            "contrast": args.contrast,
            "edited": dict(args.edited),
            "text_field": args.text_field,
            "window_words": window_words,
        },
    )
    if args.output:
        write_report(result, _path(args.output))
    if args.json:
        sys.stdout.write(dumps_report(result))
        if args.output:
            _note(f"wrote {args.output}")
    else:
        print(format_evaluation(result, color=_color()))
        if args.output:
            print(f"\nwrote {args.output}")
    return 0


RUNNERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build": _run_build,
    "score": _run_score,
    "show": _run_show,
    "metrics": _run_metrics,
    "evaluate": _run_evaluate,
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
    change = ""
    if old.reference and old.window_words is None:
        change = (
            "; unlike before, the samples are split into windows the same size as the "
            "reference's (pass --no-window to turn that off)"
        )
    print(
        "note: this form of the command is deprecated and will be removed in the next release; "
        f"use: {shlex.join(command)}{change}",
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
        if not argv[0].startswith("-") and Path(argv[0]).expanduser().exists():
            parser.print_usage(sys.stderr)
            rest = shlex.join(argv)
            print(
                f"{PROG}: error: {argv[0]} is not a command; did you mean "
                f"`{PROG} build {rest}` or `{PROG} score {rest}`?",
                file=sys.stderr,
            )
            return 2
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
