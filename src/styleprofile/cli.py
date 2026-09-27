"""Command-line entry point: profile prose, optionally scored against a saved reference."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from styleprofile.display import format_evaluation, format_summary
from styleprofile.profile import (
    StyleProfileError,
    build_profile,
    load_chunks,
    load_reference,
    window,
    write_report,
)
from styleprofile.stress import evaluate_rewording
from styleprofile.syntax import SyntaxUnavailableError, load_parser

EVALUATE = "evaluate"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styleprofile",
        description="Stylometric metrics for any prose, optionally scored against a saved profile",
        epilog=f"Run 'styleprofile {EVALUATE} --help' to stress-test LLM-likeness against "
        "edited copies of the contrast drafts.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL files, Markdown/text files, directories of them, or - for stdin",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
    _add_text_options(parser)
    parser.add_argument(
        "--reference", help="A previous styleprofile report to compute z-scores and Delta against"
    )
    parser.add_argument(
        "--top-k", type=int, default=300, help="Distribution entries kept in the report"
    )
    parser.add_argument(
        "--contrast",
        nargs="+",
        help="When building a reference: text to contrast it with (e.g. LLM drafts), used to "
        "learn which metrics separate the two and to score likeness to it",
    )
    parser.add_argument(
        "--contrast-label", default="LLM", help="Name of the contrast set in reports"
    )
    parser.add_argument(
        "--all", action="store_true", help="Show every metric in the terminal, not just key ones"
    )
    return parser


def _add_text_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--text-field", help="JSONL field holding the text (default: text, body_markdown, ...)"
    )
    parser.add_argument(
        "--window-words",
        type=int,
        help="Split inputs into ~N-word windows at paragraph breaks (500 is a good default)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Drop chunks with fewer prose words than this after windowing",
    )
    parser.add_argument(
        "--no-syntax", action="store_true", help="Skip the spaCy parser and its metrics"
    )


def _edited(value: str) -> tuple[str, str]:
    label, separator, folder = value.partition("=")
    if not separator or not label or not folder:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {value!r}")
    return label, folder


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"styleprofile {EVALUATE}",
        description="Rewording stress test: build a reference with the original LLM drafts as "
        "contrast, then score edited copies of those drafts (matched by file name) with the "
        "leave-one-draft-out weights of their originals",
    )
    parser.add_argument("--reference-inputs", nargs="+", required=True, help="The writer's corpus")
    parser.add_argument(
        "--contrast", nargs="+", required=True, help="The original, unedited LLM drafts"
    )
    parser.add_argument(
        "--edited",
        type=_edited,
        action="extend",
        nargs="+",
        required=True,
        metavar="LABEL=DIR",
        help="Folders of edited drafts with the originals' file names, "
        "e.g. light=data/light humanize=data/humanize",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument(
        "--contrast-label", default="LLM", help="Name of the contrast set in reports"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Also report the cross-validated AUC with the edited drafts added to the contrast set",
    )
    _add_text_options(parser)
    return parser


def _evaluate(args: argparse.Namespace) -> int:
    if args.window_words is not None and args.window_words < 1:
        raise StyleProfileError("--window-words must be positive")
    labels = [label for label, _ in args.edited]
    if len(set(labels)) != len(labels):
        raise StyleProfileError("each --edited LABEL must be distinct")
    if any(folder == "-" for _, folder in args.edited):
        raise StyleProfileError("--edited takes folders of files, not - (stdin)")
    if [*args.reference_inputs, *args.contrast].count("-") > 1:
        raise StyleProfileError("- (stdin) can be given only once")
    inputs = [*args.reference_inputs, *args.contrast, *(folder for _, folder in args.edited)]
    if _path(args.output) in {_path(value) for value in inputs if value != "-"}:
        raise StyleProfileError("--output is one of the inputs; choose another output path")
    reference = load_chunks(args.reference_inputs, args.text_field)
    contrast = load_chunks(args.contrast, args.text_field)
    edited = {label: load_chunks([folder], args.text_field) for label, folder in args.edited}
    if args.window_words:
        reference = window(reference, args.window_words)
        contrast = window(contrast, args.window_words)
        edited = {label: window(chunks, args.window_words) for label, chunks in edited.items()}
    result = evaluate_rewording(
        reference,
        contrast,
        edited,
        parser=None if args.no_syntax else load_parser(),
        min_words=args.min_words,
        contrast_label=args.contrast_label,
        retrain=args.retrain,
        settings={
            "reference_inputs": args.reference_inputs,
            "contrast": args.contrast,
            "edited": dict(args.edited),
            "text_field": args.text_field,
            "window_words": args.window_words,
            "syntax": not args.no_syntax,
        },
    )
    write_report(result, _path(args.output))
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(format_evaluation(result, color=color))
    print(f"\nwrote stress test to {_path(args.output)}")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.window_words is not None and args.window_words < 1:
        raise StyleProfileError("--window-words must be positive")
    if args.top_k < 1:
        raise StyleProfileError("--top-k must be positive")
    if [*args.inputs, *(args.contrast or [])].count("-") > 1:
        raise StyleProfileError("- (stdin) can be given only once")
    if args.reference and _path(args.reference) == _path(args.output):
        raise StyleProfileError("--output is the --reference file; choose another output path")
    chunks = load_chunks(args.inputs, args.text_field)
    contrast = load_chunks(args.contrast, args.text_field) if args.contrast else None
    if args.window_words:
        chunks = window(chunks, args.window_words)
        contrast = window(contrast, args.window_words) if contrast else contrast
    reference_path = _path(args.reference) if args.reference else None
    reference = load_reference(reference_path) if reference_path else None
    report = build_profile(
        chunks,
        parser=None if args.no_syntax else load_parser(),
        top_k=args.top_k,
        min_words=args.min_words,
        reference=reference,
        contrast=contrast,
        contrast_label=args.contrast_label,
        reference_path=reference_path,
        settings={
            "inputs": args.inputs,
            "text_field": args.text_field,
            "window_words": args.window_words,
            "min_words": args.min_words,
            "contrast": args.contrast,
        },
    )
    write_report(report, _path(args.output))
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(format_summary(report, reference, color=color, full=args.all))
    print(f"\nwrote style profile to {_path(args.output)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        # A subcommand only by its first word, so every existing invocation still profiles
        # its inputs; a file actually named "evaluate" can be passed as ./evaluate.
        if arguments[:1] == [EVALUATE]:
            return _evaluate(build_evaluate_parser().parse_args(arguments[1:]))
        return _run(build_parser().parse_args(arguments))
    except (StyleProfileError, SyntaxUnavailableError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
