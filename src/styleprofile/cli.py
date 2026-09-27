"""Command-line entry point: profile prose, optionally scored against a saved reference."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from styleprofile.display import format_summary
from styleprofile.profile import (
    StyleProfileError,
    build_profile,
    load_chunks,
    load_reference,
    window,
    write_report,
)
from styleprofile.syntax import SyntaxUnavailableError, load_parser


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="styleprofile",
        description="Stylometric metrics for any prose, optionally scored against a saved profile",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL files, Markdown/text files, directories of them, or - for stdin",
    )
    parser.add_argument("--output", required=True, help="JSON report path")
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
        "--reference", help="A previous styleprofile report to compute z-scores and Delta against"
    )
    parser.add_argument(
        "--no-syntax", action="store_true", help="Skip the spaCy parser and its metrics"
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


def _run(args: argparse.Namespace) -> int:
    if args.window_words is not None and args.window_words < 1:
        raise StyleProfileError("--window-words must be positive")
    if args.top_k < 1:
        raise StyleProfileError("--top-k must be positive")
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
    try:
        return _run(build_parser().parse_args(argv))
    except (StyleProfileError, SyntaxUnavailableError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
