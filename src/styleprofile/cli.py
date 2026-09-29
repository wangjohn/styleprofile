"""Command-line entry point: build a writer's reference profile once, then score drafts.

Commands:

    styleprofile build posts/ --contrast llm-drafts/ -o writer.json
    styleprofile score draft.md writer.json
    styleprofile show writer.json
    styleprofile metrics
    styleprofile evaluate posts/ --contrast llm-drafts/ --edited light=edits/
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from styleprofile import __version__, api
from styleprofile.api import (
    AUTO,
    DEFAULT_TOP_K,
    DEFAULT_WINDOW_WORDS,
    INPUT_FORMATS,
    Profile,
    ScoreResult,
    Settings,
    SettingsOverrides,
)
from styleprofile.calibration import too_short_text
from styleprofile.core import Note, NoteCode, StyleProfileError
from styleprofile.display import format_evaluation, format_summary
from styleprofile.metrics import describe
from styleprofile.profile import (
    EVALUATION,
    REFERENCE,
    VERSION,
    dumps_report,
    expand_path,
    load_report,
)

PROG = "styleprofile"
# Advice for library errors, which name the problem but never a flag.
HINTS = {
    "text_field": "pass --text-field with the JSONL field that holds the text",
    "contrast_needs_documents": "add more of the writer's documents, or build without --contrast",
    "score_as_reference": "score against the reference profile made by `styleprofile build`",
    "unmatched_edits": "give each edited file its original's name and relative path",
    "duplicate_names": "give each draft a distinct file name or JSONL id",
    "forced_jsonl": "leave out --input-format jsonl to read each file by its extension",
}
# Advice for notes, by ``Note.code``.
NOTE_HINTS = {
    NoteCode.READ_AS_HTML: "pass --input-format markdown to read it as written",
    NoteCode.READ_AS_HTML_IN_FOLDER: (
        "if any are really Markdown, give them separately with --input-format markdown, which "
        "would also apply to .html files in the folder"
    ),
    NoteCode.READ_AS_JSONL: "pass --input-format markdown to read it as prose",
}
# Library errors and notes name a setting (``setting``) as a whole word; the CLI prints the
# flag that sets it instead.
FLAGS = {
    "window_words": "--window-words",
    "min_words": "--min-words",
    "top_k": "--top-k",
}


def _path(value: str) -> Path:
    return expand_path(value).resolve()


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
        "--input-format",
        choices=INPUT_FORMATS,
        default=AUTO,
        help="how to read inputs: auto picks each file's format from its extension and "
        "content, and reads stdin as JSONL when every line is a JSON object; the others "
        "read every input that way, directory contents included (default: auto"
        + (", not the reference's: drafts are often in another format)" if inherited else ")"),
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
        help="the writer's Markdown, text, HTML or JSONL files, folders of them, or - for stdin",
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
    build.add_argument(
        "--keep-chunks",
        action="store_true",
        help="also save every chunk's metrics in the profile, for debugging (much larger)",
    )

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
    score_parser.set_defaults(top_k=None)

    show = commands.add_parser(
        "show",
        **_help_parser(
            "Show a saved reference profile, score report or evaluation report without "
            "recomputing it.",
            f"{PROG} show writer.json --all",
        ),
    )
    show.add_argument(
        "report", metavar="REPORT.json", help="a report written by build, score or evaluate"
    )
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
            f"{PROG} evaluate posts/ --contrast llm-drafts/ "
            "--edited light=edits/light humanize=edits/humanize",
            usage=f"{PROG} evaluate [options] INPUT [INPUT ...] --contrast PATH "
            "--edited LABEL=DIR [LABEL=DIR ...]",
        ),
    )
    evaluate.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="the writer's Markdown, text, HTML or JSONL files, folders of them, or - for stdin",
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
        raise argparse.ArgumentTypeError(
            f"expected LABEL=DIR, got {value!r}; put the writer's INPUTs before --edited"
        )
    return label, folder


def build_parser() -> argparse.ArgumentParser:
    return _subparsers()[0]


def _flagged(message: str, setting: str | None) -> str:
    """A library message naming a setting, with the flag that sets it in its place."""
    flag = FLAGS.get(setting or "")
    if not flag or not setting:
        return message
    return re.sub(rf"\b{re.escape(setting)}\b", flag, message)


def _notes(notes: Sequence[Note]) -> None:
    """Print notes as ``note:`` lines, except thin-reference notes, which ``build`` prints
    as warnings after its summary."""
    for note in notes:
        if note.code != NoteCode.THIN_REFERENCE:
            hint = NOTE_HINTS.get(note.code)
            message = _flagged(note.message, note.setting)
            _note(f"{message} ({hint})" if hint else message)


def _inputs_exist(values: Sequence[str]) -> None:
    """Refuse a typed input that does not exist, naming it as typed (never as text)."""
    for value in values:
        api.require_path(value, suggest_text=False)


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def _refuse_overwrite(output: str, typed: Sequence[str], sources: Sequence[str] = ()) -> None:
    """Refuse an output path that is one of the input files, typed or inside a directory."""
    target = _path(output)
    for value in typed:
        if value != "-" and _path(value) == target:
            raise StyleProfileError(f"--output would overwrite input {value}; choose another path")
    for source in sources:
        if source != "stdin" and Path(source) == target:
            raise StyleProfileError(f"--output would overwrite input {source}; choose another path")


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        window_words=args.window_words,
        min_words=args.min_words,
        text_field=args.text_field,
        syntax=False if args.no_syntax else AUTO,
        top_k=getattr(args, "top_k", DEFAULT_TOP_K),
        input_format=args.input_format,
    )


def _warn(text: str, color: bool) -> str:
    return f"\033[33m{text}\033[0m" if color else text


def _run_build(args: argparse.Namespace) -> int:
    settings = _settings(args)
    typed = [*args.inputs, *(args.contrast or [])]
    _inputs_exist(typed)
    _refuse_overwrite(args.output, typed)
    profile = api.build(
        args.inputs,
        settings,
        contrast=args.contrast,
        contrast_label=args.contrast_label,
        keep_chunks=args.keep_chunks,
    )
    _notes(profile.notes)
    # Files found inside a folder are known only once it has been read.
    _refuse_overwrite(args.output, typed, profile.sources)
    profile.save(args.output)
    print(profile.to_text(color=_color(), full=args.all))
    sys.stdout.flush()  # keep the warnings after the summary when both go to one pipe
    color_err = _color(sys.stderr)
    for note in profile.notes:
        if note.code == NoteCode.THIN_REFERENCE:
            reason = _flagged(note.message, note.setting)
            print(_warn(f"Thin reference: {reason}.", color_err), file=sys.stderr)
    print(f"\nwrote {args.output}")
    print(f"Next, score a draft against it:\n  {PROG} score <draft> {shlex.quote(args.output)}")
    return 0


def _split_score_paths(args: argparse.Namespace) -> tuple[list[str], str]:
    usage = f"{PROG} score DRAFT [DRAFT ...] REFERENCE.json"
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


def _load_score_reference(reference_arg: str) -> Profile:
    try:
        return Profile.load(reference_arg)
    except StyleProfileError as error:
        if error.code == "not_a_profile":
            raise StyleProfileError(
                f"{reference_arg} is not a style profile; the reference profile goes last: "
                f"{PROG} score DRAFT [DRAFT ...] REFERENCE.json"
            ) from error
        # Other errors, an outdated profile's included, already name it as typed.
        raise


def _headline(result: ScoreResult, samples: Sequence[str]) -> str:
    """One line with the same verdict words as the full comparison view."""
    if len(samples) == 1:
        name = "stdin" if samples[0] == "-" else samples[0]
    else:
        name = f"{len(samples)} inputs"
    if result.delta is None:
        return f"{name}: no metrics could be compared with the reference"
    if not result.judged:
        return f"{name}: {too_short_text(result.report['reference']['verdict'])}"
    verdict = result.report["reference"]["verdict"]
    left_out = verdict["chunks"] - verdict["chunks_judged"]
    # Chunks too short to judge count for nothing, which a batch line must not hide.
    note = f"; {left_out} of {verdict['chunks']} chunks not judged: too short" if left_out else ""
    parts = [f"{name}: {result.verdict} (Delta {result.delta:.2f}{note})"]
    label = result.contrast_label
    if result.likeness is not None and result.likeness_verdict is not None and label:
        likeness = f"{result.likeness_verdict.words(label)} ({result.likeness:.2f})"
        parts.append(f"{label}-likeness {likeness}")
    return "; ".join(parts)


def _run_score(args: argparse.Namespace) -> int:
    samples, reference_arg = _split_score_paths(args)
    if args.output and _path(args.output) == _path(reference_arg):
        raise StyleProfileError("--output is the reference file; choose another output path")
    _inputs_exist(samples)
    if args.output:
        _refuse_overwrite(args.output, samples)
    profile = _load_score_reference(reference_arg)
    # Flags left out are inherited from the profile.
    overrides: SettingsOverrides = {}
    if args.window_words is not None:
        overrides["window_words"] = args.window_words
    if args.min_words is not None:
        overrides["min_words"] = args.min_words
    if args.text_field:
        overrides["text_field"] = args.text_field
    if args.no_syntax:
        overrides["syntax"] = False
    if args.input_format != AUTO:
        overrides["input_format"] = args.input_format
    result = profile.score(samples, **overrides)
    # Window and syntax overrides are warned about in the report itself.
    _notes(result.notes)
    if args.output:
        _refuse_overwrite(args.output, samples, result.sources)
        result.save(args.output)
    if args.json:
        sys.stdout.write(dumps_report(result.report))
        if args.output:
            _note(f"wrote {args.output}")
    elif args.quiet:
        print(_headline(result, samples), flush=True)
        if result.warnings:
            _note(f"{_plural(len(result.warnings), 'warning')}; run without -q to see them")
    else:
        setting = result.report["reference"]["verdict"].get("setting")
        print(_flagged(result.to_text(color=_color(), full=args.all), setting))
        if args.output:
            print(f"\nwrote {args.output}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    # Errors name the path as typed: Path() drops a trailing slash or a leading ./
    report = load_report(expand_path(args.report), args.report)
    color = _color()
    if report["kind"] == EVALUATION:
        print(format_evaluation(report, color=color))
        return 0
    if report["kind"] == REFERENCE:
        print(format_summary(report, color=color, full=args.all))
        return 0
    print(format_summary(report, report["reference"]["baseline"], color=color, full=args.all))
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
    settings = _settings(args)
    labels = [label for label, _ in args.edited]
    if len(set(labels)) != len(labels):
        raise StyleProfileError("each --edited LABEL must be distinct")
    folders = [folder for _, folder in args.edited]
    if "-" in folders:
        raise StyleProfileError("--edited takes folders of files, not - (stdin)")
    typed = [*args.inputs, *args.contrast, *folders]
    _inputs_exist(typed)
    if args.output:
        _refuse_overwrite(args.output, typed)
    result = api.evaluate(
        args.inputs,
        args.contrast,
        dict(args.edited),
        settings,
        contrast_label=args.contrast_label,
        retrain=args.retrain,
    )
    _notes(result.notes)
    if args.output:
        _refuse_overwrite(args.output, typed, result.sources)
        result.save(args.output)
    if args.json:
        sys.stdout.write(dumps_report(result.report))
        if args.output:
            _note(f"wrote {args.output}")
    else:
        print(result.to_text(color=_color()))
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


def _suggest_command(argv: Sequence[str]) -> str:
    """What to run instead of a command-less line: ``score`` when it names a reference with
    --reference, ``build`` when it has --output, otherwise either."""
    guess = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    guess.add_argument("inputs", nargs="*")
    guess.add_argument("-o", "--output")
    guess.add_argument("-r", "--reference")
    for flag in (
        "--text-field",
        "--window-words",
        "--min-words",
        "--input-format",
        "--top-k",
        "--contrast-label",
    ):
        guess.add_argument(flag)
    guess.add_argument("--contrast", nargs="+")
    guess.add_argument("--no-syntax", action="store_true")
    guess.add_argument("--all", action="store_true")
    try:
        # Unknown flags are dropped: the suggestion keeps only what the command accepts.
        found, _ = guess.parse_known_intermixed_args(argv)
    except argparse.ArgumentError:
        found = None
    if found is None or not (found.reference or found.output):
        rest = shlex.join(argv)
        return f"`{PROG} build {rest}` or `{PROG} score {rest}`"
    kept = ["text_field", "window_words", "min_words", "input_format"]
    if found.reference:
        # The reference moves to the last argument; score takes no contrast or --top-k.
        command = ["score", *found.inputs, found.reference]
    else:
        command = ["build", *found.inputs]
        command += [item for path in found.contrast or [] for item in ("--contrast", path)]
        kept += ["top_k", "contrast_label"]
    if found.output:
        command += ["-o", found.output]
    for name in kept:
        if getattr(found, name) is not None:
            command += ["--" + name.replace("_", "-"), getattr(found, name)]
    command += ["--no-syntax"] * found.no_syntax + ["--all"] * found.all
    return f"`{PROG} {shlex.join(command)}`"


def _dispatch(argv: Sequence[str]) -> int:
    parser, commands = _subparsers()
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    if argv[0] not in commands:
        if not argv[0].startswith("-") and api.path_exists(argv[0]):
            parser.print_usage(sys.stderr)
            print(
                f"{PROG}: error: {argv[0]} is not a command; did you mean "
                f"{_suggest_command(argv)}?",
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
    except (StyleProfileError, OSError) as error:
        _notes(getattr(error, "notes", ()))
        code = getattr(error, "code", None)
        print(f"error: {_flagged(str(error), getattr(error, 'setting', None))}", file=sys.stderr)
        hint = HINTS.get(code or "")
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
