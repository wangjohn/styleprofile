"""Command-line entry point: build a writer's reference profile once, then score drafts.

Commands:

    styleprofile build posts/ --contrast llm-drafts/ -o writer.json
    styleprofile score draft.md writer.json
    styleprofile show writer.json
    styleprofile metrics
    styleprofile evaluate posts/ --contrast llm-drafts/ --edited light=edits/
    styleprofile cache --clear
    styleprofile setup
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import textwrap
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from styleprofile import __version__, api, demo, spacy_model
from styleprofile import cache as caching
from styleprofile.api import (
    AUTO,
    DEFAULT_TOP_K,
    DEFAULT_WINDOW_WORDS,
    INPUT_FORMATS,
    DocumentResult,
    Profile,
    ScoreResult,
    Settings,
    SettingsOverrides,
)
from styleprofile.calibration import MIN_JUDGED_WORDS, Lengths, flagged_text, too_short_text
from styleprofile.core import (
    LikenessVerdict,
    Note,
    NoteCode,
    Progress,
    StyleProfileError,
    Verdict,
)
from styleprofile.display import (
    DEFAULT_WIDTH,
    format_evaluation,
    format_summary,
    not_judged,
    severity,
)
from styleprofile.measure import AUTO_JOBS
from styleprofile.metrics import describe
from styleprofile.profile import (
    EVALUATION,
    REFERENCE,
    VERSION,
    dumps_report,
    expand_path,
    load_report,
)
from styleprofile.schema import FailedDocument, FailLevels
from styleprofile.split import SPLIT_ON
from styleprofile.status import StatusLine
from styleprofile.terminal import prepare_output, shell_join

PROG = "styleprofile"
# Advice for library errors, which name the problem but never a flag.
HINTS = {
    "text_field": "pass --text-field with the JSONL field that holds the text",
    "group_field": "pass --group-field with a JSONL field the records have, such as thread",
    "contrast_needs_documents": "add more of the writer's documents, or build without --contrast",
    "score_as_reference": "score against the reference profile made by `styleprofile build`",
    "unmatched_edits": "give each edited file its original's name and relative path",
    "duplicate_names": "give each draft a distinct file name or JSONL id",
    "forced_jsonl": "leave out --input-format jsonl to read each file by its extension",
    "setup_needs_spacy": "then run `styleprofile setup` again",
    "setup_spacy_version": (
        f"install spaCy {spacy_model.SPACY_SERIES}.x, for example by reinstalling "
        "'styleprofile[syntax]', which requires it, then run `styleprofile setup` again"
    ),
    "setup_no_installer": (
        "add pip to this environment (python -m ensurepip) and run `styleprofile setup` "
        f"again, or install the model with your package manager from {spacy_model.MODEL_URL}"
    ),
    "setup_externally_managed": (
        "install styleprofile in a virtual environment (python -m venv, pipx, or uv tool "
        "install) and run `styleprofile setup` there"
    ),
    "setup_failed": (
        "the installer's own output above says why; to install the model yourself, run "
        f"pip install '{spacy_model.MODEL_URL}'"
    ),
}
# `score` exits with this when a document reaches a --fail-above, --fail-likeness or
# --fail-flagged level.
EXIT_FAILED = 3
FAIL_ABOVE = {
    "somewhat": Verdict.SOMEWHAT_DIFFERENT,
    "clearly": Verdict.CLEARLY_DIFFERENT,
    "very": Verdict.VERY_DIFFERENT,
}
FAIL_LIKENESS = {
    "few": LikenessVerdict.FEW_TRAITS,
    "leans": LikenessVerdict.LEANS,
    "like": LikenessVerdict.LIKE_DRAFTS,
}
SCORE_EXIT_STATUS = f"""\
exit status: 0 scored, 1 error, 2 usage error, {EXIT_FAILED} a document reached a fail level
or could not be compared with the reference when any --fail-* flag was given (named
on stderr, in input order). Too short to judge never fails. A few off-voice chunks
move a whole document's verdict little: --fail-flagged catches them"""
# Library errors and notes name a setting (``setting``) as a whole word; the CLI prints the
# flag that sets it instead.
FLAGS = {
    "input_format": "--input-format",
    "window_words": "--window-words",
    "min_words": "--min-words",
    "top_k": "--top-k",
    "group_field": "--group-field",
    "pool": "--pool",
    "split_on": "--split-on",
    "jobs": "--jobs",
}


def _positive(value: str) -> int:
    """An argparse type: a whole number of 1 or more."""
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a whole number of 1 or more, not {value!r}")
    return number


def _path(value: str) -> Path:
    return expand_path(value).resolve()


def _color(stream: TextIO | None = None) -> bool:
    """NO_COLOR turns color off and FORCE_COLOR on (NO_COLOR wins); else color a terminal."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR", "0") not in ("", "0"):
        return True
    return (stream or sys.stdout).isatty()


def _width() -> int:
    """The terminal's width for views fitted to it, or 80 when stdout is not a terminal, so
    piped and logged output does not depend on the window it ran in."""
    return shutil.get_terminal_size().columns if sys.stdout.isatty() else DEFAULT_WIDTH


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
        help="drop chunks under N prose words" + (suffix or " (default: 1)"),
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
        help="read every input as this format (default: auto"
        + (", not the reference's)" if inherited else ", by each file's extension and content)"),
    )
    parser.add_argument(
        "--group-field",
        metavar="FIELD",
        help="group JSONL records into documents by this field" + suffix,
    )
    parser.add_argument(
        "--pool",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="judge short texts as one batch (default: no)"
        if inherited
        else "join short texts into windows (default: when most are under a quarter window)",
    )
    parser.add_argument(
        "--split-on",
        choices=SPLIT_ON,
        default=None if inherited else AUTO,
        metavar="HOW",
        help="heading, heading:N or rule: a verdict per part of each draft (auto, none: no split)"
        if inherited
        else "auto, heading, heading:N, rule or none: split texts into documents (default: "
        "auto, when there are too few)",
    )
    parser.add_argument(
        "--no-syntax",
        action="store_true",
        help="skip the spaCy parser and its metrics",
    )


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Flags that change how fast a run goes, never its numbers."""
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        metavar="N",
        help=f"spaCy processes (default 0: by CPUs and memory, max {AUTO_JOBS})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=f"measure every text again (see `{PROG} cache`)",
    )


def _subparsers() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Measure a writer's style, then see how far a draft drifts from it.",
        epilog=(
            f"Try the bundled samples: {PROG} demo\n\n"
            "Build a reference from the writer's texts once, then score drafts against it:\n"
            f"  {PROG} build posts/ --contrast llm-drafts/ -o writer.json\n"
            f"  {PROG} score draft.md writer.json\n"
            f"Or build and score in one command: {PROG} score draft.md --against posts/\n"
            "Short drafts need at least 75 words; use --pool to judge several together.\n\n"
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
            usage=f"{PROG} build [options] INPUT [INPUT ...] [-o PROFILE.json]",
        ),
    )
    build.formatter_class = lambda prog: argparse.RawDescriptionHelpFormatter(
        prog, width=100, max_help_position=32
    )
    build.add_argument("--verbose", action="store_true", help="show full notes and explanations")
    build.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="the writer's Markdown, text, HTML or JSONL files, folders of them, or - for stdin",
    )
    build.add_argument(
        "-o",
        "--output",
        metavar="PROFILE.json",
        help="default: first input name.profile.json; stdin needs -o",
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
    _add_run_flags(build)
    build.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help=f"distribution entries kept in the profile (default: {DEFAULT_TOP_K})",
    )
    build.add_argument("--all", action="store_true", help="show every metric, not just key ones")
    build.add_argument(
        "--by-paragraph",
        action="store_true",
        help="calibrate paragraph checks (experimental)",
    )
    build.add_argument(
        "--keep-chunks",
        action="store_true",
        help="save each chunk's metrics for debugging (much larger)",
    )

    score_parser = commands.add_parser(
        "score",
        **_help_parser(
            "Score drafts against a saved profile, or build one with --against. Settings come "
            "from the reference unless overridden. Below its shortest calibrated length "
            "(at least 75 words), use --pool to judge short texts together.",
            f"{PROG} score draft.md writer.json\n  "
            f"{PROG} score draft.md --against posts/ --contrast llm-drafts/\n  "
            f"{PROG} score -q --fail-above clearly --fail-flagged 1 -r writer.json a.md b.md"
            "   # a hook or CI",
            usage=f"{PROG} score [options] [-r REFERENCE.json] SAMPLE [SAMPLE ...] "
            "[REFERENCE.json] [--against CORPUS [CORPUS ...]]",
        ),
    )
    score_parser.formatter_class = lambda prog: argparse.RawDescriptionHelpFormatter(
        prog, width=120, max_help_position=44
    )
    score_parser.add_argument(
        "--verbose", action="store_true", help="show full notes and explanations"
    )
    score_parser.add_argument(
        "paths",
        nargs="+",
        metavar="SAMPLE ... REFERENCE.json",
        help="the texts to score (files, directories, or - for stdin), then the reference "
        "profile unless -r gives it",
    )
    score_parser.add_argument(
        "-r",
        "--reference",
        metavar="REFERENCE.json",
        help="the reference profile, if not last (for pre-commit)",
    )
    score_parser.add_argument(
        "--against",
        nargs="+",
        metavar="CORPUS",
        help="build in memory; put drafts first; excludes a saved profile or -r",
    )
    score_parser.add_argument(
        "--contrast",
        action="extend",
        nargs="+",
        metavar="PATH",
        help="contrast texts for --against; repeat for more paths",
    )
    score_parser.add_argument(
        "--contrast-label",
        default="LLM",
        metavar="NAME",
        help="name of the --against contrast set (default: LLM)",
    )
    _add_run_flags(score_parser)
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
        help="print one verdict line per document and no notes",
    )
    score_parser.add_argument(
        "--fail-above",
        choices=list(FAIL_ABOVE),
        help=f"exit {EXIT_FAILED} if a whole document is this different or more",
    )
    score_parser.add_argument(
        "--fail-likeness",
        choices=list(FAIL_LIKENESS),
        help=f"exit {EXIT_FAILED} if a whole document's likeness reaches this",
    )
    score_parser.add_argument(
        "--fail-flagged",
        type=_positive,
        metavar="N",
        help=f"exit {EXIT_FAILED} if a document has N or more chunks that read clearly "
        "different or lean LLM on their own",
    )
    score_parser.add_argument(
        "--all", action="store_true", help="show every metric and document, not just key ones"
    )
    score_parser.add_argument(
        "--by-paragraph",
        action="store_true",
        help="experimental: show where each document drifts, paragraph by paragraph",
    )
    _add_input_flags(score_parser, inherited=True)
    score_parser.set_defaults(top_k=None)
    score_parser.epilog = f"{score_parser.epilog}\n\n{SCORE_EXIT_STATUS}"

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
    show.add_argument(
        "--all", action="store_true", help="show every metric and document, not just key ones"
    )
    show.add_argument(
        "--by-paragraph",
        action="store_true",
        help="for a score report made with --by-paragraph (experimental), list every paragraph",
    )

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
            "Stress-test edited drafts with the likeness weights learned without their originals.",
            f"{PROG} evaluate posts/ --contrast llm-drafts/ "
            "--edited light=edits/light humanize=edits/humanize",
            usage=f"{PROG} evaluate [options] INPUT [INPUT ...] --contrast PATH "
            "--edited LABEL=DIR [LABEL=DIR ...]",
        ),
    )
    evaluate.add_argument("--verbose", action="store_true", help="show full notes and explanations")
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
        help="also report the AUC with the edits in the contrast set",
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
    _add_run_flags(evaluate)

    cache = commands.add_parser(
        "cache",
        **_help_parser(
            "Show where the measurement cache is, how large, and whether it can be used, or "
            "delete it. build and evaluate keep every text's measurements there (score does "
            "not), so building from a text again measures nothing; the cache holds about "
            f"{caching.MAX_BYTES // 2**20:,} MB at most, dropping the least recently used "
            f"entries first. {caching.ENVIRONMENT}=1 turns it off.",
            f"{PROG} cache --clear",
        ),
    )
    cache.add_argument("--clear", action="store_true", help="delete the cache")

    commands.add_parser(
        "setup",
        **_help_parser(
            f"Install spaCy's English model ({spacy_model.DEFAULT_MODEL} "
            f"{spacy_model.MODEL_VERSION}), which the syntax metrics need. Run it once after "
            "pip install 'styleprofile[syntax]'. It downloads the model from spaCy's releases "
            "on GitHub (about 13 MB) and installs it with pip, or with uv pip when this "
            "environment has no pip; it does nothing when the model is already installed.",
            f"{PROG} setup",
        ),
    )
    demo_parser = commands.add_parser(
        "demo",
        **_help_parser(
            "Try the bundled sample texts: copy them to a folder, build a reference and "
            "score a draft. An unchanged previous demo folder can be reused.",
            f"{PROG} demo --dir styleprofile-demo",
        ),
    )
    demo_parser.add_argument(
        "--dir",
        default="styleprofile-demo",
        metavar="DIR",
        help="where to put the samples and profile (default: ./styleprofile-demo)",
    )
    demo_parser.add_argument(
        "--no-syntax",
        action="store_true",
        help="skip the spaCy parser and its metrics",
    )
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


def _notes(notes: Sequence[Note], *, verbose: bool = False) -> None:
    """Print notes as ``note:`` lines, except thin-reference notes, which ``build`` prints
    as warnings after its summary."""
    for note in notes:
        if note.code != NoteCode.THIN_REFERENCE:
            hint = _flagged(note.code.hint, "input_format") if note.code.hint else None
            setting = note.setting or ("input_format" if note.code.hint is not None else None)
            message = _flagged(note.text(verbose=verbose), setting)
            if not verbose:
                for other in FLAGS:
                    if other != setting:
                        message = _flagged(message, other)
            _note(f"{message} ({hint})" if verbose and hint else message)


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
        group_field=args.group_field,
        pool=AUTO if args.pool is None else args.pool,
        split_on=args.split_on,
    )


def _warn(text: str, color: bool) -> str:
    return f"\033[33m{text}\033[0m" if color else text


@contextmanager
def _status(shown: bool = True) -> Iterator[Callable[[Progress], None] | None]:
    """A progress callback drawing one updating line on stderr, when it is a terminal (and
    ``shown``); otherwise None, so piped and redirected output never carries it. The line is
    erased when the run ends or fails, before anything else is printed."""
    if not (shown and sys.stderr.isatty()) or os.environ.get("TERM") == "dumb":
        yield None
        return
    line = StatusLine(sys.stderr)
    try:
        yield line
    finally:
        line.clear()


def _default_output(inputs: Sequence[str]) -> str:
    if "-" in inputs:
        raise StyleProfileError("build from stdin requires -o; give a path to save the profile")
    first = expand_path(inputs[0])
    name = (
        (first.resolve().name if first.name in ("", "..") else first.name)
        if first.is_dir()
        else first.stem
    )
    return f"{name or first.resolve().name}.profile.json"


def _thin_warnings(profile: Profile, *, verbose: bool = False) -> None:
    if not verbose:
        if any(note.code == NoteCode.THIN_REFERENCE for note in profile.notes):
            report = profile.report
            message = NoteCode.THIN_REFERENCE.message(
                "summary", f"{report['chunk_count']:,}", f"{report['word_count']:,}"
            )
            print(_warn(message, _color(sys.stderr)), file=sys.stderr)
        return
    for note in profile.notes:
        if note.code == NoteCode.THIN_REFERENCE:
            reason = _flagged(note.message, note.setting)
            print(_warn(f"Thin reference: {reason}.", _color(sys.stderr)), file=sys.stderr)


def _run_build(args: argparse.Namespace) -> int:
    args.output = args.output or _default_output(args.inputs)
    settings = _settings(args)
    typed = [*args.inputs, *(args.contrast or [])]
    _inputs_exist(typed)
    _refuse_overwrite(args.output, typed)
    with _status() as progress:
        profile = api.build(
            args.inputs,
            settings,
            contrast=args.contrast,
            contrast_label=args.contrast_label,
            keep_chunks=args.keep_chunks,
            passages=args.by_paragraph,
            progress=progress,
            jobs=args.jobs,
            cache=not args.no_cache,
        )
    _notes(profile.notes, verbose=args.verbose)
    # Files found inside a folder are known only once it has been read.
    _refuse_overwrite(args.output, typed, profile.sources)
    profile.save(args.output)
    print(profile.to_text(color=_color(), full=args.all, verbose=args.verbose))
    sys.stdout.flush()  # keep the warnings after the summary when both go to one pipe
    _thin_warnings(profile, verbose=args.verbose)
    print(f"\nwrote {args.output}")
    print(f"Next, score a draft against it:\n  {PROG} score <draft> {shell_join([args.output])}")
    return 0


def _split_score_paths(args: argparse.Namespace) -> tuple[list[str], str]:
    usage = f"{PROG} score DRAFT [DRAFT ...] REFERENCE.json"
    if args.reference is not None:
        profiles = [path for path in args.paths if path.lower().endswith(".json")]
        if profiles:
            raise StyleProfileError(
                f"{profiles[-1]} looks like a second reference profile; give the reference "
                "once, either with -r or last"
            )
        samples, reference = list(args.paths), args.reference
        usage = f"{PROG} score -r REFERENCE.json DRAFT [DRAFT ...]"
    elif len(args.paths) < 2:
        raise StyleProfileError(
            f"score needs a sample and then a reference profile: {usage}, "
            f"or {PROG} score -r REFERENCE.json DRAFT"
        )
    else:
        *samples, reference = args.paths
    flagged = args.reference is not None
    if reference == "-":
        where = "given with -r" if flagged else "the last argument"
        raise StyleProfileError(f"the reference profile ({where}) must be a file: {usage}")
    if _path(reference).is_dir():
        where = "" if flagged else "; the reference profile goes last"
        raise StyleProfileError(f"{reference} is a directory, not a profile{where}: {usage}")
    if not _path(reference).is_file():
        where = "" if flagged else "; it goes last"
        raise StyleProfileError(f"reference profile {reference} not found{where}: {usage}")
    return samples, reference


def _load_score_reference(reference_arg: str, flagged: bool = False) -> Profile:
    """The reference profile; ``flagged`` when it was given with -r rather than last."""
    try:
        return Profile.load(reference_arg)
    except StyleProfileError as error:
        if error.code == "not_a_profile":
            where = (
                "; -r takes the profile made by `styleprofile build`"
                if flagged
                else f"; the reference profile goes last: {PROG} score DRAFT [DRAFT ...] "
                "REFERENCE.json"
            )
            raise StyleProfileError(f"{reference_arg} is not a style profile{where}") from error
        # Other errors, an outdated profile's included, already name it as typed.
        raise


def _headline(
    name: str,
    delta: float | None,
    verdict: Verdict,
    likeness: float | None,
    likeness_verdict: LikenessVerdict | None,
    label: str | None,
    *,
    too_short: str | None = None,
    note: str = "",
    flagged: str = "",
) -> str:
    """One line with the same verdict words as the full comparison view; ``too_short``
    replaces the verdict and figures of a text too short to judge, and ``flagged`` ("4 of 40
    chunks read clearly different or lean LLM") ends it in parentheses."""
    if delta is None and verdict is not Verdict.TOO_SHORT:
        return f"{name}: no metrics could be compared with the reference"
    if too_short:
        return f"{name}: {too_short}"
    parts = [f"{name}: {verdict} (Delta {delta:.2f}{note})"]
    if likeness is not None and likeness_verdict is not None and label:
        parts.append(f"{label}-likeness {likeness_verdict.words(label)} ({likeness:.2f})")
    return "; ".join(parts) + (f" ({flagged})" if flagged else "")


def _flag_count(flagged: int, judged: int, chunks: int, label: str | None) -> str:
    """ "4 of 40 chunks read clearly different or lean LLM" for ``-q`` and ``failed:``
    lines when a verdict over several judged chunks has some flagged on their own, else
    ""."""
    return flagged_text(flagged, judged, chunks, label) if flagged and judged > 1 else ""


def _drifts(result: ScoreResult, name: str) -> str:
    """ "; drifts at lines 9, 15" for a document whose paragraphs drift (``passages``), or
    "; drifts throughout" when more than half of them do."""
    passages = [passage for passage in result.passages if passage.document == name]
    where = [
        f"{first}" if first == last else f"{first}-{last}"
        for passage in passages
        if passage.drifts
        for first, last in [passage.lines]
    ]
    if len(where) * 2 > len(passages):
        return "; drifts throughout"
    return f"; drifts at line{'s' * (len(where) > 1)} {', '.join(where)}" if where else ""


def _quiet_lines(result: ScoreResult, samples: Sequence[str]) -> list[str]:
    """``-q``: one line per document, furthest from the reference first, each named where
    it can be opened (``DocumentResult.shown``)."""
    label = result.contrast_label
    documents = result.documents
    if len(documents) > 1:
        ranked = sorted(
            documents, key=lambda doc: severity(doc.verdict, doc.likeness_verdict, doc.delta)
        )
        return [
            _headline(
                doc.shown,
                doc.delta,
                doc.verdict,
                doc.likeness,
                doc.likeness_verdict,
                label,
                too_short=None
                if doc.judged
                else too_short_text({"chunks": doc.chunks, "words": doc.words}),
                note=not_judged(doc.chunks, doc.chunks_judged),
                flagged=_flag_count(doc.flagged, doc.chunks_judged, doc.chunks, label),
            )
            + _drifts(result, doc.name)
            for doc in ranked
        ]
    if documents:
        name = documents[0].shown
    elif len(samples) == 1:
        name = api.STDIN_SHOWN if samples[0] == "-" else samples[0]
    else:
        name = f"{len(samples)} inputs"
    if result.verdict is Verdict.NOT_COMPARABLE:
        return [f"{name}: no metrics could be compared with the reference"]
    if not result.judged:
        return [f"{name}: {too_short_text(result.report['reference']['verdict'])}"]
    verdict = result.report["reference"]["verdict"]
    # Chunks too short to judge count for nothing, which a batch line must not hide.
    note = not_judged(verdict["chunks"], verdict["chunks_judged"])
    return [
        _headline(
            name,
            result.delta,
            result.verdict,
            result.likeness,
            result.likeness_verdict,
            label,
            note=note,
            flagged=_flag_count(
                verdict["flagged"], verdict["chunks_judged"], verdict["chunks"], label
            ),
        )
        + (_drifts(result, documents[0].name) if documents else "")
    ]


def _failed(
    args: argparse.Namespace, result: ScoreResult
) -> list[tuple[DocumentResult, FailedDocument]]:
    """The documents that reach the --fail-above, --fail-likeness or --fail-flagged level,
    in input order, each with what the report records: the verdicts that did (None for a
    check it passed), and how many of its chunks are flagged on their own, whichever check
    it failed. Each document is checked once (``api.fails``), so this is linear in their
    number."""
    above = FAIL_ABOVE.get(args.fail_above or "")
    likeness = FAIL_LIKENESS.get(args.fail_likeness or "")
    label = result.contrast_label or ""
    entries: list[tuple[DocumentResult, FailedDocument]] = []
    for doc in result.documents:
        delta_failed, likeness_failed, flagged_failed = api.fails(
            doc, above, likeness, args.fail_flagged
        )
        incomparable = (
            bool(args.fail_above or args.fail_likeness or args.fail_flagged)
            and doc.verdict is Verdict.NOT_COMPARABLE
        )
        if not (delta_failed or likeness_failed or flagged_failed or incomparable):
            continue
        entry: FailedDocument = {
            "name": doc.name,
            "path": doc.path or doc.name,
            "delta": str(doc.verdict) if delta_failed else None,
            "likeness": (
                doc.likeness_verdict.words(label)
                if likeness_failed and doc.likeness_verdict
                else None
            ),
            "flagged": doc.flagged,
            "chunks_judged": doc.chunks_judged,
        }
        if incomparable:
            entry["reason"] = "could not be compared with the reference"
        entries.append((doc, entry))
    return entries


def _failed_line(doc: DocumentResult, entry: FailedDocument, label: str | None) -> str:
    """``failed: drafts/a.md: delta very different; likeness leans LLM; 2 of 17 chunks read
    clearly different or lean LLM``: one per document, named as ``-q`` names it. The count
    of chunks flagged on their own is there whenever it has some, whichever check failed;
    for a document of one judged chunk only when nothing else is, since its verdict is that
    chunk's own (the JSON entry always has it)."""
    reason = entry.get("reason")
    if reason:
        return f"failed: {doc.shown}: {reason}"
    parts = [f"delta {entry['delta']}"] if entry["delta"] else []
    parts += [f"likeness {entry['likeness']}"] if entry["likeness"] else []
    if entry["flagged"] and (entry["chunks_judged"] > 1 or not parts):
        parts.append(flagged_text(entry["flagged"], entry["chunks_judged"], doc.chunks, label))
    return f"failed: {doc.shown}: " + "; ".join(parts)


def _is_saved_report(value: str) -> bool:
    if value == "-":
        return False
    try:
        load_report(expand_path(value), value)
    except StyleProfileError as error:
        return error.code == "outdated"
    return True


def _against_reference(args: argparse.Namespace) -> tuple[Profile, str]:
    if args.reference is not None or any(_is_saved_report(path) for path in args.paths):
        raise StyleProfileError(
            "--against cannot be combined with a reference profile or -r; "
            "give only drafts before --against"
        )
    typed = [*args.paths, *args.against, *(args.contrast or [])]
    if typed.count("-") > 1:
        raise StyleProfileError("stdin can be read only once; give files for the other inputs")
    _inputs_exist(typed)
    if args.output:
        _refuse_overwrite(args.output, typed)
    build_args = argparse.Namespace(**vars(args))
    defaults = Settings()
    for name in ("window_words", "min_words", "split_on", "top_k"):
        if getattr(build_args, name) is None:
            setattr(build_args, name, getattr(defaults, name))
    started = time.perf_counter()
    with _status(not (args.json or args.quiet)) as progress:
        profile = api.build(
            args.against,
            _settings(build_args),
            contrast=args.contrast,
            contrast_label=args.contrast_label,
            passages=args.by_paragraph,
            progress=progress,
            jobs=args.jobs,
            cache=not args.no_cache,
        )
    elapsed = time.perf_counter() - started
    if args.output:
        _refuse_overwrite(args.output, typed, profile.sources)
    command = [PROG, "build", *args.against]
    if args.by_paragraph:
        command += ["--by-paragraph"]
    for path in args.contrast or []:
        command += ["--contrast", path]
    if args.contrast_label != "LLM":
        command += ["--contrast-label", args.contrast_label]
    for name in ("window_words", "min_words", "text_field", "group_field", "split_on"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if args.input_format != AUTO:
        command += ["--input-format", args.input_format]
    if args.no_syntax:
        command += ["--no-syntax"]
    if args.pool is not None:
        command += ["--pool" if args.pool else "--no-pool"]
    output = "reference.profile.json" if "-" in args.against else _default_output(args.against)
    command += ["-o", output]
    report = profile.report
    summary = (
        f"Reference: built from {_plural(report['document_count'], 'document')} "
        f"({report['word_count']:,} words) in {elapsed:.1f} s, not saved. "
        f"To reuse it: {shell_join(command)}"
    )
    if not args.verbose:
        summary = (
            f"Reference: built from {_plural(report['document_count'], 'document')} "
            f"({report['word_count']:,} words) in {elapsed:.1f} s, not saved.\n"
            f"To reuse: `{shell_join(command)}`"
        )
    _notes(profile.notes, verbose=args.verbose)
    _thin_warnings(profile, verbose=args.verbose)
    return profile, summary


def _short_text_help(profile: Profile, result: ScoreResult) -> str | None:
    lengths = Lengths(profile.report)
    floor = (
        min(anchor.covers for anchor in lengths.anchors)
        if lengths.calibrated
        else lengths.uncalibrated_words / 2
    )
    shortest = max(MIN_JUDGED_WORDS, math.ceil(floor))
    if result.documents and all(doc.words < shortest for doc in result.documents):
        return _flagged(NoteCode.SHORT_TEXTS.message("minimum", f"{shortest:,}"), "pool")
    return None


def _run_score(args: argparse.Namespace) -> int:
    summary = None
    if args.against:
        profile, summary = _against_reference(args)
        samples, reference_arg = list(args.paths), "the in-memory reference"
    else:
        if args.contrast:
            raise StyleProfileError(
                "--contrast needs --against; rebuild the saved reference with --contrast instead"
            )
        samples, reference_arg = _split_score_paths(args)
        profile = _load_score_reference(reference_arg, flagged=args.reference is not None)
    if args.output and not args.against and _path(args.output) == _path(reference_arg):
        raise StyleProfileError("--output is the reference file; choose another output path")
    _inputs_exist(samples)
    if args.output:
        _refuse_overwrite(args.output, samples)
    if args.by_paragraph and not (profile.report.get("calibration") or {}).get("drift"):
        command = [
            PROG,
            "build",
            "WRITER_TEXTS",
            "--by-paragraph",
            "-o",
            reference_arg,
        ]
        contrast = profile.report["settings"].get("contrast")
        if contrast:
            command += ["--contrast", "CONTRAST_TEXTS"]
        if not profile.has_syntax:
            command += ["--no-syntax"]
        guidance = NoteCode.PARAGRAPH_PROFILE.note("default", shell_join(command))
        _note(guidance.text(verbose=args.verbose))
        if not args.verbose:
            print(f"  `{shell_join(command)}`", file=sys.stderr)
        args.by_paragraph = False
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
    if args.group_field:
        overrides["group_field"] = args.group_field
    if args.pool is not None:
        overrides["pool"] = args.pool
    if args.split_on is not None:
        overrides["split_on"] = args.split_on
    with _status(not (args.json or args.quiet)) as progress:
        result = profile.score(
            samples, progress=progress, passages=args.by_paragraph, jobs=args.jobs, **overrides
        )
    if (
        args.fail_likeness
        and not profile.report.get("contrast")
        and any(doc.judged for doc in result.documents)
    ):
        raise StyleProfileError(
            f"--fail-likeness needs a reference built with --contrast, and {reference_arg} "
            "has none; rebuild it with --contrast, or use --fail-above"
        )
    # Window and syntax overrides are warned about in the report itself.
    _notes(result.notes, verbose=args.verbose)
    failed = _failed(args, result)
    if args.fail_above or args.fail_likeness or args.fail_flagged:
        # Recorded for CI, which reads --json or -o: the levels asked, and who reached them.
        # argparse allows only FAIL_ABOVE's and FAIL_LIKENESS's keys.
        levels: FailLevels = {
            "above": args.fail_above,
            "likeness": args.fail_likeness,
            "flagged": args.fail_flagged,
        }
        result.report["fail"] = levels
        result.report["failed"] = [entry for _, entry in failed]
    if args.output:
        _refuse_overwrite(args.output, samples, result.sources)
        result.save(args.output)
    help_short = _short_text_help(profile, result)
    if summary:
        print(summary, file=sys.stderr if args.json else sys.stdout)
    if args.json:
        if help_short:
            _note(help_short)
        sys.stdout.write(dumps_report(result.report))
        if args.output:
            _note(f"wrote {args.output}")
    elif args.quiet:
        print("\n".join(_quiet_lines(result, samples)), flush=True)
        if result.warnings:
            _note(
                NoteCode.WARNING_COUNT.message("default", _plural(len(result.warnings), "warning"))
            )
    else:
        setting = result.report["reference"]["verdict"].get("setting")
        text = result.to_text(
            color=_color(),
            full=args.all,
            width=_width(),
            by_paragraph=args.by_paragraph,
            verbose=args.verbose,
        )
        print(_flagged(text, setting))
        if args.output:
            print(f"\nwrote {args.output}")
    if help_short and not args.json:
        print(help_short)
    if failed:
        sys.stdout.flush()  # keep these after the report when both go to one pipe
        for doc, entry in failed:
            print(_failed_line(doc, entry, result.contrast_label), file=sys.stderr)
        return EXIT_FAILED
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
    baseline = report["reference"]["baseline"]
    text = format_summary(
        report,
        baseline,
        color=color,
        full=args.all,
        width=_width(),
        by_paragraph=args.by_paragraph,
    )
    print(text)
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
    columns = min(shutil.get_terminal_size().columns, 120) if sys.stdout.isatty() else 120
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
    with _status(not args.json) as progress:
        result = api.evaluate(
            args.inputs,
            args.contrast,
            dict(args.edited),
            settings,
            contrast_label=args.contrast_label,
            retrain=args.retrain,
            progress=progress,
            jobs=args.jobs,
            cache=not args.no_cache,
        )
    _notes(result.notes, verbose=args.verbose)
    if args.output:
        _refuse_overwrite(args.output, typed, result.sources)
        result.save(args.output)
    if args.json:
        sys.stdout.write(dumps_report(result.report))
        if args.output:
            _note(f"wrote {args.output}")
    else:
        print(result.to_text(color=_color(), verbose=args.verbose))
        if args.output:
            print(f"\nwrote {args.output}")
    return 0


def _run_cache(args: argparse.Namespace) -> int:
    path, size, entries, problem = caching.describe()
    held = f"{entries:,} {'entry' if entries == 1 else 'entries'}, {size / 2**20:,.1f} MB"
    if args.clear:
        if caching.clear():
            print(f"deleted the measurement cache ({held})")
        else:
            print(f"the measurement cache is already empty ({path})")
        return 0
    print(f"measurement cache: {path}")
    limit = caching.MAX_BYTES // 2**20
    if caching.disabled_by_environment():
        print(f"  off: {caching.ENVIRONMENT} is set")
    elif problem:
        print(f"  unavailable: {problem}")
    print(f"  {held} (at most about {limit:,} MB)" if size else "  empty")
    return 0


def _run_setup(args: argparse.Namespace) -> int:
    del args
    name = f"{spacy_model.DEFAULT_MODEL} {spacy_model.MODEL_VERSION}"
    status = spacy_model.model_status()
    if status.ready:
        print(f"{name} is already installed (spaCy {status.spacy_version}); nothing to do.")
        return 0
    spacy_model.check_spacy(status)
    found = f", replacing {status.model_version}" if status.model_version else ""
    print(f"Installing {name}{found} from github.com/explosion/spacy-models ...", flush=True)
    command = spacy_model.install_model()
    _note(f"ran: {shell_join(command)}")
    print(f"Installed {name}. Syntax metrics are on for new profiles and scores.")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    directory, samples = demo.prepare(args.dir)
    # Keep the ordinary build/score output and explicit CLI build cache behavior.
    root = Path(args.dir).expanduser()
    # A relative folder starting with '-' would be parsed as a build/score option.
    if str(root).startswith("-"):
        root = directory
    profile = (root / demo.PROFILE).as_posix()
    syntax = ["--no-syntax"] if args.no_syntax else []
    _dispatch(
        [
            "build",
            (root / "writer").as_posix(),
            "--contrast",
            (root / "llm-drafts").as_posix(),
            "-o",
            profile,
            *syntax,
        ]
    )
    demo.record(directory, samples)
    code = _dispatch(["score", (root / "draft.md").as_posix(), profile])
    print("\nNext: try it on your own texts")
    print(f"  {PROG} build posts/ --contrast llm-drafts/ -o writer.json")
    print(f"  {PROG} score draft.md writer.json")
    print(f"  {PROG} score draft.md --against posts/")
    return code


RUNNERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build": _run_build,
    "score": _run_score,
    "show": _run_show,
    "metrics": _run_metrics,
    "evaluate": _run_evaluate,
    "cache": _run_cache,
    "setup": _run_setup,
    "demo": _run_demo,
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
        "--group-field",
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
        rest = shell_join(argv)
        return f"`{PROG} build {rest}` or `{PROG} score {rest}`"
    kept = ["text_field", "group_field", "window_words", "min_words", "input_format"]
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
    return f"`{PROG} {shell_join(command)}`"


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
    prepare_output()
    try:
        return _dispatch(sys.argv[1:] if argv is None else list(argv))
    except (StyleProfileError, OSError) as error:
        _notes(
            getattr(error, "notes", ()),
            verbose="--verbose" in (sys.argv[1:] if argv is None else argv),
        )
        code = getattr(error, "code", None)
        print(f"error: {_flagged(str(error), getattr(error, 'setting', None))}", file=sys.stderr)
        hint = HINTS.get(code or "")
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
