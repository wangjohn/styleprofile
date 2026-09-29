"""Run the benchmark cases, report them against the plan's targets, and gate on a base.

    python bench/run.py [--case NAME ...] [--quick] [--repeat N] [--against REF [--check] [--pr N]]

Each case generates its corpus if needed (``bench/gen.py``), builds a reference from it with
``styleprofile build`` and scores ``examples/draft.md`` against that reference, each in a fresh
Python process with its own empty cache and home directories, so no run is timed off a cache
another run filled. For both steps it records wall time, CPU time (user + system) and peak
resident memory, all from ``os.wait4`` (which works on macOS and Linux); for the build it also
records the profile's size. Peak memory and CPU time cover the main process only, not any
workers it starts: PR 13, which adds spaCy's ``n_process``, is responsible for measuring the
whole process tree.

``--against REF`` also benchmarks the package as of git revision ``REF`` (its ``src/`` is
exported to ``bench/work/base/``), with this checkout's harness, corpora and draft, in the same
Python environment. Base and change runs are interleaved (base, change, change, base, ...) so a
runner that slows down mid-job slows both. Time is the fastest of ``--repeat`` runs and memory
the largest. ``--check`` then exits 1 when the change is worse than its base by more than the
margins in ``bench/targets.py``, unless ``bench/accepted.toml`` accepts it for this pull
request (``--pr``, detected in GitHub Actions); see ``bench/gate.py``.

Results go to ``bench/results.json``. The tables are printed (and appended to
``$GITHUB_STEP_SUMMARY`` in GitHub Actions): the plan's targets are a report, never a gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from gate import Verdict, inert, judge, load_accepted
from gen import generate
from targets import BACKSTOP, CASES, MARGINS, METRICS, Case

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DRAFT = ROOT / "examples" / "draft.md"
WORK = HERE / "work"  # git-ignored: profiles, reports and logs of the last run, and the base
RESULTS = HERE / "results.json"  # git-ignored
MB = 1_000_000
CHANGE, BASE = "change", "base"


class RunFailed(Exception):
    pass


def _peak_bytes(maxrss: int) -> int:
    """ru_maxrss is in bytes on macOS and in kibibytes on Linux."""
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def _env(fresh: Path, src: Path) -> dict[str, str]:
    # A fresh home and cache for every run: a cache (PR 13) must never give cache-hit timings.
    # PYTHONPATH comes before site-packages, so it wins over the editable install of this
    # checkout; the change gets it too, so both import the same way.
    home = fresh / "home"
    home.mkdir()
    return {
        **os.environ,
        "NO_COLOR": "1",
        "PYTHONPATH": str(src),
        "HOME": str(home),
        "XDG_CACHE_HOME": str(fresh / "cache"),
        "XDG_DATA_HOME": str(fresh / "data"),
        "XDG_STATE_HOME": str(fresh / "state"),
    }


def measure(command: list[str], log: Path, src: Path) -> tuple[float, float, float]:
    """Run ``command`` to completion with the package from ``src``; return its wall time (s),
    CPU time (s) and peak memory (MB)."""
    with (
        tempfile.TemporaryDirectory(prefix="styleprofile-bench-") as fresh,
        log.open("w", encoding="utf-8") as out,
    ):
        env = _env(Path(fresh), src)
        start = time.perf_counter()
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        # wait4 reaps this one child and returns its own resource usage, where getrusage
        # would give the total or maximum over every child so far.
        _, status, rusage = os.wait4(process.pid, 0)
        elapsed = time.perf_counter() - start
    process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode != 0:
        tail = log.read_text(encoding="utf-8").strip().splitlines()[-10:]
        raise RunFailed(f"{' '.join(command)} exited {process.returncode}:\n" + "\n".join(tail))
    cpu = rusage.ru_utime + rusage.ru_stime
    return elapsed, cpu, _peak_bytes(rusage.ru_maxrss) / MB


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def export_base(ref: str) -> tuple[str, Path]:
    """Export ``src/`` as of ``ref`` to ``bench/work/base/<sha>/``; return the sha and the
    exported ``src``."""
    sha = _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    folder = WORK / BASE / sha
    src = folder / "src"
    if not (folder / ".complete").exists():
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / "src.tar"
        _git("archive", f"--output={archive}", sha, "src")
        subprocess.run(["tar", "-xf", str(archive), "-C", str(folder)], check=True)
        archive.unlink()
        (folder / ".complete").touch()
    # Make sure the base really is what gets imported.
    with tempfile.TemporaryDirectory(prefix="styleprofile-bench-") as fresh:
        where = subprocess.run(
            [sys.executable, "-c", "import styleprofile; print(styleprofile.__file__)"],
            env=_env(Path(fresh), src),
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if not Path(where).resolve().is_relative_to(src.resolve()):
        raise SystemExit(f"the base's styleprofile was not the one imported ({where})")
    return sha, src


def _best(results: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        if key not in results:
            results[key] = value
        else:
            results[key] = (min if METRICS[key].best_is_min else max)(results[key], value)


def run_case(case: Case, trees: dict[str, Path], repeat: int) -> dict[str, dict[str, float]]:
    """Each tree's metrics for ``case``: the fastest of ``repeat`` runs, the largest memory.

    ``trees`` maps a name to the ``src`` directory to import; with two, their runs alternate
    in an ABBA order. A tree whose run fails is dropped from the result.
    """
    corpus = generate(case.corpus)
    stats = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
    results: dict[str, dict[str, float]] = {
        name: {"words": stats["words"], "contrast_words": stats["contrast_words"]} for name in trees
    }
    names = list(trees)
    failed: set[str] = set()
    for attempt in range(repeat):
        order = names if attempt % 2 == 0 else names[::-1]
        for name in order:
            if name in failed:
                continue
            folder = WORK / case.name / name
            folder.mkdir(parents=True, exist_ok=True)
            profile = folder / "reference.json"
            styleprofile = [sys.executable, "-m", "styleprofile"]
            build = [
                *styleprofile,
                "build",
                str(corpus / "writer"),
                "--contrast",
                str(corpus / "contrast"),
                "-o",
                str(profile),
            ]
            if not case.syntax:
                build.append("--no-syntax")
            score = [
                *styleprofile,
                "score",
                str(DRAFT),
                str(profile),
                "-o",
                str(folder / "score.json"),
            ]
            label = f"{case.name} ({name})" if len(trees) > 1 else case.name
            suffix = f"-{attempt + 1}" if repeat > 1 else ""
            try:
                print(f"  {label}: build{suffix}", file=sys.stderr, flush=True)
                build_s, build_cpu_s, build_mb = measure(
                    build, folder / f"build{suffix}.log", trees[name]
                )
                print(f"  {label}: score{suffix}", file=sys.stderr, flush=True)
                score_s, score_cpu_s, score_mb = measure(
                    score, folder / f"score{suffix}.log", trees[name]
                )
            except RunFailed as error:
                if name == CHANGE:
                    raise SystemExit(str(error)) from None
                print(f"the base failed, so it is not compared:\n{error}", file=sys.stderr)
                failed.add(name)
                continue
            _best(
                results[name],
                {
                    "build_s": build_s,
                    "build_cpu_s": build_cpu_s,
                    "build_mb": build_mb,
                    "profile_mb": profile.stat().st_size / MB,
                    "score_s": score_s,
                    "score_cpu_s": score_cpu_s,
                    "score_mb": score_mb,
                },
            )
    return {name: values for name, values in results.items() if name not in failed}


def _cell(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f} {unit}" if value < 100 else f"{value:,.0f} {unit}"


def report_table(results: dict[str, dict[str, float]]) -> tuple[str, list[str]]:
    """A Markdown table of every measured metric beside its target, and warnings for the
    values over ``BACKSTOP`` x their target. Targets never fail a run."""
    lines = ["| Case | Metric | Measured | Target | Status |", "|---|---|---|---|---|"]
    warnings: list[str] = []
    for name, measured in results.items():
        case = CASES[name]
        for metric, info in METRICS.items():
            value = measured[metric]
            target = case.targets.get(metric)
            if target is None:
                status = ""
            elif value <= target:
                status = "meets target"
            else:
                status = f"{value / target:.1f}x target"
                if value > BACKSTOP * target:
                    status += f", over {BACKSTOP:g}x"
                    warnings.append(
                        f"{name} {info.label} is {_cell(value, info.unit)}, over {BACKSTOP:g}x "
                        f"its target of {_cell(target, info.unit)}"
                    )
            cells = [name, info.label, _cell(value, info.unit), _cell(target, info.unit), status]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), warnings


def compare_table(verdicts: list[Verdict], pairs: dict[str, dict[str, dict[str, float]]]) -> str:
    """A Markdown table of change against base: every metric, and the gate's verdicts."""
    lines = [
        "| Case | Metric | Base | This change | Ratio | Allowed | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    judged = {(v.case, v.metric): v for v in verdicts}
    for name, pair in pairs.items():
        for metric, info in METRICS.items():
            base, change = pair[BASE][metric], pair[CHANGE][metric]
            verdict = judged.get((name, metric))
            ratio = f"{change / base:.3f}x" if base else "—"
            allowed = f"{verdict.allowed:.2f}x" if verdict else "—"
            status = verdict.status if verdict else "not gated"
            cells = [name, info.label, _cell(base, info.unit), _cell(change, info.unit)]
            lines.append("| " + " | ".join([*cells, ratio, allowed, status]) + " |")
    return "\n".join(lines)


def this_pr() -> int | None:
    """The pull request under test in GitHub Actions: from the merge ref of a pull_request
    run, or from the subject of the commit a push to main is testing (a squash merge ends in
    "(#N)"; a merge commit starts "Merge pull request #N")."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        match = re.fullmatch(r"refs/pull/(\d+)/merge", os.environ.get("GITHUB_REF", ""))
        return int(match.group(1)) if match else None
    subject = _git("log", "-1", "--format=%s", "HEAD")
    match = re.search(r"\(#(\d+)\)\s*$", subject) or re.match(r"Merge pull request #(\d+)", subject)
    return int(match.group(1)) if match else None


def _spacy_available() -> bool:
    return importlib.util.find_spec("spacy") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--case", action="append", choices=list(CASES), help="run this case; repeatable"
    )
    parser.add_argument(
        "--quick", action="store_true", help="run only the cases CI runs (medium-nosyntax)"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs per case (and per side); time is the fastest"
    )
    parser.add_argument(
        "--against", metavar="REF", help="also benchmark git revision REF and compare with it"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="with --against: exit 1 if the change is worse than its base beyond the margins",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="apply bench/accepted.toml's entries for this pull request (detected in CI)",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.check and not args.against:
        parser.error("--check needs --against: the gate compares with a base")
    pr = args.pr if args.pr is not None else this_pr()
    try:
        entries = load_accepted()
    except ValueError as error:
        raise SystemExit(f"bench/accepted.toml is invalid: {error}") from None

    trees = {CHANGE: ROOT / "src"}
    base_sha = None
    if args.against:
        base_sha, trees[BASE] = export_base(args.against)
        print(f"base: {args.against} ({base_sha[:12]})", file=sys.stderr)
        # The base goes first in each ABBA round's opening pair.
        trees = {BASE: trees[BASE], CHANGE: trees[CHANGE]}

    names = args.case or [name for name, case in CASES.items() if case.ci or not args.quick]
    runs: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        case = CASES[name]
        if case.syntax and not _spacy_available():
            print(f"skipping {name}: it needs spaCy (uv sync --extra syntax)", file=sys.stderr)
            continue
        print(f"{name}: {case.about}", file=sys.stderr, flush=True)
        runs[name] = run_case(case, trees, args.repeat)

    results = {name: pair[CHANGE] for name, pair in runs.items()}
    pairs = {name: pair for name, pair in runs.items() if BASE in pair}
    verdicts = [
        judge(name, metric, pair[BASE][metric], pair[CHANGE][metric], entries, pr)
        for name, pair in pairs.items()
        for metric in MARGINS
    ]

    report = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.machine(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "spacy": _spacy_available(),
        },
        "repeat": args.repeat,
        "cases": results,
    }
    if args.against:
        report["base"] = {
            "ref": args.against,
            "sha": base_sha,
            "cases": {name: pair[BASE] for name, pair in pairs.items()},
        }
        report["pr"] = pr
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    text, warnings = report_table(results)
    sections = [f"### Against the plan's targets (a report, not a gate)\n\n{text}"]
    if args.against:
        if pairs:
            base_label = f"`{args.against}` ({(base_sha or '')[:12]})"
            sections.insert(
                0,
                f"### This change against its base, {base_label}\n\n"
                + compare_table(verdicts, pairs),
            )
        for name in runs.keys() - pairs.keys():
            warnings.append(f"{name}: the base failed to run, so this case is not compared")
        warnings += [v.warning for v in verdicts if v.warning]
        warnings += inert(entries, pr)
    print("\n\n".join(sections))
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")

    problems: list[str] = []
    if args.check:
        failed = [v.problem() for v in verdicts if not v.passed]
        if failed:
            problems.append(
                "Worse than the base beyond the gate's margin. If the cost is intended, "
                "declare it in bench/accepted.toml (see README, Development):\n  "
                + "\n  ".join(failed)
            )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write("## Benchmark\n\n" + "\n\n".join(sections) + "\n")
            out.writelines(f"\n> [!WARNING]\n> {warning}\n" for warning in warnings)
            out.writelines(f"\n```\n{problem}\n```\n" for problem in problems)
    sys.stdout.flush()  # keep the problems after the table when both go to one pipe
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for warning in warnings:
        print(f"::warning::{warning}" if in_actions else f"\nwarning: {warning}", file=sys.stderr)
    for problem in problems:
        print("\n" + problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
