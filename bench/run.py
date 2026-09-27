"""Run the benchmark cases and compare them with the plan's performance targets.

    python bench/run.py [--case NAME ...] [--quick] [--repeat N] [--check]

Each case generates its corpus if needed (``bench/gen.py``), builds a reference from it with
``styleprofile build`` and scores ``examples/draft.md`` against that reference, each in a fresh
Python process. For both steps it records wall time and the child's peak resident memory (from
``os.wait4``, which works on macOS and Linux); for the build it also records the profile's size.
Results go to ``bench/results.json`` and a table is printed beside the targets in
``bench/targets.py``. ``--check`` exits 1 when a value is over its CI regression budget.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gen import generate
from targets import CASES, METRICS, Case

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DRAFT = ROOT / "examples" / "draft.md"
WORK = HERE / "work"  # git-ignored: profiles, reports and logs of the last run
RESULTS = HERE / "results.json"  # git-ignored
MB = 1_000_000


def _peak_bytes(maxrss: int) -> int:
    """ru_maxrss is in bytes on macOS and in kibibytes on Linux."""
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def measure(command: list[str], log: Path) -> tuple[float, float]:
    """Run ``command`` to completion; return its wall time (s) and peak memory (MB)."""
    env = {**os.environ, "NO_COLOR": "1"}
    with log.open("w", encoding="utf-8") as out:
        start = time.perf_counter()
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        # wait4 reaps this one child and returns its own resource usage, where getrusage
        # would give the maximum over every child so far.
        _, status, rusage = os.wait4(process.pid, 0)
        elapsed = time.perf_counter() - start
    process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode != 0:
        tail = log.read_text(encoding="utf-8").strip().splitlines()[-10:]
        raise SystemExit(f"{' '.join(command)} exited {process.returncode}:\n" + "\n".join(tail))
    return elapsed, _peak_bytes(rusage.ru_maxrss) / MB


def run_case(case: Case, repeat: int) -> dict[str, float]:
    """The case's metrics: the fastest of ``repeat`` runs, and the largest peak memory."""
    corpus = generate(case.corpus)
    stats = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
    folder = WORK / case.name
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
    score = [*styleprofile, "score", str(DRAFT), str(profile), "-o", str(folder / "score.json")]
    result: dict[str, float] = {"words": stats["words"], "contrast_words": stats["contrast_words"]}
    for attempt in range(repeat):
        suffix = f"-{attempt + 1}" if repeat > 1 else ""
        print(f"  {case.name}: build{suffix}", file=sys.stderr, flush=True)
        build_s, build_mb = measure(build, folder / f"build{suffix}.log")
        print(f"  {case.name}: score{suffix}", file=sys.stderr, flush=True)
        score_s, score_mb = measure(score, folder / f"score{suffix}.log")
        for key, value, best in (
            ("build_s", build_s, min),
            ("build_mb", build_mb, max),
            ("score_s", score_s, min),
            ("score_mb", score_mb, max),
        ):
            result[key] = best(result[key], value) if key in result else value
    result["profile_mb"] = profile.stat().st_size / MB
    return result


def _cell(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f} {unit}" if value < 100 else f"{value:,.0f} {unit}"


def table(results: dict[str, dict[str, float]]) -> tuple[str, list[str]]:
    """A Markdown table of every measured metric, and the metrics over their CI budget."""
    lines = [
        "| Case | Metric | Measured | Target | CI budget | Status |",
        "|---|---|---|---|---|---|",
    ]
    over: list[str] = []
    for name, measured in results.items():
        case = CASES[name]
        for metric, (unit, label) in METRICS.items():
            value = measured[metric]
            target = case.targets.get(metric)
            budget = case.budget(metric)
            if target is None:
                status = ""
            elif value <= target:
                status = "meets target"
            elif budget is None or value <= budget:
                status = f"{value / target:.1f}x target"
            else:
                status = f"{value / target:.1f}x target, OVER BUDGET"
                over.append(f"{name} {label}: {_cell(value, unit)} > {_cell(budget, unit)}")
            cells = [name, label, _cell(value, unit), _cell(target, unit), _cell(budget, unit)]
            lines.append("| " + " | ".join([*cells, status]) + " |")
    return "\n".join(lines), over


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
        "--repeat", type=int, default=1, help="runs per case; wall time is the fastest (default 1)"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if a value is over its CI regression budget"
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    names = args.case or [name for name, case in CASES.items() if case.ci or not args.quick]
    results: dict[str, dict[str, float]] = {}
    for name in names:
        case = CASES[name]
        if case.syntax and not _spacy_available():
            print(f"skipping {name}: it needs spaCy (uv sync --extra syntax)", file=sys.stderr)
            continue
        print(f"{name}: {case.about}", file=sys.stderr, flush=True)
        results[name] = run_case(case, args.repeat)
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
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text, over = table(results)
    print(text)
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    if args.check and over:
        print("\nOver the CI regression budget:\n  " + "\n  ".join(over), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
