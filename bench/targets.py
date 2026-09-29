"""Benchmark cases, their performance targets, and the margins of CI's regression gate.

The targets come from the "Performance targets" table of the engineering plan (measured on a
4-core laptop):

| Case                                      | Target                           |
|-------------------------------------------|----------------------------------|
| Score one draft, including spaCy load     | < 1s                             |
| Build 200k words, with spaCy              | < 12s, with progress             |
| Build 1M words, without spaCy             | < 8s                             |
| Build 20k JSONL comments, without spaCy   | < 10s                            |
| Reference profile size                    | < 1 MB regardless of corpus size |
| Peak memory at 1M words                   | < 300 MB                         |

Each case below builds a reference from one corpus of ``bench/gen.py`` (with its contrast set),
then scores ``examples/draft.md`` against it. Targets marked "derived" are not in the table
but follow from it; see the comments beside them.

The targets are a *report*, not a gate: GitHub's shared runners vary by up to ~75% in wall time
from one run of the same code to the next, so no absolute budget is both tight and reliable.
CI instead gates on a *relative* comparison (``run.py --against``): it benchmarks the change's
base in the same job, on the same runner, interleaving base and change runs, and fails when the
change is worse than its base by more than ``MARGINS`` allow. Time is judged on CPU time
(user + system), which a busy neighbour on the runner inflates far less than wall time; wall
time is still measured and reported. A change that knowingly costs more declares it in
``bench/accepted.toml``. Values over ``BACKSTOP`` x their target only raise a warning.

CPU time and peak memory cover the whole process tree, since a build with spaCy parses in
worker processes (see ``run.py``). A ``warm`` case measures a build whose measurement cache an
untimed build of the same corpus has just filled: what rebuilding after a small change costs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    unit: str
    label: str
    # Keep the smallest value over repeated runs (time, whose noise only ever adds) rather than
    # the largest (memory, where the worst run is the one that matters).
    best_is_min: bool


# Metric keys, their units and labels, in the order the tables list them.
METRICS = {
    "build_s": Metric("s", "build wall time", best_is_min=True),
    "build_cpu_s": Metric("s", "build CPU time", best_is_min=True),
    "build_mb": Metric("MB", "build peak memory", best_is_min=False),
    "profile_mb": Metric("MB", "reference profile size", best_is_min=False),
    "score_s": Metric("s", "score wall time", best_is_min=True),
    "score_cpu_s": Metric("s", "score CPU time", best_is_min=True),
    "score_mb": Metric("MB", "score peak memory", best_is_min=False),
}


@dataclass(frozen=True)
class Margin:
    ratio: float  # the most the change may exceed its base, as change / base
    # A difference at most this large (in the metric's unit) is noise whatever the ratio: it
    # keeps a tiny value (say a 0.2s CPU time) from failing on a few milliseconds.
    floor: float


# The relative gate: only these metrics are judged against the base. Wall time is not (CPU
# time stands in for it). The ratios come from seven A/A runs of the CI job, where base and
# change are the same code (#19): build time swung between 2.47s and 3.23s from run to run,
# but CPU time stayed within 1% of its base's in every run, peak memory within 0.3%, and
# profile size is deterministic. The margins leave over ten times that noise, so they don't
# flake, while catching a 15% slowdown or a 10% memory increase.
MARGINS = {
    "build_cpu_s": Margin(ratio=1.15, floor=0.05),
    "build_mb": Margin(ratio=1.10, floor=5.0),
    "profile_mb": Margin(ratio=1.05, floor=0.0),
    "score_cpu_s": Margin(ratio=1.15, floor=0.05),
    "score_mb": Margin(ratio=1.10, floor=5.0),
}

# Values over this multiple of their target get a warning in the report; they never fail CI.
BACKSTOP = 3.0


@dataclass(frozen=True)
class Case:
    name: str
    corpus: str  # a corpus name from bench/gen.py
    syntax: bool  # build (and so score) with the spaCy parser
    about: str
    # The plan's targets, keyed by metric. Time targets are wall time, as the plan states them.
    targets: dict[str, float]
    ci: bool = False  # run by the CI benchmark job and `--quick`
    warm: bool = False  # build once, untimed, to fill the measurement cache first
    flags: tuple[str, ...] = ()  # more build options


PROFILE_MB = 1.0  # "< 1 MB regardless of corpus size"
SCORE_S = 1.0  # "Score one draft, including spaCy load: < 1s"
PEAK_MB_1M = 300.0  # "Peak memory at 1M words: < 300 MB"

CASES = {
    case.name: case
    for case in (
        Case(
            "medium",
            corpus="medium",
            syntax=True,
            about="build 200k words with spaCy; score one draft including spaCy load",
            targets={"build_s": 12.0, "profile_mb": PROFILE_MB, "score_s": SCORE_S},
        ),
        Case(
            "medium-warm",
            corpus="medium",
            syntax=True,
            about="build 200k words with spaCy again, from a warm measurement cache",
            # Not in the plan: a rebuild after a small change should take seconds.
            targets={"build_s": 3.0},
            warm=True,
        ),
        Case(
            "medium-nosyntax",
            corpus="medium",
            syntax=False,
            about="build 200k words without spaCy; score one draft",
            targets={
                # Derived: the 1M-word target (8s) scaled linearly to 200k words.
                "build_s": 1.6,
                # Derived: the 1M-word memory target holds for a fifth of the words too.
                "build_mb": PEAK_MB_1M,
                "profile_mb": PROFILE_MB,
                "score_s": SCORE_S,
            },
            ci=True,
        ),
        Case(
            "big",
            corpus="big",
            syntax=False,
            about="build 1M words without spaCy; score one draft",
            targets={
                "build_s": 8.0,
                "build_mb": PEAK_MB_1M,
                "profile_mb": PROFILE_MB,
                "score_s": SCORE_S,
            },
        ),
        Case(
            "comments",
            corpus="comments",
            syntax=False,
            about="build 20k JSONL comments (1M words) without spaCy; score one draft",
            targets={
                "build_s": 10.0,
                # Derived: the comments corpus is also about 1M words.
                "build_mb": PEAK_MB_1M,
                "profile_mb": PROFILE_MB,
                "score_s": SCORE_S,
            },
            # Pooled into windows by default (plan PR 10).
        ),
        Case(
            "comments-nopool",
            corpus="comments",
            syntax=False,
            about="build 20k JSONL comments unpooled (20k documents) without spaCy",
            # The same targets: 20,000 documents of one comment each must be fast and lean too.
            targets={
                "build_s": 10.0,
                "build_mb": PEAK_MB_1M,
                "profile_mb": PROFILE_MB,
                "score_s": SCORE_S,
            },
            flags=("--no-pool",),
        ),
    )
}
