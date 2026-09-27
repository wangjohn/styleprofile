"""Benchmark cases and their performance targets: the one place both live.

Rule: a PR that improves a metric must lower or delete its baseline here in the same PR.

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

CI enforces a *regression budget* rather than the target itself, because several targets are
aspirational and are met only by later PRs (13 and 14):

    budget = max(CI_MARGIN x target, NOISE[metric] x baseline)

where ``baseline`` is what the metric measured on the CI runner (GitHub's ``ubuntu-24.04``)
when the budget was last set, and ``NOISE`` is the metric's noise margin: 1.5 for wall time
and memory, 1.05 for the deterministic profile size. The first term is the plan's "fail above
2x the target", which a met target falls back to; the second keeps an unmet target from
failing CI while still catching a regression.

The ratchet is enforced: ``run.py --check`` also fails when a baseline is *stale*, that is when
the metric is back within 2x its target (the baseline is no longer needed) or below
``baseline / NOISE`` (it improved by more than noise). The PR that made the improvement must
then lower or delete the baseline, so the budget tightens towards 2x the target and the gain
can't quietly be lost again. Time and memory baselines are only judged stale on the CI runner
(in GitHub Actions), where they were measured; profile size is judged on any machine.

Peak memory is the main ``styleprofile`` process's alone (``ru_maxrss`` from ``os.wait4``),
not its process tree. That is exact while styleprofile runs in one process; PR 13, which adds
spaCy's ``n_process`` workers, is responsible for measuring the whole tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CI_MARGIN = 2.0
# How far a metric may drift from its baseline through noise alone, per metric.
NOISE = {
    "build_s": 1.5,
    "build_mb": 1.5,
    "profile_mb": 1.05,  # deterministic but for path lengths in the stored sources
    "score_s": 1.5,
    "score_mb": 1.5,
}
# Metrics that don't depend on the machine, so any run can judge their baselines.
DETERMINISTIC = {"profile_mb"}

# Metric keys, their units, and how to label them in the results table.
METRICS = {
    "build_s": ("s", "build wall time"),
    "build_mb": ("MB", "build peak memory"),
    "profile_mb": ("MB", "reference profile size"),
    "score_s": ("s", "score wall time"),
    "score_mb": ("MB", "score peak memory"),
}


@dataclass(frozen=True)
class Case:
    name: str
    corpus: str  # a corpus name from bench/gen.py
    syntax: bool  # build (and so score) with the spaCy parser
    about: str
    targets: dict[str, float]
    # Measured on the CI runner; only for cases that CI runs. Lower a value (or delete it once
    # the metric is within 2x its target) whenever a PR improves on it; --check insists.
    baseline: dict[str, float] = field(default_factory=dict)
    ci: bool = False  # run by the CI benchmark job and `--quick`

    def budget(self, metric: str) -> float | None:
        """The most CI accepts for ``metric``; None when CI doesn't run the case or no target."""
        if not self.ci or metric not in self.targets:
            return None
        return max(CI_MARGIN * self.targets[metric], NOISE[metric] * self.baseline.get(metric, 0))

    def stale(self, metric: str, value: float, on_runner: bool) -> str | None:
        """Why ``metric``'s baseline should be lowered or deleted, or None if it shouldn't.

        Time and memory baselines belong to the CI runner, so only a run there (``on_runner``)
        can find them stale; a faster laptop would always look like an improvement.
        """
        if not self.ci or metric not in self.baseline or metric not in self.targets:
            return None
        if metric not in DETERMINISTIC and not on_runner:
            return None
        baseline = self.baseline[metric]
        if value <= CI_MARGIN * self.targets[metric]:
            return "is within 2x its target, so its baseline is no longer needed; delete it"
        if value < baseline / NOISE[metric]:
            return f"improved on its baseline ({baseline:g}) by more than noise; lower it"
        return None


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
            # Only values over 2x target need a baseline; the rest fall back to 2x target.
            # build_s meets its target on a laptop (1.3s) but the shared runner is about 2.8x
            # slower; PRs 13 and 14 should bring it under 3.2s there and let the baseline go.
            # profile_mb goes with PR 14, which drops per-chunk rows from the profile.
            baseline={"build_s": 3.72, "profile_mb": 2.21},
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
        ),
    )
}
