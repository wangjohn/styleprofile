"""Benchmark cases and their performance targets: the one place both live.

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

    budget = max(CI_MARGIN x target, BASELINE_MARGIN x baseline)

where ``baseline`` is what the metric measured on GitHub's ``ubuntu-latest`` runner when the
budget was last set. The first term is the plan's "fail above 2x the target", which a met
target falls back to; the second keeps an unmet target from failing CI while still catching a
regression of more than 50%. When a PR makes a metric faster or smaller, it lowers (or
deletes) that baseline in the same PR, so the budget tightens towards 2x the target and the
gain can't quietly be lost again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CI_MARGIN = 2.0
BASELINE_MARGIN = 1.5

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
    # Measured on GitHub's ubuntu-latest runner; only for cases that CI runs. Lower a value
    # (or delete it once the target is met) whenever a PR improves on it.
    baseline: dict[str, float] = field(default_factory=dict)
    ci: bool = False  # run by the CI benchmark job and `--quick`

    def budget(self, metric: str) -> float | None:
        """The most CI accepts for ``metric``; None when CI doesn't run the case or no target."""
        if not self.ci or metric not in self.targets:
            return None
        return max(CI_MARGIN * self.targets[metric], BASELINE_MARGIN * self.baseline.get(metric, 0))


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
            # Only the unmet target needs a baseline; the met ones fall back to 2x target.
            # PR 14 (no per-chunk rows in the profile) should let this baseline go.
            baseline={"profile_mb": 2.22},
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
