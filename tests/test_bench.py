"""The benchmark ratchet in ``bench/targets.py``: when a baseline is stale."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bench_targets", ROOT / "bench" / "targets.py")
assert _spec and _spec.loader
targets: Any = importlib.util.module_from_spec(_spec)
sys.modules["bench_targets"] = targets  # dataclasses look their module up while defining
_spec.loader.exec_module(targets)


def _case() -> Any:
    # Target 1.6s, so 2x target is 3.2s; baseline 2.94s gives a 4.41s budget.
    return targets.Case(
        "t", corpus="medium", syntax=False, about="", targets={"build_s": 1.6},
        baseline={"build_s": 2.94}, ci=True,
    )  # fmt: skip


def test_a_baseline_is_deleted_only_with_room_for_noise() -> None:
    case = _case()
    assert case.budget("build_s") == 2.94 * 1.5
    # Just under 2x target is not stale: deleting the baseline would leave a 3.2s budget
    # against a noisy 2.94s, so the run would flake.
    assert case.stale("build_s", 2.94, on_runner=True) is None
    assert case.stale("build_s", 3.1, on_runner=True) is None
    # Comfortably under (x 1.5 still within 3.2s): the baseline is no longer needed.
    assert "delete it" in case.stale("build_s", 2.1, on_runner=True)
    assert case.stale("build_s", 2.1, on_runner=False) is None, "only the CI runner judges time"


def test_a_baseline_beaten_by_more_than_noise_must_be_lowered() -> None:
    case = targets.Case(
        "t", corpus="medium", syntax=False, about="", targets={"build_s": 1.0},
        baseline={"build_s": 6.0}, ci=True,
    )  # fmt: skip
    assert "lower it" in case.stale("build_s", 3.9, on_runner=True)
    assert case.stale("build_s", 4.1, on_runner=True) is None


def test_the_ci_baselines_are_not_stale_at_their_own_value() -> None:
    for case in targets.CASES.values():
        for metric, baseline in case.baseline.items():
            assert case.stale(metric, baseline, on_runner=True) is None, (case.name, metric)
