"""CI's relative benchmark gate (``bench/gate.py``) and its accepted regressions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))  # the bench scripts import each other as top-level modules
gate: Any = importlib.import_module("gate")
targets: Any = importlib.import_module("targets")
run: Any = importlib.import_module("run")

CASE = "medium-nosyntax"


def _entry(**overrides: Any) -> Any:
    fields = {
        "pr": 15,
        "case": CASE,
        "metric": "build_cpu_s",
        "ratio": 1.9,
        "reason": "Length-aware verdicts bootstrap each length bucket.",
    }
    return gate.Accepted(**{**fields, **overrides})


def test_the_gate_allows_noise_and_fails_a_regression() -> None:
    assert targets.MARGINS["build_cpu_s"].ratio == 1.25
    assert gate.judge(CASE, "build_cpu_s", 2.0, 2.4, [], None).passed
    verdict = gate.judge(CASE, "build_cpu_s", 2.0, 2.6, [], None)
    assert not verdict.passed
    assert verdict.status == "REGRESSION"
    assert "1.30x" in verdict.problem()
    # Faster is always fine.
    assert gate.judge(CASE, "build_cpu_s", 2.0, 1.0, [], None).passed


def test_a_tiny_difference_is_noise_whatever_the_ratio() -> None:
    # Score takes a few hundredths of a second, where 20 ms is a large ratio but only noise.
    assert gate.judge(CASE, "score_cpu_s", 0.05, 0.09, [], None).passed
    assert not gate.judge(CASE, "score_cpu_s", 0.05, 0.13, [], None).passed
    # Profile size is deterministic: no floor.
    assert not gate.judge(CASE, "profile_mb", 0.050, 0.053, [], None).passed


def test_an_accepted_regression_passes_only_for_its_own_pr_and_up_to_its_ratio() -> None:
    entries = [_entry()]
    verdict = gate.judge(CASE, "build_cpu_s", 2.0, 3.4, entries, 15)
    assert verdict.passed
    assert verdict.status == "accepted regression (#15)"
    over = gate.judge(CASE, "build_cpu_s", 2.0, 4.0, entries, 15)
    assert not over.passed
    assert "accepts up to 1.9x for #15" in over.problem()
    # Another PR, or no PR at all, gets the normal margin.
    assert not gate.judge(CASE, "build_cpu_s", 2.0, 3.4, entries, 16).passed
    assert not gate.judge(CASE, "build_cpu_s", 2.0, 3.4, entries, None).passed
    # The entry is for one case and metric only.
    assert not gate.judge(CASE, "build_mb", 100.0, 150.0, entries, 15).passed


def test_stale_entries_are_flagged() -> None:
    entries = [_entry()]
    unneeded = gate.judge(CASE, "build_cpu_s", 2.0, 2.1, entries, 15)
    assert unneeded.passed
    assert "delete the entry" in unneeded.warning
    assert gate.inert(entries, 15) == []
    [warning] = gate.inert(entries, 16)
    assert "#15" in warning
    assert "delete it once #15 has merged" in warning


def _load(tmp_path: Path, text: str) -> list[Any]:
    path = tmp_path / "accepted.toml"
    path.write_text(text, encoding="utf-8")
    return gate.load_accepted(path)


ENTRY = """
[[regression]]
pr = 15
case = "medium-nosyntax"
metric = "build_cpu_s"
ratio = 2
reason = "Length-aware verdicts."
"""


def test_accepted_toml_is_parsed(tmp_path: Path) -> None:
    assert _load(tmp_path, "# nothing accepted\n") == []
    assert gate.load_accepted(tmp_path / "missing.toml") == []
    [entry] = _load(tmp_path, ENTRY)
    assert entry == _entry(ratio=2.0, reason="Length-aware verdicts.")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("build_cpu_s", "build_s"), "is not gated"),
        (('"medium-nosyntax"', '"medium-fast"'), "unknown case"),
        (("ratio = 2", "ratio = 1.1"), "within the normal margin"),
        (('"Length-aware verdicts."', '"  "'), "give a reason"),
        (("pr = 15", 'pr = "15"'), "pr must be a int"),
        (('reason = "Length-aware verdicts."', ""), "needs exactly the keys"),
        (("[[regression]]", "[[regressions]]"), "unknown key"),
    ],
)
def test_accepted_toml_mistakes_are_named(
    tmp_path: Path, change: tuple[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, ENTRY.replace(*change))


def test_a_duplicate_entry_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _load(tmp_path, ENTRY + ENTRY)


def test_the_repository_accepted_toml_is_valid() -> None:
    gate.load_accepted()


def test_every_gated_metric_is_measured_and_every_target_is_a_metric() -> None:
    assert set(targets.MARGINS) <= set(targets.METRICS)
    for case in targets.CASES.values():
        assert set(case.targets) <= set(targets.METRICS), case.name
    assert any(case.ci for case in targets.CASES.values())


def test_repeated_runs_keep_the_fastest_time_and_the_largest_memory() -> None:
    results: dict[str, float] = {}
    run._best(results, {"build_cpu_s": 2.0, "build_mb": 100.0})
    run._best(results, {"build_cpu_s": 1.5, "build_mb": 120.0})
    run._best(results, {"build_cpu_s": 1.8, "build_mb": 110.0})
    assert results == {"build_cpu_s": 1.5, "build_mb": 120.0}


def test_the_pr_is_detected_from_a_pull_request_merge_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/15/merge")
    assert run.this_pr() == 15
    monkeypatch.delenv("GITHUB_ACTIONS")
    assert run.this_pr() is None, "only in GitHub Actions"
