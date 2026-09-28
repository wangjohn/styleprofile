"""CI's relative regression gate: judge a change's benchmark against its base's.

A metric in ``targets.MARGINS`` fails when the change is worse than its base by more than the
metric's margin: ``change > base x ratio`` and ``change - base > floor``. A change that knowingly
costs more declares it in ``bench/accepted.toml``, one entry per case and metric::

    [[regression]]
    pr = 15                      # the pull request that accepts the cost
    case = "medium-nosyntax"     # a case in bench/targets.py
    metric = "build_cpu_s"       # a metric in targets.MARGINS
    ratio = 1.9                  # the most this PR may cost, as change / base
    reason = "Length-aware verdicts bootstrap each length bucket separately."

An entry applies only to the pull request it names (on the PR's own runs, and on the push to
main that merges it), so it cannot silently widen the gate for any later change. Once that PR
has merged its entry is inert, and the gate warns on every other run until someone deletes it.
The gate also warns when an entry's own PR didn't need it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from targets import CASES, MARGINS, METRICS

ACCEPTED = Path(__file__).resolve().parent / "accepted.toml"
FIELDS = {"pr": int, "case": str, "metric": str, "ratio": float, "reason": str}


@dataclass(frozen=True)
class Accepted:
    pr: int
    case: str
    metric: str
    ratio: float
    reason: str


def load_accepted(path: Path = ACCEPTED) -> list[Accepted]:
    """The entries in ``path``; raises ValueError naming the first problem in it."""
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path.name}: {error}") from None
    unknown = set(data) - {"regression"}
    if unknown:
        raise ValueError(f"{path.name}: unknown key(s) {sorted(unknown)}; use [[regression]]")
    entries: list[Accepted] = []
    seen: set[tuple[int, str, str]] = set()
    for number, raw in enumerate(data.get("regression", []), start=1):
        where = f"{path.name}, entry {number}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a [[regression]] table")
        if set(raw) != set(FIELDS):
            raise ValueError(f"{where}: needs exactly the keys {', '.join(FIELDS)}")
        for key, kind in FIELDS.items():
            value = raw[key]
            ok = isinstance(value, (int, float)) if kind is float else isinstance(value, kind)
            if not ok or isinstance(value, bool):
                raise ValueError(f"{where}: {key} must be a {kind.__name__}")
        entry = Accepted(
            pr=raw["pr"],
            case=raw["case"],
            metric=raw["metric"],
            ratio=float(raw["ratio"]),
            reason=raw["reason"].strip(),
        )
        if entry.case not in CASES:
            raise ValueError(f"{where}: unknown case {entry.case!r}; one of {sorted(CASES)}")
        if entry.metric not in MARGINS:
            raise ValueError(f"{where}: {entry.metric!r} is not gated; one of {sorted(MARGINS)}")
        if entry.ratio <= MARGINS[entry.metric].ratio:
            raise ValueError(
                f"{where}: ratio {entry.ratio:g} is within the normal margin "
                f"({MARGINS[entry.metric].ratio:g}); no entry is needed"
            )
        if not entry.reason:
            raise ValueError(f"{where}: give a reason")
        key = (entry.pr, entry.case, entry.metric)
        if key in seen:
            raise ValueError(
                f"{where}: duplicate entry for #{entry.pr} {entry.case} {entry.metric}"
            )
        seen.add(key)
        entries.append(entry)
    return entries


@dataclass(frozen=True)
class Verdict:
    case: str
    metric: str
    base: float
    change: float
    allowed: float  # the ratio this change may reach
    passed: bool
    accepted: Accepted | None  # the entry that let it pass, or that it exceeded
    warning: str | None  # an entry that wasn't needed

    @property
    def ratio(self) -> float:
        return self.change / self.base if self.base else float("inf") if self.change else 1.0

    @property
    def status(self) -> str:
        if self.passed and self.accepted:
            return f"accepted regression (#{self.accepted.pr})"
        if self.passed:
            return "ok"
        return "REGRESSION"

    def problem(self) -> str:
        unit = METRICS[self.metric].unit
        label = f"{self.case} {METRICS[self.metric].label}"
        text = (
            f"{label}: {self.change:.3g} {unit} is {self.ratio:.2f}x its base's "
            f"{self.base:.3g} {unit}; the most allowed is {self.allowed:.2f}x"
        )
        if self.accepted:
            entry = self.accepted
            text += f" (bench/accepted.toml accepts up to {entry.ratio:g}x for #{entry.pr})"
        return text


def judge(
    case: str, metric: str, base: float, change: float, entries: list[Accepted], pr: int | None
) -> Verdict:
    """Judge ``metric`` of ``case``; ``entries`` apply only when their ``pr`` is this one."""
    margin = MARGINS[metric]
    within = change <= base * margin.ratio or change - base <= margin.floor
    entry = next((e for e in entries if e.pr == pr and e.case == case and e.metric == metric), None)
    ratio = change / base if base else float("inf")
    if within:
        warning = None
        if entry:
            warning = (
                f"bench/accepted.toml: #{entry.pr} accepts {case} {metric} up to "
                f"{entry.ratio:g}x, but it measured {ratio:.2f}x, within the normal "
                f"{margin.ratio:g}x; delete the entry"
            )
        return Verdict(case, metric, base, change, margin.ratio, True, None, warning)
    if entry:
        return Verdict(case, metric, base, change, entry.ratio, ratio <= entry.ratio, entry, None)
    return Verdict(case, metric, base, change, margin.ratio, False, None, None)


def inert(entries: list[Accepted], pr: int | None) -> list[str]:
    """Warnings for the entries that don't apply to this change: they belong to another PR."""
    return [
        f"bench/accepted.toml: the entry for #{e.pr} ({e.case} {e.metric}) doesn't apply to "
        f"{f'#{pr}' if pr else 'this change'}; delete it once #{e.pr} has merged"
        for e in entries
        if e.pr != pr
    ]
