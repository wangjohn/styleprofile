"""Personal preferences learned from rated passages, alongside unchanged style scores.

Rhetorical cues are observations with evidence, not universal AI markers. Ratings train
separate voice and revision targets. Source-group holdouts are development diagnostics;
feedback used to design these features cannot also be an independent evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from styleprofile import Profile, Text
from styleprofile.calibration import MIN_JUDGED_WORDS
from styleprofile.core import StyleProfileError
from styleprofile.schema import ReferenceReport
from styleprofile.surface import WORD, prose
from styleprofile.surface import sentences as split_sentences

VERSION = 1
FEATURES = (
    "distance",
    "contrast",
    "hypothetical_prompts",
    "observation_prefaces",
    "short_leadin_pairs",
    "lesson_statements",
)
LABELS = {
    "hypothetical_prompts": "Hypothetical reader prompt",
    "observation_prefaces": "Observation preface",
    "short_leadin_pairs": "Short sentence followed by a longer sentence",
    "lesson_statements": "Explicit lesson statement",
}
PATTERNS = {
    "hypothetical_prompts": re.compile(
        r"^(?:now\s+)?(?:picture|imagine)\b|"
        r"^consider\s+(?:(?:a|an|two|three|several)\s+)?hypothetical\b",
        re.I,
    ),
    "observation_prefaces": re.compile(
        r"^(?:there(?:['\N{RIGHT SINGLE QUOTATION MARK}]s| is) a pattern\b|i keep noticing\b|"
        r"(?:one|a) (?:pattern|thing) i (?:keep )?(?:notice|noticed|noticing)\b)",
        re.I,
    ),
    "lesson_statements": re.compile(
        r"\b(?:that|this) (?:is|was) (?:the )?(?:point|lesson|key|difference)\b", re.I
    ),
}
REVISION = {"keep": 0, "light": 1, "substantial": 2}
WARNING = (
    "Experimental personal preference estimate, not an AI probability or an editing verdict. "
    "Features were informed by these judgments; source-group holdouts are post-hoc diagnostics. "
    "Fresh rated passages are needed to establish improvement."
)


def rhetoric(text: str) -> dict[str, Any]:
    """Rates per 1,000 prose words and exact spans in the returned normalized text."""
    normalized = prose(text).text
    n = len(WORD.findall(normalized))
    if not n:
        raise StyleProfileError("feedback scoring needs nonempty prose")
    counts = dict.fromkeys(LABELS, 0)
    evidence = []
    offset = 0
    for block in normalized.split("\n\n"):
        sentences = split_sentences(block)
        cursor = 0
        for index, sentence in enumerate(sentences):
            start = block.find(sentence, cursor)
            if start < 0:
                raise StyleProfileError("could not locate a rhetorical evidence span")
            cursor = start + len(sentence)
            found = [name for name, pattern in PATTERNS.items() if pattern.search(sentence)]
            if (
                index + 1 < len(sentences)
                and len(WORD.findall(sentence)) <= 8
                and len(WORD.findall(sentences[index + 1])) >= 9
            ):
                found.append("short_leadin_pairs")
            for name in found:
                counts[name] += 1
                evidence.append(
                    {
                        "metric": name,
                        "label": LABELS[name],
                        "start": offset + start,
                        "end": offset + cursor,
                        "text": sentence,
                    }
                )
        offset += len(block) + 2
    return {
        "text": normalized,
        "words": n,
        "counts": counts,
        "rates_per_1k": {name: value * 1000 / n for name, value in counts.items()},
        "evidence": evidence,
    }


def features(text: str, reference: Profile) -> tuple[list[float], dict[str, Any]]:
    observed = rhetoric(text)
    score = reference.score(Text(observed["text"], "feedback-input"))
    if score.delta is None or score.likeness is None:
        raise StyleProfileError("feedback learning requires a comparable contrast reference")
    values = [score.delta, score.likeness, *observed["rates_per_1k"].values()]
    observed["baseline"] = {
        "distance": score.delta,
        "contrast": score.likeness,
        "distance_label": score.verdict.value,
        "contrast_label": score.likeness_verdict.value if score.likeness_verdict else None,
    }
    return values, observed


def _fit(x: list[list[float]], y: list[float]) -> dict[str, Any]:
    """Fixed ridge penalty; only the training rows determine scaling and intercept."""
    means = [statistics.mean(col) for col in zip(*x, strict=True)]
    scales = [statistics.pstdev(col) or 1.0 for col in zip(*x, strict=True)]
    z = [[(v - m) / s for v, m, s in zip(row, means, scales, strict=True)] for row in x]
    intercept = statistics.mean(y)
    p = len(means)
    matrix = [
        [sum(row[i] * row[j] for row in z) + (8.0 if i == j else 0.0) for j in range(p)]
        for i in range(p)
    ]
    rhs = [
        sum(row[i] * (target - intercept) for row, target in zip(z, y, strict=True))
        for i in range(p)
    ]
    augmented = [[*row, b] for row, b in zip(matrix, rhs, strict=True)]
    for i in range(p):
        pivot = max(range(i, p), key=lambda j: abs(augmented[j][i]))
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        divisor = augmented[i][i]
        augmented[i] = [v / divisor for v in augmented[i]]
        for j in range(p):
            if j != i:
                multiple = augmented[j][i]
                augmented[j] = [
                    a - multiple * b for a, b in zip(augmented[j], augmented[i], strict=True)
                ]
    return {
        "means": means,
        "scales": scales,
        "intercept": intercept,
        "weights": [row[-1] for row in augmented],
        "ridge_penalty": 8.0,
    }


def _predict(model: dict[str, Any], row: list[float], low: float, high: float) -> float:
    value = model["intercept"] + sum(
        w * (v - m) / s
        for w, v, m, s in zip(model["weights"], row, model["means"], model["scales"], strict=True)
    )
    return max(low, min(high, value))


class FeedbackProfile:
    """Optional, portable preference layer. A saved profile embeds its base reference."""

    def __init__(self, report: dict[str, Any]) -> None:
        if report.get("kind") != "feedback-profile" or report.get("version") != VERSION:
            raise StyleProfileError("unsupported feedback profile")
        if report.get("features") != list(FEATURES):
            raise StyleProfileError("feedback metrics differ; rebuild from saved ratings")
        if set(report.get("models", {})) != {"voice", "revision"} or set(
            report.get("deployment", {})
        ) != {"voice", "revision"}:
            raise StyleProfileError(
                "feedback profile needs both target models and enablement flags"
            )
        self.report = report
        self.reference = Profile(cast(ReferenceReport, report["reference"]))
        for model in report["models"].values():
            for name in ["weights", "means", "scales"]:
                if len(model[name]) != len(FEATURES) or not all(
                    math.isfinite(v) for v in model[name]
                ):
                    raise StyleProfileError("invalid feedback model parameters")
            if any(s <= 0 for s in model["scales"]) or not math.isfinite(model["intercept"]):
                raise StyleProfileError("invalid feedback model scaling")

    @classmethod
    def learn(cls, examples: Sequence[dict[str, Any]], reference: Profile) -> FeedbackProfile:
        records = list(examples)
        if len(records) < 6:
            raise StyleProfileError("feedback learning needs at least six rated passages")
        ids, groups, x, voice, revision, history = set(), [], [], [], [], []
        for record in records:
            if not isinstance(record.get("id"), str) or not record["id"] or record["id"] in ids:
                raise StyleProfileError("feedback examples need unique nonempty string IDs")
            ids.add(record["id"])
            if not isinstance(record.get("source_post_id"), str) or not record["source_post_id"]:
                raise StyleProfileError("every feedback example needs its source_post_id")
            if type(record.get("voice")) is not int or record["voice"] not in range(1, 6):
                raise StyleProfileError("voice ratings must be integers from 1 to 5")
            if record.get("revision") not in REVISION:
                raise StyleProfileError("revision must be keep, light or substantial")
            if not isinstance(record.get("text"), str) or not isinstance(record.get("notes"), str):
                raise StyleProfileError("feedback examples need text and notes")
            row, _ = features(record["text"], reference)
            x.append(row)
            group = record.get("topic_family", record["source_post_id"])
            if not isinstance(group, str) or not group:
                raise StyleProfileError("topic_family must be a nonempty string when supplied")
            groups.append(group)
            voice.append(float(record["voice"]))
            revision.append(float(REVISION[record["revision"]]))
            history.append(
                {k: record[k] for k in ["id", "source_post_id", "voice", "revision", "notes"]}
            )
            history[-1]["holdout_group"] = group
            history[-1]["text_sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
        if len(set(groups)) < 3:
            raise StyleProfileError("feedback learning needs at least three source groups")
        diagnostics: dict[str, Any] = {
            "scope": "post-hoc source-group holdouts; not fresh validation",
            "groups": len(set(groups)),
            "folds": [],
        }
        errors: dict[str, list[float]] = {}
        for group in sorted(set(groups)):
            train = [i for i, g in enumerate(groups) if g != group]
            test = [i for i, g in enumerate(groups) if g == group]
            diagnostics["folds"].append(
                {
                    "held_out_group": group,
                    "train_ids": [history[i]["id"] for i in train],
                    "test_ids": [history[i]["id"] for i in test],
                }
            )
            for target, y, low, high in [("voice", voice, 1, 5), ("revision", revision, 0, 2)]:
                for family, dimensions in [("baseline", 2), ("with_rhetoric", len(FEATURES))]:
                    model = _fit([x[i][:dimensions] for i in train], [y[i] for i in train])
                    losses = [
                        abs(_predict(model, x[i][:dimensions], low, high) - y[i]) for i in test
                    ]
                    errors.setdefault(f"{target}/{family}", []).append(statistics.mean(losses))
                constant = statistics.mean(y[i] for i in train)
                errors.setdefault(f"{target}/mean_only", []).append(
                    statistics.mean(abs(constant - y[i]) for i in test)
                )
        diagnostics["source_weighted_mae"] = {k: statistics.mean(v) for k, v in errors.items()}
        deployment = {
            target: diagnostics["source_weighted_mae"][f"{target}/with_rhetoric"]
            < diagnostics["source_weighted_mae"][f"{target}/mean_only"]
            for target in ["voice", "revision"]
        }
        return cls(
            {
                "version": VERSION,
                "kind": "feedback-profile",
                "features": list(FEATURES),
                "reference": reference.report,
                "examples": history,
                "models": {"voice": _fit(x, voice), "revision": _fit(x, revision)},
                "diagnostics": diagnostics,
                "deployment": deployment,
                "warning": WARNING,
            }
        )

    def score(self, text: str) -> dict[str, Any]:
        observed = rhetoric(text)
        if observed["words"] < MIN_JUDGED_WORDS:
            observed["baseline"] = None
            return {
                "kind": "feedback-score",
                "version": VERSION,
                "voice_estimate_1_to_5": None,
                "revision_estimate_0_to_2": None,
                "estimate_enabled": {"voice": False, "revision": False},
                "rated_examples": len(self.report["examples"]),
                "rhetoric": observed,
                "personal_associations": [],
                "warning": f"Under {MIN_JUDGED_WORDS} words; estimates withheld. " + WARNING,
            }
        row, observed = features(text, self.reference)
        adjustments = []
        for metric in LABELS:
            index = FEATURES.index(metric)
            model = self.report["models"]["voice"]
            if observed["counts"][metric]:
                adjustments.append(
                    {
                        "metric": metric,
                        "label": LABELS[metric],
                        "voice_weight": model["weights"][index],
                        "interpretation": "association with your ratings, not a causal judgment",
                    }
                )
        return {
            "kind": "feedback-score",
            "version": VERSION,
            "voice_estimate_1_to_5": (
                _predict(self.report["models"]["voice"], row, 1, 5)
                if self.report["deployment"]["voice"]
                else None
            ),
            "revision_estimate_0_to_2": (
                _predict(self.report["models"]["revision"], row, 0, 2)
                if self.report["deployment"]["revision"]
                else None
            ),
            "estimate_enabled": self.report["deployment"],
            "rated_examples": len(self.report["examples"]),
            "rhetoric": observed,
            "personal_associations": adjustments,
            "warning": WARNING,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> FeedbackProfile:
        try:
            return cls(json.loads(Path(path).read_text()))
        except (KeyError, TypeError, ValueError) as error:
            raise StyleProfileError(f"invalid feedback profile: {error}") from error
