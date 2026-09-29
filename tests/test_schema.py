"""The TypedDicts in ``schema`` match the reports the code writes, and ``load_report`` refuses
a report whose shape its kind does not have, naming the part."""

from __future__ import annotations

import copy
import importlib
import json
import re
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

import pytest

import styleprofile as sp
from styleprofile import schema
from styleprofile.cli import main
from styleprofile.profile import Chunk, build_reference, dumps_report, load_report, score
from styleprofile.schema import (
    EvaluationReport,
    Problem,
    ReferenceReport,
    ScoreReport,
    SyntaxUsed,
    find_problem,
)
from styleprofile.syntax import SyntaxUnavailableError, load_parser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
gen: Any = importlib.import_module("gen")

ROOT = Path(__file__).resolve().parent.parent
WRITER, CONTRAST, DRAFT = (
    ROOT / "examples/writer",
    ROOT / "examples/llm-drafts",
    ROOT / "examples/draft.md",
)
POSTS = [
    "I don't know what I expected. We tried it anyway, and it mostly worked for a while.",
    "You can't plan everything. We shipped it on a Tuesday, and I think that was right.",
]
# 36 words: under 75, so too short to judge.
SHORT = (
    "The stones shift every winter, and the posts lean a little further each spring. My "
    "father kept his fence by walking it in April with a bar and a bucket of wedges, and so "
    "do I."
)


@dataclass
class Walk:
    """Checks a value against a type strictly: every key declared, every required key
    present, every leaf of the declared type. Records which TypedDict keys it saw."""

    problems: list[str] = field(default_factory=list)
    seen: set[tuple[str, str]] = field(default_factory=set)

    def check(self, value: object, hint: Any, path: str) -> None:
        if is_typeddict(hint):
            self._typeddict(value, hint, path)
            return
        origin, args = get_origin(hint), get_args(hint)
        if origin in (Union, types.UnionType):
            # Pick the arm that fits: None, or the one whose check finds nothing wrong.
            for arm in args:
                trial = Walk()
                trial.check(value, arm, path)
                if not trial.problems:
                    self.seen |= trial.seen
                    return
            self.problems.append(f"{path}: {value!r:.60} fits none of {hint}")
        elif origin is Literal:
            if value not in args:
                self.problems.append(f"{path}: {value!r} is not one of {args}")
        elif origin is list:
            if not isinstance(value, list):
                self.problems.append(f"{path}: expected a list, got {type(value).__name__}")
                return
            for index, item in enumerate(value):
                self.check(item, args[0], f"{path}[{index}]")
        elif origin is dict:
            if not isinstance(value, dict):
                self.problems.append(f"{path}: expected a dict, got {type(value).__name__}")
                return
            for key, item in value.items():
                self.check(key, args[0], f"{path} key")
                self.check(item, args[1], f"{path}.{key}")
        elif hint is type(None):
            if value is not None:
                self.problems.append(f"{path}: expected None, got {value!r:.60}")
        elif hint is float:
            # JSON writes a whole float as an int only if it was one; either is a number.
            if isinstance(value, bool) or not isinstance(value, int | float):
                self.problems.append(f"{path}: expected a number, got {value!r:.60}")
        elif hint in (int, str, bool):
            if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
                self.problems.append(f"{path}: expected {hint.__name__}, got {value!r:.60}")
        else:
            raise AssertionError(f"{path}: the walker does not know {hint}")

    def _typeddict(self, value: object, schema_type: Any, path: str) -> None:
        if not isinstance(value, dict):
            self.problems.append(f"{path}: expected {schema_type.__name__}, got {value!r:.60}")
            return
        hints = get_type_hints(schema_type)
        for key in schema_type.__required_keys__ - value.keys():
            self.problems.append(f"{path}: {schema_type.__name__} requires {key!r}")
        for key, item in value.items():
            if key not in hints:
                self.problems.append(f"{path}: {key!r} is not in {schema_type.__name__}")
                continue
            self.seen.add((schema_type.__name__, key))
            self.check(item, hints[key], f"{path}.{key}" if path else key)


def _declared(root: type) -> Iterator[tuple[str, str]]:
    """Every (TypedDict, key) reachable from ``root``."""
    visited: set[type] = set()

    def visit(hint: Any) -> Iterator[tuple[str, str]]:
        if is_typeddict(hint):
            if hint in visited:
                return
            visited.add(hint)
            for key, item in get_type_hints(hint).items():
                yield hint.__name__, key
                yield from visit(item)
        for arg in get_args(hint):
            yield from visit(arg)

    yield from visit(root)


def _spacy() -> bool:
    try:
        load_parser()
    except SyntaxUnavailableError:
        return False
    return True


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> list[tuple[str, Any, type]]:
    """Real reports of every kind and shape, each as returned and as saved."""
    tmp = tmp_path_factory.mktemp("reports")
    made: list[tuple[str, Any, type]] = []
    # With spaCy when it is installed, so ``syntax_used`` is a parser as well as None.
    contrast = sp.build(WRITER, contrast=CONTRAST)
    made += [
        ("reference with contrast", contrast.report, ReferenceReport),
        ("score with contrast", contrast.score(DRAFT, passages=True).report, ScoreReport),
        ("score of two files", contrast.score([DRAFT, WRITER / "old-maps.md"]).report, ScoreReport),
        # No verdict: every chunk is too short to judge.
        (
            "score of a short text",
            contrast.score(sp.Text(SHORT), passages=True).report,
            ScoreReport,
        ),
        # A verdict that leaves out a chunk too short to judge.
        (
            "score of files with a short one",
            contrast.score([DRAFT, sp.Text(SHORT), WRITER / "old-maps.md"]).report,
            ScoreReport,
        ),
    ]
    # A reference large enough for paragraph thresholds (``calibration.drift`` tails, and a
    # score's ``thresholds``): the demo corpus, remixed from the essays.
    demo = gen.generate("demo", out=tmp)
    large = sp.build(demo / "writer", sp.Settings(syntax=False), contrast=demo / "contrast")
    made += [
        ("reference large enough for paragraph thresholds", large.report, ReferenceReport),
        ("score with paragraph thresholds", large.score(DRAFT, passages=True).report, ScoreReport),
    ]
    plain = sp.build(WRITER, sp.Settings(syntax=False))
    made += [
        ("reference without contrast", plain.report, ReferenceReport),
        ("score without contrast", plain.score(DRAFT, passages=True).report, ScoreReport),
    ]
    # One document: no reliability or calibration, and a baseline without them. Its chunks
    # are kept, as ``--keep-chunks`` does.
    single = sp.build(
        WRITER / "old-maps.md", sp.Settings(syntax=False, window_words=100), keep_chunks=True
    )
    made += [
        ("reference of one document", single.report, ReferenceReport),
        ("score against one document", single.score(DRAFT, passages=True).report, ScoreReport),
        (
            "short text against one document",
            single.score(sp.Text(SHORT), passages=True).report,
            ScoreReport,
        ),
    ]
    # The lower-level functions record only the settings they are given.
    chunks = [Chunk(f"post{index}", "posts", text) for index, text in enumerate(POSTS)]
    lower = build_reference(chunks)
    made += [
        ("lower-level reference", lower, ReferenceReport),
        ("lower-level score", score([Chunk("d", "d", POSTS[0])], lower), ScoreReport),
    ]
    # An edited set that skips one draft, so its partial-set keys are written too.
    edited = tmp / "light"
    edited.mkdir()
    for path in sorted(CONTRAST.glob("*.md"))[1:]:
        text = path.read_text(encoding="utf-8").replace(" — ", ", ")
        (edited / path.name).write_text(text, encoding="utf-8")
    evaluation = sp.evaluate(
        WRITER, CONTRAST, {"light": edited}, sp.Settings(syntax=False), retrain=True
    )
    made.append(("evaluation with retrain", evaluation.report, EvaluationReport))
    plain_evaluation = sp.evaluate(WRITER, CONTRAST, {"light": edited}, sp.Settings(syntax=False))
    made.append(("evaluation", plain_evaluation.report, EvaluationReport))
    # Fail flags record ``fail`` and ``failed``; the LLM draft reaches both levels.
    reference = tmp / "writer.json"
    sp.build(WRITER, contrast=CONTRAST).save(reference)
    failed = tmp / "failed.json"
    argv = ["score", "--fail-above", "clearly", "--fail-likeness", "few", "-o", str(failed)]
    assert main([*argv, str(CONTRAST / "old-maps.md"), str(DRAFT), str(reference)]) == 3
    made.append(("score with fail flags", json.loads(failed.read_text("utf-8")), ScoreReport))
    return made + [
        (f"{name}, saved", json.loads(dumps_report(report)), kind) for name, report, kind in made
    ]


def test_real_reports_match_the_schema(reports: list[tuple[str, Any, type]]) -> None:
    for name, report, kind in reports:
        walk = Walk()
        walk.check(report, kind, "")
        assert not walk.problems, f"{name}:\n" + "\n".join(walk.problems[:20])
        assert find_problem(report, kind) is None


def test_every_schema_key_is_written_by_some_report(reports: list[tuple[str, Any, type]]) -> None:
    """A key the schema declares but no report writes is stale, or its report is missing
    from ``reports``."""
    seen: set[tuple[str, str]] = set()
    for _, report, kind in reports:
        walk = Walk()
        walk.check(report, kind, "")
        seen |= walk.seen
    declared = {
        key for root in (ReferenceReport, ScoreReport, EvaluationReport) for key in _declared(root)
    }
    if not _spacy():
        declared -= {(SyntaxUsed.__name__, key) for key in get_type_hints(SyntaxUsed)}
    assert declared - seen == set()


def test_optional_keys_are_not_required() -> None:
    """``__required_keys__`` is right on every supported Python, which needs the schema
    module to keep its annotations unquoted."""
    assert "contrast" not in ReferenceReport.__required_keys__
    assert "calibration" not in ReferenceReport.__required_keys__
    assert "chunks" not in ReferenceReport.__required_keys__
    assert {"summary", "document_count", "kind"} <= ReferenceReport.__required_keys__
    assert "likeness" not in schema.ChunkScore.__required_keys__
    assert "calibration" in schema.ChunkScore.__required_keys__
    assert {"delta", "effective", "reliability"}.isdisjoint(
        schema.LengthCalibration.__required_keys__
    )
    assert "reliability" not in get_type_hints(schema.BaselineLength)
    assert "window_words" not in schema.ReportSettings.__required_keys__
    assert {"top_k", "syntax_used"} <= schema.ReportSettings.__required_keys__
    assert "edits" not in schema.SetSummary.__required_keys__


def test_find_problem_names_the_path_and_what_is_wrong() -> None:
    report = _score_report()
    assert find_problem(report, ScoreReport) is None
    assert find_problem({}, ScoreReport) == Problem("chunk_count", "missing")
    delta = report["chunks"][0]["reference"].pop("delta")
    assert find_problem(report, ScoreReport) == Problem("chunks[0].reference.delta", "missing")
    report["chunks"][0]["reference"]["delta"] = delta
    report["reference"]["baseline"] = None
    assert find_problem(report, ScoreReport) == Problem("reference.baseline", "null, not an object")
    # Null where the type allows it is fine, and so is a key the schema does not know.
    report = _score_report()
    report["reference"]["baseline"]["calibration"] = None
    report["reference"]["documents"] = []
    assert find_problem(report, ScoreReport) is None


def _score_report() -> dict[str, Any]:
    return json.loads(dumps_report(_reference().score(DRAFT).report))


def _reference() -> sp.Profile:
    return sp.build(WRITER, sp.Settings(syntax=False))


@pytest.fixture(scope="module")
def saved(reports: list[tuple[str, Any, type]]) -> dict[str, Any]:
    """A saved reference with contrast, a score against it, and an evaluation with retrain."""
    by_name = {name: report for name, report, _ in reports}
    return {
        "reference": by_name["reference with contrast, saved"],
        "score": by_name["score with contrast, saved"],
        "evaluation": by_name["evaluation with retrain, saved"],
    }


DELETE = object()
# Hand edits of saved reports, each refused with the part it names and what is wrong with
# it; ``"missing"`` reads "has no ...", anything else "has an unreadable ... (reason)".
CORRUPTIONS: list[tuple[str, tuple[str | int, ...], object, str, str]] = [
    # A key missing, at any depth.
    ("score", ("reference", "baseline"), DELETE, "reference.baseline", "missing"),
    ("score", ("chunks", 0, "reference", "z"), DELETE, "chunks[0].reference.z", "missing"),
    (
        "score",
        ("reference", "baseline", "contrast", "calibration", "reference"),
        DELETE,
        "reference.baseline.contrast.calibration.reference",
        "missing",
    ),
    (
        "score",
        ("reference", "baseline", "contrast", "calibration", "bootstrap", "method"),
        DELETE,
        "reference.baseline.contrast.calibration.bootstrap.method",
        "missing",
    ),
    ("reference", ("summary",), DELETE, "summary", "missing"),
    (
        "reference",
        ("calibration", "delta", "by_group"),
        DELETE,
        "calibration.delta.by_group",
        "missing",
    ),
    ("reference", ("settings", "syntax_used"), DELETE, "settings.syntax_used", "missing"),
    ("reference", ("contrast", "effects"), DELETE, "contrast.effects", "missing"),
    (
        "evaluation",
        ("retrain", "by_set", "light", "before"),
        DELETE,
        "retrain.by_set.light.before",
        "missing",
    ),
    # Length calibration (plan PR 6): a reference's lengths, a chunk's own, and the verdict.
    (
        "reference",
        ("calibration", "by_length", "75", "delta", "p99"),
        DELETE,
        "calibration.by_length.75.delta.p99",
        "missing",
    ),
    (
        "score",
        ("chunks", 0, "reference", "calibration"),
        DELETE,
        "chunks[0].reference.calibration",
        "missing",
    ),
    ("score", ("reference", "verdict", "judged"), DELETE, "reference.verdict.judged", "missing"),
    # A part one section names and another lacks.
    (
        "score",
        ("reference", "baseline", "summary", "sentence_shape"),
        DELETE,
        "reference.baseline.summary.sentence_shape",
        "missing",
    ),
    ("evaluation", ("signals", 0, "edited", "light"), DELETE, "signals[0].edited.light", "missing"),
    # A null where the type has none. Before, a null baseline showed the score as a profile.
    ("score", ("reference", "baseline"), None, "reference.baseline", "null, not an object"),
    ("score", ("chunks", 0, "reference"), None, "chunks[0].reference", "null, not an object"),
    (
        "score",
        ("reference", "baseline", "calibration", "delta"),
        None,
        "reference.baseline.calibration.delta",
        "null, not an object",
    ),
    ("reference", ("contrast",), None, "contrast", "null, not an object"),
    ("reference", ("reliability",), None, "reliability", "null, not an object"),
    (
        "score",
        ("reference", "verdict", "delta"),
        None,
        "reference.verdict.delta",
        "null, not an object",
    ),
    # The wrong kind of part.
    (
        "reference",
        ("calibration", "by_length"),
        [],
        "calibration.by_length",
        "a list, not an object",
    ),
    (
        "score",
        ("reference", "baseline", "calibration", "by_length", "75"),
        "x",
        "reference.baseline.calibration.by_length.75",
        "a string, not an object",
    ),
    ("score", ("reference", "baseline"), "x", "reference.baseline", "a string, not an object"),
    ("score", ("chunks",), {}, "chunks", "an object, not a list"),
    ("reference", ("summary",), [], "summary", "a list, not an object"),
    ("reference", ("calibration",), 3, "calibration", "a number, not an object"),
    (
        "reference",
        ("settings", "syntax_used"),
        "x",
        "settings.syntax_used",
        "a string, not an object",
    ),
    (
        "reference",
        ("summary", "sentence_shape"),
        "x",
        "summary.sentence_shape",
        "a string, not an object",
    ),
    ("evaluation", ("sets",), [], "sets", "a list, not an object"),
    ("evaluation", ("retrain",), "x", "retrain", "a string, not an object"),
    # A report's own numbers, which rendering compares and formats.
    ("score", ("chunk_count",), "7", "chunk_count", "a string, not a number"),
    ("score", ("reference", "delta_mean"), "far", "reference.delta_mean", "a string, not a number"),
    (
        "score",
        ("chunks", 0, "reference", "calibration", "judged"),
        "yes",
        "chunks[0].reference.calibration.judged",
        "a string, not true or false",
    ),
    (
        "reference",
        ("calibration", "by_length", "75", "words"),
        "75",
        "calibration.by_length.75.words",
        "a string, not a number",
    ),
]
AGAIN = {
    "reference": "rebuild it with `styleprofile build`",
    "score": "score it again with `styleprofile score`",
    "evaluation": "run `styleprofile evaluate` again",
}


@pytest.mark.parametrize(
    ("kind", "at", "value", "named", "reason"),
    CORRUPTIONS,
    ids=[f"{kind}-{named}-{reason.split(',')[0]}" for kind, _, _, named, reason in CORRUPTIONS],
)
def test_edited_reports_are_refused_cleanly(
    saved: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    at: tuple[str | int, ...],
    value: object,
    named: str,
    reason: str,
) -> None:
    report: Any = copy.deepcopy(saved[kind])
    part = report
    for step in at[:-1]:
        part = part[step]
    if value is DELETE:
        del part[at[-1]]
    else:
        part[at[-1]] = value
    monkeypatch.chdir(tmp_path)
    Path("old.json").write_text(json.dumps(report), encoding="utf-8")
    what = (
        f"has no {named}, which this version of styleprofile needs"
        if reason == "missing"
        else f"has an unreadable {named} ({reason})"
    )
    expected = (
        f"old.json {what}: it was made by an earlier development version, or edited; {AGAIN[kind]}"
    )

    with pytest.raises(sp.StyleProfileError) as error:
        load_report(Path("old.json"))
    assert (str(error.value), error.value.code) == (expected, "outdated")

    assert main(["show", "old.json"]) == 1
    assert capsys.readouterr().err == f"error: {expected}\n"
    if kind == "reference":
        Path("draft.md").write_text(POSTS[0], encoding="utf-8")
        assert main(["score", "draft.md", "old.json"]) == 1
        assert capsys.readouterr().err == f"error: {expected}\n"
        with pytest.raises(sp.StyleProfileError, match=f"^the profile {re.escape(what)}"):
            sp.Profile(report)


@pytest.mark.parametrize("version", ["6", None, 6.0, True])
def test_a_version_that_is_not_a_whole_number_is_unreadable(
    saved: dict[str, Any], version: object
) -> None:
    report = {**saved["reference"], "version": version}
    if version is None:
        del report["version"]
    with pytest.raises(sp.StyleProfileError) as error:
        sp.Profile(report)  # type: ignore[arg-type]
    assert error.value.code == "outdated"
    assert str(error.value).startswith(
        "the profile is not a style profile this version of styleprofile can read"
    )


def test_the_walker_catches_undeclared_keys_and_wrong_types() -> None:
    report = _score_report()
    report["reference"]["documents"] = []
    report["chunks"][0]["reference"]["delta"] = "far"
    walk = Walk()
    walk.check(report, ScoreReport, "")
    assert walk.problems == [
        "chunks[0].reference.delta: 'far' fits none of float | None",
        "reference: 'documents' is not in ReferenceScore",
    ]
