"""Additive report minors preserve numbers while incompatible scientific majors refuse."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile.cli import main
from styleprofile.profile import (
    EVALUATION_VERSION,
    MINOR_VERSION,
    VERSION,
    dumps_report,
    load_report,
)

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "examples/writer"
CONTRAST = ROOT / "examples/llm-drafts"
DRAFT = ROOT / "examples/draft.md"
KINDS = ("reference", "score", "evaluation")


@pytest.fixture(scope="module")
def reports() -> dict[str, Any]:
    profile = sp.build(WRITER, syntax=False, contrast=CONTRAST)
    result = profile.score(DRAFT)
    evaluation = sp.evaluate(WRITER, CONTRAST, {"unchanged": CONTRAST}, sp.Settings(syntax=False))
    return {
        "reference": json.loads(dumps_report(profile.report)),
        "score": json.loads(dumps_report(result.report)),
        "evaluation": json.loads(dumps_report(evaluation.report)),
    }


def saved(tmp_path: Path, report: Any) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("minor", [None, 0, MINOR_VERSION, MINOR_VERSION + 1])
def test_legacy_older_current_and_newer_minors_load(
    kind: str,
    minor: int | None,
    reports: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = copy.deepcopy(reports[kind])
    if minor is None:
        del report["minor_version"]
    else:
        report["minor_version"] = minor
    report["future_field"] = {"keep": [1, 2, 3]}
    path = saved(tmp_path, report)
    notes = []
    loaded = load_report(path, notes=notes)
    assert loaded == report, "loading preserves versions and unknown fields"
    newer = minor is not None and minor > MINOR_VERSION
    assert bool(notes) is newer
    if newer:
        assert notes[0].code == sp.NoteCode.NEWER_REPORT_VERSION
    if kind == "reference":
        result = sp.load(path)
        assert result.report == report
        scored = result.score(DRAFT)
        assert json.loads(dumps_report(scored.report))["documents"] == reports["score"]["documents"]
        assert json.loads(dumps_report(scored.report))["chunks"] == reports["score"]["chunks"]
        assert any(n.code == sp.NoteCode.NEWER_REPORT_VERSION for n in scored.notes) is newer
        assert main(["score", str(DRAFT), str(path), "--json"]) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["documents"] == reports["score"]["documents"]
        assert ("some fields may be ignored" in captured.err) is newer
    elif kind == "score":
        assert loaded["kind"] == "score"
        result = sp.ScoreResult(loaded)
        assert result.delta == sp.ScoreResult(reports[kind]).delta
    else:
        assert loaded["kind"] == "evaluation"
        result = sp.Evaluation(loaded)
    assert any(n.code == sp.NoteCode.NEWER_REPORT_VERSION for n in result.notes) is newer
    destination = tmp_path / "roundtrip.json"
    result.save(destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert main(["show", str(path)]) == 0
    captured = capsys.readouterr()
    assert ("some fields may be ignored" in captured.err) is newer


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("offset", [-1, 1])
def test_different_majors_refuse_in_library_and_cli(
    kind: str,
    structured: bool,
    offset: int,
    reports: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = copy.deepcopy(reports[kind])
    expected = EVALUATION_VERSION if kind == "evaluation" else VERSION
    report["version"] = expected + offset
    if not structured:
        del report["minor_version"]
    path = saved(tmp_path, report)
    with pytest.raises(sp.StyleProfileError, match="report version") as error:
        load_report(path)
    assert error.value.code == "outdated"
    if kind == "reference":
        with pytest.raises(sp.StyleProfileError, match="rebuild it"):
            sp.load(path)
        assert main(["score", str(DRAFT), str(path)]) == 1
    else:
        assert main(["show", str(path)]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("major", [None, "8", 8.0, True, 0, -1, {"major": 8, "minor": 1}])
def test_malformed_majors_refuse(kind, major, reports, tmp_path):
    report = copy.deepcopy(reports[kind])
    report["version"] = major
    if major is None:
        del report["version"]
    with pytest.raises(sp.StyleProfileError) as error:
        load_report(saved(tmp_path, report))
    assert error.value.code == "outdated"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("minor", [None, "1", 1.0, True, -1, []])
def test_malformed_minors_refuse(kind, minor, reports, tmp_path):
    report = copy.deepcopy(reports[kind])
    report["minor_version"] = minor
    with pytest.raises(sp.StyleProfileError) as error:
        load_report(saved(tmp_path, report))
    assert error.value.code == "outdated"


def test_legacy_optional_defaults_and_missing_required_data(reports, tmp_path, capsys):
    report = copy.deepcopy(reports["reference"])
    del report["minor_version"]
    for field in ("window_words", "pool", "split_on", "group_field"):
        del report["settings"][field]
    profile = sp.load(saved(tmp_path, report))
    assert profile.settings == sp.Settings(
        window_words=0, syntax=False, text_field=report["settings"]["text_field"]
    )
    assert profile.score(DRAFT).judged
    score = copy.deepcopy(reports["score"])
    del score["minor_version"]
    del score["passages"]
    path = saved(tmp_path, score)
    loaded = load_report(path)
    assert loaded["kind"] == "score"
    assert sp.ScoreResult(loaded).passages == ()
    assert main(["show", str(path), "--by-paragraph"]) == 0
    assert (
        "Paragraph checks were not saved; score again with --by-paragraph."
        in capsys.readouterr().out
    )
    del report["summary"]
    with pytest.raises(sp.StyleProfileError, match="has no summary"):
        sp.load(saved(tmp_path, report))
