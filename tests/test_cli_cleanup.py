"""The command forms and report formats that are no longer accepted, and what users see."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import styleprofile
from styleprofile import StyleProfileError, load_report, report_kind
from styleprofile.cli import main

POSTS = [
    "I don't know what I expected. We tried it anyway, and it mostly worked for a while.",
    "You can't plan everything. We shipped it on a Tuesday, and I think that was right.",
    "It's a small thing. So I wrote it down, because I'll forget it otherwise, like always.",
]
DRAFT = (
    "In today's fast-paced world, it is crucial to leverage synergies. Moreover, stakeholders "
    "must foster alignment across teams to unlock transformative value."
)


@pytest.fixture
def workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Path:
    """posts/, draft.md, writer.json (a reference) and draft.json (a score report)."""
    posts = tmp_path / "posts"
    posts.mkdir()
    for index, text in enumerate(POSTS):
        (posts / f"post{index}.md").write_text(text, encoding="utf-8")
    (tmp_path / "draft.md").write_text(DRAFT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["build", "posts", "-o", "writer.json", "--no-syntax"]) == 0
    assert main(["score", "draft.md", "writer.json", "-o", "draft.json"]) == 0
    capsys.readouterr()
    return tmp_path


def _without_kind(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    del report["kind"]
    path.write_text(json.dumps(report), encoding="utf-8")


def test_the_flat_form_gets_the_not_a_command_hint(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["posts/", "--output", "x.json"]) == 2
    err = capsys.readouterr().err
    assert "posts/ is not a command; did you mean `styleprofile build posts/ --output x.json`" in (
        err
    )
    assert "`styleprofile score posts/ --output x.json`?" in err
    assert not (workspace / "x.json").exists()


def test_score_has_no_reference_flag(capsys: pytest.CaptureFixture[str], workspace: Path) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["score", "draft.md", "--reference", "writer.json"])
    assert exited.value.code == 2
    assert "unrecognized arguments: --reference" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "report"),
    [
        (["show", "writer.json"], "writer.json"),
        (["show", "draft.json"], "draft.json"),
        (["score", "draft.md", "writer.json"], "writer.json"),
    ],
)
def test_reports_without_a_kind_are_refused_with_rebuild_advice(
    workspace: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], report: str
) -> None:
    _without_kind(workspace / report)

    assert main(arguments) == 1
    err = capsys.readouterr().err
    assert err.startswith(
        f"error: {report} is not a style profile this version of styleprofile can read; "
        "rebuild it\n"
    )
    assert "hint: run `styleprofile build` again" in err


def test_the_library_requires_a_kind(workspace: Path) -> None:
    _without_kind(workspace / "writer.json")
    with pytest.raises(StyleProfileError, match="rebuild it") as raised:
        load_report(workspace / "writer.json")
    assert raised.value.code == "outdated"
    with pytest.raises(StyleProfileError, match="no known kind"):
        report_kind({"summary": {}})
    assert report_kind(load_report(workspace / "draft.json")) == "score"


def test_build_profile_is_gone() -> None:
    assert "build_profile" not in styleprofile.__all__
    assert not hasattr(styleprofile, "build_profile")
