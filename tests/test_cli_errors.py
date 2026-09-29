"""Errors for command lines and reports styleprofile does not accept, and how they say to fix
them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from styleprofile import StyleProfileError
from styleprofile.cli import main
from styleprofile.profile import load_report, report_kind

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


@pytest.mark.parametrize(
    ("arguments", "suggestion"),
    [
        (["posts/", "--output", "x.json"], "styleprofile build posts/ -o x.json"),
        (
            ["posts/", "--out=x.json", "--no-syntax", "--top-k", "50", "--bogus"],
            "styleprofile build posts/ -o x.json --top-k 50 --no-syntax",
        ),
        (
            ["posts/", "--contrast", "draft.md", "--output", "x.json"],
            "styleprofile build posts/ --contrast draft.md -o x.json",
        ),
        (
            ["draft.md", "--reference", "writer.json", "--output", "x.json"],
            "styleprofile score draft.md writer.json -o x.json",
        ),
        (
            ["draft.md", "--reference=writer.json", "--out", "x.json", "--window-words", "100"],
            "styleprofile score draft.md writer.json -o x.json --window-words 100",
        ),
        (
            ["draft.md", "-r", "writer.json", "--top-k", "5", "--no-syntax"],
            "styleprofile score draft.md writer.json --no-syntax",
        ),
    ],
)
def test_the_flat_form_gets_one_working_command(
    workspace: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], suggestion: str
) -> None:
    assert main(arguments) == 2
    err = capsys.readouterr().err
    assert f"{arguments[0]} is not a command; did you mean `{suggestion}`?" in err
    assert not (workspace / "x.json").exists()

    # The suggestion runs as given.
    command = suggestion.split()[1:]
    if "--no-syntax" not in command:
        command.append("--no-syntax")
    assert main(command) == 0


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
    # The message carries the remedy, so there is no hint.
    assert err == (
        f"error: {report} is not a style profile this version of styleprofile can read; make it "
        "again with `styleprofile build` (or `score` or `evaluate`, whichever wrote it)\n"
    )


def test_the_library_requires_a_kind(workspace: Path) -> None:
    _without_kind(workspace / "writer.json")
    with pytest.raises(StyleProfileError, match="make it again") as raised:
        load_report(workspace / "writer.json")
    assert raised.value.code == "outdated"
    with pytest.raises(StyleProfileError, match="no known kind"):
        report_kind({"summary": {}})
    assert report_kind(load_report(workspace / "draft.json")) == "score"


def test_a_draft_matching_its_reference_has_no_notable_differences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    essay = tmp_path / "essay.md"
    essay.write_text("\n\n".join(POSTS * 8), encoding="utf-8")
    reference = tmp_path / "writer.json"
    command = ["build", str(essay), "-o", str(reference), "--no-syntax", "--window-words", "100"]
    assert main(command) == 0
    capsys.readouterr()

    assert main(["score", str(essay), str(reference)]) == 0
    out = capsys.readouterr().out
    assert "Overall: close" in out
    assert "No metric differs by 1 sd or more on average." in out
