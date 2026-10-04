"""The installed-demo path and refusal to overwrite a user's files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from styleprofile import api, demo
from styleprofile.cache import MeasurementCache
from styleprofile.cli import build_parser, main


def test_demo_builds_scores_and_can_be_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[bool] = []
    build = api.build

    def cached_build(*args: object, **kwargs: object) -> api.Profile:
        calls.append(kwargs.get("cache") is True)
        return build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(api, "build", cached_build)
    for _ in range(2):
        assert main(["demo", "--no-syntax"]) == 0
        output = capsys.readouterr().out
        assert "Delta" in output
        assert output.split("Next: try it on your own texts\n", 1)[1].splitlines() == [
            "  styleprofile build posts/ --contrast llm-drafts/ -o writer.json",
            "  styleprofile score draft.md writer.json",
            "  styleprofile score draft.md --against posts/",
        ]
    directory = tmp_path / "styleprofile-demo"
    profile = json.loads((directory / demo.PROFILE).read_text(encoding="utf-8"))
    assert profile["settings"]["syntax_used"] is None
    assert profile["contrast"]
    assert (directory / "writer" / "old-maps.md").is_file()
    assert all(calls)


@pytest.mark.parametrize("kind", ["unmarked", "edited", "extra", "profile", "invalid-marker"])
def test_demo_refuses_user_files(
    kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "demo"
    if kind == "unmarked":
        directory.mkdir()
        (directory / "draft.md").write_text("My own draft", encoding="utf-8")
    else:
        directory, _ = demo.prepare(str(directory))
        if kind == "edited":
            (directory / "draft.md").write_text("My own draft", encoding="utf-8")
        elif kind == "extra":
            (directory / "notes.txt").write_text("Keep this", encoding="utf-8")
        elif kind == "profile":
            (directory / demo.PROFILE).write_text("My own profile", encoding="utf-8")
        else:
            (directory / demo.MARKER).write_text("{}", encoding="utf-8")
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert main(["demo", "--dir", str(directory), "--no-syntax"]) == 1
    assert "choose another folder with --dir" in capsys.readouterr().err
    after = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert after == before


def test_demo_refuses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("Keep this", encoding="utf-8")
    assert main(["demo", "--dir", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == "Keep this"


def test_demo_refuses_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks requires permission on this platform")
    assert main(["demo", "--dir", str(link)]) == 1
    assert list(target.iterdir()) == []
    directory, _ = demo.prepare(str(tmp_path / "demo"))
    (directory / "draft.md").unlink()
    (directory / "draft.md").symlink_to(target / "draft.md")
    assert main(["demo", "--dir", str(directory)]) == 1
    assert list(target.iterdir()) == []


def test_demo_resource_fallback_and_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.setattr(demo, "files", lambda package: tmp_path)
    monkeypatch.setattr(demo, "__file__", str(root.parent / "src" / "styleprofile" / "demo.py"))
    directory, samples = demo.prepare(str(tmp_path / "empty"))
    assert samples["draft.md"] == (root / "draft.md").read_bytes()
    assert (directory / "draft.md").read_bytes() == samples["draft.md"]


def test_demo_is_in_root_help() -> None:
    assert "Try the bundled samples: styleprofile demo" in build_parser().format_help()


def test_demo_continues_when_the_cache_has_a_disk_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STYLEPROFILE_NO_CACHE", "1")
    assert main(["demo", "--no-syntax"]) == 0
    measured = capsys.readouterr()
    attempts = []

    def unavailable(self: MeasurementCache) -> sqlite3.Connection:
        attempts.append(self.path)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.delenv("STYLEPROFILE_NO_CACHE")
    monkeypatch.setattr(MeasurementCache, "_open", unavailable)
    assert main(["demo", "--no-syntax"]) == 0
    fallback = capsys.readouterr()
    assert attempts
    assert fallback.out == measured.out
    assert "Overall: close" in fallback.out
    assert "Cache unavailable (disk I/O error)" in fallback.err
    assert "so this run went on without it" in fallback.err


def test_demo_accepts_a_directory_starting_with_a_dash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--dir=-samples", "--no-syntax"]) == 0
    assert "Overall: close" in capsys.readouterr().out
    assert (tmp_path / "-samples" / demo.PROFILE).is_file()
