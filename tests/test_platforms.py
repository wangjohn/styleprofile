"""CLI streams, command hints and platform defaults without native OS dependencies."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from styleprofile import cache, measure, terminal
from styleprofile.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("encoding,expected", [("utf-8", "█"), ("cp1252", "#"), ("ascii", "#")])
def test_glyphs_fit_the_output_encoding(encoding: str, expected: str) -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding)
    chosen = terminal.glyphs(stream)
    assert chosen["bar"] == expected
    "".join(chosen.values()).encode(encoding)


def test_output_replaces_unencodable_names_on_both_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    output, errors = io.BytesIO(), io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="ascii")
    stderr = io.TextIOWrapper(errors, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    terminal.prepare_output()
    print("日本語")
    print("日本語", file=sys.stderr)
    stdout.flush()
    stderr.flush()
    assert output.getvalue() == b"???\n"
    assert errors.getvalue() == b"???\n"


def test_piped_cli_uses_readable_cp1252_output(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "STYLEPROFILE_NO_CACHE": "1"}
    reference = tmp_path / "writer.json"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "styleprofile",
            "build",
            str(ROOT / "examples/writer"),
            "--no-syntax",
            "--contrast",
            str(ROOT / "examples/llm-drafts"),
            "-o",
            str(reference),
        ],
        env=env,
        capture_output=True,
        check=True,
    )
    assert b"Traceback" not in built.stderr
    scored = subprocess.run(
        [
            sys.executable,
            "-m",
            "styleprofile",
            "score",
            str(ROOT / "examples/draft.md"),
            str(reference),
        ],
        env=env,
        capture_output=True,
        check=True,
    )
    text = scored.stdout.decode("cp1252")
    assert "By area" in text and "Biggest differences" in text
    assert "#" in text and "each ^ or v" in text and "Delta /" in text
    assert b"Traceback" not in scored.stderr


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_shell_hints_use_platform_quoting(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminal.sys, "platform", platform)
    command = ["styleprofile", "build", "writer's essays", "-o", 'my "profile".json']
    if platform == "win32":
        assert terminal.shell_join(command) == subprocess.list2cmdline(command)
    else:
        import shlex

        assert terminal.shell_join(command) == shlex.join(command)


@pytest.mark.parametrize(
    "platform,suffix",
    [
        ("win32", "AppData/Local/styleprofile/Cache"),
        ("darwin", "Library/Caches/styleprofile"),
        ("linux", ".cache/styleprofile"),
    ],
)
def test_cache_platform_paths_and_xdg_override(
    platform: str,
    suffix: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cache.sys, "platform", platform)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert cache.cache_dir() == tmp_path / suffix
    assert main(["cache"]) == 0
    assert str(tmp_path / suffix) in capsys.readouterr().out
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    if platform == "win32":
        assert cache.cache_dir() == tmp_path / "local/styleprofile/Cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.cache_dir() == tmp_path / "xdg/styleprofile"


def test_missing_memory_probe_falls_back_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(measure.os, "sysconf", raising=False)
    assert measure.memory_jobs() is None
