"""One-command CLI paths preserve the saved-profile pipeline and safe file handling."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from styleprofile import Settings, api
from styleprofile.cli import main

ROOT = Path(__file__).resolve().parent.parent
WRITER = str(ROOT / "examples/writer")
CONTRAST = str(ROOT / "examples/llm-drafts")
DRAFT = str(ROOT / "examples/draft.md")


def test_default_folder_output(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["build", WRITER + "/", "--no-syntax"]) == 0
    assert (tmp_path / "writer.profile.json").is_file()
    assert "wrote writer.profile.json" in capsys.readouterr().out


def test_default_file_output_and_stdin_requirement(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "archive.notes.md"
    source.write_text(
        (ROOT / "examples/writer/old-maps.md").read_text(encoding="utf-8") * 3, encoding="utf-8"
    )
    assert (
        main(["build", str(source), "--no-syntax", "--split-on", "none", "--window-words", "300"])
        == 0
    )
    assert (tmp_path / "archive.notes.profile.json").is_file()
    monkeypatch.setattr("sys.stdin", io.StringIO(source.read_text(encoding="utf-8")))
    assert main(["build", "-", "--no-syntax"]) == 1
    assert "stdin requires -o" in capsys.readouterr().err


def test_against_matches_two_step_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["build", WRITER, "--contrast", CONTRAST, "--no-syntax", "-o", "ref.json"]) == 0
    capsys.readouterr()
    assert main(["score", DRAFT, "ref.json", "--json"]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert (
        main(["score", DRAFT, "--against", WRITER, "--contrast", CONTRAST, "--no-syntax", "--json"])
        == 0
    )
    captured = capsys.readouterr()
    instant = json.loads(captured.out)
    assert instant["reference"]["verdict"] == saved["reference"]["verdict"]
    assert instant["documents"] == saved["documents"]
    assert instant["chunks"] == saved["chunks"]
    assert "Reference: built from 7 documents (4,496 words)" in captured.err
    assert "not saved. To reuse it:" in captured.err
    assert not (tmp_path / "writer.profile.json").exists()


@pytest.mark.parametrize("extra", [["-r", "ref.json"], ["ref.json"]])
def test_against_excludes_saved_reference(extra, capsys):
    assert main(["score", DRAFT, *extra, "--against", WRITER, "--no-syntax"]) == 1
    assert "--against cannot be combined" in capsys.readouterr().err


def test_against_cache_warnings_and_reuse_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []
    original = api.build

    def build(*args, **kwargs):
        calls.append(kwargs["cache"])
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "build", build)
    for flag, expected in [([], True), (["--no-cache"], False)]:
        assert (
            main(
                [
                    "score",
                    DRAFT,
                    "--against",
                    WRITER,
                    "--no-syntax",
                    "--window-words",
                    "250",
                    "--no-pool",
                    *flag,
                    "-q",
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert "Thin reference:" in captured.err
        assert "--window-words 250 --no-syntax --no-pool -o writer.profile.json" in captured.out
        assert calls[-1] is expected


def test_against_refuses_overwrite_of_discovered_source(tmp_path, capsys):
    output = str(Path(WRITER) / "old-maps.md")
    assert main(["score", DRAFT, "--against", WRITER, "--no-syntax", "-o", output]) == 1
    assert "--output would overwrite input" in capsys.readouterr().err


def test_short_help_uses_actual_length(tmp_path, capsys):
    profile = api.build(WRITER, Settings(syntax=False), cache=False)
    calibration = profile.report["calibration"]
    calibration["by_length"] = {}
    calibration["chunk_words"] = 400
    reference = tmp_path / "ref.json"
    profile.save(reference)
    draft = tmp_path / "short.md"
    draft.write_text("This is my short draft. " * 20, encoding="utf-8")
    assert main(["score", str(draft), str(reference), "-q"]) == 0
    assert "styleprofile judges 200 words or more;" in capsys.readouterr().out
    draft.write_text("This is my longer draft. " * 45, encoding="utf-8")
    assert main(["score", str(draft), str(reference), "-q"]) == 0
    assert "score several short texts together" not in capsys.readouterr().out


def test_pool_gets_a_verdict_for_short_drafts(tmp_path, capsys):
    drafts = []
    for index in range(10):
        path = tmp_path / f"{index}.md"
        path.write_text(
            f"Day {index}: " + "I thought we had time. We tried the old way again. " * 3,
            encoding="utf-8",
        )
        drafts.append(str(path))
    assert main(["score", str(tmp_path), "--against", WRITER, "--no-syntax", "--pool", "-q"]) == 0
    captured = capsys.readouterr()
    assert "Delta " in captured.out
    assert "score several short texts together" not in captured.out


@pytest.mark.parametrize("words, shown", [(74, True), (75, False)])
def test_short_help_boundary(tmp_path, capsys, words, shown):
    reference = tmp_path / "ref.json"
    api.build(WRITER, Settings(syntax=False), cache=False).save(reference)
    draft = tmp_path / "draft.md"
    draft.write_text("word " * words, encoding="utf-8")
    assert main(["score", str(draft), str(reference), "-q"]) == 0
    assert ("styleprofile judges 75 words or more" in capsys.readouterr().out) is shown


def test_short_help_requires_every_draft_to_be_short(tmp_path, capsys):
    reference = tmp_path / "ref.json"
    api.build(WRITER, Settings(syntax=False), cache=False).save(reference)
    drafts = []
    for words in (30, 100):
        draft = tmp_path / f"{words}.md"
        draft.write_text("word " * words, encoding="utf-8")
        drafts.append(str(draft))
    assert main(["score", *drafts, str(reference), "-q"]) == 0
    assert "score several short texts together" not in capsys.readouterr().out


def test_against_rejects_repeated_stdin(capsys):
    assert main(["score", "-", "--against", "-", "--no-syntax"]) == 1
    assert "stdin can be read only once" in capsys.readouterr().err
