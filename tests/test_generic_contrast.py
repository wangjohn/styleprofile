"""Bundled draft mechanics; empirical acceptance failures are recorded in docs/method.md."""

from collections import Counter
from pathlib import Path

import pytest

import styleprofile as sp
from styleprofile.api import _generic_texts
from styleprofile.cli import main
from styleprofile.surface import prose, words

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples/writer"


def test_bundled_set_has_24_original_drafts_in_seven_genres() -> None:
    texts = _generic_texts()
    assert len(texts) == 24
    genres = Counter((t.name or "").split("-")[1] for t in texts)
    assert genres == {
        "essay": 4,
        "blog": 4,
        "howto": 4,
        "opinion": 3,
        "review": 3,
        "newsletter": 3,
        "story": 3,
    }
    assert all(400 <= len(words(" ".join(prose(t.text).blocks))) <= 900 for t in texts)
    assert len({t.text for t in texts}) == 24
    sample_texts = {
        p.read_text(encoding="utf-8") for p in (ROOT / "examples/llm-drafts").glob("*.md")
    }
    assert not sample_texts.intersection(t.text for t in texts)
    folder = ROOT / "src/styleprofile/data/generic-contrast"
    assert sum(p.stat().st_size for p in folder.iterdir()) < 150_000
    assert "MIT" in (folder / "README.md").read_text(encoding="utf-8")


def test_generic_build_adds_explicit_drafts_and_preserves_library_cache_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    generic = sp.build(WRITER, generic_contrast=True, syntax=False)
    expected = sp.build(
        WRITER, contrast=_generic_texts(), contrast_label="generic LLM drafts", syntax=False
    )
    assert generic.report["contrast"] == expected.report["contrast"]
    assert generic.report["contrast"]["sources"] == 24
    combined = sp.build(
        WRITER, generic_contrast=True, contrast=ROOT / "examples/llm-drafts", syntax=False
    )
    assert combined.report["contrast"]["sources"] == 29
    assert combined.report["contrast"]["label"] == "generic LLM drafts"
    strings = [p.read_text(encoding="utf-8") for p in sorted(WRITER.glob("*.md"))]
    assert (
        sp.build_texts(strings, generic_contrast=True, syntax=False).report["contrast"]
        == generic.report["contrast"]
    )
    assert not (tmp_path / "cache").exists()
    assert "drafts from your own briefs are better" in generic.to_text()
    assert "docs/contrast.md" in generic.to_text(full=True)
    path = tmp_path / "generic.json"
    generic.save(path)
    assert "drafts from your own briefs are better" in sp.load(path).to_text()


def test_build_advice_and_cli_generic_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = sp.build(WRITER, syntax=False)
    assert (
        plain.to_text()
        .splitlines()[-1]
        .startswith("For LLM-likeness, build with --generic-contrast")
    )
    assert "generic_contrast" not in plain.report["settings"]
    output = tmp_path / "generic.json"
    assert (
        main(
            [
                "build",
                str(WRITER),
                "--generic-contrast",
                "--no-syntax",
                "--no-cache",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert "drafts from your own briefs are better" in capsys.readouterr().out
    assert sp.load(output).report["contrast"]["sources"] == 24
