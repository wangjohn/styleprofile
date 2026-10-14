"""The bundled generic contrast drafts and the register folds; the evaluation against the
bars is in docs/method.md (``bench/eval/generic_contrast.py``)."""

from collections import Counter
from pathlib import Path

import pytest

import styleprofile as sp
from styleprofile.api import _generic_folders
from styleprofile.cli import main
from styleprofile.surface import prose, words
from styleprofile.weighting import cross_validate

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples/writer"
LLM = ROOT / "examples/llm-drafts"
FOLDER = ROOT / "src/styleprofile/data/generic-contrast"
REGISTERS = {"explainer", "blog", "essay", "newsletter", "opinion", "humanlike"}


def _drafts() -> list[Path]:
    return sorted(p for p in FOLDER.glob("*/*.md"))


def test_the_corpus_has_sixty_drafts_in_six_registers_and_four_length_bands() -> None:
    drafts = _drafts()
    assert len(drafts) == 60
    assert Counter(p.parent.name for p in drafts) == dict.fromkeys(REGISTERS, 10)
    texts = [p.read_text(encoding="utf-8") for p in drafts]
    lengths = [len(words(" ".join(prose(t).blocks))) for t in texts]
    bands = Counter(
        "150-300" if n < 300 else "300-600" if n < 600 else "600-1000" if n < 1000 else "1000+"
        for n in lengths
    )
    assert bands == {"150-300": 15, "300-600": 15, "600-1000": 15, "1000+": 15}
    assert min(lengths) >= 150 and max(lengths) <= 1500
    assert len(set(texts)) == 60
    assert sum(p.stat().st_size for p in FOLDER.rglob("*") if p.is_file()) < 400_000
    assert "MIT" in (FOLDER / "README.md").read_text(encoding="utf-8")
    samples = {p.read_text(encoding="utf-8") for p in [*LLM.glob("*.md"), *WRITER.glob("*.md")]}
    assert not samples.intersection(texts)


def test_the_human_register_has_no_em_dashes() -> None:
    for path in (FOLDER / "humanlike").glob("*.md"):
        assert "—" not in path.read_text(encoding="utf-8"), path.name


def test_generic_folders_are_the_register_folders() -> None:
    assert {Path(f).name for f in _generic_folders()} == REGISTERS


def test_generic_build_is_labelled_and_combines_with_explicit_drafts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    generic = sp.build(WRITER, generic_contrast=True, syntax=False)
    assert generic.report["contrast"]["label"] == "generic LLM"
    assert generic.report["contrast"]["sources"] == 60
    assert generic.report["settings"]["generic_contrast"] is True
    combined = sp.build(WRITER, generic_contrast=True, contrast=LLM, syntax=False)
    assert combined.report["contrast"]["sources"] == 65
    strings = [p.read_text(encoding="utf-8") for p in sorted(WRITER.glob("*.md"))]
    from_strings = sp.build_texts(strings, generic_contrast=True, syntax=False)
    assert from_strings.report["contrast"]["sources"] == 60
    assert not (tmp_path / "cache").exists()
    assert "generic" in generic.to_text().lower()
    path = tmp_path / "generic.json"
    generic.save(path)
    assert sp.load(path).report["settings"]["generic_contrast"] is True


def test_a_profile_without_generic_contrast_has_no_trace_of_it() -> None:
    plain = sp.build(WRITER, contrast=LLM, syntax=False)
    assert "generic_contrast" not in plain.report["settings"]
    again = sp.build(WRITER, contrast=LLM, syntax=False)
    assert plain.report == again.report


def test_the_cli_flag_builds_a_generic_profile(tmp_path: Path) -> None:
    output = tmp_path / "generic.json"
    args = ["build", str(WRITER), "--generic-contrast", "--no-syntax", "--no-cache"]
    assert main([*args, "-o", str(output)]) == 0
    assert sp.load(output).report["contrast"]["sources"] == 60


def test_groups_leave_out_a_whole_register_and_default_to_documents() -> None:
    reference = [{("g", "m"): float(i % 3)} for i in range(6)]
    reference_sources = [f"r{i // 2}" for i in range(6)]
    contrast = [{("g", "m"): 3.0 + i} for i in range(4)]
    documents = ["a", "b", "c", "d"]
    by_document = cross_validate(reference, reference_sources, contrast, documents)
    assert set(by_document.contrast_folds) == set(documents)
    assert cross_validate(
        reference, reference_sources, contrast, documents, documents
    ).contrast_scores == pytest.approx(by_document.contrast_scores)
    groups = ["x", "x", "y", "y"]
    by_group = cross_validate(reference, reference_sources, contrast, documents, groups)
    assert set(by_group.contrast_folds) == {"x", "y"}


def test_generic_likeness_is_an_indicative_number_without_verdict_words(tmp_path: Path) -> None:
    profile = sp.build(WRITER, generic_contrast=True, syntax=False)
    text = profile.score(LLM).to_text(full=True)
    assert "generic LLM-likeness (indicative):" in text
    for words_ in ("leans", "like the generic", "a few generic", "Most generic LLM-like"):
        assert words_ not in text
    own = sp.build(WRITER, contrast=LLM, syntax=False).score(LLM).to_text()
    assert "(indicative)" not in own


def test_fail_likeness_refuses_a_generic_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "generic.json"
    sp.build(WRITER, generic_contrast=True, syntax=False).save(path)
    code = main(["score", str(LLM), str(path), "--fail-likeness", "leans", "--quiet"])
    assert code != 0
    assert "indicative" in capsys.readouterr().err


def test_quiet_lines_show_an_indicative_number_for_generic_contrast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "generic.json"
    sp.build(WRITER, generic_contrast=True, syntax=False).save(path)
    assert main(["score", "-q", str(LLM), str(path)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        assert "generic LLM likeness (indicative) " in line
        for words_ in ("leans", "like the generic", "a few generic", "drafts drafts"):
            assert words_ not in line
    one = tmp_path / "one.md"
    one.write_text(next(LLM.glob("*.md")).read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["score", "-q", str(one), str(path)]) == 0
    assert "likeness (indicative)" in capsys.readouterr().out


def test_library_results_flag_generic_likeness_as_indicative() -> None:
    generic = sp.build(WRITER, generic_contrast=True, syntax=False).score(LLM)
    assert generic.indicative is True
    assert generic.contrast_label == "generic LLM"
    assert generic.documents and all(d.indicative for d in generic.documents)
    assert generic.likeness_verdict is not None  # still the stored verdict, for those who want it
    own = sp.build(WRITER, contrast=LLM, syntax=False).score(LLM)
    assert own.indicative is False and not any(d.indicative for d in own.documents)
    none = sp.build(WRITER, syntax=False).score(LLM)
    assert none.indicative is False
