"""``styleprofile evaluate``, with deterministic fake editors in place of a model."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from styleprofile import Chunk, StyleProfileError
from styleprofile.cli import main
from styleprofile.evaluate import evaluate_rewording, match_key, ngram_changed
from styleprofile.weighting import ZScores, cross_validate, fold_scores

POOL = [
    "We shipped the first version in March, and nobody used it.",
    "I think the problem was obvious, but we didn't want to see it.",
    "So we talked to twelve customers, one at a time, over coffee.",
    "Most of them liked the idea, though few had the problem.",
    "It's easy to build something clever, and hard to build something needed.",
    "Our second attempt was smaller, slower and much more useful.",
    "You learn more from one angry email than from ten polite ones, honestly.",
    "The team was tired, but the numbers finally moved.",
    "I kept a notebook of every complaint, sorted by how often it came up.",
    "We cut half the features, and nobody noticed.",
    "That was humbling, of course, and also a relief.",
    "If I did it again, I'd start with the notebook.",
    "Pricing was the hardest part, not the code.",
    "We charged too little at first, then too much, then about right.",
    "The best advice came from a customer, not an investor.",
    "Hiring slowly felt wrong at the time, but it saved us later.",
    "I still don't know if the market was ready, or if we were early.",
    "Every launch felt smaller than the one before it, somehow.",
    "Support tickets told us more than any dashboard, by far.",
    "We wrote the docs ourselves, badly at first, then better.",
]


def _essay(rng: random.Random, sentences: int = 9) -> str:
    chosen = rng.sample(POOL, sentences)
    return " ".join(chosen[:5]) + "\n\n" + " ".join(chosen[5:])


def _dashed(text: str) -> str:
    """LLM-style: the same prose with its commas turned into em dashes."""
    return text.replace(", ", " — ")


def _strip_dashes(text: str) -> str:
    """A fake humanizer that removes the top signal, em dashes."""
    return text.replace(" — ", ", ").replace("—", ", ")


def _identity(text: str) -> str:
    return text


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    rng = random.Random(7)
    author, drafts = tmp_path / "author", tmp_path / "drafts"
    author.mkdir()
    drafts.mkdir()
    for index in range(8):
        (author / f"post{index}.md").write_text(_essay(rng), encoding="utf-8")
    for index in range(5):
        (drafts / f"draft{index}.md").write_text(_dashed(_essay(rng)), encoding="utf-8")
    return author, drafts


def _edit(drafts: Path, output: Path, edit: Callable[[str], str]) -> None:
    """Edited copies with the originals' file names, as any editing tool would write them."""
    output.mkdir()
    for path in sorted(drafts.glob("*.md")):
        (output / path.name).write_text(edit(path.read_text(encoding="utf-8")), encoding="utf-8")


def _evaluate(author: Path, drafts: Path, edited: dict[str, Path], **options: Any) -> Any:
    from styleprofile.profile import load_chunks

    return evaluate_rewording(
        load_chunks([str(author)]),
        load_chunks([str(drafts)]),
        {label: load_chunks([str(folder)]) for label, folder in edited.items()},
        parser=None,
        **options,
    )


def test_fold_scoring_never_uses_a_documents_own_weights() -> None:
    rng = random.Random(0)
    key, other = ("punctuation", "em"), ("shape", "length")
    reference: list[ZScores] = [{key: rng.gauss(0, 1), other: rng.gauss(0, 1)} for _ in range(12)]
    reference_sources = [f"r{index // 2}" for index in range(12)]
    contrast: list[ZScores] = [{key: 3 + index, other: rng.gauss(1, 1)} for index in range(4)]
    contrast_sources = ["a", "b", "c", "d"]
    learned = cross_validate(reference, reference_sources, contrast, contrast_sources)

    # A document's fold is exactly what the other documents alone would learn...
    for held_out in contrast_sources:
        kept = [index for index, source in enumerate(contrast_sources) if source != held_out]
        alone = cross_validate(
            reference,
            reference_sources,
            [contrast[index] for index in kept],
            [contrast_sources[index] for index in kept],
        )
        assert learned.contrast_folds[held_out][0] == pytest.approx(alone.effects)
    # ...so changing the document itself changes every fold but its own.
    changed = [dict(row) for row in contrast]
    changed[0][key] = 500.0
    relearned = cross_validate(reference, reference_sources, changed, contrast_sources)
    assert relearned.contrast_folds["a"] == learned.contrast_folds["a"]
    assert relearned.contrast_folds["b"] != learned.contrast_folds["b"]

    # An edited copy of "a" is scored with a's fold, and an unknown document is an error.
    assert fold_scores(contrast[:1], ["a"], learned.contrast_folds) == learned.contrast_scores[:1]
    with pytest.raises(KeyError, match="no held-out fold"):
        fold_scores(contrast[:1], ["zzz"], learned.contrast_folds)


def test_identity_rewriter_reproduces_the_original_auc(tmp_path: Path) -> None:
    author, drafts = _corpus(tmp_path)
    _edit(drafts, tmp_path / "same", _identity)

    result = _evaluate(author, drafts, {"same": tmp_path / "same"}, retrain=True)

    original, same = result["sets"]["original"], result["sets"]["same"]
    assert original["auc"] == pytest.approx(1.0)
    assert same["auc"] == original["auc"] and same["auc_ci"] == original["auc_ci"]
    assert same["by_draft"] == original["by_draft"]
    assert same["edits"]["ngram13_changed_median"] == 0
    top = result["signals"][0]
    assert top["metric"] == "punctuation.em_dashes_per_1k"
    assert top["edited"]["same"]["remaining"] == pytest.approx(1.0)
    assert result["retrain"]["by_set"]["original"]["before"] == original["auc"]


def test_removing_the_top_signal_lowers_the_auc_and_shows_it_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    author, drafts = _corpus(tmp_path)
    _edit(drafts, tmp_path / "humanize", _strip_dashes)
    output = tmp_path / "stress.json"

    arguments = [
        "evaluate",
        str(author),
        "--contrast",
        str(drafts),
        "--edited",
        f"humanize={tmp_path / 'humanize'}",
        f"same={drafts}",
        "--no-syntax",
        "--retrain",
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    result = json.loads(output.read_text(encoding="utf-8"))

    original, humanized = result["sets"]["original"], result["sets"]["humanize"]
    assert result["sets"]["same"]["auc"] == original["auc"]
    assert humanized["auc"] < original["auc"] - 0.3
    assert humanized["flagged"] < original["flagged"]
    dashes = next(s for s in result["signals"] if s["metric"] == "punctuation.em_dashes_per_1k")
    assert dashes["edited"]["humanize"]["remaining"] < 0.1
    assert humanized["edits"]["ngram13_changed_median"] > 0.5

    shown = capsys.readouterr()
    assert "REWORDING STRESS TEST" in shown.out
    assert "Em dashes" in shown.out and "% gone" in shown.out
    assert "weaker evidence" in shown.out and "Retrained" in shown.out
    # The "same" set is the drafts folder itself, which is worth a note.
    assert "same: 5 files" in shown.err and "not an edit of them" in shown.err

    # The saved report is its own kind: show renders it, and it is no reference.
    assert result["kind"] == "evaluation"
    assert main(["show", str(output)]) == 0
    assert "REWORDING STRESS TEST" in capsys.readouterr().out
    assert main(["score", str(author / "post0.md"), str(output)]) == 1
    assert "evaluation report" in capsys.readouterr().err
    # evaluate is one command among the others; build still runs on its own.
    assert main(["build", str(author), "--no-syntax", "-o", str(tmp_path / "p.json")]) == 0


def test_edited_drafts_are_matched_to_originals_by_name(tmp_path: Path) -> None:
    author, drafts = _corpus(tmp_path)
    assert match_key(Chunk("sub/draft1.md#w3", "x", "")) == "sub/draft1.md"

    partial = tmp_path / "partial"
    _edit(drafts, partial, _strip_dashes)
    (partial / "draft4.md").unlink()
    result = _evaluate(author, drafts, {"partial": partial})
    assert result["sets"]["partial"]["missing"] == ["draft4.md"]
    assert "original_auc_same_drafts" in result["sets"]["partial"]
    assert any("no edited copy" in warning for warning in result["warnings"])

    (partial / "stranger.md").write_text("A stranger's post, with commas.", encoding="utf-8")
    with pytest.raises(StyleProfileError, match="no original"):
        _evaluate(author, drafts, {"partial": partial})
    with pytest.raises(StyleProfileError, match="unedited"):
        _evaluate(author, drafts, {"original": drafts})

    from styleprofile.profile import load_chunks

    twice = [
        *load_chunks([str(drafts)]),
        Chunk("draft0.md", str(tmp_path / "other" / "draft0.md"), "Another, different draft."),
    ]
    with pytest.raises(StyleProfileError, match="contrast: 1 name"):
        evaluate_rewording(
            load_chunks([str(author)]), twice, {"x": load_chunks([str(drafts)])}, parser=None
        )
    # Two edited copies of one draft would be averaged into one verdict; refuse instead.
    doubled = [*load_chunks([str(drafts)]), Chunk("draft0.md", "edits.jsonl", "Another copy.")]
    with pytest.raises(StyleProfileError, match="x: 1 name"):
        evaluate_rewording(
            load_chunks([str(author)]), load_chunks([str(drafts)]), {"x": doubled}, parser=None
        )
    repeated = [Chunk("draft0.md", "e.jsonl", "One copy."), Chunk("draft0.md", "e.jsonl", "Two.")]
    with pytest.raises(StyleProfileError, match="x: 1 name"):
        evaluate_rewording(
            load_chunks([str(author)]), load_chunks([str(drafts)]), {"x": repeated}, parser=None
        )


def test_partial_sets_are_compared_with_the_drafts_they_cover(tmp_path: Path) -> None:
    """Unedited copies of the least dashed drafts must not look like dashes were removed."""
    author, drafts = _corpus(tmp_path)
    ranked = sorted(drafts.glob("*.md"), key=lambda path: path.read_text().count("—"))
    subset = tmp_path / "subset"
    subset.mkdir()
    for path in ranked[:2]:
        (subset / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    result = _evaluate(author, drafts, {"subset": subset})
    assert result["sets"]["subset"]["partial"] is True
    from styleprofile.display import format_evaluation

    assert "subset*" in format_evaluation(result) and "* subset edited only" in format_evaluation(
        result
    )

    dashes = next(s for s in result["signals"] if s["metric"] == "punctuation.em_dashes_per_1k")
    entry = dashes["edited"]["subset"]
    assert entry["remaining"] == pytest.approx(1.0)
    assert entry["original_z"] < dashes["original_z"]
    assert result["sets"]["subset"]["auc"] == result["sets"]["subset"]["original_auc_same_drafts"]


def test_short_edits_are_not_reported_missing(tmp_path: Path) -> None:
    author, drafts = _corpus(tmp_path)
    edited = tmp_path / "edited"
    _edit(drafts, edited, _identity)
    (edited / "draft1.md").write_text("Too short.", encoding="utf-8")

    result = _evaluate(author, drafts, {"edited": edited}, min_words=20)

    assert result["sets"]["edited"]["missing"] == []
    assert result["sets"]["edited"]["too_short"] == ["draft1.md"]
    assert any("not scored" in warning for warning in result["warnings"])


def test_short_originals_skip_their_edits(tmp_path: Path) -> None:
    """An original under --min-words has no fold; its edit is skipped, not "unmatched"."""
    author, drafts = _corpus(tmp_path)
    (drafts / "draft9.md").write_text("Too short.", encoding="utf-8")
    edited = tmp_path / "edited"
    _edit(drafts, edited, _identity)

    result = _evaluate(author, drafts, {"edited": edited}, min_words=20)

    assert result["sets"]["edited"]["skipped"] == ["draft9.md"]
    assert result["sets"]["edited"]["auc"] == result["sets"]["original"]["auc"]
    assert any("too little prose" in warning for warning in result["warnings"])


def test_an_unscorable_set_does_not_stop_the_others(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    author, drafts = _corpus(tmp_path)
    _edit(drafts, tmp_path / "empty", lambda text: "Too short.")
    _edit(drafts, tmp_path / "same", _identity)

    result = _evaluate(
        author, drafts, {"empty": tmp_path / "empty", "same": tmp_path / "same"}, min_words=20
    )

    empty = result["sets"]["empty"]
    assert empty["drafts"] == 0 and empty["auc"] is None
    assert empty["likeness_median_chunks"] is None and len(empty["too_short"]) == 5
    assert result["sets"]["same"]["auc"] == result["sets"]["original"]["auc"]
    output = tmp_path / "report.json"
    from styleprofile.profile import write_report

    write_report(result, output)
    assert main(["show", str(output)]) == 0
    assert "empty" in capsys.readouterr().out


def test_contrast_is_measured_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from styleprofile import profile

    author, drafts = _corpus(tmp_path)
    _edit(drafts, tmp_path / "same", _identity)
    measured: list[int] = []
    original = profile._measure

    def counting(chunks: Any, parser: Any, min_words: int, **options: Any) -> Any:
        measured.append(len(chunks))
        return original(chunks, parser, min_words, **options)

    monkeypatch.setattr(profile, "_measure", counting)
    result = _evaluate(author, drafts, {"same": tmp_path / "same"})
    # The reference, the contrast drafts and the edited set, once each.
    assert measured == [8, 5, 5]
    reference = profile.build_reference(
        profile.load_chunks([str(author)]), parser=None, contrast=profile.load_chunks([str(drafts)])
    )
    assert result["sets"]["original"]["auc"] == reference["contrast"]["calibration"]["auc"]


def test_evaluate_rejects_stdin_twice_and_overwriting_an_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    author, drafts = _corpus(tmp_path)
    base = ["evaluate", "--no-syntax", "--edited", f"x={drafts}"]
    stdin_twice = ["evaluate", "-", "--no-syntax", "--contrast", "-", "--edited", f"x={drafts}"]
    stdin_twice += ["--output", "o.json"]
    assert main(stdin_twice) == 1
    assert "only once" in capsys.readouterr().err
    stdin_edits = [*base, "y=-", "--contrast", str(drafts), str(author)]
    assert main([*stdin_edits, "--output", str(tmp_path / "o.json")]) == 1
    assert "not - (stdin)" in capsys.readouterr().err
    target = drafts / "draft0.md"
    clobber = [*base, "--contrast", str(target), str(author)]
    assert main([*clobber, "--output", str(target)]) == 1
    assert "would overwrite input" in capsys.readouterr().err
    # A writer's input after --edited is taken as an edited set; say where inputs go.
    with pytest.raises(SystemExit):
        main(["evaluate", "--contrast", str(drafts), "--edited", f"x={drafts}", str(author)])
    assert "before --edited" in capsys.readouterr().err


def test_ngram_changed_counts_verbatim_sequences() -> None:
    words = " ".join(f"w{index}" for index in range(30))
    assert ngram_changed(words, words) == 0
    assert ngram_changed(words, words.upper()) == 1
    assert ngram_changed("too short", "too short") is None
