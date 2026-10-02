"""Per-document verdicts when several documents are scored at once, and --fail-above."""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile.cli import EXIT_FAILED, _failed, main
from styleprofile.core import NoteCode
from styleprofile.display import CLOSE_ROWS, format_summary, shorten
from styleprofile.profile import load_report, write_report
from styleprofile.schema import DocumentEntry

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples" / "writer"
DRAFTS = ROOT / "examples" / "llm-drafts"
OWN, DRAFT = WRITER / "sharpening.md", DRAFTS / "old-maps.md"
# The same files as typed from the repository root, as a user or a hook would.
OWN_TYPED, DRAFT_TYPED = "examples/writer/sharpening.md", "examples/llm-drafts/old-maps.md"


@pytest.fixture(scope="module")
def profile() -> sp.Profile:
    return sp.build(WRITER, sp.Settings(syntax=False), contrast=DRAFTS)


@pytest.fixture(scope="module")
def saved(profile: sp.Profile, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("per-document") / "writer.json"
    profile.save(path)
    return path


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A reference built without a contrast set."""
    path = tmp_path_factory.mktemp("plain") / "plain.json"
    sp.build(WRITER, sp.Settings(syntax=False)).save(path)
    return path


@pytest.fixture
def at_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from the repository root with color off, so paths are typed as a user would."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("NO_COLOR", "1")


def _block(out: str, title: str) -> list[str]:
    return out.split(title, 1)[1].split("\n\n", 1)[0].splitlines()


def test_each_document_gets_its_own_verdict(profile: sp.Profile) -> None:
    result = profile.score([OWN, DRAFT])
    own, draft = result.documents
    assert (own.name, own.path, own.verdict, own.chunks) == (
        "sharpening.md",
        "sharpening.md",
        sp.Verdict.CLOSE,
        1,
    )
    assert (draft.name, draft.verdict) == ("old-maps.md", sp.Verdict.VERY_DIFFERENT)
    assert own.likeness_verdict is sp.LikenessVerdict.LIKE_REFERENCE
    assert draft.likeness_verdict is sp.LikenessVerdict.LIKE_DRAFTS
    # The pooled figures hide the close essay.
    assert result.verdict is sp.Verdict.CLEARLY_DIFFERENT
    assert own.delta is not None and draft.delta is not None
    assert result.delta == pytest.approx(
        (own.delta * own.chunks + draft.delta * draft.chunks) / (own.chunks + draft.chunks),
        abs=1e-6,
    )
    assert own.words + draft.words == result.report["word_count"]
    assert draft.differences and all(isinstance(z, float) for _, z in draft.differences)
    assert draft.differences[0][0].startswith(("vocabulary.", "markdown."))
    assert draft.signals and len(draft.signals) <= 3
    metric, z, share = next(iter(draft.signals))
    assert "." in metric and z > 0 and share > 0
    assert draft.location == str(DRAFT) and draft.shown == str(DRAFT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        draft.name = "other"  # type: ignore[misc]


def test_a_document_scores_the_same_alone_or_with_others(profile: sp.Profile) -> None:
    """Grouping judges each document exactly as scoring it alone would."""
    together = profile.score([OWN, DRAFT]).documents
    for path, document in zip([OWN, DRAFT], together, strict=True):
        alone = profile.score(path)
        (single,) = alone.documents
        assert single == document
        assert (alone.delta, alone.verdict) == (document.delta, document.verdict)
        assert alone.likeness_verdict is document.likeness_verdict


def test_windows_of_one_document_are_judged_together(profile: sp.Profile) -> None:
    result = profile.score([OWN, DRAFT], window_words=200)
    own, draft = result.documents
    assert own.chunks > 1 and draft.chunks > 1
    assert own.chunks + draft.chunks == result.chunk_count
    rows = [row for row in result.report["chunks"] if row["id"].startswith("old-maps.md#w")]
    assert draft.words == sum(row["metrics"]["size"]["words"] or 0 for row in rows)
    assert draft.delta == pytest.approx(
        sum(row["reference"]["delta"] or 0 for row in rows) / len(rows), abs=1e-6
    )


def test_files_with_the_same_name_stay_apart(profile: sp.Profile) -> None:
    result = profile.score([WRITER / "old-maps.md", DRAFT, OWN])
    assert [document.name for document in result.documents] == [
        "old-maps.md",
        "old-maps (2).md",
        "sharpening.md",
    ]
    assert [document.location for document in result.documents] == [
        str(WRITER / "old-maps.md"),
        str(DRAFT),
        str(OWN),
    ]


def test_a_directory_names_every_file_inside_it(profile: sp.Profile) -> None:
    result = profile.score([WRITER, DRAFT])
    names = [document.name for document in result.documents]
    assert names[:-1] == sorted(f"writer/{path.name}" for path in WRITER.glob("*.md"))
    assert names[-1] == "old-maps.md"
    assert result.documents[0].location == str(WRITER / "fence-lines.md")


def test_names_are_unique_and_the_same_under_any_hash_seed(tmp_path: Path) -> None:
    """The review's case: a folder holding writer/old-maps.md, scored with two more files
    named old-maps.md, gave duplicate names under some hash seeds."""
    folder = tmp_path / "root" / "writer"
    folder.mkdir(parents=True)
    (folder / "old-maps.md").write_text(DRAFT.read_text(encoding="utf-8"), encoding="utf-8")
    reference = tmp_path / "writer.json"
    sp.build(WRITER, sp.Settings(syntax=False)).save(reference)
    script = (
        "import json, sys; import styleprofile as sp; "
        "result = sp.Profile.load(sys.argv[1]).score(sys.argv[2:]); "
        "print(json.dumps([d.name for d in result.documents]))"
    )
    argv = [str(reference), str(tmp_path / "root"), str(WRITER / "old-maps.md"), str(DRAFT)]
    seen = set()
    for seed in ("0", "1", "2", "3", "4", "5"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", script, *argv],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout
        seen.add(out)
    (names,) = [json.loads(out) for out in seen]
    assert names == ["root/writer/old-maps.md", "old-maps.md", "old-maps (2).md"]


def test_jsonl_records_are_named_by_file_and_id(profile: sp.Profile, tmp_path: Path) -> None:
    essay, draft = OWN.read_text(encoding="utf-8"), DRAFT.read_text(encoding="utf-8")
    records = tmp_path / "posts.jsonl"
    records.write_text(
        "\n".join(
            json.dumps(record)
            for record in ({"id": "a", "text": essay}, {"text": draft}, {"id": 7, "text": essay})
        ),
        encoding="utf-8",
    )
    result = profile.score(records)
    assert [document.name for document in result.documents] == [
        "posts.jsonl:a",
        "posts.jsonl:2",
        "posts.jsonl:7",
    ]
    assert {document.path for document in result.documents} == {"posts.jsonl"}
    assert result.documents[1].verdict is sp.Verdict.VERY_DIFFERENT
    # The command line names each record after the file as typed.
    assert [document.shown for document in result.documents] == [
        f"{records}:a",
        f"{records}:2",
        f"{records}:7",
    ]


def test_jsonl_records_are_told_apart_in_quiet_and_failed_lines(
    saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    essay, draft = OWN.read_text(encoding="utf-8"), DRAFT.read_text(encoding="utf-8")
    records = tmp_path / "exports" / "dup.jsonl"
    records.parent.mkdir()
    lines = [
        {"id": "same", "text": draft},
        {"id": "same", "text": draft},
        {"id": "ok", "text": essay},
        {"text": draft},
    ]
    records.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    argv = ["score", "-q", "--fail-above", "clearly", str(records), str(saved)]
    assert main(argv) == EXIT_FAILED
    captured = capsys.readouterr()
    quiet = [line.split(": ", 1)[0] for line in captured.out.splitlines()]
    assert sorted(quiet) == sorted(f"{records}:{name}" for name in ("same@1", "same@2", "ok", "4"))
    failed = [line for line in captured.err.splitlines() if line.startswith("failed")]
    assert failed == [
        f"failed: {records}:{name}: delta very different" for name in ("same@1", "same@2", "4")
    ]


def test_records_sharing_an_id_stay_separate_documents(profile: sp.Profile, tmp_path: Path) -> None:
    essay, draft = OWN.read_text(encoding="utf-8"), DRAFT.read_text(encoding="utf-8")
    records = tmp_path / "dup.jsonl"
    lines = [
        {"id": "same", "text": essay},
        {"id": "same", "text": draft},
        {"id": "x", "text": essay},
        {"id": "x#w2", "text": draft},
    ]
    records.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    result = profile.score(records)
    names = [document.name for document in result.documents]
    # The id is shown as given; only chunk ids keep the escape.
    assert names == ["dup.jsonl:same@1", "dup.jsonl:same@2", "dup.jsonl:x", "dup.jsonl:x#w2"]
    assert result.report["chunks"][-1]["id"].startswith("x%23w2")
    verdicts = [document.verdict for document in result.documents]
    assert verdicts == [sp.Verdict.CLOSE, sp.Verdict.VERY_DIFFERENT] * 2
    (note,) = [note for note in result.notes if note.code is NoteCode.REPEATED_ID]
    assert "dup.jsonl share ids (2 share 'same')" in note.message


def test_texts_are_documents_by_name(profile: sp.Profile) -> None:
    essay, draft = OWN.read_text(encoding="utf-8"), DRAFT.read_text(encoding="utf-8")
    result = profile.score([sp.Text(essay, name="mine"), sp.Text(draft), sp.Text(essay, "y#w1")])
    assert [document.name for document in result.documents] == ["mine", "text1", "y#w1"]
    assert all(document.location is None for document in result.documents)


def test_stdin_and_a_file(
    saved: Path, at_root: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    stdin = io.TextIOWrapper(io.BytesIO(DRAFT.read_bytes()))
    monkeypatch.setattr(sys, "stdin", stdin)
    assert main(["score", "-q", "-", OWN_TYPED, str(saved)]) == 0
    first, second = capsys.readouterr().out.splitlines()
    assert first.startswith("<stdin>: very different (Delta ")
    assert second.startswith(f"{OWN_TYPED}: close (Delta ")


def test_stdin_and_a_file_called_stdin_are_told_apart(
    saved: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "stdin").write_bytes(OWN.read_bytes())
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(DRAFT.read_bytes())))
    assert main(["score", "-q", "--fail-above", "clearly", "-", "stdin", str(saved)]) == 3
    captured = capsys.readouterr()
    first, second = captured.out.splitlines()
    assert first.startswith("<stdin>: very different (Delta ")
    assert second.startswith("stdin: close (Delta ")
    assert "failed: <stdin>: delta very different" in captured.err


def test_the_json_report_lists_documents(saved: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", "--json", str(OWN), str(DRAFT), str(saved)]) == 0
    report = json.loads(capsys.readouterr().out)
    documents = {document["name"]: document for document in report["documents"]}
    assert list(documents) == ["sharpening.md", "old-maps.md"]
    draft = documents["old-maps.md"]
    assert draft["path"] == "old-maps.md"
    assert draft["verdict"] == "very different"
    assert draft["likeness_verdict"] == "like the drafts"
    assert {"chunks", "words", "delta", "likeness", "differences", "signals"} <= set(draft)
    assert set(draft["differences"][0]) == {"metric", "z"}
    assert set(draft["signals"][0]) == {"metric", "z", "contribution"}
    assert documents["sharpening.md"]["verdict"] == "close"
    # No fail flag, nothing about failing.
    assert "failed" not in report and "fail" not in report


def test_the_table_and_quiet_lines_put_the_furthest_first(
    saved: Path, tmp_path: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "mixed.json"
    assert main(["score", OWN_TYPED, DRAFT_TYPED, str(saved), "-o", str(report)]) == 0
    out = capsys.readouterr().out
    table = _block(out, "By document")
    assert table[1] == "  2 documents: 1 very different, 1 close"
    # A live run names each file as typed, as -q does (cut in the middle to fit).
    assert table[3].split()[0] == shorten(DRAFT_TYPED, len(table[3].split()[0]))
    assert table[3].split()[0].startswith("examples/") and "very different" in table[3]
    assert "most different in:" in table[4]
    assert table[5].split()[0] == shorten(OWN_TYPED, len(table[5].split()[0]))
    assert " close " in table[5]
    assert all(len(line) <= 80 for line in table)
    assert "Across 2 documents: clearly different" in out
    assert "Overall:" not in out

    # show renders the same table from the saved report, with the saved names.
    assert main(["show", str(report)]) == 0
    shown = _block(capsys.readouterr().out, "By document")
    assert shown[3].lstrip().startswith("old-maps.md ")
    assert shown[5].lstrip().startswith("sharpening.md ")
    assert [line.split()[-3:] for line in shown] == [line.split()[-3:] for line in table]

    # -q names each file where it can be opened, as with one document.
    assert main(["score", "-q", OWN_TYPED, DRAFT_TYPED, str(saved)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith(f"{DRAFT_TYPED}: very different (Delta ")
    assert lines[1].startswith(f"{OWN_TYPED}: close (Delta ")
    assert main(["score", "-q", OWN_TYPED, str(saved)]) == 0
    assert capsys.readouterr().out.startswith(f"{OWN_TYPED}: close (Delta ")


def test_one_document_keeps_the_single_view(
    saved: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["score", DRAFT_TYPED, str(saved)]) == 0
    out = capsys.readouterr().out
    assert "By document" not in out and "Overall: very different" in out


def test_the_reference_can_come_first_for_hooks(
    saved: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """pre-commit appends file names to the command, so the reference goes before them."""
    argv = ["score", "-q", "--fail-above", "clearly", "-r", str(saved), OWN_TYPED, DRAFT_TYPED]
    assert main(argv) == EXIT_FAILED
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2
    assert captured.err.splitlines()[-1] == f"failed: {DRAFT_TYPED}: delta very different"
    assert main(["score", "-q", "--fail-above", "clearly", "-r", str(saved), OWN_TYPED]) == 0


@pytest.mark.parametrize(
    ("level", "inputs", "code"),
    [
        ("clearly", [OWN_TYPED, DRAFT_TYPED], EXIT_FAILED),
        ("very", [OWN_TYPED, DRAFT_TYPED], EXIT_FAILED),
        ("somewhat", [OWN_TYPED], 0),
        ("clearly", [OWN_TYPED, "examples/writer/old-maps.md"], 0),
    ],
)
def test_fail_above_exits_3_when_any_document_reaches_the_level(
    saved: Path,
    at_root: None,
    capsys: pytest.CaptureFixture[str],
    level: str,
    inputs: list[str],
    code: int,
) -> None:
    argv = ["score", "-q", "--fail-above", level, *inputs, str(saved)]
    assert main(argv) == code
    err = capsys.readouterr().err
    if code:
        assert f"failed: {DRAFT_TYPED}: delta very different\n" in err
    else:
        assert "failed:" not in err


def test_fail_above_applies_with_json_and_the_full_view(
    saved: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["score", "--json", "--fail-above", "clearly", OWN_TYPED, DRAFT_TYPED, str(saved)]
    assert main(argv) == EXIT_FAILED
    report = json.loads(capsys.readouterr().out)  # stdout stays pure JSON
    assert report["fail"] == {"above": "clearly", "likeness": None, "flagged": None}
    # The entry counts its chunks flagged on their own, whichever check it failed.
    assert report["failed"] == [
        {
            "name": "old-maps.md",
            "path": "old-maps.md",
            "delta": "very different",
            "likeness": None,
            "flagged": 1,
            "chunks_judged": 1,
        }
    ]
    assert main(["score", "--fail-above", "very", OWN_TYPED, DRAFT_TYPED, str(saved)]) == 3


def test_both_fail_flags_give_one_line_per_document(
    saved: Path, tmp_path: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "failed.json"
    argv = ["score", "-q", "--fail-above", "clearly", "--fail-likeness", "leans"]
    argv += [OWN_TYPED, DRAFT_TYPED, str(saved), "-o", str(report)]
    assert main(argv) == EXIT_FAILED
    failed = [line for line in capsys.readouterr().err.splitlines() if line.startswith("failed")]
    assert failed == [f"failed: {DRAFT_TYPED}: delta very different; likeness like the LLM drafts"]
    saved_report = json.loads(report.read_text(encoding="utf-8"))
    assert saved_report["failed"][0]["likeness"] == "like the LLM drafts"


def test_fail_likeness(
    saved: Path, plain: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["score", "-q", "--fail-likeness", "leans", OWN_TYPED, DRAFT_TYPED, str(saved)]
    assert main(argv) == EXIT_FAILED
    assert f"failed: {DRAFT_TYPED}: likeness like the LLM drafts\n" in capsys.readouterr().err
    assert main(["score", "-q", "--fail-likeness", "few", OWN_TYPED, str(saved)]) == 0

    assert main(["score", "-q", "--fail-likeness", "few", OWN_TYPED, str(plain)]) == 1
    err = capsys.readouterr().err
    assert "--fail-likeness needs a reference built with --contrast" in err

    with pytest.raises(SystemExit) as exit_:
        main(["score", "--fail-above", "slightly", OWN_TYPED, str(saved)])
    assert exit_.value.code == 2


# 45 prose words in the LLM drafts' register, too short to judge. Its indicative Delta is
# about 17 and its likeness about 40: judged, it would fail both flags at any level.
SHORT_DRAFT = (
    "## Key Takeaways\n\n"
    "- **Leverage** comprehensive documentation to streamline organizational communication.\n"
    "- **Facilitate** meaningful collaboration through transparent, actionable frameworks.\n"
    "- **Prioritize** sustainable implementation of innovative methodologies.\n\n"
    "Ultimately, these considerations underscore the fundamental importance of establishing "
    "robust, scalable infrastructure that empowers stakeholders to navigate increasingly "
    "complex environments with confidence and demonstrable effectiveness.\n"
)


def test_a_short_document_is_too_short_to_judge_and_never_fails(
    saved: Path, tmp_path: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    short = tmp_path / "short.md"
    short.write_text(SHORT_DRAFT, encoding="utf-8")
    result = sp.Profile.load(saved).score([short, DRAFT])
    note, draft = result.documents
    assert note.words < 75
    assert (note.judged, note.verdict, note.likeness_verdict) == (
        False,
        sp.Verdict.TOO_SHORT,
        sp.LikenessVerdict.TOO_SHORT,
    )
    assert note.reason and note.reason.startswith("under 75 words")
    assert draft.judged and draft.verdict is sp.Verdict.VERY_DIFFERENT
    # Judged alone, the same way.
    assert sp.Profile.load(saved).score(short).verdict is sp.Verdict.TOO_SHORT

    # Both flags at their lowest level: only the long draft fails.
    argv = ["score", "-q", "--fail-above", "somewhat", "--fail-likeness", "few"]
    assert main([*argv, str(short), DRAFT_TYPED, str(saved)]) == EXIT_FAILED
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == f"{short}: too short to judge ({note.words} words)"
    failed = [line for line in captured.err.splitlines() if line.startswith("failed")]
    assert failed == [f"failed: {DRAFT_TYPED}: delta very different; likeness like the LLM drafts"]
    # Next to the writer's own essay, nothing fails.
    assert main([*argv, str(short), OWN_TYPED, str(saved)]) == 0
    assert "failed" not in capsys.readouterr().err

    # The table lists it last, with no figures.
    assert main(["score", str(short), DRAFT_TYPED, str(saved)]) == 0
    table = _block(capsys.readouterr().out, "By document")
    assert table[1] == "  2 documents: 1 very different, 1 too short to judge"
    assert table[-1].split()[0].endswith("short.md")
    assert table[-1].split()[1:] == [
        str(note.words),
        *f"too short to judge ({note.words} words)".split(),
    ]


def test_a_document_that_is_not_comparable_never_fails(profile: sp.Profile) -> None:
    result = profile.score([OWN, DRAFT])
    for entry in result.report["documents"]:
        # Judged on length, but with no metric in common: no Delta verdict.
        entry["delta"] = None
        entry["verdict"] = str(sp.Verdict.NOT_COMPARABLE)
    assert all(document.judged for document in result.documents)
    assert result.failing(sp.Verdict.SOMEWHAT_DIFFERENT, sp.LikenessVerdict.FEW_TRAITS) == ()


def test_names_are_cut_in_the_middle() -> None:
    spring = "spring/a-really-quite-long-file-name-for-a-blog-post.md"
    autumn = "autumn/a-really-quite-long-file-name-for-a-blog-post.md"
    assert shorten(spring, 24) == "spring/…r-a-blog-post.md"
    assert shorten(spring, 24) != shorten(autumn, 24)
    assert all(len(shorten(name, 24)) == 24 for name in (spring, autumn))
    assert shorten("short.md", 24) == "short.md"


def _document(name: str, verdict: str, delta: float) -> DocumentEntry:
    return {
        "name": name,
        "path": name,
        "chunks": 1,
        "words": 600,
        "judged": True,
        "chunks_judged": 1,
        "reason": None,
        "delta": delta,
        "verdict": verdict,
        "likeness": None,
        "likeness_verdict": None,
        "flagged": 0,
        "differences": [],
    }


def test_close_rows_are_capped_unless_all_are_asked_for(profile: sp.Profile) -> None:
    result = profile.score([OWN, DRAFT])
    report = result.report
    report["documents"] = [_document("far.md", "very different", 4.0)] + [
        _document(f"near-{index:02d}.md", "close", 0.5) for index in range(15)
    ]
    table = _block(format_summary(report, result.report["reference"]["baseline"]), "By document")
    assert table[1] == "  16 documents: 1 very different, 15 close"
    rows = [line for line in table[3:] if line.startswith("  ") and ".md" in line]
    assert len(rows) == 1 + CLOSE_ROWS
    assert table[-1] == f"  … and {15 - CLOSE_ROWS} more close (--all lists them)"
    full = format_summary(report, result.report["reference"]["baseline"], full=True)
    assert len([line for line in _block(full, "By document") if "near-" in line]) == 15


def test_a_score_report_from_before_per_document_verdicts_is_refused(
    profile: sp.Profile, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A report of this version from before plan PR 7: the schema needs ``documents``.
    report: dict[str, Any] = dict(profile.score([OWN, DRAFT]).report)
    del report["documents"]
    path = tmp_path / "old-score.json"
    write_report(report, path)
    with pytest.raises(sp.StyleProfileError, match="has no documents") as error:
        load_report(path)
    assert error.value.code == "outdated"
    assert main(["show", str(path)]) == 1
    assert "score it again with `styleprofile score`" in capsys.readouterr().err


def test_a_document_judged_on_some_of_its_chunks_says_so(
    saved: Path, at_root: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # 60-word windows: some of each file's windows fall under 75 words.
    argv = ["score", "--window-words", "60", DRAFT_TYPED, OWN_TYPED, str(saved)]
    result = sp.Profile.load(saved).score([DRAFT, OWN], window_words=60)
    draft = result.documents[0]
    assert draft.judged and 0 < draft.chunks_judged < draft.chunks
    note = f"{draft.chunks - draft.chunks_judged} of {draft.chunks} chunks not judged: too short"

    assert main([*argv, "-q"]) == 0
    quiet = capsys.readouterr().out.splitlines()
    assert quiet[0].startswith(f"{DRAFT_TYPED}: clearly different (Delta ")
    assert f"; {note})" in quiet[0]

    assert main(argv) == 0
    table = _block(capsys.readouterr().out, "By document")
    assert table[3].split()[0].endswith("old-maps.md")
    assert table[4] == f"      {note}"


def test_failing_is_linear_in_the_number_of_documents(profile: sp.Profile) -> None:
    result = profile.score([OWN, DRAFT])
    far = result.report["documents"][1]
    assert far["verdict"] == "very different"
    result.report["documents"] = [{**far, "name": f"far-{index}.md"} for index in range(2000)]
    args = argparse.Namespace(fail_above="clearly", fail_likeness="few", fail_flagged=None)
    start = time.perf_counter()
    failed = _failed(args, result)
    # Quadratic, this took about 16 s; linear, a few milliseconds.
    assert time.perf_counter() - start < 2.0
    assert len(failed) == 2000
    assert len(result.failing(sp.Verdict.CLEARLY_DIFFERENT)) == 2000
