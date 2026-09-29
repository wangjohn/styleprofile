"""Splitting one big file into documents (plan PR 11): where a text divides (headings, rules,
never inside code or front matter), which split is chosen, stand-in documents for a text
with neither, and what that does for calibration, contrast, scoring and the command line."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile.cli import main
from styleprofile.formats import html_to_markdown
from styleprofile.profile import chunk_document, document_label
from styleprofile.split import (
    HEADING,
    RULE,
    STAND_INS,
    markers,
    part_id,
    plan_split,
    stand_in_groups,
)

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "examples" / "writer"
DRAFTS = ROOT / "examples" / "llm-drafts"
DRAFT = ROOT / "examples" / "draft.md"
SURFACE = sp.Settings(syntax=False)
WINDOW = 500


def _essays() -> list[str]:
    return [path.read_text(encoding="utf-8") for path in sorted(WRITER.glob("*.md"))]


def _without_headings(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.startswith("#"))


def _prose(words: int, seed: int = 0) -> str:
    """Paragraphs of the essays' prose, about ``words`` words, starting at a different place
    for each ``seed``."""
    paragraphs = [
        block
        for essay in _essays()
        for block in essay.split("\n\n")
        if block.strip() and not block.startswith("#")
    ]
    taken: list[str] = []
    index = seed * 3
    while sum(len(block.split()) for block in taken) < words:
        taken.append(paragraphs[index % len(paragraphs)])
        index += 1
    return "\n\n".join(taken)


def _coded(notes: tuple[sp.Note, ...], code: sp.NoteCode) -> list[str]:
    return [note.message for note in notes if note.code == code]


@pytest.fixture
def writer_file(tmp_path: Path) -> Path:
    """The sample essays in one Markdown file, each under its ``# Title``."""
    path = tmp_path / "writer.md"
    path.write_text("\n\n".join(_essays()), encoding="utf-8")
    return path


@pytest.fixture
def bare_file(tmp_path: Path) -> Path:
    """The same text with no headings: nothing marks where one essay ends."""
    path = tmp_path / "bare.md"
    path.write_text("\n\n".join(map(_without_headings, _essays())), encoding="utf-8")
    return path


# Where a text divides.


def test_headings_and_rules_are_found() -> None:
    text = "\n".join(
        [
            "# One",
            "",
            "Text.",
            "",
            "## Two ##",
            "",
            "---",
            "",
            "* * *",
            "",
            "___",
            "#hashtag, not a heading",
            "    # indented code, not a heading",
        ]
    )
    found = markers(text)
    assert [(m.line, m.kind, m.level, m.title) for m in found] == [
        (0, HEADING, 1, "One"),
        (4, HEADING, 2, "Two"),
        (6, RULE, 0, ""),
        (8, RULE, 0, ""),
        (10, RULE, 0, ""),
    ]


def test_front_matter_is_not_a_rule() -> None:
    text = "---\ntitle: A post\ntags: [a, b]\n---\n\nText.\n\n---\n\nMore text.\n"
    assert [(m.line, m.kind) for m in markers(text)] == [(7, RULE)]


def test_markers_inside_code_blocks_do_not_split() -> None:
    text = "\n".join(
        [
            "Text.",
            "",
            "```bash",
            "# a comment",
            "---",
            "```",
            "",
            "~~~",
            "***",
            "~~~",
            "",
            "````",
            "```",
            "# still code",
            "````",
            "",
            "# Real",
        ]
    )
    assert [(m.line, m.kind) for m in markers(text)] == [(16, HEADING)]


def test_an_unclosed_fence_hides_what_follows_unless_it_is_bare() -> None:
    # A fence with a language runs to the end, as markdown_blocks reads it.
    assert markers("Text.\n\n```python\n# code\n\n# More code\n") == []
    # A bare ``` that never closes is dropped (often a section break); what follows is read.
    assert [m.title for m in markers("Text.\n\n```\n\n# Next\n")] == ["Next"]


def test_a_dash_line_under_a_paragraph_is_a_heading_not_a_rule() -> None:
    text = "Chapter One\n===========\n\nText.\n\nA Section\n---\n\nText.\n\n---\n\nText.\n"
    found = [(m.line, m.kind, m.level, m.title) for m in markers(text)]
    assert found == [
        (0, HEADING, 1, "Chapter One"),
        (5, HEADING, 2, "A Section"),
        (10, RULE, 0, ""),
    ]
    # Under a list item it is a rule.
    assert [(m.line, m.kind) for m in markers("- item\n---\n")] == [(1, RULE)]


def test_crlf_text_splits_as_lf_text() -> None:
    text = "\n\n".join(f"# Part {n}\n\n{_prose(400, n)}" for n in range(4))
    lf, crlf = plan_split(text, WINDOW), plan_split(text.replace("\n", "\r\n"), WINDOW)
    assert lf is not None and crlf is not None
    assert lf.texts == crlf.texts and [p.title for p in crlf.parts] == [
        f"Part {n}" for n in range(4)
    ]
    assert markers("# A\r\n\r\ntext\r\n\r\n---\r\n") == markers("# A\n\ntext\n\n---\n")


# Which split is chosen.


def test_the_top_heading_level_that_splits_is_used() -> None:
    # A title, then chapters with sections: the title alone does not split, so chapters.
    chapters = "\n\n".join(
        f"## Chapter {n}\n\n{_prose(300, n)}\n\n### Section\n\n{_prose(300, n + 1)}"
        for n in range(5)
    )
    plan = plan_split(f"# The Book\n\n{chapters}", WINDOW)
    assert plan is not None and (plan.kind, plan.level) == (HEADING, 2)
    # The title joins the first chapter rather than being a part of its own.
    assert len(plan.parts) == 5 and plan.texts[0].startswith("# The Book")
    assert plan.describe() == "its level-2 headings"


def test_the_coarsest_usable_split_wins() -> None:
    # Issues separated by rules, each with headed sections: the issues are the documents.
    issues = "\n\n---\n\n".join(
        "\n\n".join(f"# Section {n}.{s}\n\n{_prose(300, n + s)}" for s in range(3))
        for n in range(4)
    )
    plan = plan_split(issues, WINDOW)
    assert plan is not None and (plan.kind, len(plan.parts)) == (RULE, 4)
    assert plan.describe() == "its rules"
    # Asked for headings, it splits at them.
    headed = plan_split(issues, WINDOW, "heading")
    assert headed is not None and (headed.kind, len(headed.parts)) == (HEADING, 12)
    assert plan_split(issues, WINDOW, "none") is None


def test_small_parts_join_their_neighbours_and_micro_splits_are_refused() -> None:
    text = "\n\n".join(
        [
            _prose(40),  # a preamble
            f"# One\n\n{_prose(400, 1)}",
            f"# Interlude\n\n{_prose(60, 2)}",
            f"# Two\n\n{_prose(400, 3)}",
            f"# Three\n\n{_prose(400, 4)}",
            f"# Coda\n\n{_prose(50, 5)}",
        ]
    )
    plan = plan_split(text, WINDOW)
    assert plan is not None
    # A merged part is named by the larger: the interlude joins Two, the coda Three.
    assert [part.title for part in plan.parts] == ["One", "Two", "Three"]
    assert all(part.words >= WINDOW / 2 for part in plan.parts)
    assert sum(len(part) for part in plan.texts) + len(plan.texts) - 1 == len(text)
    # Headings every short paragraph mark paragraphs, not pieces of writing: no split.
    dense = "\n\n".join(f"### Q{n}\n\n{_prose(80, n)}" for n in range(30))
    assert plan_split(dense, WINDOW) is None
    # Too few parts for the automatic split, enough when asked.
    two = f"# A\n\n{_prose(600)}\n\n# B\n\n{_prose(600, 4)}"
    assert plan_split(two, WINDOW) is None
    assert plan_split(two, WINDOW, "heading", min_parts=2) is not None


def test_html_articles_and_headings_split() -> None:
    posts = [_prose(400, n) for n in range(4)]

    def page(body: str) -> str:
        return f"<html><body><nav>Home</nav>{body}<footer>(c)</footer></body></html>"

    def paragraphs(text: str) -> str:
        return "".join(f"<p>{block}</p>" for block in text.split("\n\n"))

    articles = page("".join(f"<article>{paragraphs(post)}</article>" for post in posts))
    plan = plan_split(html_to_markdown(articles), WINDOW)
    assert plan is not None and (plan.kind, len(plan.parts)) == (RULE, 4)
    headed = page(
        "<main>"
        + "".join(f"<h2>Post {n}</h2>{paragraphs(p)}" for n, p in enumerate(posts))
        + "</main>"
    )
    plan = plan_split(html_to_markdown(headed), WINDOW)
    assert plan is not None and (plan.kind, plan.level) == (HEADING, 2)
    assert [part.title for part in plan.parts] == [f"Post {n}" for n in range(4)]


def test_part_ids_and_labels() -> None:
    assert part_id("book.md", 3, "Mud Season!") == "book.md#3-mud-season"
    assert part_id("book.md", 12) == "book.md#12"
    # A part is never read as a window (#w2) or a grouped record (#r2).
    assert part_id("x", 2, "W2") == "x#2-w2"
    assert document_label("book.md", "book.md#3-mud-season") == "book.md#3-mud-season"
    assert document_label("book (2).md", "book.md#3") == "book (2).md#3"
    assert document_label("posts/2024/a.md", "2024/a.md#1-intro") == "posts/2024/a.md#1-intro"
    assert document_label("stdin", "stdin#2") == "stdin#2"
    assert document_label("<text>", "text1#2") == "text1#2"


def test_stand_in_groups_are_even_and_consecutive() -> None:
    groups = stand_in_groups(20)
    assert {len(group) for group in groups} == {2, 3} and len(groups) == STAND_INS
    assert [index for group in groups for index in group] == list(range(20))
    assert [list(group) for group in stand_in_groups(3)] == [[0], [1], [2]]
    assert len(stand_in_groups(1000)) == STAND_INS


# Building from one file.


def test_one_file_of_essays_splits_into_its_essays_and_calibrates(writer_file: Path) -> None:
    split = sp.build(writer_file, SURFACE, contrast=DRAFTS)
    assert _coded(split.notes, sp.NoteCode.SPLIT) == [
        "split writer.md into 7 documents at its level-1 headings"
    ]
    assert split.report["document_count"] == 7 and "calibration" in split.report
    assert split.report["settings"]["split_on"] == "auto"
    assert split.report["settings"]["split_used"] == ["heading"]
    assert split.report["contrast"] is not None
    # The same essays as seven files give the same reference.
    files = sp.build(WRITER, SURFACE, contrast=DRAFTS)
    assert files.report["settings"]["split_used"] == []
    for key in ("summary", "calibration", "reliability", "contrast", "chunk_count"):
        assert split.report[key] == files.report[key], key  # type: ignore[literal-required]


def test_one_file_is_refused_a_contrast_only_when_splitting_is_off(writer_file: Path) -> None:
    with pytest.raises(sp.StyleProfileError) as error:
        sp.build(writer_file, sp.Settings(syntax=False, split_on="none"), contrast=DRAFTS)
    assert error.value.code == "contrast_needs_documents"


def test_a_file_without_headings_gets_stand_in_documents(bare_file: Path) -> None:
    profile = sp.build(bare_file, SURFACE, contrast=DRAFTS)
    [note] = _coded(profile.notes, sp.NoteCode.STAND_INS)
    windows = profile.report["chunk_count"]
    assert windows == STAND_INS and note == (
        "bare.md has no headings or rules that divide it into 3 or more parts of at least "
        "half a window, so each of its 8 windows stands in for a document"
    )
    assert profile.report["document_count"] == min(windows, STAND_INS) >= 7
    assert profile.report["settings"]["split_used"] == ["stand-in"]
    assert profile.report["settings"]["contrast_split_used"] == []
    assert "calibration" in profile.report and profile.report["contrast"] is not None
    # What that means is said once, in the report, so show repeats it; scores don't.
    [warning] = [w for w in profile.warnings if "stand-in" in w]
    assert warning == (
        "held-out calibration comes from 8 stand-in documents, consecutive parts of one file "
        "(bare.md): calibration from them is less sensitive, so short off-voice passages are "
        "caught less often; mark where pieces begin with headings or rules, or give them as "
        "separate files"
    )
    assert warning in profile.to_text()
    assert not any("stand-in" in w for w in profile.score(DRAFT).warnings)


def test_a_one_file_contrast_set_is_split_and_recorded(tmp_path: Path) -> None:
    drafts = [path.read_text(encoding="utf-8") for path in sorted(DRAFTS.glob("*.md"))]
    headed = tmp_path / "llm.md"
    headed.write_text("\n\n".join(drafts), encoding="utf-8")
    profile = sp.build(WRITER, SURFACE, contrast=headed)
    assert "split contrast llm.md into 5 documents at its level-1 headings" in _coded(
        profile.notes, sp.NoteCode.SPLIT
    )
    assert profile.report["settings"]["contrast_split_used"] == ["heading"]
    assert profile.report["settings"]["split_used"] == []
    bare = tmp_path / "bare-llm.md"
    bare.write_text("\n\n".join(map(_without_headings, drafts)), encoding="utf-8")
    profile = sp.build(WRITER, SURFACE, contrast=bare)
    assert profile.report["settings"]["contrast_split_used"] == ["stand-in"]
    [warning] = [w for w in profile.warnings if "stand-in" in w]
    assert warning.startswith("the contrast set's documents are") and "drafts" in warning
    assert "writer" not in warning


def test_too_few_headings_fall_back_to_stand_ins_with_an_accurate_note(tmp_path: Path) -> None:
    path = tmp_path / "two.md"
    path.write_text(
        f"# Half 1\n\n{_prose(2500)}\n\n# Half 2\n\n{_prose(2500, 5)}", encoding="utf-8"
    )
    [note] = _coded(sp.build(path, SURFACE).notes, sp.NoteCode.STAND_INS)
    assert note.startswith("two.md has no headings or rules that divide it into 3 or more parts")


def test_stand_ins_are_consecutive_windows(bare_file: Path) -> None:
    text = bare_file.read_text(encoding="utf-8") * 3  # about 27 windows
    bare_file.write_text(text, encoding="utf-8")
    profile = sp.build(bare_file, SURFACE, keep_chunks=True)
    [note] = _coded(profile.notes, sp.NoteCode.STAND_INS)
    windows = profile.report["chunk_count"]
    assert f"so its {windows} windows are grouped into 8 stand-in documents of consecutive " in note
    ids = [row["id"] for row in profile.report["chunks"]]  # type: ignore[typeddict-item]
    groups = [chunk_id.split("#")[1] for chunk_id in ids]
    assert groups == sorted(groups, key=int) and len(set(groups)) == STAND_INS
    assert ids[0] == "bare.md#1#w1" and profile.report["document_count"] == STAND_INS


def test_a_short_file_is_not_split(tmp_path: Path) -> None:
    path = tmp_path / "short.md"
    path.write_text(f"{_prose(300)}\n\n---\n\n{_prose(300, 2)}", encoding="utf-8")
    profile = sp.build(path, SURFACE)
    assert profile.report["document_count"] == 1 and profile.report["settings"]["split_used"] == []
    assert not _coded(profile.notes, sp.NoteCode.SPLIT)
    assert not _coded(profile.notes, sp.NoteCode.STAND_INS)


def _bench_medium(out: Path) -> Path:
    spec = importlib.util.spec_from_file_location("bench_gen", ROOT / "bench" / "gen.py")
    assert spec and spec.loader
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.generate("medium", out)


def test_a_bench_corpus_in_one_file_splits_into_its_documents(tmp_path: Path) -> None:
    corpus = _bench_medium(tmp_path / "corpus")
    files = sorted((corpus / "writer").glob("*.md"))[:50]
    one = tmp_path / "archive.md"
    one.write_text(
        "\n\n".join(path.read_text(encoding="utf-8") for path in files), encoding="utf-8"
    )
    profile = sp.build(one, SURFACE)
    assert _coded(profile.notes, sp.NoteCode.SPLIT) == [
        "split archive.md into 50 documents at its level-1 headings"
    ]
    assert profile.report["document_count"] == 50
    assert profile.report["summary"] == sp.build(files, SURFACE).report["summary"]


# What is split.


def _manuscripts(folder: Path, count: int, chapters: int = 8, words: int = 1400) -> Path:
    """``count`` manuscripts of ``chapters`` chapters under ``# Chapter N`` headings."""
    folder.mkdir()
    for n in range(count):
        body = "\n\n".join(
            f"# Chapter {c}\n\n{_prose(words, n * chapters + c)}" for c in range(1, chapters + 1)
        )
        (folder / f"book{n}.md").write_text(body, encoding="utf-8")
    return folder


def _calibrated(profile: sp.Profile) -> list[str]:
    lengths = profile.report.get("calibration", {}).get("by_length", {})
    return sorted((length for length, entry in lengths.items() if "delta" in entry), key=int)


def test_fewer_than_ten_documents_split_at_their_headings(tmp_path: Path) -> None:
    folder = _manuscripts(tmp_path / "books", 3)
    profile = sp.build(folder, SURFACE)
    assert _coded(profile.notes, sp.NoteCode.SPLIT) == [
        "split 3 texts into 24 documents at their headings, since 3 documents are too few "
        "to calibrate well"
    ]
    assert profile.report["document_count"] == 24
    assert _calibrated(profile) == ["75", "150", "300"]
    assert profile.report["settings"]["split_used"] == ["heading"]
    unsplit = sp.build(folder, sp.Settings(syntax=False, split_on="none"))
    assert unsplit.report["document_count"] == 3 and _calibrated(unsplit) == []
    # Never into stand-ins: a manuscript without headings among them stays whole.
    (folder / "bare.md").write_text(_prose(5000, 7), encoding="utf-8")
    mixed = sp.build(folder, SURFACE)
    assert mixed.report["document_count"] == 25 and "stand-in" not in str(mixed.report["settings"])


def test_manuscripts_that_calibrate_already_still_split_like_their_chapters(
    tmp_path: Path,
) -> None:
    folder = _manuscripts(tmp_path / "books", 5, chapters=6, words=2400)
    assert _calibrated(sp.build(folder, sp.Settings(syntax=False, split_on="none"))) == [
        "75",
        "150",
        "300",
    ]
    profile = sp.build(folder, SURFACE)
    assert profile.report["document_count"] == 30
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    for book in sorted(folder.glob("*.md")):
        for number, text in enumerate(book.read_text(encoding="utf-8").split("\n\n# ")):
            (chapters / f"{book.stem}-{number:02d}.md").write_text(
                text if number == 0 else "# " + text, encoding="utf-8"
            )
    files = sp.build(chapters, SURFACE)
    for key in ("summary", "calibration", "reliability"):
        assert profile.report[key] == files.report[key], key  # type: ignore[literal-required]


def test_ten_or_more_documents_are_never_split_automatically(tmp_path: Path) -> None:
    folder = _manuscripts(tmp_path / "posts", 10, chapters=4, words=300)
    profile = sp.build(folder, SURFACE)
    assert profile.report["document_count"] == 10
    assert not _coded(profile.notes, sp.NoteCode.SPLIT)
    # Asked for, each file is split at its headings.
    asked = sp.build(folder, sp.Settings(syntax=False, split_on="heading"))
    assert _coded(asked.notes, sp.NoteCode.SPLIT) == [
        "split 10 texts into 40 documents at their headings"
    ]
    assert asked.report["document_count"] == 40


def test_blog_posts_with_sections_stay_whole_while_manuscripts_split(tmp_path: Path) -> None:
    # Five posts of about 1,500 words in ~375-word ## sections: sections of one post share
    # its topic, so between the bands a part must hold a whole window, and they stay whole.
    posts = tmp_path / "posts"
    posts.mkdir()
    for n in range(5):
        sections = "\n\n".join(f"## Section {s}\n\n{_prose(375, n * 4 + s)}" for s in range(4))
        (posts / f"{n}.md").write_text(f"# Post {n}\n\n{sections}", encoding="utf-8")
    profile = sp.build(posts, SURFACE)
    assert profile.report["document_count"] == 5
    assert not _coded(profile.notes, sp.NoteCode.SPLIT)
    # Asked for, they split at their sections, which are over half a window.
    assert (
        sp.build(posts, sp.Settings(syntax=False, split_on="heading")).report["document_count"]
        == 20
    )
    # Three manuscripts' ~1,000-word chapters still split.
    books = _manuscripts(tmp_path / "books", 3, words=1000)
    assert sp.build(books, SURFACE).report["document_count"] == 24


def test_texts_whose_sections_are_short_stay_whole(tmp_path: Path) -> None:
    # Headings every short paragraph divide nothing: a few such posts are left as they are.
    folder = tmp_path / "posts"
    folder.mkdir()
    for n in range(3):
        sections = "\n\n".join(f"## Q{q}\n\n{_prose(80, n + q)}" for q in range(8))
        (folder / f"{n}.md").write_text(sections, encoding="utf-8")
    profile = sp.build(folder, SURFACE)
    assert profile.report["document_count"] == 3
    assert not _coded(profile.notes, sp.NoteCode.SPLIT)


def test_a_folder_of_one_file_and_stdin_split(
    tmp_path: Path, writer_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "one"
    folder.mkdir()
    writer_file.rename(folder / "all.md")
    assert sp.build(folder, SURFACE).report["document_count"] == 7
    text = (folder / "all.md").read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", type("In", (), {"buffer": _Bytes(text.encode())})())
    profile = sp.build("-", SURFACE)
    assert _coded(profile.notes, sp.NoteCode.SPLIT) == [
        "split stdin into 7 documents at its level-1 headings"
    ]
    library = sp.build(sp.Text(text, "book"), SURFACE)
    assert _coded(library.notes, sp.NoteCode.SPLIT) == [
        "split book into 7 documents at its level-1 headings"
    ]


class _Bytes:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


def test_jsonl_records_are_never_split(tmp_path: Path) -> None:
    text = "\n\n".join(_essays())
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({"id": "all", "text": text}) + "\n", encoding="utf-8")
    profile = sp.build(path, SURFACE)
    assert profile.report["document_count"] == 1 and profile.report["settings"]["split_used"] == []
    heading = sp.build(path, sp.Settings(syntax=False, split_on="heading"))
    assert heading.report["document_count"] == 1


def test_asking_for_a_split_that_is_not_there_is_noted(bare_file: Path) -> None:
    profile = sp.build(bare_file, sp.Settings(syntax=False, split_on="heading"))
    assert _coded(profile.notes, sp.NoteCode.SPLIT) == [
        "bare.md has no headings that split it into parts of at least half a window, so it "
        "is kept whole"
    ]
    assert profile.report["document_count"] == 1


def test_scene_breaks_do_not_beat_chapters() -> None:
    chapters = []
    for n in range(1, 13):
        body = _prose(700, n)
        if n in (3, 7, 10):
            body += f"\n\n* * *\n\n{_prose(400, n + 20)}"
        chapters.append(f"# Chapter {n}\n\n{body}")
    plan = plan_split("# The Long Valley\n\nby A. Writer\n\n" + "\n\n".join(chapters), WINDOW)
    assert plan is not None and (plan.kind, len(plan.parts)) == (HEADING, 12)
    # The title and byline join Chapter 1, which keeps its name.
    assert [part.title for part in plan.parts][:2] == ["Chapter 1", "Chapter 2"]
    # Rules win only where they sit before headings: a newsletter's issues.
    issues = "\n\n---\n\n".join(
        f"title: Issue {n}\ndate: 2024-01-0{n}\n\n# The week\n\n{_prose(300, n)}\n\n"
        f"# Reading\n\n{_prose(300, n + 1)}"
        for n in range(1, 6)
    )
    plan = plan_split(issues, WINDOW)
    assert plan is not None and plan.kind == RULE
    # A part after a rule is named by its title: line, or else its first heading.
    assert [part.title for part in plan.parts] == [f"Issue {n}" for n in range(1, 6)]
    plain = plan_split(issues.replace("title: ", "about: "), WINDOW)
    assert plain is not None and {part.title for part in plain.parts} == {"The week"}


def test_a_metadata_block_inside_a_text_is_not_a_heading() -> None:
    text = "Intro.\n\n---\ntitle: Issue 2\ndate: 2024-01-02\n---\n\n# The week\n"
    assert [(m.line, m.kind) for m in markers(text)] == [(2, RULE), (7, HEADING)]


def test_a_heading_level_can_be_asked_for(tmp_path: Path) -> None:
    book = "\n\n".join(
        f"# Part {p}\n\n"
        + "\n\n".join(f"## Chapter {p}.{c}\n\n{_prose(600, c)}" for c in range(1, 5))
        for p in ("I", "II", "III")
    )
    assert len(plan_split(book, WINDOW).parts) == 3  # type: ignore[union-attr]
    chapters = plan_split(book, WINDOW, "heading:2", 2)
    assert chapters is not None and len(chapters.parts) == 12
    assert (
        chapters.parts[0].title == "Chapter I.1" and chapters.describe() == "its level-2 headings"
    )
    path = tmp_path / "book.md"
    path.write_text(book, encoding="utf-8")
    result = sp.build(WRITER, SURFACE).score(path, split_on="heading:2")
    assert len(result.documents) == 12
    assert result.documents[4].name == "book.md#5-chapter-ii-1"
    # A level deeper than the book has cuts at its deepest, and says so.
    deepest = plan_split(book, WINDOW, "heading:3", 2)
    assert deepest is not None and deepest.describe() == "its level-2 headings"
    assert len(deepest.parts) == 12
    kept = sp.build(WRITER, SURFACE).score(DRAFT, split_on="heading:3")
    assert _coded(kept.notes, sp.NoteCode.SPLIT) == [
        "draft.md has no level-3 headings that split it into parts of at least half a "
        "window, so it is kept whole"
    ]


def test_plain_text_chapter_lines_are_headings(tmp_path: Path) -> None:
    labels = ["Chapter 1", "CHAPTER II", "Chapter Three", "Chapter 4: The End"]
    text = "\n\n".join(f"{label}\n\n{_prose(600, n)}" for n, label in enumerate(labels))
    found = markers(text, plain=True)
    assert [(m.kind, m.level, m.title) for m in found] == [(HEADING, 2, t) for t in labels]
    assert markers(text) == []  # only in plain text
    # Only alone between blank lines: not a sentence that starts with the word.
    assert markers("Chapter 12 of the report says so.\n", plain=True) == []
    assert markers("Text.\nChapter 3\nMore text.\n", plain=True) == []
    assert [m.level for m in markers("Part One\n\nBook II\n", plain=True)] == [1, 1]
    # Roman numerals only when well formed: "Part mild" is a sentence fragment.
    # ("Book mix" is MIX, 1009, and stays a heading.)
    for line in ("Part mild", "Part did", "Chapter IIII", "Part VX"):
        assert markers(f"Text.\n\n{line}\n\nMore.\n", plain=True) == [], line
    for line in ("Chapter XIV", "PART MCMXC", "Book iv", "Chapter xl"):
        assert len(markers(f"Text.\n\n{line}\n\nMore.\n", plain=True)) == 1, line
    path = tmp_path / "novel.txt"
    path.write_text(text, encoding="utf-8")
    assert _coded(sp.build(path, SURFACE).notes, sp.NoteCode.SPLIT) == [
        "split novel.txt into 4 documents at its level-2 headings"
    ]


def test_parts_that_repeat_another_are_dropped(tmp_path: Path) -> None:
    issue = f"# Issue A\n\n{_prose(600)}"
    path = tmp_path / "dups.md"
    path.write_text(
        "\n\n".join(
            [issue, f"# Issue B\n\n{_prose(600, 4)}", issue, f"# Issue C\n\n{_prose(600, 8)}"]
        ),
        encoding="utf-8",
    )
    profile = sp.build(path, SURFACE)
    assert profile.report["document_count"] == 3
    [note] = _coded(profile.notes, sp.NoteCode.DUPLICATES)
    assert "dups.md#3-issue-a repeats dups.md#1-issue-a" in note


def test_the_split_suggestion_is_never_circular(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "preface.md"
    chapters = "\n\n".join(f"# Chapter {n}\n\n{_prose(300, n)}" for n in range(1, 8))
    path.write_text(f"# Preface\n\n{_prose(6000)}\n\n{chapters}", encoding="utf-8")
    thin = _coded(sp.build(path, SURFACE).notes, sp.NoteCode.THIN_REFERENCE)
    held = [message for message in thin if message.startswith("one document holds")]
    assert held and not any("split" in message for message in held)
    # A whole file among others is told how, with each flag written out.
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "big.md").write_text(f"{_prose(6000)}", encoding="utf-8")
    for n in range(2):
        (folder / f"{n}.md").write_text(_prose(300, n), encoding="utf-8")
    code = main(["build", str(folder), "-o", str(tmp_path / "p.json"), "--no-syntax"])
    assert code == 0
    # Wrapped lines may break at a hyphen, so compare without whitespace.
    shown = "".join(capsys.readouterr().err.split())
    assert "orsplitthatone(--split-onheadingor--split-onrulesplits" in shown


def test_split_on_is_a_setting() -> None:
    with pytest.raises(sp.StyleProfileError) as error:
        sp.Settings(split_on="chapter")  # type: ignore[arg-type]
    assert error.value.setting == "split_on"
    settings = sp.Settings(split_on="rule")
    assert sp.Settings.from_report(settings.to_report()) == settings


# Scoring and the command line.


def test_scoring_splits_only_when_asked(writer_file: Path) -> None:
    profile = sp.build(WRITER, SURFACE, contrast=DRAFTS)
    whole = profile.score(writer_file)
    assert [document.name for document in whole.documents] == ["writer.md"]
    assert whole.report["settings"]["split_on"] == "none"
    assert profile.score(writer_file, split_on="auto").documents == whole.documents
    chapters = profile.score(writer_file, split_on="heading")
    assert [document.name for document in chapters.documents][:2] == [
        "writer.md#1-fence-lines",
        "writer.md#2-mud-season",
    ]
    assert len(chapters.documents) == 7 and chapters.report["settings"]["split_used"] == ["heading"]
    assert _coded(chapters.notes, sp.NoteCode.SPLIT) == [
        "split writer.md into 7 documents at its level-1 headings"
    ]


def test_cli_builds_one_file_with_a_contrast_and_scores_its_chapters(
    writer_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "writer.json"
    assert (
        main(["build", str(writer_file), "--contrast", str(DRAFTS), "-o", str(out), "--no-syntax"])
        == 0
    )
    captured = capsys.readouterr()
    assert "note: split writer.md into 7 documents at its level-1 headings" in captured.err
    assert "(7 documents)" in captured.out
    assert main(["score", "-q", str(writer_file), str(out), "--split-on", "heading"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 and all(str(writer_file) + "#" in line for line in lines), lines
    code = main(
        ["build", str(writer_file), "--split-on", "none", "--contrast", str(DRAFTS), "-o", str(out)]
    )
    assert code == 1 and "contrast set needs" in capsys.readouterr().err


def test_evaluate_splits_the_writers_file(writer_file: Path, tmp_path: Path) -> None:
    edited = tmp_path / "edited"
    edited.mkdir()
    for path in sorted(DRAFTS.glob("*.md"))[:2]:
        (edited / path.name).write_text(path.read_text(encoding="utf-8").lower(), encoding="utf-8")
    result = sp.evaluate(writer_file, DRAFTS, {"lower": edited}, SURFACE)
    assert _coded(result.notes, sp.NoteCode.SPLIT) == [
        "split writer.md into 7 documents at its level-1 headings"
    ]
    assert result.report["settings"]["split_used"] == ["heading"]


def test_chunks_keep_their_documents_through_windowing(writer_file: Path) -> None:
    from styleprofile.api import _chunked  # pyright: ignore[reportPrivateUsage]
    from styleprofile.profile import load_chunks

    notes: list[sp.Note] = []
    cut = _chunked(load_chunks([str(writer_file)]), SURFACE, False, notes, split="calibrate")
    assert len({chunk_document(chunk) for chunk in cut.windows}) == 7
    assert cut.split == ("heading",)
