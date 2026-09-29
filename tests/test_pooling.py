"""Documents and pooling (plan PR 10): which texts make up one document, how short texts
are joined into windows without crossing documents, and the ids that name them."""

from __future__ import annotations

import importlib
import io
import json
import random
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import styleprofile as sp
from styleprofile import api
from styleprofile.calibration import (
    MIN_JUDGED_WORDS,
    chunk_level,
    cut,
    plan_pieces,
    single_parts,
    whole_parts,
)
from styleprofile.cli import main
from styleprofile.profile import (
    Chunk,
    _pack,  # pyright: ignore[reportPrivateUsage]
    base_id,
    chunk_document,
    drop_duplicates,
    group_id,
    load_chunks,
    pair_key,
    pool,
    pools_by_default,
    window,
)
from styleprofile.surface import words as prose_words

ROOT = Path(__file__).resolve().parent.parent
THREADS = ("t1", "t2", "t3", "t4")
# The comments below have a median of about 50 words, under a quarter of this.
WINDOW = 300
SETTINGS = sp.Settings(window_words=WINDOW, syntax=False)
GROUPED = sp.Settings(window_words=WINDOW, syntax=False, group_field="thread")


def _sentences(folder: Path) -> list[str]:
    """The prose sentences of the sample essays, in order."""
    paragraphs = [
        paragraph.replace("\n", " ")
        for path in sorted(folder.glob("*.md"))
        for paragraph in path.read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip() and not paragraph.startswith(("#", "---"))
    ]
    return [s for p in paragraphs for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]


def _texts(folder: Path, count: int, words: int) -> list[str]:
    """``count`` texts of at least ``words`` words, from consecutive sentences."""
    sentences = _sentences(folder)
    texts: list[str] = []
    position = 0
    while len(texts) < count:
        taken: list[str] = []
        while len(" ".join(taken).split()) < words:
            taken.append(sentences[position % len(sentences)])
            position += 1
        texts.append(" ".join(taken))
    return texts


def _write(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _comments(path: Path, count: int = 144, words: int = 40, **extra: Any) -> Path:
    """Short comments in four threads, interleaved, each signed with its thread's marker."""
    texts = _texts(ROOT / "examples/writer", count, words)
    records = [
        {
            "id": f"c{index:03d}",
            "thread": THREADS[index % len(THREADS)],
            "text": f"{text} Signed {THREADS[index % len(THREADS)]}mark.",
            **extra,
        }
        for index, text in enumerate(texts)
    ]
    return _write(path, records)


def _drafts(path: Path, count: int = 24, **extra: Any) -> tuple[Path, list[dict[str, Any]]]:
    """Short LLM-style drafts: the sample drafts' sentences, commas turned to em dashes."""
    texts = _texts(ROOT / "examples/llm-drafts", count, 40)
    records = [
        {"id": f"d{index:02d}", **extra, "text": text.replace(", ", " — ")}
        for index, text in enumerate(texts)
    ]
    return _write(path, records), records


def _bench_comments() -> Callable[[random.Random, int], str]:
    """The comments generator of ``bench/gen.py``: short texts in the sample writer's voice."""
    sys.path.insert(0, str(ROOT / "bench"))
    try:
        gen: Any = importlib.import_module("gen")
    finally:
        sys.path.remove(str(ROOT / "bench"))
    pool = gen._pool(ROOT / "examples/writer")  # pyright: ignore[reportPrivateUsage]
    return lambda rng, words: gen._comment(rng, pool, words)  # pyright: ignore[reportPrivateUsage]


def _markers(text: str) -> set[str]:
    return set(re.findall(r"Signed (\w+)mark", text))


def _flat(length: int, word: str = "word") -> str:
    return " ".join([word] * length) + "."


def _coded(notes: Sequence[sp.Note], code: sp.NoteCode) -> list[str]:
    return [note.message for note in notes if note.code == code]


def _pooled(notes: Sequence[sp.Note]) -> list[str]:
    return _coded(notes, sp.NoteCode.POOLED)


# What a document is.


def test_grouped_records_are_one_document_per_group(tmp_path: Path) -> None:
    path = _comments(tmp_path / "comments.jsonl")
    records = load_chunks([str(path)], group_field="thread")
    assert records[0].id == "thread=t1#r1" and records[1].id == "thread=t2#r2"
    assert len({chunk_document(chunk) for chunk in records}) == len(THREADS)
    # Windowing keeps each record in its group's document.
    assert {chunk_document(chunk) for chunk in window(records, WINDOW)} == {
        chunk_document(chunk) for chunk in records
    }

    profile = sp.build(path, GROUPED, keep_chunks=True)
    rows = profile.report["chunks"]
    assert {base_id(row["id"]) for row in rows} == {f"thread={name}" for name in THREADS}
    assert {row["source"] for row in rows} == {"comments.jsonl"}
    assert profile.report["document_count"] == len(THREADS)
    assert profile.report["calibration"]["sources"] == len(THREADS)
    assert _pooled(profile.notes) == [
        "joined 144 records into 24 windows of about 300 words, since the median record is "
        "under a quarter of a window, never across thread groups"
    ]
    assert profile.report["settings"]["group_field"] == "thread"
    assert profile.report["settings"]["pool_used"] is True


def test_group_windows_never_cross_groups(tmp_path: Path) -> None:
    """Interleaved records join their own group's records only, all of them, in order."""
    path = _comments(tmp_path / "comments.jsonl")
    records = load_chunks([str(path)], group_field="thread")
    windows = pool(records, WINDOW).windows
    for chunk in windows:
        (thread,) = _markers(chunk.text)
        assert re.fullmatch(rf"thread={thread}#w\d+", chunk.id)
    assert [chunk.id for chunk in windows[:7]] == [
        *(f"thread=t1#w{number}" for number in range(1, 7)),
        "thread=t2#w1",
    ]
    t1 = [chunk.text for chunk in windows if chunk.id.startswith("thread=t1#")]
    assert "\n\n".join(t1) == "\n\n".join(chunk.text for chunk in records[::4])


def test_a_group_spans_the_files_of_one_input(tmp_path: Path) -> None:
    """A thread split across monthly exports in one folder is one document; the same values
    in two inputs typed separately are separate documents."""
    folder = tmp_path / "exports"
    first = _comments(folder / "2024-01.jsonl")
    second = _comments(folder / "2024-02.jsonl")
    one_input = pool(load_chunks([str(folder)], group_field="thread"), WINDOW).windows
    assert len({chunk_document(chunk) for chunk in one_input}) == len(THREADS)
    assert all(len(_markers(chunk.text)) == 1 for chunk in one_input)
    two_inputs = pool(load_chunks([str(first), str(second)], group_field="thread"), WINDOW)
    assert len({chunk_document(chunk) for chunk in two_inputs.windows}) == 2 * len(THREADS)


def test_documents_are_never_read_back_from_ids(tmp_path: Path) -> None:
    """Group values and record ids that look like id suffixes, or like the missing-value
    group, are documents of their own."""
    text = _flat(30)
    values = ["ada", "ada#r3", "ada#w1", "a=b", "(none)", None, "  ", 1, "1"]
    path = _write(tmp_path / "g.jsonl", [{"thread": value, "text": text} for value in values])
    chunks = load_chunks([str(path)], group_field="thread")
    documents = [chunk_document(chunk) for chunk in chunks]
    # ada, ada#r3, ada#w1, a=b and "(none)" apart; None and blank together; 1 and "1" one.
    assert len(set(documents)) == 7
    assert documents[5] == documents[6] and documents[7] == documents[8]
    assert documents[4] != documents[5]
    assert chunks[5].id == chunks[6].id.replace("#r7", "#r6") == "thread=(none)#r6"

    ids = ["post#r1", "post#r2", "reply#w1", "reply#w2"]
    path = _write(tmp_path / "u.jsonl", [{"id": i, "text": text} for i in ids])
    chunks = load_chunks([str(path)])
    assert len({chunk_document(chunk) for chunk in chunks}) == 4
    # Windows keep their text's document, whatever its id says.
    assert [chunk_document(w) for w in window(chunks, WINDOW)] == list(map(chunk_document, chunks))


def test_ungrouped_windows_are_documents_and_never_cross_files(tmp_path: Path) -> None:
    folder = tmp_path / "comments"
    _comments(folder / "a.jsonl", count=12)
    _comments(folder / "b.jsonl", count=12)
    chunks = load_chunks([str(folder)])
    joined = pool(chunks, WINDOW)
    by_file: dict[str, list[str]] = {}
    for chunk in joined.windows:
        by_file.setdefault(Path(chunk.source).name, []).append(chunk.id)
    # Each file's windows cover its records, c000 to c011, in consecutive ranges, named
    # with their file, since ids restart in each file of a folder.
    assert by_file == {
        "a.jsonl": ["a.jsonl:c000..c005", "a.jsonl:c006..c011"],
        "b.jsonl": ["b.jsonl:c000..c005", "b.jsonl:c006..c011"],
    }
    assert len({chunk.id for chunk in joined.windows}) == 4
    assert len({chunk_document(chunk) for chunk in joined.windows}) == 4
    assert joined.together == 24
    assert joined.joined["a.jsonl:c004"] == "a.jsonl:c000..c005"
    # A file given directly names its records by id alone.
    alone = load_chunks([str(folder / "a.jsonl")])
    assert [chunk.id for chunk in alone[:2]] == ["c000", "c001"]
    texts = "\n\n".join(chunk.text for chunk in joined.windows[:2])
    assert texts == "\n\n".join(chunk.text for chunk in chunks[:12])
    # Files are told apart by their real path, not by the name a report saves for them.
    twins = [Chunk(f"t{i}", "comments.jsonl", _flat(20), f"/{i}/comments.jsonl") for i in (1, 2)]
    assert [chunk.id for chunk in pool(twins, WINDOW).windows] == ["t1#w1", "t2#w1"]


def test_long_texts_are_windowed_alone_and_keep_their_document() -> None:
    short = [Chunk(f"s{index}", "f.jsonl", _flat(20)) for index in range(6)]
    long = Chunk("long", "f.jsonl", "\n\n".join(_flat(60) for _ in range(5)))
    joined = pool([*short[:3], long, *short[3:]], 100)
    assert [chunk.id for chunk in joined.windows] == [
        "s0..s2",
        "long#w1",
        "long#w2",
        "long#w3",
        "s3..s5",
    ]
    assert {chunk_document(chunk) for chunk in joined.windows[1:4]} == {chunk_document(long)}
    # A lone short text has nothing to join: it is windowed as window() would.
    alone = pool([Chunk("only", "f.jsonl", _flat(10))], 100)
    assert [chunk.id for chunk in alone.windows] == ["only#w1"] and alone.together == 0


def test_pack_is_the_one_size_rule() -> None:
    """``_pack``, shared by window() and pool(): close at window_words, close early rather
    than pass one and a half windows, merge a short tail, and keep glued items together."""
    assert _pack([60, 60, 60, 60], 100, [False] * 4) == [[0, 1], [2, 3]]
    assert _pack([60, 100], 100, [False] * 2) == [[0], [1]]
    assert _pack([120, 20], 100, [False] * 2) == [[0, 1]]
    assert _pack([60, 60, 60], 100, [False, False, True]) == [[0, 1, 2]]
    assert _pack([], 100, []) == [[]]


def test_windows_are_about_window_words() -> None:
    """Joined windows follow window()'s size rule."""
    chunks = [Chunk(f"t{index}", "f.jsonl", _flat(30)) for index in range(11)]
    assert [len(chunk.text.split()) for chunk in pool(chunks, 100).windows] == [120, 120, 90]
    chunks = [Chunk(f"t{index}", "f.jsonl", _flat(45)) for index in range(5)]
    assert [len(chunk.text.split()) for chunk in pool(chunks, 100).windows] == [135, 90]
    chunks = [Chunk(f"t{index}", "f.jsonl", _flat(40)) for index in range(7)]
    assert [len(chunk.text.split()) for chunk in pool(chunks, 100).windows] == [120, 160]


def test_missing_groups_form_one_documented_bucket(tmp_path: Path) -> None:
    path = _comments(tmp_path / "comments.jsonl")
    records = _read(path)
    for record in records[:4]:
        del record["thread"]
    records[4]["thread"] = None
    records[5]["thread"] = ""
    _write(path, records)
    chunks = load_chunks([str(path)], group_field="thread")
    assert group_id("thread", None) == "thread=(none)"
    assert sum(chunk.id.startswith("thread=(none)#") for chunk in chunks) == 6

    profile = sp.build(path, GROUPED)
    assert _coded(profile.notes, sp.NoteCode.MISSING_GROUP) == [
        f"{path}: 6 of 144 records have no thread, so they are read as one document, thread=(none)"
    ]
    assert profile.report["calibration"]["sources"] == len(THREADS) + 1


def test_a_group_field_no_record_has_is_an_error_naming_the_fields(tmp_path: Path) -> None:
    path = _comments(tmp_path / "comments.jsonl", author="ada")
    with pytest.raises(sp.StyleProfileError) as error:
        sp.build(path, sp.Settings(window_words=WINDOW, syntax=False, group_field="thred"))
    assert error.value.code == "group_field"
    assert str(error.value) == (
        f"{path}: no record has a value in the group field 'thred'; its records have the "
        "fields author, id, thread"
    )
    nested = _comments(tmp_path / "nested.jsonl", topic={"id": 3})
    with pytest.raises(sp.StyleProfileError, match="holds a dict") as error:
        load_chunks([str(nested)], group_field="topic")
    assert error.value.code == "group_field"


# Duplicates are dropped before pooling.


def test_duplicate_records_are_dropped_before_pooling(tmp_path: Path) -> None:
    path = _comments(tmp_path / "comments.jsonl")
    records = _read(path)
    copies = [{**record, "id": f"copy{index}"} for index, record in enumerate(records[:4])]
    _write(path, [*records, *copies])
    profile = sp.build(path, SETTINGS)
    (duplicates,) = _coded(profile.notes, sp.NoteCode.DUPLICATES)
    assert duplicates.startswith("dropped 4 documents that repeat another")
    assert _pooled(profile.notes)[0].startswith("joined 144 records into 23 windows")


def test_grouped_records_are_deduplicated_one_by_one(tmp_path: Path) -> None:
    """With a group field a document is a whole group, but a comment posted twice, in one
    group or across two, is still a duplicate; short records never are."""
    path = _comments(tmp_path / "comments.jsonl")
    records = _read(path)
    reposted = [
        {**records[0], "id": "again0"},  # t1 again, in its own thread
        {**records[1], "id": "again1", "thread": "t3"},  # t2's comment, reposted in t3
        {"id": "ok0", "thread": "t1", "text": "Thanks, that helps a lot."},
        {"id": "ok1", "thread": "t2", "text": "Thanks, that helps a lot."},
    ]
    _write(path, [*records, *reposted])
    profile = sp.build(path, GROUPED)
    assert _coded(profile.notes, sp.NoteCode.DUPLICATES) == [
        "dropped 2 records that repeat another word for word, keeping the first copy (for "
        "example, line 145 (thread=t1) in comments.jsonl repeats line 1 (thread=t1) in "
        "comments.jsonl)"
    ]
    # The two short thank-yous (under 20 words) are both kept.
    assert _pooled(profile.notes)[0].startswith("joined 146 records into")
    # Files are still compared whole.
    folder = tmp_path / "posts"
    folder.mkdir()
    for name in ("a.md", "b.md"):
        (folder / name).write_text(records[0]["text"], encoding="utf-8")
    kept, dropped = drop_duplicates(load_chunks([str(folder)]))
    assert [chunk.id for chunk in kept] == ["a.md"] and dropped is not None


# When to pool.


def test_a_folder_of_short_files_is_pooled(tmp_path: Path) -> None:
    """One-tweet files in a folder join into windows, in sorted order, like records."""
    folder = tmp_path / "tweets"
    folder.mkdir()
    texts = _texts(ROOT / "examples/writer", 300, 30)
    for index, text in enumerate(texts):
        (folder / f"t{index:03d}.md").write_text(f"{text} Number {index}.", encoding="utf-8")
    profile = sp.build(folder, SETTINGS)
    (message,) = _pooled(profile.notes)
    made = re.match(
        r"joined 300 files into (\d+) windows of about 300 words, since the median file is "
        r"under a quarter of a window\. With no group_field, each window counts",
        message,
    )
    assert made and int(made[1]) >= 30
    assert profile.report["document_count"] == int(made[1])
    assert not any("fewer than 150 words" in warning for warning in profile.warnings)

    joined = pool(load_chunks([str(folder)]), WINDOW)
    assert joined.windows[0].id.startswith("t000.md..")
    assert joined.windows[-1].id.endswith("..t299.md")
    # Never across typed inputs: two folders, or files named one by one.
    other = tmp_path / "more"
    other.mkdir()
    (other / "u.md").write_text(texts[0], encoding="utf-8")
    both = pool(load_chunks([str(folder), str(other)]), WINDOW).windows
    assert both[-1].id == "u.md#w1"
    named = [str(folder / "t000.md"), str(folder / "t001.md")]
    assert [chunk.id for chunk in pool(load_chunks(named), WINDOW).windows] == [
        "t000.md#w1",
        "t001.md#w1",
    ]


def test_auto_leaves_a_handful_of_short_texts_alone_and_says_so(tmp_path: Path) -> None:
    """Pooling eight short files would leave one window and no held-out calibration, so
    "auto" keeps them apart, with a note; asking for it pools anyway."""
    folder = tmp_path / "posts"
    folder.mkdir()
    for index, text in enumerate(_texts(ROOT / "examples/writer", 8, 40)):
        (folder / f"p{index}.md").write_text(text, encoding="utf-8")
    alone = sp.build(folder, SETTINGS)
    assert alone.report["chunk_count"] == 8
    (note,) = [note for note in alone.notes if note.code == sp.NoteCode.POOLED]
    assert note.setting == "pool"
    assert note.message == (
        "the median file is under a quarter of a window, but joining them would make only "
        "1 window, too few to calibrate, so each is kept on its own; use pool to join them "
        "anyway"
    )
    forced = sp.build(folder, sp.Settings(window_words=WINDOW, syntax=False, pool=True))
    assert _pooled(forced.notes)[0].startswith("joined 8 files")
    assert forced.report["chunk_count"] < 8


@pytest.mark.parametrize(("words", "pools"), [(74, True), (75, False)])
def test_auto_pools_below_a_quarter_window(words: int, pools: bool, tmp_path: Path) -> None:
    """The median text, in prose words, against a quarter of window_words."""
    chunks = [Chunk(f"t{index}", "f.jsonl", _flat(words)) for index in range(9)]
    assert pools_by_default(chunks, WINDOW) is pools
    # The median decides: a few long texts do not stop the pooling.
    mixed = [*chunks, *(Chunk(f"l{index}", "f.jsonl", _flat(900)) for index in range(4))]
    assert pools_by_default(mixed, WINDOW) is pools
    assert not pools_by_default(chunks, 0) and not pools_by_default([], WINDOW)

    # Texts of exactly ``words`` prose words, cut from the sample essays; each ends in its
    # own word, since the essays repeat after a while and copies would be dropped.
    essays = _texts(ROOT / "examples/writer", 90, 90)
    texts = [
        " ".join([*prose_words(text)[: words - 1], f"unique{index}"]) + "."
        for index, text in enumerate(essays)
    ]
    assert {len(prose_words(text)) for text in texts} == {words}
    path = _write(tmp_path / "c.jsonl", [{"id": f"r{i}", "text": t} for i, t in enumerate(texts)])
    profile = sp.build(path, SETTINGS)
    assert profile.report["settings"]["pool_used"] is pools
    assert bool(_pooled(profile.notes)) is pools


def test_pool_setting_overrides_auto(tmp_path: Path) -> None:
    path = _comments(tmp_path / "comments.jsonl")
    unpooled = sp.build(path, sp.Settings(window_words=WINDOW, syntax=False, pool=False))
    assert unpooled.report["chunk_count"] == 144
    assert unpooled.report["settings"]["pool_used"] is False
    assert any("fewer than 150 words" in warning for warning in unpooled.warnings)

    pooled = sp.build(path, sp.Settings(window_words=WINDOW, syntax=False, pool=True))
    assert pooled.report["chunk_count"] == 23
    assert not any("fewer than 150 words" in warning for warning in pooled.warnings)
    (note,) = [note for note in pooled.notes if note.code == sp.NoteCode.POOLED]
    assert note.setting == "group_field"
    assert note.message == (
        "joined 144 records into 23 windows of about 300 words. With no group_field, each "
        "window counts as one document for held-out calibration: if the records come from "
        "different threads or authors, and especially if those are interleaved, its ranges "
        "are too narrow and verdicts on new text can be far too harsh. These records have "
        "thread (4 values): pass group_field thread if each value is a separate source"
    )
    # Long texts have nothing to join, so pooling them changes nothing and says nothing.
    essays = sp.build(ROOT / "examples/writer", sp.Settings(syntax=False, pool=True))
    assert essays.report["settings"]["pool_used"] is False
    assert not _pooled(essays.notes)
    # Without windows, "auto" never pools, and asking to pool fails before any reading.
    assert sp.build(path, sp.Settings(window_words=0, syntax=False)).report["chunk_count"] == 144
    impossible = sp.Settings(window_words=0, syntax=False, pool=True)
    with pytest.raises(sp.StyleProfileError, match="window_words must be above 0") as error:
        sp.build(tmp_path / "not-there.jsonl", impossible)
    assert (error.value.code, error.value.setting) == ("invalid_setting", "window_words")
    with pytest.raises(sp.StyleProfileError, match="pool must be"):
        sp.Settings(pool="yes")  # pyright: ignore[reportArgumentType]


def test_the_note_suggests_fields_that_look_like_sources(tmp_path: Path) -> None:
    """Fields with 2 or more values, and at most a third as many as records, are named;
    ids, the text and one-value fields are not."""
    path = _comments(tmp_path / "c.jsonl", lang="en")
    records = _read(path)
    for index, record in enumerate(records):
        record["conversation_id"] = f"conv{index % 12}"
    _write(path, records)
    (message,) = _pooled(sp.build(path, SETTINGS).notes)
    assert message.endswith(
        "These records have thread (4 values), conversation_id (12 values): pass group_field "
        "thread if each value is a separate source"
    )


def test_stdin_records_are_called_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _comments(tmp_path / "c.jsonl").read_bytes()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    (message,) = _pooled(sp.build("-", SETTINGS).notes)
    assert message.startswith("joined 144 records into 23 windows")


def test_contrast_pools_with_the_writer(tmp_path: Path) -> None:
    writer = _comments(tmp_path / "writer.jsonl")
    contrast, _ = _drafts(tmp_path / "drafts.jsonl")
    profile = sp.build(writer, SETTINGS, contrast=contrast)
    made = re.fullmatch(
        r"joined 24 contrast records into (\d) windows of about 300 words",
        _pooled(profile.notes)[1],
    )
    assert made
    assert profile.report["contrast"]["sources"] == int(made[1])


# Group fields that cannot work are named.


def test_one_group_is_a_thin_reference_that_names_the_field(tmp_path: Path) -> None:
    path = _comments(tmp_path / "c.jsonl")
    _write(path, [{**record, "thread": "all"} for record in _read(path)])
    profile = sp.build(path, GROUPED)
    thin = [note for note in profile.notes if note.code == sp.NoteCode.THIN_REFERENCE]
    assert thin[0] == sp.Note(
        "it comes from 1 document, so it has no held-out calibration and cannot learn a "
        "contrast: every record has the same thread; leave out group_field, or group by a "
        "finer field",
        sp.NoteCode.THIN_REFERENCE,
        setting="group_field",
    )


def test_single_record_groups_say_to_coarsen_the_field(tmp_path: Path) -> None:
    path = _comments(tmp_path / "c.jsonl")
    _write(path, [{**r, "thread": f"x{i}"} for i, r in enumerate(_read(path))])
    profile = sp.build(path, GROUPED)
    (note,) = _coded(profile.notes, sp.NoteCode.GROUPING)
    assert re.fullmatch(
        r"most thread groups are short \(a median of \d+ words\), so pooling has little to "
        r"join and their records are measured nearly alone; group by a coarser field, or "
        r"leave out group_field",
        note,
    )
    assert not _pooled(profile.notes)


def test_one_giant_group_is_a_thin_reference(tmp_path: Path) -> None:
    """A giant group's held-out range rests on the tiny group's windows: fewer than the
    pieces and documents a length's range needs (``calibration``)."""
    path = _comments(tmp_path / "c.jsonl")
    records = _read(path)
    _write(path, [{**r, "thread": "big" if i >= 6 else "small"} for i, r in enumerate(records)])
    profile = sp.build(path, GROUPED)
    thin = _coded(profile.notes, sp.NoteCode.THIN_REFERENCE)
    assert any(
        re.fullmatch(
            r"one document holds 2\d of its 2\d chunks, so its held-out range rests on the "
            r"other 1 chunk, where a range needs 20 from 3 or more documents; add documents "
            r"of a similar size, or split that one",
            message,
        )
        for message in thin
    ), thin
    # Ordinary groups are not flagged.
    assert not any(
        "one document holds" in m
        for m in _coded(
            sp.build(_comments(tmp_path / "even.jsonl"), GROUPED).notes, sp.NoteCode.THIN_REFERENCE
        )
    )


def test_lengths_are_calibrated_on_pooled_windows_by_document(tmp_path: Path) -> None:
    """Length calibration cuts its pieces from the pooled windows and holds out whole
    documents: groups with a group field, windows without one."""
    path = _comments(tmp_path / "c.jsonl")
    grouped = sp.build(path, GROUPED).report["calibration"]["by_length"]
    assert {entry["documents"] for entry in grouped.values()} == {len(THREADS)}
    ungrouped = sp.build(path, SETTINGS).report["calibration"]["by_length"]
    assert all(entry["documents"] >= 20 for entry in ungrouped.values())


# Scoring many short texts.


@dataclass(frozen=True)
class _OwnComments:
    """A pooled reference built from the writer's comments, and more of the writer's
    comments it has not seen."""

    reference: sp.Profile
    held_out: list[str]
    folder: Path


@pytest.fixture(scope="module")
def own_comments(tmp_path_factory: pytest.TempPathFactory) -> _OwnComments:
    folder = tmp_path_factory.mktemp("own")
    comment = _bench_comments()
    rng = random.Random(5)
    train = [comment(rng, 50) for _ in range(3000)]
    seen = set(train)
    held_out: list[str] = []
    while len(held_out) < 600:
        text = comment(rng, 50)
        if text not in seen:
            seen.add(text)
            held_out.append(text)
    writer = _write(
        folder / "writer.jsonl", [{"id": f"w{i}", "text": t} for i, t in enumerate(train)]
    )
    reference = sp.build(writer, sp.Settings(syntax=False))
    assert reference.report["settings"]["pool_used"] is True
    return _OwnComments(reference, held_out, folder)


def _records(folder: Path, name: str, texts: Sequence[str]) -> Path:
    return _write(folder / name, [{"id": f"h{i}", "text": t} for i, t in enumerate(texts)])


def test_the_writers_own_short_records_are_judged_fairly_against_a_pooled_reference(
    own_comments: _OwnComments,
) -> None:
    """Plan PR 10's acceptance for scoring (review finding B1): against a pooled reference,
    the writer's own held-out records scored one by one either abstain or are judged at
    their own length, and at most 5% read "clearly different" or worse; pooled, the batch
    reads close."""
    reference = own_comments.reference
    drafts = _records(own_comments.folder, "held.jsonl", own_comments.held_out[:300])
    alone = reference.score(drafts)
    assert alone.chunk_count == 300
    levels = [chunk_level(row) for row in alone.report["chunks"]]
    judged = [level for level in levels if level is not None]
    for row, level in zip(alone.report["chunks"], levels, strict=True):
        words = row["metrics"]["size"]["words"]
        assert (level is None) is (not row["reference"]["calibration"]["judged"])
        if words < MIN_JUDGED_WORDS:
            assert level is None  # too short to judge
    assert len(judged) >= 20
    assert sum(level >= 2 for level in judged) <= 0.05 * len(judged)
    assert _coded(alone.notes, sp.NoteCode.SHORT_TEXTS)

    batch = reference.score(drafts, pool=True)
    assert batch.verdict == sp.Verdict.CLOSE
    assert not batch.warnings or not any("pool" in w for w in batch.warnings)


@dataclass(frozen=True)
class _Batches:
    """For one reference (one seed): how many batches of the writer's own comments were
    scored, and in how many each (mode, area) or (mode, headline) read "somewhat different"
    or worse."""

    count: int
    flagged: dict[tuple[str, str], int]


# References, each from its own seed: one reference's areas err together, so the bounds are
# on the mean over several and on the worst of them. The bounds were set on seeds 1 to 32;
# the test runs on others, so it does not pass for the seeds it was tuned on.
BY_AREA_SEEDS = range(101, 109)


@pytest.fixture(scope="module")
def own_batches(tmp_path_factory: pytest.TempPathFactory) -> list[_Batches]:
    comment = _bench_comments()
    found: list[_Batches] = []
    for seed in BY_AREA_SEEDS:
        folder = tmp_path_factory.mktemp(f"seed{seed}")
        rng = random.Random(seed)
        train = [comment(rng, 50) for _ in range(3000)]
        seen = set(train)
        held_out: list[str] = []
        while len(held_out) < 600:
            text = comment(rng, 50)
            if text not in seen:
                seen.add(text)
                held_out.append(text)
        writer = _records(folder, "writer.jsonl", train)
        reference = sp.build(writer, sp.Settings(syntax=False))
        batches = [held_out[start : start + 20] for start in range(0, len(held_out), 20)]
        flagged: dict[tuple[str, str], int] = {}
        for number, texts in enumerate(batches):
            drafts = _records(folder, f"batch{number}.jsonl", texts)
            for mode, pooled in (("alone", False), ("pooled", True)):
                result = reference.score(drafts, pool=pooled).report["reference"]["verdict"]
                off = [
                    area for area, entry in result["by_group"].items() if (entry["level"] or 0) >= 1
                ]
                if result["verdict"] not in ("close", "too short to judge"):
                    off.append("headline")
                for area in off:
                    flagged[(mode, area)] = flagged.get((mode, area), 0) + 1
        found.append(_Batches(len(batches), flagged))
    return found


def test_the_writers_own_comments_read_close_by_area(own_batches: list[_Batches]) -> None:
    """Not just the headline: batches of 20 of the writer's own comments, scored one by one
    or pooled against a pooled reference, read close in every area.

    An area's range is the 95th percentile of the reference's own held-out windows, so 5% of
    batches per area is nominal. Over 32 references (seeds 1 to 32, 30 batches each), each
    area read "somewhat different" or worse in 1.4% to 4.3% of batches on average and 20% at
    worst for one reference; the headline in 2.2% (one by one) and 2.1% (pooled), 10% at
    worst. Here, over 8 other references, an area may average 7% and reach 25% in one; the
    headline 5% and 15%. With 6 areas, some area reads "somewhat" in about 1 batch in 7:
    each errs at about its nominal rate, and they err apart."""
    keys = {key for batches in own_batches for key in batches.flagged}
    for key in sorted(keys):
        shares = [batches.flagged.get(key, 0) / batches.count for batches in own_batches]
        mean, worst = sum(shares) / len(shares), max(shares)
        limit = (0.05, 0.15) if key[1] == "headline" else (0.07, 0.25)
        assert mean <= limit[0] and worst <= limit[1], (key, f"{mean:.1%}", f"{worst:.0%}")


def test_length_calibration_cuts_pooled_windows_at_record_boundaries() -> None:
    """Pieces of a pooled window are what people score: single records of the length when
    there are enough (``calibration.single_parts``), else runs of whole records, and a record
    long enough on its own (``calibration.whole_parts``)."""
    parts = [_flat(40, f"r{index}") for index in range(6)]
    assert whole_parts(parts, 75) == parts  # a second record would pass 75 words
    assert whole_parts(parts, 150) == ["\n\n".join(parts[:3]), "\n\n".join(parts[3:])]
    long = "\n\n".join(_flat(60, "long") for _ in range(4))
    # A run shorter than half a piece, before or after a long record, is not a piece.
    assert whole_parts([parts[0], long, parts[1]], 150) == cut(long, 150)
    # A pooled window carries its records, so build cuts it along them.
    chunks = [Chunk(f"c{i}", "f.jsonl", _flat(40, f"w{i}")) for i in range(30)]
    windows = pool(chunks, WINDOW).windows
    first = windows[0].parts or ()
    assert len(first) > 1 and first == tuple(chunk.text for chunk in chunks[: len(first)])
    planned = plan_pieces(
        [w.text for w in windows], [300] * len(windows), [75], 10_000, [w.parts for w in windows]
    )
    assert {text for _, _, text in planned} <= {chunk.text for chunk in chunks}
    # With 20 or more records of 75 words or more, from 3 or more documents, the 75-word
    # pieces are those records alone, as the records judged at 75 words are; with fewer,
    # runs of records.
    assert single_parts([parts[0], _flat(80, "a"), _flat(130, "b")], 75) == [
        _flat(80, "a"),
        *cut(_flat(130, "b"), 75),
    ]
    mixed = [Chunk(f"m{i}", "f.jsonl", _flat(80 if i % 3 == 0 else 30, f"m{i}")) for i in range(90)]
    windows = pool(mixed, WINDOW).windows
    long_ones = {chunk.text for chunk in mixed if len(chunk.text.split()) >= 75}

    def planned_75(documents: list[str]) -> set[str]:
        texts = [w.text for w in windows]
        return {
            text
            for _, _, text in plan_pieces(
                texts, [300] * len(texts), [75], 100_000, [w.parts for w in windows], documents
            )
        }

    assert len(long_ones) >= 20
    assert planned_75([str(index) for index in range(len(windows))]) == long_ones
    two_documents = planned_75(["a" if index % 2 else "b" for index in range(len(windows))])
    assert two_documents != long_ones  # too few documents: runs of records instead


def test_score_judges_each_record_unless_asked_to_pool(tmp_path: Path) -> None:
    writer = _comments(tmp_path / "writer.jsonl")
    drafts = _comments(tmp_path / "drafts.jsonl", count=12)
    pooled = sp.build(writer, GROUPED)
    alone = pooled.score(drafts)
    assert alone.chunk_count == 12
    assert alone.report["settings"]["pool"] is False
    assert alone.report["settings"]["group_field"] == "thread"  # inherited
    (note,) = [note for note in alone.notes if note.code == sp.NoteCode.SHORT_TEXTS]
    assert note.setting == "pool"
    assert note.message == (
        "these 12 texts are short (the median is under a quarter of a window), so each is "
        "judged at its own length, and those under 75 words get no verdict; use pool to "
        "judge them as one batch"
    )
    assert not any("pool" in warning for warning in alone.warnings)
    # The inherited group field names each record's document.
    assert {base_id(row["id"]) for row in alone.report["chunks"]} == {
        f"thread={name}" for name in THREADS
    }

    # Per-document results follow the groups, one by one or pooled (plan PR 7's documents).
    names = [f"drafts.jsonl:thread={name}" for name in THREADS]
    assert [document["name"] for document in alone.report["documents"]] == names
    assert [document["chunks"] for document in alone.report["documents"]] == [3] * 4

    batch = pooled.score(drafts, pool=True)
    assert [document["name"] for document in batch.report["documents"]] == names
    assert batch.chunk_count == 4  # one window per thread
    assert batch.report["settings"]["pool_used"] is True
    assert not any(note.code == sp.NoteCode.SHORT_TEXTS for note in batch.notes)
    assert _pooled(batch.notes) == [
        "joined 12 records into 4 windows of about 300 words, never across thread groups"
    ]

    # Pooled drafts against a reference that was not pooled read too close: a warning.
    unpooled = sp.build(writer, sp.Settings(window_words=WINDOW, syntax=False, pool=False))
    assert not any("pool" in warning for warning in unpooled.score(drafts).warnings)
    assert unpooled.score(drafts, pool=True).warnings[-1] == (
        "these texts were joined into windows, but the reference's were not, so they read "
        "closer to it than they are; score them without pooling"
    )

    # Drafts without the inherited group field are read, with a note, not refused.
    plain_file = _write(tmp_path / "plain.jsonl", [{"id": "p", "text": _flat(40)}])
    assert _coded(pooled.score(plain_file).notes, sp.NoteCode.MISSING_GROUP)
    with pytest.raises(sp.StyleProfileError, match="no record has a value"):
        pooled.score(plain_file, group_field="thread")


# Evaluate: edited drafts pool as their originals did.


def test_evaluate_pairs_pooled_drafts_with_their_edits(tmp_path: Path) -> None:
    writer = _comments(tmp_path / "writer.jsonl")
    contrast, originals = _drafts(tmp_path / "drafts" / "drafts.jsonl")
    # Edits change lengths, so pooling them by length would cut different windows.
    edited = [
        {**record, "text": record["text"].replace(" — ", ", ") + " More." * (index % 7)}
        for index, record in enumerate(originals)
    ]
    same = _write(tmp_path / "same" / "drafts.jsonl", originals)
    humanized = _write(tmp_path / "humanized" / "drafts.jsonl", edited)
    result = sp.evaluate(writer, contrast, {"same": same, "humanized": humanized}, SETTINGS)
    sets = result.report["sets"]
    names = [draft["draft"] for draft in sets["original"]["by_draft"]]
    assert names[0].startswith("d00..") and names[-1].endswith("..d23")
    for label in ("same", "humanized"):
        assert [draft["draft"] for draft in sets[label]["by_draft"]] == names
        assert sets[label]["missing"] == []
    assert sets["same"]["auc"] == sets["original"]["auc"]
    assert result.report["settings"]["pool_used"] is True
    assert _pooled(result.notes)[1] == (
        f"joined 24 original records into {len(names)} windows of about 300 words"
    )


def test_evaluate_pairs_records_by_file_when_ids_restart(tmp_path: Path) -> None:
    """Two original files with ids 0..N each: every edited window pairs with its own file's
    original window, so the edits' halved length shows."""
    writer = _comments(tmp_path / "writer.jsonl")
    texts = _texts(ROOT / "examples/llm-drafts", 48, 40)
    for name, part in (("a.jsonl", texts[:24]), ("b.jsonl", texts[24:])):
        rows = [
            {"id": str(i), "text": f"{t.replace(', ', ' — ')} File {name}."}
            for i, t in enumerate(part)
        ]
        _write(tmp_path / "orig" / name, rows)
        halves = [
            {**r, "text": " ".join(r["text"].split()[: len(r["text"].split()) // 2])} for r in rows
        ]
        _write(tmp_path / "edit" / name, halves)
    originals = pool(load_chunks([str(tmp_path / "orig")]), WINDOW)
    edits = pool(load_chunks([str(tmp_path / "edit")]), WINDOW, like=originals)
    assert len(edits.windows) == len(originals.windows)
    assert [pair_key(w) for w in edits.windows] == [pair_key(w) for w in originals.windows]
    for edited_window in edits.windows:
        files = set(re.findall(r"File (\w)\.jsonl", edited_window.text))
        assert len(files) <= 1

    result = sp.evaluate(writer, tmp_path / "orig", {"half": tmp_path / "edit"}, SETTINGS)
    half = result.report["sets"]["half"]
    assert half["missing"] == [] and half["drafts"] == len(originals.windows)
    assert half["edits"]["word_ratio_median"] == pytest.approx(0.5, abs=0.05)


def test_evaluate_pairs_grouped_drafts_by_group(tmp_path: Path) -> None:
    writer = _comments(tmp_path / "writer.jsonl")
    contrast, records = _drafts(tmp_path / "drafts" / "drafts.jsonl")
    for index, record in enumerate(records):
        record["thread"] = f"d{index % 6}"
    _write(contrast, records)
    edited = _write(
        tmp_path / "edited" / "drafts.jsonl",
        [{**record, "text": record["text"].replace(" — ", ", ")} for record in records],
    )
    result = sp.evaluate(writer, contrast, {"edited": edited}, GROUPED)
    sets = result.report["sets"]
    expected = [f"thread=d{index}" for index in range(6)]
    assert [draft["draft"] for draft in sets["original"]["by_draft"]] == expected
    assert [draft["draft"] for draft in sets["edited"]["by_draft"]] == expected
    assert result.report["reference"]["documents"] == len(THREADS)


def test_drafts_without_the_group_field_are_read_ungrouped(tmp_path: Path) -> None:
    """LLM drafts rarely carry the writer's thread: a grouped writer with an ungrouped
    JSONL contrast set builds, each draft (or pooled window) a document of its own."""
    writer = _comments(tmp_path / "writer.jsonl")
    contrast, _ = _drafts(tmp_path / "llm.jsonl", count=96)
    profile = sp.build(writer, GROUPED, contrast=contrast)
    assert _coded(profile.notes, sp.NoteCode.MISSING_GROUP) == [
        f"{contrast}: no record has a thread, so each record is read as a document of its own"
    ]
    assert profile.report["document_count"] == len(THREADS)
    assert profile.report["contrast"]["sources"] > 2
    # The writer's own texts still must have it: a missing field there is likely a typo.
    with pytest.raises(sp.StyleProfileError, match="no record has a value"):
        sp.build(contrast, GROUPED)


def test_evaluate_pairs_records_of_single_files_by_id(tmp_path: Path) -> None:
    """A JSONL file of drafts and an edited copy under another name pair by record id."""
    writer = _comments(tmp_path / "writer.jsonl")
    contrast, originals = _drafts(tmp_path / "llm.jsonl", count=48)
    halves = [
        {**r, "text": " ".join(r["text"].split()[: len(r["text"].split()) // 2])} for r in originals
    ]
    edited = _write(tmp_path / "llm-half.jsonl", halves)
    for settings in (SETTINGS, sp.Settings(window_words=WINDOW, syntax=False, pool=False)):
        half = sp.evaluate(writer, contrast, {"half": edited}, settings).report["sets"]["half"]
        assert half["missing"] == []
        assert half["edits"]["word_ratio_median"] == pytest.approx(0.5, abs=0.05)


def test_evaluate_pairs_one_record_groups_and_groups_across_files(tmp_path: Path) -> None:
    """A group's windows are always named by the group, so an edited set that keeps one
    record of each group pairs; and a group pairs by its value, whichever files of a folder
    hold its records."""
    writer = _comments(tmp_path / "writer.jsonl")
    _, records = _drafts(tmp_path / "all.jsonl", count=48)
    for index, record in enumerate(records):
        record["thread"] = f"d{index % 8}"
    contrast = _write(tmp_path / "orig" / "a.jsonl", records[:24]).parent
    _write(contrast / "b.jsonl", records[24:])
    # Edited: one record per thread, the threads split across the files the other way.
    firsts = {record["thread"]: record for record in reversed(records)}
    edited = [{**r, "text": r["text"].replace(" — ", ", ")} for r in firsts.values()]
    _write(tmp_path / "edit" / "a.jsonl", edited[4:])
    _write(tmp_path / "edit" / "b.jsonl", edited[:4])
    result = sp.evaluate(writer, contrast, {"one": tmp_path / "edit"}, GROUPED)
    sets = result.report["sets"]
    expected = [f"thread=d{index}" for index in range(8)]
    assert [draft["draft"] for draft in sets["original"]["by_draft"]] == expected
    assert [draft["draft"] for draft in sets["one"]["by_draft"]] == expected
    assert sets["one"]["missing"] == []
    # Windows of one-record groups are named by the group too.
    one = [{"thread": "solo", "text": _flat(20)}, {"thread": "pair", "text": _flat(20)}]
    one.append({"thread": "pair", "text": _flat(20)})
    path = _write(tmp_path / "one.jsonl", one)
    windows = pool(load_chunks([str(path)], group_field="thread"), WINDOW).windows
    assert [chunk.id for chunk in windows] == ["thread=solo#w1", "thread=pair#w1"]


def test_only_fields_named_like_sources_are_suggested(tmp_path: Path) -> None:
    """A tweet export's language, app and counts have few values but name no source."""
    texts = _texts(ROOT / "examples/writer", 144, 40)
    tweets = [
        {
            "id": str(index),
            "created_at": f"2026-01-{index % 28 + 1:02d}",
            "lang": "en" if index % 5 else "es",
            "favorite_count": index % 7,
            "retweet_count": index % 14,
            "source": "web" if index % 2 else "phone",
            "i": index % 40,
            "text": text,
        }
        for index, text in enumerate(texts)
    ]
    path = _write(tmp_path / "tweets.jsonl", tweets)
    pooled = sp.Settings(window_words=WINDOW, syntax=False, pool=True)
    (message,) = _pooled(sp.build(path, pooled).notes)
    assert "These records have" not in message and "far too harsh" in message
    # Counts named like a source are not sources.
    for tweet in tweets:
        tweet["reply_count"] = int(tweet["id"]) % 11
        tweet["user_followers"] = int(tweet["id"]) % 13
    _write(path, tweets)
    (message,) = _pooled(sp.build(path, pooled).notes)
    assert "These records have" not in message
    # A writer's field is a fallback, after a thread's; camelCase names count too.
    for tweet in tweets:
        tweet["user_id"] = f"u{int(tweet['id']) % 3}"
        tweet["inReplyToId"] = f"p{int(tweet['id']) % 9}"
    _write(path, tweets)
    (message,) = _pooled(sp.build(path, pooled).notes)
    assert message.endswith(
        "These records have inReplyToId (9 values), user_id (3 values): pass group_field "
        "inReplyToId if each value is a separate source"
    )


def test_a_group_split_across_inputs_typed_one_by_one_is_noted(tmp_path: Path) -> None:
    folder = tmp_path / "exports"
    # Different texts each month, so neither file's records are duplicates of the other's.
    months = [_comments(folder / f"2024-0{m}.jsonl", words=36 + 8 * m) for m in (1, 2)]
    split = sp.build(months, GROUPED)
    assert _coded(split.notes, sp.NoteCode.GROUPING) == [
        "records with the same thread (such as 't1') are in more than one input, and each "
        "input's records are separate documents; to keep a group together, give the folder "
        "that holds them"
    ]
    assert split.report["document_count"] == 2 * len(THREADS)
    together = sp.build(folder, GROUPED)
    assert not _coded(together.notes, sp.NoteCode.GROUPING)
    assert together.report["document_count"] == len(THREADS)


def _llm_records(count: int, **extra: Any) -> list[dict[str, Any]]:
    """``count`` distinct LLM-style drafts of about 40 words (each numbered, so none is a
    duplicate of another)."""
    texts = _texts(ROOT / "examples/llm-drafts", count, 40)
    return [
        {"id": str(index), **extra, "text": f"{text.replace(', ', ' — ')} Draft {index}."}
        for index, text in enumerate(texts)
    ]


def _halved(record: dict[str, Any]) -> dict[str, Any]:
    split = record["text"].split()
    return {**record, "text": " ".join(split[: max(8, len(split) // 2)])}


def test_a_partial_edit_of_pooled_drafts_compares_like_with_like(tmp_path: Path) -> None:
    """An edited set that halves every 10th of 400 drafts is compared with just those
    originals, rebuilt into their windows: its length reads x0.5, as when each draft is
    scored alone, not x0.05 against whole windows."""
    writer = _comments(tmp_path / "writer.jsonl")
    originals = _llm_records(400)
    contrast = _write(tmp_path / "llm.jsonl", originals)
    edited = _write(tmp_path / "half.jsonl", [_halved(r) for r in originals[::10]])
    pooled = sp.evaluate(writer, contrast, {"half": edited}, SETTINGS)
    alone = sp.evaluate(
        writer,
        contrast,
        {"half": edited},
        sp.Settings(window_words=WINDOW, syntax=False, pool=False),
    )
    assert pooled.report["settings"]["pool_used"] is True
    assert alone.report["settings"]["pool_used"] is False
    edits = pooled.report["sets"]["half"]["edits"]
    assert edits["word_ratio_median"] == pytest.approx(0.5, abs=0.03)
    main = alone.report["sets"]["half"]["edits"]
    assert edits["word_ratio_median"] == pytest.approx(main["word_ratio_median"], abs=0.02)
    assert edits["ngram13_changed_median"] == pytest.approx(
        main["ngram13_changed_median"], abs=0.05
    )
    # Signal survival compares the edits with the same texts unedited: no habit reads far
    # past gone or stronger, as a 25-word record against a 300-word window would.
    for signal in pooled.report["signals"]:
        for entry in signal["edited"].values():
            assert entry["remaining"] is None or -1.0 < entry["remaining"] < 2.0


def test_a_partial_edit_of_grouped_drafts_compares_like_with_like(tmp_path: Path) -> None:
    """One record of each of 40 threads halved: each thread's edit is compared with that
    record's original, not the whole thread."""
    writer = _comments(tmp_path / "writer.jsonl")
    originals = [
        {**record, "thread": f"t{index // 10}"} for index, record in enumerate(_llm_records(400))
    ]
    contrast = _write(tmp_path / "llm.jsonl", originals)
    edited = _write(tmp_path / "half.jsonl", [_halved(r) for r in originals[::10]])
    result = sp.evaluate(writer, contrast, {"half": edited}, GROUPED)
    half = result.report["sets"]["half"]
    assert half["missing"] == []
    assert half["edits"]["word_ratio_median"] == pytest.approx(0.5, abs=0.03)


def test_a_folder_of_drafts_pairs_with_edits_given_as_a_file(tmp_path: Path) -> None:
    """Originals read from a folder (``a.jsonl:17``) and edits from the file given directly
    (``17``) pair by id while ids are unique across the folder, pooled or not."""
    writer = _comments(tmp_path / "writer.jsonl")
    originals = _llm_records(48)
    contrast = _write(tmp_path / "llm" / "a.jsonl", originals).parent
    edited = _write(tmp_path / "edits" / "a.jsonl", [_halved(r) for r in originals])
    for settings in (SETTINGS, sp.Settings(window_words=WINDOW, syntax=False, pool=False)):
        half = sp.evaluate(writer, contrast, {"half": edited}, settings).report["sets"]["half"]
        assert half["missing"] == []
        assert half["edits"]["word_ratio_median"] == pytest.approx(0.5, abs=0.05)


def test_an_ungrouped_contrast_note_does_not_claim_groups(tmp_path: Path) -> None:
    writer = _comments(tmp_path / "writer.jsonl")
    contrast = _write(tmp_path / "llm.jsonl", _llm_records(96))
    notes = _pooled(sp.build(writer, GROUPED, contrast=contrast).notes)
    assert notes[0].endswith("never across thread groups")
    assert re.fullmatch(r"joined 96 contrast records into \d+ windows of about 300 words", notes[1])


# The command line.


def test_cli_flags_group_and_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _comments(tmp_path / "writer.jsonl")
    _comments(tmp_path / "drafts.jsonl", count=12)
    base = ["build", "writer.jsonl", "--no-syntax", "--window-words", str(WINDOW)]
    assert main([*base, "-o", "grouped.json", "--group-field", "thread"]) == 0
    err = capsys.readouterr().err
    assert "note: joined 144 records into 24 windows" in err and "thread groups" in err
    grouped = json.loads((tmp_path / "grouped.json").read_text(encoding="utf-8"))
    assert (grouped["settings"]["group_field"], grouped["settings"]["pool"]) == ("thread", "auto")

    assert main([*base, "-o", "plain.json"]) == 0
    err = capsys.readouterr().err
    assert "With no --group-field, each window counts" in err
    assert "pass --group-field thread if each value" in err
    assert main([*base, "-o", "unpooled.json", "--no-pool"]) == 0
    captured = capsys.readouterr()
    assert "joined" not in captured.err and "fewer than 150 words" in captured.out

    assert main(["score", "drafts.jsonl", "unpooled.json", "-q"]) == 0
    assert "use --pool to judge them as one batch" in capsys.readouterr().err
    assert main(["score", "drafts.jsonl", "grouped.json", "--pool", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["chunk_count"] == 4 and report["settings"]["pool"] is True
    # The writer's own comments cross no fail level, per record or pooled per thread; each
    # document names its thread.
    for pooling in ([], ["--pool"]):
        argv = ["score", "-q", "--fail-above", "clearly", "--fail-flagged", "1", *pooling]
        assert main([*argv, "drafts.jsonl", "grouped.json"]) == 0
        assert "drafts.jsonl:thread=t1" in capsys.readouterr().out

    assert main([*base, "-o", "none.json", "--window-words", "0", "--pool"]) == 1
    assert "error: --window-words must be above 0 to pool" in capsys.readouterr().err
    assert main([*base, "-o", "typo.json", "--group-field", "thred"]) == 1
    err = capsys.readouterr().err
    assert "have the fields id, thread" in err
    assert "hint: pass --group-field with a JSONL field the records have" in err


def test_settings_overrides_include_the_new_fields() -> None:
    assert {"group_field", "pool"} <= set(api.SettingsOverrides.__annotations__)
    assert sp.Settings.from_report({"group_field": "thread", "pool": False}) == sp.Settings(
        window_words=0, group_field="thread", pool=False
    )
