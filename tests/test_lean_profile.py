"""Reference profiles without per-chunk rows, and reports that never save absolute paths."""

from __future__ import annotations

import getpass
import io
import json
import shutil
from pathlib import Path

import pytest

from styleprofile import StyleProfileError
from styleprofile.cli import main
from styleprofile.corpus.ids import document_of
from styleprofile.corpus.reading import SourceNames, load_chunks, root_name
from styleprofile.corpus.windows import window
from styleprofile.reference import build_reference
from styleprofile.reports import VERSION

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A copy of examples/ under tmp_path: writer/, llm-drafts/ and draft.md."""
    shutil.copytree(EXAMPLES / "writer", tmp_path / "writer")
    shutil.copytree(EXAMPLES / "llm-drafts", tmp_path / "llm-drafts")
    shutil.copy(EXAMPLES / "draft.md", tmp_path / "draft.md")
    return tmp_path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_version_marks_the_lean_format() -> None:
    assert VERSION >= 6  # the lean format arrived in 6


def test_references_leave_out_chunk_rows_unless_asked(corpus: Path) -> None:
    chunks = window(load_chunks([str(corpus / "writer")]), 200)
    contrast = window(load_chunks([str(corpus / "llm-drafts")]), 200)
    lean = build_reference(chunks, parser=None, contrast=contrast)
    kept = build_reference(chunks, parser=None, contrast=contrast, keep_chunks=True)

    assert "chunks" not in lean
    assert len(kept["chunks"]) == kept["chunk_count"] == lean["chunk_count"]
    # Nothing else depends on the rows: the reports agree apart from them.
    assert {key: value for key, value in kept.items() if key != "chunks"} == lean
    documents = {document_of(chunk.source, chunk.id) for chunk in chunks}
    assert lean["document_count"] == len(documents) == 7
    assert lean["calibration"]["sources"] == 7


def test_build_keeps_chunks_only_with_the_flag(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(corpus)
    build = ["build", "writer", "--contrast", "llm-drafts", "--no-syntax"]
    assert main([*build, "-o", "lean.json"]) == 0
    assert main([*build, "-o", "full.json", "--keep-chunks"]) == 0
    lean, full = _json(corpus / "lean.json"), _json(corpus / "full.json")

    assert "chunks" not in lean and lean["document_count"] == 7
    assert {row["source"] for row in full["chunks"]} == {
        f"writer/{path.name}" for path in (corpus / "writer").iterdir()
    }
    assert (corpus / "lean.json").stat().st_size < (corpus / "full.json").stat().st_size

    # A lean reference scores, shows, and its score report keeps its per-chunk rows.
    capsys.readouterr()
    assert main(["score", "draft.md", "lean.json", "-o", "draft.json"]) == 0
    assert "LLM-likeness" in capsys.readouterr().out
    scored = _json(corpus / "draft.json")
    assert scored["chunks"] and scored["chunks"][0]["source"] == "draft.md"
    assert scored["document_count"] == 1
    assert main(["show", "lean.json", "--all"]) == 0
    assert main(["show", "draft.json"]) == 0


def _no_absolute_paths(text: str, root: Path) -> None:
    for absolute in {str(root), str(root.resolve())}:
        assert absolute not in text


def test_no_absolute_paths_in_build_or_score_output(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = corpus / "writer.json"
    report = corpus / "draft.json"
    build = ["build", str(corpus / "writer"), "--contrast", str(corpus / "llm-drafts")]
    assert main([*build, "--no-syntax", "--keep-chunks", "-o", str(reference)]) == 0
    capsys.readouterr()
    assert main(["score", str(corpus / "draft.md"), str(reference), "-o", str(report)]) == 0
    capsys.readouterr()
    assert main(["score", "--json", str(corpus / "draft.md"), str(reference)]) == 0
    printed = capsys.readouterr().out

    for text in (reference.read_text(encoding="utf-8"), report.read_text(encoding="utf-8")):
        _no_absolute_paths(text, corpus)
    _no_absolute_paths(printed, corpus)
    built, scored = _json(reference), _json(report)
    # Absolute inputs are saved by their final name.
    assert built["settings"]["inputs"] == ["writer"]
    assert built["settings"]["contrast"] == ["llm-drafts"]
    assert {row["source"].split("/")[0] for row in built["chunks"]} == {"writer"}
    assert scored["reference"]["path"] == "writer.json"
    assert scored["settings"]["inputs"] == ["draft.md"]


def test_no_absolute_paths_in_evaluate_output(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(corpus / "llm-drafts", corpus / "plain")
    output = corpus / "evaluation.json"
    command = ["evaluate", str(corpus / "writer"), "--contrast", str(corpus / "llm-drafts")]
    command += ["--edited", f"plain={corpus / 'plain'}", "--no-syntax", "-o", str(output)]
    assert main(command) == 0
    _no_absolute_paths(output.read_text(encoding="utf-8"), corpus)
    assert _json(output)["settings"]["edited"] == {"plain": ["plain"]}


def test_inputs_are_saved_by_their_final_name_however_typed(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nested = corpus / "more" / "deeper"
    nested.mkdir(parents=True)
    maps = (corpus / "writer" / "old-maps.md").read_text(encoding="utf-8")
    # Its own last line, since build drops documents whose text repeats another's.
    (nested / "maps.md").write_text(maps + "\nA copy kept deeper.\n", encoding="utf-8")
    (corpus / "sub").mkdir()
    monkeypatch.chdir(corpus / "sub")
    # A ..-relative path and a ./ path save the same names as a plain one would.
    command = ["build", "../writer/", "./../more", "--no-syntax", "--keep-chunks"]
    assert main([*command, "-o", "../out/w.json"]) == 0
    built = _json(corpus / "out" / "w.json")
    sources = {row["source"] for row in built["chunks"]}
    assert "more/deeper/maps.md" in sources and "writer/sharpening.md" in sources
    assert not any(".." in source for source in sources)
    assert built["settings"]["inputs"] == ["writer", "more"]

    capsys.readouterr()
    assert main(["score", "../writer/old-maps.md", "../out/w.json", "--json"]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["reference"]["path"] == "w.json"
    assert {row["source"] for row in scored["chunks"]} == {"old-maps.md"}
    assert scored["settings"]["inputs"] == ["old-maps.md"]


def test_the_working_directory_saves_only_paths_inside_it(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(corpus / "writer")
    chunks = load_chunks(["."])
    assert {chunk.source for chunk in chunks} == {path.name for path in Path().glob("*.md")}
    assert root_name(".") == "" and root_name("./") == ""


def test_home_and_filesystem_roots_save_as_input(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_home = str(Path.home())
    user = getpass.getuser()
    home = corpus / user  # a home directory named like the user, as on most systems
    shutil.copytree(corpus / "writer", home)
    monkeypatch.setenv("HOME", str(home))
    assert root_name("~") == root_name(str(home)) == root_name("/") == "input"
    assert root_name("~/old-maps.md") == "old-maps.md"

    reference = corpus / "w.json"
    assert main(["build", "~", "--no-syntax", "--keep-chunks", "-o", str(reference)]) == 0
    text = reference.read_text(encoding="utf-8")
    built = json.loads(text)
    assert built["settings"]["inputs"] == ["input"]
    assert {row["source"].split("/")[0] for row in built["chunks"]} == {"input"}
    for leak in (str(home), real_home, f'"{user}', f"{user}/"):
        assert leak not in text


def test_any_home_directory_saves_as_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Other users' homes, or yours when HOME points elsewhere (sudo, CI), are user names too.
    monkeypatch.setenv("HOME", str(tmp_path))
    for home in ("/home/alice", "/Users/bob", "/root", "/home/alice/"):
        assert root_name(home) == "input", home
    assert root_name("/home/alice/posts") == "posts"
    assert root_name("/Users/bob/notes.md") == "notes.md"
    assert root_name("/home") == "home"


def test_an_input_given_twice_is_listed_once(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(corpus)
    command = ["build", "writer", "writer", "writer/old-maps.md", "--no-syntax", "-o", "w.json"]
    assert main(command) == 0
    assert _json(corpus / "w.json")["settings"]["inputs"] == ["writer"]


def test_unknown_users_home_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    missing = "~no-such-user-for-styleprofile/x.md"
    with pytest.raises(StyleProfileError):
        load_chunks([missing])
    assert main(["build", missing, "--no-syntax", "-o", "unused.json"]) == 1
    assert "error:" in capsys.readouterr().err


def test_symlinks_and_stdin(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    link = corpus / "linked-posts"
    link.symlink_to(corpus / "writer", target_is_directory=True)
    reference = corpus / "w.json"
    # The link names its sources, and the same files through the real path are read once.
    command = ["build", str(link), str(corpus / "writer"), "--no-syntax", "--keep-chunks"]
    assert main([*command, "-o", str(reference)]) == 0
    assert "already given" in capsys.readouterr().err
    built = _json(reference)
    assert {row["source"].split("/")[0] for row in built["chunks"]} == {"linked-posts"}
    assert built["document_count"] == 7
    # The real path gave no sources of its own, so settings leave it out.
    assert built["settings"]["inputs"] == ["linked-posts"]
    _no_absolute_paths(reference.read_text(encoding="utf-8"), corpus)

    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"Some words from stdin.")))
    assert main(["score", "-", str(reference), "--json"]) == 0
    scored = json.loads(capsys.readouterr().out)
    assert scored["settings"]["inputs"] == ["stdin"]
    assert {row["source"] for row in scored["chunks"]} == {"stdin"}


def _same_named_folders(tmp_path: Path) -> tuple[str, str]:
    """Two folders both called posts, each with the 7 writer files. The second copies end
    with a line of their own, since build drops documents whose text repeats another's."""
    for side in ("a", "b"):
        shutil.copytree(EXAMPLES / "writer", tmp_path / side / "posts")
    for path in (tmp_path / "b" / "posts").glob("*.md"):
        path.write_text(path.read_text(encoding="utf-8") + "\nA second copy.\n", "utf-8")
    return str(tmp_path / "a" / "posts"), str(tmp_path / "b" / "posts")


def test_same_named_inputs_stay_separate_documents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = _same_named_folders(tmp_path)

    chunks = load_chunks([first, second])
    assert {chunk.source.split("/")[0] for chunk in chunks} == {"posts", "posts (2)"}
    # Separate calls without shared names save the same sources twice, but documents are
    # keyed on the files they came from, so they still count as 14.
    apart = [*load_chunks([first]), *load_chunks([second])]
    assert len({chunk.source for chunk in apart}) == 7
    lean = build_reference(window(apart, 500), parser=None)
    assert lean["document_count"] == 14 and lean["calibration"]["sources"] == 14
    assert build_reference(window(chunks, 500), parser=None) == lean
    # Sharing names gives what one call gives.
    names = SourceNames()
    shared = [*load_chunks([first], names=names), *load_chunks([second], names=names)]
    assert [chunk.source for chunk in shared] == [chunk.source for chunk in chunks]
    assert names.roots == {"-": "stdin", first: "posts", second: "posts (2)"}
    # The same folder again keeps its name.
    assert load_chunks([first], names=names)[0].source.startswith("posts/")

    reference = tmp_path / "w.json"
    command = ["build", first, second, "--no-syntax", "--keep-chunks", "-o", str(reference)]
    assert main(command) == 0
    built = _json(reference)
    assert built["document_count"] == 14
    # Settings name the roots the sources were actually saved under.
    assert built["settings"]["inputs"] == ["posts", "posts (2)"]
    assert {row["source"].split("/")[0] for row in built["chunks"]} == {"posts", "posts (2)"}


def test_same_named_files_number_before_the_extension(tmp_path: Path) -> None:
    for side in ("a", "b"):
        (tmp_path / side).mkdir()
        shutil.copy(EXAMPLES / "writer" / "old-maps.md", tmp_path / side / "notes.md")
    chunks = load_chunks([str(tmp_path / "a" / "notes.md"), str(tmp_path / "b" / "notes.md")])
    assert [chunk.source for chunk in chunks] == ["notes.md", "notes (2).md"]


def test_a_file_given_twice_is_still_read_once(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(corpus)
    command = ["build", "writer", str(corpus / "writer" / "old-maps.md"), "--no-syntax"]
    assert main([*command, "-o", "w.json"]) == 0
    assert "already given; using it once" in capsys.readouterr().err
    assert _json(corpus / "w.json")["document_count"] == 7


def test_root_name() -> None:
    assert root_name("posts/") == "posts"
    assert root_name("./notes/a.md") == "a.md"
    assert root_name("../shared/posts") == "posts"
    assert root_name("posts/../x.md") == "x.md"
    assert root_name("/home/me/posts") == "posts"
    assert root_name("~/posts/a.md") == "a.md"
    assert root_name("-") == "stdin"
