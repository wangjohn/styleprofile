"""Input formats: HTML read as Markdown, format detection, and the skipped-file report."""

from __future__ import annotations

import io
import json
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from styleprofile.cli import main
from styleprofile.core import Note, NoteCode, StyleProfileError
from styleprofile.formats import html_to_markdown, looks_like_html, looks_like_jsonl
from styleprofile.profile import (
    Chunk,
    build_reference,
    drop_duplicates,
    load_chunks,
    score,
    window,
)
from styleprofile.schema import ReferenceReport
from styleprofile.surface import prose

WRITER = Path(__file__).resolve().parent.parent / "examples" / "writer"


def _stdin(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(text.encode("utf-8"))))


def _wrapped(markdown: str) -> str:
    """The review's case: every non-blank line of a Markdown essay wrapped in <p>...</p>."""
    return "\n".join(f"<p>{line}</p>" for line in markdown.split("\n") if line.strip())


def _inline(line: str) -> str:
    line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
    line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
    line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
    line = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', line)
    return line.replace("&", "&amp;").replace("\N{EM DASH}", "&mdash;")


def _page(markdown: str) -> str:
    """A realistic blog export: one line, no blank lines, nested divs and site chrome."""
    body = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        heading = re.match(r"(#{1,6}) (.*)", block)
        if heading:
            level = len(heading.group(1))
            body.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif block:
            body.append(f"<p>{_inline(' '.join(block.split()))}</p>")
    return (
        "<!DOCTYPE html><html><head><title>Blog</title><style>p { color: red; }</style>"
        '</head><body><header><h1>My Blog</h1></header><nav><a href="/">Home</a> '
        '<a href="/about">About me and this site</a></nav><div class="wrap"><div class="post">'
        + "".join(body)
        + "</div></div><footer><p>Copyright me. All rights reserved.</p></footer>"
        "<script>console.log('hi');</script></body></html>"
    )


def _blocks(markdown: str) -> list[str]:
    """The essay as HTML blocks: headings and paragraphs, with inline markup."""
    html = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        heading = re.match(r"(#{1,6}) (.*)", block)
        if heading:
            level = len(heading.group(1))
            html.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif block:
            html.append(f"<p>{_inline(' '.join(block.split()))}</p>")
    return html


def _pandoc(markdown: str) -> str:
    """Pandoc's HTML: one block per line, no blank lines, hard-wrapped at 72 columns."""
    return "\n".join("\n".join(textwrap.wrap(block, 72)) for block in _blocks(markdown)) + "\n"


def _article(markdown: str) -> str:
    """A theme's post page (Jekyll minima, WordPress): the title in the article's <header>,
    a date line, a comment section, and related-post cards in an aside."""
    title, *body = _blocks(markdown)
    return (
        "<!DOCTYPE html><html><head><title>Post</title></head><body>"
        '<header class="site-header"><a href="/">My Blog</a></header><main>'
        f'<article class="post"><header class="entry-header">{title}'
        '<p class="meta">May 1, 2024</p></header>'
        f'<div class="entry-content">{"".join(body)}</div>'
        '<section id="comments" class="comments-area"><h2>3 Comments</h2>'
        '<article class="comment-body"><p>Great post!!! lol</p></article>'
        "<form><label>Name</label><textarea></textarea></form></section></article>"
        '<aside><article class="card"><h3>Related post</h3><p>Another story.</p></article>'
        "</aside></main><footer><p>Copyright me.</p></footer></body></html>"
    )


def _delta(text: str, reference: ReferenceReport) -> float:
    chunks = window([Chunk("essay", "essay", text)], 500)
    delta = score(chunks, reference, parser=None)["reference"]["delta_mean"]
    assert delta is not None
    return delta


@pytest.fixture(scope="module")
def writer_reference() -> ReferenceReport:
    return build_reference(window(load_chunks([str(WRITER)]), 500), parser=None)


@pytest.mark.parametrize(
    "dress",
    [_wrapped, _page, _pandoc, _article],
    ids=["p-per-line", "blog-page", "pandoc-wrapped", "article-header"],
)
def test_html_essays_score_like_their_markdown(
    writer_reference: ReferenceReport, dress: Any, tmp_path: Path
) -> None:
    """Acceptance: HTML versions of the writer's essays, saved as .txt so they are found by
    sniffing, score within 0.05 Delta of the Markdown originals (before HTML was converted,
    the p-per-line essays scored ~4)."""
    for path in sorted(WRITER.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        saved = tmp_path / f"{path.stem}.txt"
        saved.write_text(dress(markdown), encoding="utf-8")
        notes: list[Note] = []
        [chunk] = load_chunks([str(saved)], notes=notes)
        assert [note.code for note in notes] == [NoteCode.READ_AS_HTML], path.name
        original = _delta(markdown, writer_reference)
        assert abs(_delta(chunk.text, writer_reference) - original) < 0.05, path.name


def test_pandoc_html_on_stdin_is_detected(
    writer_reference: ReferenceReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    markdown = (WRITER / "sharpening.md").read_text(encoding="utf-8")
    _stdin(monkeypatch, _pandoc(markdown))
    notes: list[Note] = []
    [chunk] = load_chunks(["-"], notes=notes)
    assert [note.code for note in notes] == [NoteCode.READ_AS_HTML]
    original = _delta(markdown, writer_reference)
    assert abs(_delta(chunk.text, writer_reference) - original) < 0.05


def test_html_files_read_through_load_chunks(tmp_path: Path) -> None:
    markdown = (WRITER / "sharpening.md").read_text(encoding="utf-8")
    (tmp_path / "sharpening.html").write_text(_wrapped(markdown), encoding="utf-8")
    (tmp_path / "page.htm").write_text(_page(markdown), encoding="utf-8")
    chunks = load_chunks([str(tmp_path)])
    assert [chunk.id for chunk in chunks] == ["page.htm", "sharpening.html"]
    paragraphs = len(prose(markdown).paragraphs)
    assert all(len(prose(chunk.text).paragraphs) == paragraphs for chunk in chunks)


def test_block_elements_become_markdown() -> None:
    html = (
        "<div><div><h2>A &amp; B</h2><p>One <b>bold</b> and <em>soft</em> word, a "
        '<a href="https://x.test">link</a> and <code>x = 1</code>.</p>'
        "<ul><li>First</li><li>Second<ol><li>nested</li></ol></li></ul>"
        "<ol start='3'><li>three</li><li>four</li></ol>"
        "<blockquote><p>Quoted words.</p></blockquote>"
        "<pre><code>def f():\n    return &lt;1&gt;</code></pre>"
        "line one<br>line two<br><br>new paragraph<hr><p>After the rule.</p></div></div>"
    )
    assert html_to_markdown(html) == (
        "## A & B\n\n"
        "One **bold** and *soft* word, a [link](https://x.test) and `x = 1`.\n\n"
        "- First\n- Second\n  1. nested\n\n"
        "3. three\n4. four\n\n"
        "> Quoted words.\n\n"
        "```\ndef f():\n    return <1>\n```\n\n"
        "line one\nline two\n\nnew paragraph\n\n---\n\nAfter the rule.\n"
    )


def test_formatting_metrics_survive_conversion() -> None:
    markdown = "Some **bold** text, a [link](https://x.test) and `code` here.\n\n- one\n- two\n"
    html = (
        "<p>Some <strong> bold </strong>text, a <a href='https://x.test'>link</a> and "
        "<code>code</code> here.</p><ul><li>one</li><li>two</li></ul>"
    )
    original, converted = prose(markdown), prose(html_to_markdown(html))
    assert converted == original


def test_chrome_is_dropped_and_article_preferred() -> None:
    html = (
        "<html><head><title>Site</title><script>var a = '<p>no</p>';</script></head><body>"
        "<header>Site name</header><nav><ul><li>Home</li></ul></nav>"
        "<aside>Popular posts</aside><noscript>Enable JS</noscript>"
        "<div>Sidebar text outside the article.</div>"
        "<article><p>The post itself.</p><aside>Share this</aside></article>"
        "<footer>Copyright</footer></body></html>"
    )
    assert html_to_markdown(html) == "The post itself.\n"
    main_only = "<p>Chrome.</p><main><p>Main text.</p></main><p>More chrome.</p>"
    assert html_to_markdown(main_only) == "Main text.\n"
    assert html_to_markdown("<p>No container.</p>") == "No container.\n"


def test_entities_are_decoded() -> None:
    html = "<p>Fish &amp; chips &mdash; &#8220;fresh&#8221; &hellip; caf&eacute;&nbsp;now</p>"
    assert html_to_markdown(html) == (
        "Fish & chips \N{EM DASH} \N{LEFT DOUBLE QUOTATION MARK}fresh"
        "\N{RIGHT DOUBLE QUOTATION MARK} \N{HORIZONTAL ELLIPSIS} caf\u00e9 now\n"
    )


def test_html_sniffing_leaves_markdown_with_some_html_alone() -> None:
    assert looks_like_html(_wrapped("One.\n\nTwo.\n\nThree."))
    assert looks_like_html(_page("A paragraph."))
    assert looks_like_html("<!doctype html><p>Short.</p>")
    markdown = "# Title\n\nA paragraph<br>with a break.\n\n<details>\n<summary>More</summary>\n"
    markdown += "\nHidden text.\n\n</details>\n\nAnother paragraph.\n"
    assert not looks_like_html(markdown)
    fenced = "Intro.\n\n```html\n<div>\n<p>a</p>\n<p>b</p>\n</div>\n```\n"
    assert not looks_like_html(fenced)
    assert not looks_like_html("")


def test_markdown_with_a_little_html_is_read_as_markdown(tmp_path: Path) -> None:
    """Real Markdown often carries a few inline tags or one HTML block; it stays Markdown."""
    essay = tmp_path / "essay.md"
    essay.write_text(
        "# Notes from the ridge\n\n"
        '<div align="center">\n  <img src="ridge.jpg" alt="The ridge at dawn">\n</div>\n\n'
        "We left before light.<br>\nThe trail was **frozen** in places.\n\n"
        '![map](map.png) The map, as ever, was wrong. <img src="x.png"> Still, we went on.\n\n'
        "By noon the fog lifted,<br>and the valley opened below us.\n",
        encoding="utf-8",
    )
    notes: list[Note] = []
    [chunk] = load_chunks([str(essay)], notes=notes)
    assert notes == []
    assert chunk.text == essay.read_text(encoding="utf-8")


def test_markdown_file_dense_with_html_is_read_as_html(tmp_path: Path) -> None:
    essay = tmp_path / "essay.md"
    essay.write_text(_wrapped("# Title\n\nFirst one.\n\nSecond one."), encoding="utf-8")
    notes: list[Note] = []
    [chunk] = load_chunks([str(essay)], notes=notes)
    assert chunk.text == "# Title\n\nFirst one.\n\nSecond one.\n"
    assert notes == [Note(f"{essay} looks like HTML, so it is read as HTML", NoteCode.READ_AS_HTML)]
    [raw] = load_chunks([str(essay)], input_format="markdown")
    assert raw.text.startswith("<p># Title</p>")


def test_stdin_jsonl_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"id": "a", "text": "First post."}, {"text": "Second post."}]
    _stdin(monkeypatch, "\n".join(json.dumps(record) for record in records) + "\n\n")
    notes: list[Note] = []
    chunks = load_chunks(["-"], notes=notes)
    assert chunks == [
        Chunk("a", "stdin", "First post."),
        Chunk("stdin:2", "stdin", "Second post."),
    ]
    assert [note.code for note in notes] == [NoteCode.READ_AS_JSONL]

    _stdin(monkeypatch, '{"text": "Looks like JSON."}\nbut this line is prose.\n')
    [chunk] = load_chunks(["-"])
    assert chunk.text.startswith('{"text"')

    _stdin(monkeypatch, '{"id": 1, "text": "Forced prose."}\n')
    [chunk] = load_chunks(["-"], input_format="markdown")
    assert chunk == Chunk("stdin", "stdin", '{"id": 1, "text": "Forced prose."}\n')

    _stdin(monkeypatch, '{"body": "x"}\n')
    with pytest.raises(StyleProfileError, match="stdin:1: no string field") as error:
        load_chunks(["-"], text_field="text")
    assert error.value.code == "text_field"


def test_jsonl_sniffing() -> None:
    assert looks_like_jsonl('{"a": 1}\n\n{"b": 2}\n')
    assert not looks_like_jsonl('{"a": 1}\n[1, 2]\n')
    assert not looks_like_jsonl('{"a": 1}\nplain words\n')
    assert not looks_like_jsonl("{not json}\n")
    assert not looks_like_jsonl("\n\n")


def test_forced_formats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stdin(monkeypatch, "<div>One.</div><div>Two.</div>")
    assert load_chunks(["-"], input_format="html")[0].text == "One.\n\nTwo.\n"
    records = tmp_path / "records.txt"
    records.write_text('{"text": "From a text file."}\n', encoding="utf-8")
    assert load_chunks([str(records)], input_format="jsonl")[0].text == "From a text file."
    page = tmp_path / "page.html"
    page.write_text("<p>Kept as written.</p>", encoding="utf-8")
    assert load_chunks([str(page)], input_format="markdown")[0].text == "<p>Kept as written.</p>"
    with pytest.raises(StyleProfileError, match="unknown input format 'docx'"):
        load_chunks([str(page)], input_format="docx")


def test_directory_walk_reports_skipped_files(tmp_path: Path) -> None:
    (tmp_path / "post.md").write_text("A post.", encoding="utf-8")
    for index in range(3):
        (tmp_path / f"draft{index}.docx").write_bytes(b"PK")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    for name in ("cover.png", "site.css", "app.js", "feed.xml", "LICENSE"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"\0")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("x", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    notes: list[Note] = []
    assert [chunk.id for chunk in load_chunks([str(tmp_path)], notes=notes)] == ["post.md"]
    assert notes == [
        Note(
            f"skipped 3 .docx, 1 .pdf and 5 other files in {tmp_path}; convert the .docx and "
            ".pdf files to Markdown, text or HTML first (for example with pandoc)",
            NoteCode.SKIPPED_FILES,
        )
    ]


def test_directory_walk_is_quiet_about_assets(tmp_path: Path) -> None:
    (tmp_path / "post.md").write_text("A post.", encoding="utf-8")
    for name in ("cover.png", "photo.JPG", "app.js", "tool.py", "Thumbs.db", "notes.txt~"):
        (tmp_path / name).write_bytes(b"x")
    notes: list[Note] = []
    assert [chunk.id for chunk in load_chunks([str(tmp_path)], notes=notes)] == ["post.md"]
    assert notes == []
    (tmp_path / "chapter.rst").write_text("Title\n=====\n", encoding="utf-8")
    load_chunks([str(tmp_path)], notes=notes)
    assert [note.message.split(" in ")[0] for note in notes] == ["skipped 1 .rst and 6 other files"]
    assert "convert the .rst file to Markdown" in notes[0].message


def test_directory_with_nothing_usable_says_what_it_has(tmp_path: Path) -> None:
    for name in ("a.docx", "b.docx", "c.pdf", "notes", "d.rtf", "e.odt", "f.pages", "g.png"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(StyleProfileError) as error:
        load_chunks([str(tmp_path)])
    assert str(error.value) == (
        f"{tmp_path} contains no .md, .markdown, .txt, .html, .htm, or .jsonl files; it has "
        "2 .docx, 1 .pdf, 1 .rtf, 1 .odt and 3 other files, so convert those documents to "
        "Markdown, text or HTML first (for example with pandoc)"
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "logo.svg").write_bytes(b"<svg/>")
    with pytest.raises(StyleProfileError, match=r"it has only images, code and other files"):
        load_chunks([str(assets)])


def test_cli_builds_from_html_and_prints_notes(tmp_path: Path, capsys: Any) -> None:
    posts = tmp_path / "posts"
    posts.mkdir()
    for path in sorted(WRITER.glob("*.md")):
        (posts / f"{path.stem}.html").write_text(
            _page(path.read_text(encoding="utf-8")), encoding="utf-8"
        )
    (posts / "cover.png").write_bytes(b"\x89PNG")
    (posts / "draft.docx").write_bytes(b"PK")
    output = tmp_path / "writer.json"
    build = ["build", str(posts), "-o", str(output), "--no-syntax", "--input-format", "html"]
    assert main([*build, "--verbose"]) == 0
    err = capsys.readouterr().err
    assert (
        f"note: skipped 1 .docx and 1 other file in {posts}; convert the .docx file to "
        "Markdown, text or HTML first (for example with pandoc)\n" in err
    )
    assert json.loads(output.read_text(encoding="utf-8"))["settings"]["input_format"] == "html"
    # score does not inherit the format: the draft below is sniffed, which forced HTML
    # would not note.
    draft = tmp_path / "draft.md"
    draft.write_text(_wrapped((WRITER / "mud-season.md").read_text(encoding="utf-8")))
    assert main(["score", str(draft), str(output), "--json", "--verbose"]) == 0
    captured = capsys.readouterr()
    assert (
        f"note: {draft} looks like HTML, so it is read as HTML (pass --input-format markdown "
        "to read it as written)" in captured.err
    )
    assert json.loads(captured.out)["reference"]["delta_mean"] < 1


def test_cli_reads_jsonl_from_stdin(
    tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        {"id": path.stem, "text": path.read_text(encoding="utf-8")}
        for path in sorted(WRITER.glob("*.md"))
    ]
    _stdin(monkeypatch, "".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "writer.json"
    assert main(["build", "-", "-o", str(output), "--no-syntax"]) == 0
    assert "note: stdin is JSON objects, one per line" in capsys.readouterr().err
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["calibration"]["sources"] == len(records)


def test_a_command_less_line_keeps_its_input_format(tmp_path: Path, capsys: Any) -> None:
    posts = tmp_path / "posts"
    posts.mkdir()
    (posts / "post.html").write_text("<p>A post.</p>", encoding="utf-8")
    output = tmp_path / "x.json"
    assert main([str(posts), "--output", str(output), "--input-format", "html"]) == 2
    suggestion = f"styleprofile build {posts} -o {output} --input-format html"
    assert f"did you mean `{suggestion}`?" in capsys.readouterr().err
    draft = tmp_path / "draft.md"
    draft.write_text("A draft.", encoding="utf-8")
    assert main([str(draft), "-r", "writer.json", "--input-format", "markdown"]) == 2
    suggestion = f"styleprofile score {draft} writer.json --input-format markdown"
    assert f"did you mean `{suggestion}`?" in capsys.readouterr().err


# Review regressions: pages and folders that used to give empty, flattened or doubled text.


def test_a_head_without_its_end_tag_ends_at_the_body() -> None:
    """HTML5 lets </head> be left out; minifiers do, and the page used to convert to nothing."""
    page = "<html><head><title>Blog</title><meta charset=utf-8><body><p>The post.</p></body>"
    assert html_to_markdown(page) == "The post.\n"
    assert html_to_markdown("<head><title>T</title><p>No body tag.</p>") == "No body tag.\n"


def test_html_with_no_text_is_noted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "post.md").write_text("A post with words.", encoding="utf-8")
    (tmp_path / "empty.html").write_text("<script>app()</script><div></div>", encoding="utf-8")
    notes: list[Note] = []
    load_chunks([str(tmp_path)], notes=notes)
    assert notes == [
        Note(
            f"{tmp_path / 'empty.html'} has no readable text after conversion from HTML",
            NoteCode.EMPTY_HTML,
        )
    ]
    _stdin(monkeypatch, "<html><body><nav>Home</nav></body></html>")
    load_chunks(["-"], notes=notes)
    assert notes[-1].code == NoteCode.EMPTY_HTML


def test_main_content_selection() -> None:
    # Related-post cards inside dropped chrome are not candidates.
    aside = (
        "<main><h1>Real post</h1><p>The body.</p></main>"
        "<aside><article><h3>Other post</h3><p>Teaser.</p></article></aside>"
    )
    assert html_to_markdown(aside) == "# Real post\n\nThe body.\n"
    # The post body in a <div> beside a small article card: <main> holds the post.
    body = " ".join(["Words of the post."] * 20)
    card = f"<main><div><p>{body}</p></div><article><p>A card.</p></article></main>"
    assert html_to_markdown(card).startswith("Words of the post.")
    # One article with most of the text is the post; smaller ones are cards.
    post = f"<article><p>{body}</p></article><article><p>Related card.</p></article>"
    assert html_to_markdown(post) == f"{body}\n"
    # An index page: several excerpts of similar length are all read.
    index = "".join(f"<article><h2>Post {n}</h2><p>Excerpt {n}.</p></article>" for n in range(3))
    assert html_to_markdown(index).count("Excerpt") == 3
    # An empty chosen container falls back to the body.
    empty = "<body><article></article><div><p>Text outside it.</p></div></body>"
    assert html_to_markdown(empty) == "Text outside it.\n"


def test_post_headers_are_kept_and_comments_dropped() -> None:
    page = _article("# Title\n\nThe first paragraph of the post.")
    assert html_to_markdown(page) == "# Title\n\nThe first paragraph of the post.\n"
    wordpress = (
        '<div id="content"><article class="post"><p>The post.</p></article>'
        '<ol class="comment-list"><li id="comment-2"><article id="div-comment-2" '
        'class="comment-body"><p>Nice!</p></article></li></ol>'
        '<div id="respond"><h3>Leave a reply</h3><form><label>Comment</label></form></div></div>'
    )
    assert html_to_markdown(wordpress) == "The post.\n"


def test_links_around_blocks_leave_no_markup() -> None:
    substack = (
        "<p>Intro.</p><a class='image-link' href='https://x.test/img'><div><picture>"
        "<img src='a.png'></picture></div></a><p>After.</p>"
    )
    assert html_to_markdown(substack) == "Intro.\n\nAfter.\n"
    card = "<a href='/post'><div><h3>Card title</h3><p>Card excerpt.</p></div></a>"
    assert html_to_markdown(card) == "### Card title\n\nCard excerpt.\n"


def test_inline_markup_edge_cases() -> None:
    assert html_to_markdown("<p>A<strong></strong>B <em> </em>C <b>bold </b>next</p>") == (
        "AB C **bold** next\n"
    )
    assert html_to_markdown("<b><p>one</p><p>two</p></b>") == "**one**\n\n**two**\n"
    anchors = (
        "<h2 id='part'>Part<a class='anchor' href='#part'>#</a></h2>"
        "<h3>Sub<a href='#sub'>\N{PILCROW SIGN}</a></h3><p>Text<a href='#fn1'>1</a>.</p>"
    )
    assert html_to_markdown(anchors) == "## Part\n\n### Sub\n\nText.\n"


def test_paragraph_text_is_escaped_on_pages_with_html_structure() -> None:
    page = (
        "<h1>Title</h1><p># of users rose.</p><p>1. Not a list.</p><p>- nor this</p>"
        "<p>> nor a quote</p><blockquote><p>- quoted</p></blockquote>"
    )
    markdown = html_to_markdown(page)
    assert markdown == (
        "# Title\n\n\\# of users rose.\n\n1\\. Not a list.\n\n\\- nor this\n\n"
        "\\> nor a quote\n\n> \\- quoted\n"
    )
    parsed = prose(markdown)
    assert (parsed.headings, parsed.list_items, len(parsed.paragraphs)) == (1, 0, 5)


def test_sniffing_leaves_real_markdown_with_html_blocks_alone() -> None:
    readme = (
        '<p align="center">\n  <img src="logo.png" width="200">\n</p>\n'
        '<h1 align="center">MyTool</h1>\n\nA tool that does things.\n\n## Install\n\n'
        "    pip install mytool\n"
    )
    details = (
        "# Title\n\nParagraph one.\n\n<details>\n<summary>More</summary>\n<p>Hidden.</p>\n"
        "</details>\n\nParagraph two.\n"
    )
    table = "Some intro.\n\n<table>\n<tr><td>a</td><td>b</td></tr>\n</table>\n"
    mdx = (
        "import { Chart } from '../components/Chart'\n\n# My post\n\n"
        '<div className="callout">\n  <p>Note this.</p>\n</div>\n\nSome prose.\n'
    )
    for markdown in (readme, details, table, mdx):
        assert not looks_like_html(markdown)
    fragment = "<h2>Intro</h2>\n<p>My post body.</p>\n<p>Second paragraph.</p>\n"
    assert looks_like_html(fragment)
    assert looks_like_html("<p>One.</p>\n\n<div>\n\nInside a div.\n\n</div>\n<p>Two.</p>\n")
    assert looks_like_html("<!-- saved -->\n<!DOCTYPE html>\n<p>x</p>")
    assert looks_like_html('<?xml version="1.0"?>\n<html><body><p>x</p></body></html>')


def test_a_blank_line_in_text_is_a_paragraph_break() -> None:
    """Markdown misread as HTML keeps its paragraphs rather than collapsing into one."""
    assert html_to_markdown("<div>\nFirst paragraph.\n\nSecond paragraph.\n</div>") == (
        "First paragraph.\n\nSecond paragraph.\n"
    )


def test_conversion_is_linear() -> None:
    """Blogger posts and mail archives are one <div> of <br> lines; 5 MB took 12 s."""
    line = "The quick brown fox jumps over the lazy dog near the old stone fence. "
    lines = f"<div>{f'{line}<br>{chr(10)}' * 70_000}</div>"
    inline = f"<div>{f'{line}<b>bold</b> <a href=/x>link</a> ' * 55_000}</div>"
    for html in (lines, inline):
        assert len(html) > 5_000_000
        start = time.perf_counter()
        html_to_markdown(html)
        assert time.perf_counter() - start < 15


def _site(root: Path, folders: dict[str, dict[str, str]]) -> None:
    for folder, files in folders.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (root / folder / name).write_text(text, encoding="utf-8")


def test_jekyll_and_hugo_trees_skip_generated_folders(tmp_path: Path) -> None:
    essays = {path.name: path.read_text(encoding="utf-8") for path in WRITER.glob("*.md")}
    built = {f"{Path(name).stem}.html": _page(text) for name, text in essays.items()}
    jekyll = tmp_path / "jekyll"
    _site(
        jekyll,
        {
            "_posts": essays,
            "_site/posts": built,
            "_layouts": {"default.html": "<html>{% include head.html %}{{ content }}</html>"},
            "_includes": {"head.html": "<head><title>{{ page.title }}</title></head>"},
        },
    )
    notes: list[Note] = []
    chunks = load_chunks([str(jekyll)], notes=notes)
    assert sorted(chunk.id for chunk in chunks) == sorted(f"_posts/{name}" for name in essays)
    assert notes == [
        Note(
            f"left out static-site output and template folders in {jekyll}: _includes, "
            "_layouts and _site; name one directly to read it",
            NoteCode.SKIPPED_DIRS,
        )
    ]
    # Named directly, a generated folder is read.
    assert len(load_chunks([str(jekyll / "_site")])) == len(essays)

    hugo = tmp_path / "hugo"
    _site(
        hugo,
        {
            "content/posts": essays,
            "public/posts": built,
            "layouts/_default": {"single.html": "<main>{{ .Content }}</main>"},
            "themes/ananke/layouts": {"list.html": "{{ range .Pages }}{{ end }}"},
            "resources/_gen": {"x.json": "{}"},
        },
    )
    notes = []
    chunks = load_chunks([str(hugo)], notes=notes)
    assert len(chunks) == len(essays)
    assert [note.message.split(": ")[1] for note in notes] == [
        "layouts, public and themes; name one directly to read it"
    ]


def test_duplicate_documents_are_dropped(tmp_path: Path) -> None:
    essay = (WRITER / "sharpening.md").read_text(encoding="utf-8")
    other = (WRITER / "old-maps.md").read_text(encoding="utf-8")
    chunks = [
        Chunk("sharpening.md", "posts/sharpening.md", essay),
        Chunk("old-maps.md", "posts/old-maps.md", other),
        Chunk("sharpening.html", "site/sharpening.html", html_to_markdown(_page(essay))),
        Chunk("a", "records.jsonl", "Thanks!"),
        Chunk("b", "records.jsonl", "Thanks!"),
    ]
    seen: dict[bytes, str] = {}
    kept, note = drop_duplicates(chunks, seen)
    assert [chunk.id for chunk in kept] == ["sharpening.md", "old-maps.md", "a", "b"]
    assert note == Note(
        "dropped 1 document that repeats another word for word, keeping the first copy (for "
        "example, site/sharpening.html repeats posts/sharpening.md)",
        NoteCode.DUPLICATES,
    )
    # A later set (the contrast drafts) is checked against the texts already seen.
    kept, note = drop_duplicates([Chunk("x", "drafts.jsonl", other)], seen)
    assert kept == []
    assert note is not None
    assert "record x in drafts.jsonl repeats posts/old-maps.md" in note.message


def test_cli_build_drops_duplicates_across_reference_and_contrast(
    tmp_path: Path, capsys: Any
) -> None:
    posts, drafts = tmp_path / "posts", tmp_path / "drafts"
    posts.mkdir()
    drafts.mkdir()
    for path in sorted(WRITER.glob("*.md")):
        (posts / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (posts / "copy.html").write_text(_page((WRITER / "sharpening.md").read_text()), "utf-8")
    for path in sorted((WRITER.parent / "llm-drafts").glob("*.md")):
        (drafts / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (drafts / "stolen.md").write_text((WRITER / "old-maps.md").read_text(), encoding="utf-8")
    output = tmp_path / "writer.json"
    command = ["build", str(posts), "--contrast", str(drafts), "-o", str(output), "--no-syntax"]
    assert main([*command, "--verbose"]) == 0
    err = capsys.readouterr().err
    assert "note: dropped 1 document that repeats another word for word" in err
    assert err.count("note: dropped 1 document") == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["calibration"]["sources"] == len(list(WRITER.glob("*.md")))


def test_sniffed_files_are_named_in_the_note(tmp_path: Path, capsys: Any) -> None:
    for name in ("a.md", "b.md", "c.txt", "d.md"):
        (tmp_path / name).write_text(_wrapped("One.\n\nTwo.\n\nThree."), encoding="utf-8")
    (tmp_path / "e.md").write_text("Plain Markdown.", encoding="utf-8")
    notes: list[Note] = []
    load_chunks([str(tmp_path)], notes=notes)
    assert notes == [
        Note(
            f"{tmp_path / 'a.md'}, {tmp_path / 'b.md'}, {tmp_path / 'c.txt'} and 1 more look "
            "like HTML, so they are read as HTML",
            NoteCode.READ_AS_HTML_IN_FOLDER,
        )
    ]
    assert "would also apply to .html files" in (NoteCode.READ_AS_HTML_IN_FOLDER.hint or "")


def test_forced_jsonl_errors_say_it_was_forced(tmp_path: Path, capsys: Any) -> None:
    essay = tmp_path / "essay.md"
    essay.write_text("An essay, not records.", encoding="utf-8")
    with pytest.raises(StyleProfileError) as error:
        load_chunks([str(essay)], input_format="jsonl")
    assert error.value.code == "forced_jsonl"
    assert str(error.value).endswith("essay.md was read as JSONL because that format was asked for")
    assert main(["build", str(essay), "-o", str(tmp_path / "x.json"), "--input-format", "jsonl"])
    err = capsys.readouterr().err
    assert "hint: leave out --input-format jsonl to read each file by its extension" in err


def test_evaluate_pairs_edits_by_stem(tmp_path: Path, capsys: Any) -> None:
    posts, drafts, edits = tmp_path / "posts", tmp_path / "drafts", tmp_path / "edits"
    for folder in (posts, drafts, edits):
        folder.mkdir()
    for path in sorted(WRITER.glob("*.md")):
        (posts / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    for path in sorted((WRITER.parent / "llm-drafts").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        (drafts / path.name).write_text(text, encoding="utf-8")
        (edits / f"{path.stem}.html").write_text(_page(text), encoding="utf-8")
    command = ["evaluate", str(posts), "--contrast", str(drafts), "--edited", f"html={edits}"]
    assert main([*command, "--no-syntax"]) == 0


def test_the_library_threads_input_format_and_drops_duplicates(tmp_path: Path) -> None:
    import styleprofile as sp

    with pytest.raises(sp.StyleProfileError) as error:
        sp.Settings(input_format="docx")
    assert (error.value.code, error.value.setting) == ("invalid_setting", "input_format")
    posts = tmp_path / "posts"
    posts.mkdir()
    for path in sorted(WRITER.glob("*.md")):
        (posts / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (posts / "copy.txt").write_text(_pandoc((WRITER / "sharpening.md").read_text()), "utf-8")
    profile = sp.build(posts, sp.Settings(syntax=False))
    codes = [note.code for note in profile.notes]
    assert NoteCode.READ_AS_HTML_IN_FOLDER in codes and NoteCode.DUPLICATES in codes
    # The duplicate note names documents by their saved sources, never by path.
    [duplicates] = [note for note in profile.notes if note.code == NoteCode.DUPLICATES]
    assert "posts/sharpening.md repeats posts/copy.txt" in duplicates.message
    assert str(tmp_path) not in duplicates.message
    assert profile.report["settings"]["input_format"] == "auto"
    forced = sp.build(posts, sp.Settings(syntax=False, input_format="markdown"))
    assert forced.settings.input_format == "markdown"
    # score reads drafts in auto unless told otherwise: the HTML draft is detected.
    draft = tmp_path / "draft.txt"
    draft.write_text(_pandoc((WRITER / "mud-season.md").read_text()), encoding="utf-8")
    result = forced.score(draft)
    assert result.report["settings"]["input_format"] == "auto"
    assert [note.code for note in result.notes] == [NoteCode.READ_AS_HTML]


# Second review: article dominance, WebForms, unclosed inline tags, site folders, widgets.


def test_an_article_is_read_alone_only_when_it_dominates() -> None:
    excerpt = "<p>" + " ".join(["word"] * 40) + ".</p>"
    two = f"<main><article><h2>A</h2>{excerpt}</article><article><h2>B</h2>{excerpt}</article>"
    assert html_to_markdown(two + "</main>").count("## ") == 2
    full = "".join(f"<p>{' '.join(['full'] * 60)}.</p>" for _ in range(3))
    teaser = "<p>" + " ".join(["teaser"] * 25) + ".</p>"
    index = f"<main><article><h2>New</h2>{full}</article>" + "".join(
        f"<article><h2>Old {n}</h2>{teaser}</article>" for n in range(4)
    )
    assert html_to_markdown(index + "</main>").count("## ") == 5


def test_a_page_wrapped_in_a_form_is_read() -> None:
    """ASP.NET WebForms wraps the whole body in one <form>; only its controls are dropped."""
    page = (
        '<!doctype html><html><body><form id="form1" method="post" action="./">'
        '<input type="hidden" name="__VIEWSTATE" value="x"><div class="post"><h1>Title</h1>'
        "<p>First paragraph.</p><p>Second paragraph.</p></div>"
        "<label for='q'>Search</label><input id='q'><select><option>One</option></select>"
        "<button>Go</button><textarea>Draft</textarea></form></body></html>"
    )
    assert html_to_markdown(page) == "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n"


def test_subscribe_widgets_are_dropped() -> None:
    page = (
        "<article><p>The post.</p><div class='subscription-widget-wrap'><p>Subscribe now</p>"
        "</div><div class='subscribe'><p>Get new posts</p></div>"
        "<div id='newsletter-signup'><p>Join</p></div></article>"
    )
    assert html_to_markdown(page) == "The post.\n"


def test_unclosed_inline_tags_stay_linear() -> None:
    """``<p><b>text`` repeated, never closed, took 49.7 s at 1.5 MB."""
    sentence = "The quick brown fox jumps over the lazy dog near the old stone fence. "
    html = f"<p><b>{sentence}" * 66_000
    assert len(html) > 5_000_000
    start = time.perf_counter()
    markdown = html_to_markdown(html)
    assert time.perf_counter() - start < 15
    assert markdown.startswith(f"**{sentence.strip()}**\n\n")


def test_site_output_folders_skip_only_html(tmp_path: Path) -> None:
    essays = tmp_path / "essays"
    _site(essays, {"public": {"one.md": "An essay kept in public.", "one.html": "<p>Built.</p>"}})
    notes: list[Note] = []
    chunks = load_chunks([str(essays)], notes=notes)
    assert [chunk.id for chunk in chunks] == ["public/one.md"]
    assert [note.code for note in notes] == [NoteCode.SKIPPED_DIRS]


def test_a_folder_with_only_site_output_names_it(tmp_path: Path) -> None:
    blog = tmp_path / "blog"
    _site(blog, {"_site": {"a.html": "<p>x</p>"}, "_layouts": {"b.html": "{{ content }}"}})
    with pytest.raises(StyleProfileError) as error:
        load_chunks([str(blog)])
    assert error.value.code == "only_generated"
    assert str(error.value) == (
        f"{blog} has readable files only in static-site output and template folders "
        "(_layouts and _site), which are left out; name one directly to read it"
    )
