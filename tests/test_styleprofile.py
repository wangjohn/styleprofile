from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from styleprofile import (
    Chunk,
    StyleProfileError,
    build_profile,
    build_reference,
    load_chunks,
    load_reference,
    surface_metrics,
    window,
)
from styleprofile.cli import main
from styleprofile.display import describe_delta, label, value
from styleprofile.profile import _prepare
from styleprofile.profile import _score as _score_with
from styleprofile.surface import (
    Metrics,
    char_trigrams,
    jensen_shannon,
    masked_bigrams,
    mattr,
    mtld,
    prose,
    sentences,
)

AUTHOR = (
    "I think the honest answer is that most tools are built for engineers, by engineers. "
    "That's fine, but it means the person closest to the problem (usually not an engineer) "
    "gets left out. We noticed this at work. It's probably the most common failure I see.\n\n"
    "So we built something small. It isn't clever, and it doesn't need to be."
)
GENERIC = (
    "In today's rapidly evolving landscape, organizations must leverage robust solutions. "
    "Moreover, it is worth noting that comprehensive frameworks are crucial. Additionally, "
    "teams must foster collaboration. This is not just a tool, but a transformation."
)
MARKDOWN = """# A heading

First paragraph with a [link](https://example.com) and **bold** text and `code`.

```python
print("ignored entirely")
```

- One list item.
- Another item
  that continues here.

| table | row |
|---|---|
"""


def test_prose_strips_markup_but_counts_it() -> None:
    parsed = prose(MARKDOWN)

    assert parsed.headings == 1
    assert parsed.list_items == 2
    assert parsed.links == 1
    assert parsed.bold == 1
    assert parsed.code_spans == 1
    assert parsed.paragraphs == ["First paragraph with a link and bold text and ."]
    assert parsed.blocks[1:] == ["One list item.", "Another item that continues here."]
    assert "print" not in parsed.text
    assert "table" not in parsed.text


def test_sentences_respect_quotes_and_abbreviations() -> None:
    assert sentences('He said "stop." Then we left. See e.g. Mr. Smith now. Done! Why?') == [
        'He said "stop."',
        "Then we left.",
        "See e.g. Mr. Smith now.",
        "Done!",
        "Why?",
    ]


def test_surface_metrics_are_rates_and_shares() -> None:
    metrics = surface_metrics(AUTHOR)
    words = metrics["size"]["words"]

    assert words == 60
    assert metrics["size"]["sentences"] == 6
    assert metrics["size"]["paragraphs"] == 2
    assert metrics["punctuation"]["parentheses_per_1k"] == pytest.approx(1000 / words)
    assert metrics["voice"]["contractions_per_1k"] == pytest.approx(4 * 1000 / words)
    assert metrics["voice"]["and_but_so_openers_pct"] == pytest.approx(100 / 6)
    assert metrics["voice"]["llm_markers_per_1k"] == 0
    assert metrics["vocabulary"]["mattr_100"] is None
    assert metrics["punctuation"]["curly_apostrophe_pct"] == 0

    generic = surface_metrics(GENERIC)
    assert (generic["voice"]["llm_markers_per_1k"] or 0) > 0
    assert generic["voice"]["transition_openers_pct"] == pytest.approx(200 / 4)
    assert (generic["voice"]["not_just_but_per_1k"] or 0) > 0


def test_lexical_diversity_needs_enough_words() -> None:
    varied = [f"w{index}" for index in range(120)]
    repetitive = ["same", "word"] * 60

    assert mattr(varied) == 1.0
    assert mattr(repetitive) == pytest.approx(0.02)
    assert mtld(varied[:40]) is None
    assert (mtld(varied) or 0) > (mtld(repetitive) or 0)


def test_distributions_mask_content_words() -> None:
    bigrams = masked_bigrams("The cat sat on the mat.")

    assert bigrams == {"the ·": 2, "· on": 1, "on the": 1}
    assert char_trigrams("Ab c")["ab "] == 1


def test_jensen_shannon_bounds() -> None:
    assert jensen_shannon({"a": 1.0}, {"a": 1.0}) == 0
    assert jensen_shannon({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert jensen_shannon({}, {"a": 1.0}) is None


def test_window_splits_at_paragraphs_and_merges_short_tail() -> None:
    text = "\n\n".join(" ".join(["word"] * 40) for _ in range(5))
    pieces = window([Chunk("post", "src", text)], 100)

    assert [piece.id for piece in pieces] == ["post#w1", "post#w2"]
    assert [len(piece.text.split()) for piece in pieces] == [120, 80]

    tail = window([Chunk("post", "src", text + "\n\nshort")], 100)
    assert [len(piece.text.split()) for piece in tail] == [120, 81]


def test_load_chunks_reads_jsonl_directories_and_files(tmp_path: Path) -> None:
    records = tmp_path / "posts.jsonl"
    records.write_text(
        json.dumps({"id": "a", "body_markdown": AUTHOR})
        + "\n\n"
        + json.dumps({"output": GENERIC})
        + "\n",
        encoding="utf-8",
    )
    folder = tmp_path / "essays"
    (folder / "nested").mkdir(parents=True)
    (folder / "one.md").write_text(AUTHOR, encoding="utf-8")
    (folder / "nested" / "two.txt").write_text(GENERIC, encoding="utf-8")
    (folder / "skip.json").write_text("{}", encoding="utf-8")

    chunks = load_chunks([str(records), str(folder)])

    assert [chunk.id for chunk in chunks] == ["a", "posts.jsonl:3", "nested/two.txt", "one.md"]
    assert chunks[1].text == GENERIC

    with pytest.raises(StyleProfileError, match="no string field among missing"):
        load_chunks([str(records)], text_field="missing")


def test_profile_scores_chunks_against_a_reference() -> None:
    author = [
        Chunk("a1", "src", AUTHOR),
        Chunk(
            "a2", "src", "I don't know. But we tried it anyway, and it mostly worked (for a while)."
        ),
        Chunk("a3", "src", "You can't plan everything. We shipped it, and I think that was right."),
        Chunk(
            "a4", "src", "It's a small thing. So I wrote it down, because I'll forget otherwise."
        ),
    ]
    reference = build_profile(author, parser=None, settings={"window_words": None})

    assert reference["chunk_count"] == 4
    assert reference["summary"]["voice"]["contractions_per_1k"]["n"] == 4
    assert set(reference["distributions"]) == {"masked_bigram", "char_trigram"}

    scored = build_profile(
        [Chunk("author", "src", AUTHOR), Chunk("generic", "src", GENERIC)],
        parser=None,
        reference=reference,
        settings={"window_words": None},
    )
    author_score, generic_score = (row["reference"] for row in scored["chunks"])

    assert generic_score["delta"] > author_score["delta"]
    assert (
        generic_score["divergence"]["masked_bigram"] > author_score["divergence"]["masked_bigram"]
    )
    assert generic_score["delta_by_group"]["voice"] > author_score["delta_by_group"]["voice"]
    unseen = {row["metric"] for row in generic_score["unseen_in_reference"]}
    assert "voice.llm_markers_per_1k" in unseen
    assert "voice.llm_markers_per_1k" not in {
        row["metric"] for row in author_score["unseen_in_reference"]
    }
    assert "size" not in generic_score["z"]
    assert scored["reference"]["delta_mean"] == pytest.approx(
        (author_score["delta"] + generic_score["delta"]) / 2
    )
    assert any("fewer than 150 words" in warning for warning in scored["warnings"])


def test_legacy_flat_form_still_works_with_a_deprecation_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "essay.md"
    source.write_text("\n\n".join([AUTHOR] * 6), encoding="utf-8")
    reference = tmp_path / "reference.json"
    output = tmp_path / "report.json"
    command = [str(source), "--no-syntax", "--window-words", "100"]

    assert main([*command, "--output", str(reference)]) == 0
    built = capsys.readouterr()
    assert "deprecated" in built.err
    assert f"use: styleprofile build {source} -o {reference} --no-syntax --window-words 100" in (
        built.err
    )
    assert main([*command, "--reference", str(reference), "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["kind"] == "score"
    assert report["chunk_count"] == 3
    assert report["settings"]["window_words"] == 100
    assert report["settings"]["syntax"] is None
    scored = capsys.readouterr()
    assert f"use: styleprofile score {source} {reference} -o {output}" in scored.err
    assert "STYLE COMPARISON" in scored.out
    assert "Overall: close" in scored.out
    assert "No metric differs by 1 sd or more on average." in scored.out

    # A reference built by the flat form keeps its old default of no windowing.
    assert main([str(source), "--no-syntax", "--output", str(reference)]) == 0
    assert "--no-window" in capsys.readouterr().err
    assert json.loads(reference.read_text(encoding="utf-8"))["chunk_count"] == 1


def test_syntax_metrics_when_spacy_is_installed() -> None:
    pytest.importorskip("spacy")
    from styleprofile import SyntaxUnavailableError, load_parser

    try:
        parser = load_parser()
    except SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")
    report = build_profile([Chunk("a", "src", AUTHOR)], parser=parser)
    chunk = report["chunks"][0]["metrics"]

    assert chunk["syntax"]["parse_depth_mean"] > 0
    assert chunk["sentence_openers"]["opens_pronoun_pct"] > 0
    assert "pos_trigram" in report["distributions"]
    assert report["settings"]["syntax"]["model"] == "en_core_web_sm"


def test_display_labels_and_units() -> None:
    assert label("em_dashes_per_1k") == ("Em dashes", "/1k")
    assert label("fw_however_per_1k") == ('"however"', "/1k")
    assert label("new_metric") == ("New metric", "")
    assert value(45.567, "/1k") == "45.6 /1k"
    assert value(3.33, "%") == "3.3%"
    assert value(25.4, "%") == "25%"
    assert value(19.03, "words") == "19.0 words"
    assert value(0.4812, "") == "0.48"
    assert value(None, "%") == "-"
    assert [describe_delta(d) for d in (0.75, 1.2, 1.74, 6.9)] == [
        "close",
        "somewhat different",
        "clearly different",
        "very different",
    ]


def test_profile_and_comparison_views_are_readable() -> None:
    from styleprofile import format_summary

    author = [
        Chunk("a1", "src", AUTHOR),
        Chunk("a2", "src", "I don't know. But we tried it anyway, and it mostly worked."),
        Chunk("a3", "src", "You can't plan everything. We shipped it, and I think that was right."),
    ]
    reference = build_profile(author, parser=None)
    brief = format_summary(reference)

    assert brief.startswith("STYLE PROFILE   3 chunks")
    assert "usual range" in brief
    assert "Average sentence length" in brief
    assert '"the"' not in brief
    assert "Pass --all to see every metric." in brief
    assert "\033[" not in brief

    full = format_summary(reference, full=True)
    assert '"the"' in full
    assert "--all" not in full

    scored = build_profile([Chunk("g", "src", GENERIC)], parser=None, reference=reference)
    comparison = format_summary(scored, reference, color=True)

    assert "Overall: " in comparison
    assert "Biggest differences" in comparison
    assert "All metrics" not in comparison
    assert "\033[1m" in comparison
    assert "All metrics" in format_summary(scored, reference, full=True)


def test_distance_colors_only_mark_what_is_far(monkeypatch: pytest.MonkeyPatch) -> None:
    from styleprofile import format_summary
    from styleprofile.display import delta_level, z_level

    assert [delta_level(d) for d in (0.75, 1.2, 1.74, 6.9)] == [0, 1, 2, 3]
    assert [z_level(z) for z in (0.4, -1.5, 2.2, -8.0)] == [0, 1, 2, 3]

    reference = build_profile(
        [
            Chunk("a1", "src", AUTHOR),
            Chunk("a2", "src", "I don't know. But we tried it anyway, and it mostly worked."),
            Chunk(
                "a3", "src", "You can't plan everything. We shipped it, and I think it was right."
            ),
        ],
        parser=None,
    )
    scored = build_profile([Chunk("g", "src", GENERIC)], parser=None, reference=reference)

    monkeypatch.setenv("COLORTERM", "truecolor")
    deep = "\033[1;38;2;184;70;26m"
    truecolor = format_summary(scored, reference, color=True)
    assert deep in truecolor
    assert "\033[31m" not in truecolor and "\033[32m" not in truecolor

    monkeypatch.delenv("COLORTERM")
    assert "\033[1;38;5;130m" in format_summary(scored, reference, color=True)

    report, handmade_reference = _handmade_comparison()
    close = format_summary(report, handmade_reference, color=True)
    head, differences = close.split("Biggest differences")
    assert "Overall: \033[0m\033[1mclose" in head
    assert "38;" not in head
    assert "LLM marker words" in differences
    assert "reference never varies" in differences
    assert "\033[1;38;5;130m▲▲▲" in differences
    assert "Hedges" not in differences


def _handmade_comparison() -> tuple[dict[str, Any], dict[str, Any]]:
    """A close sample with one metric the reference never varies on."""
    reference = {
        "chunk_count": 5,
        "summary": {
            "voice": {
                "llm_markers_per_1k": {"mean": 0.0, "sd": 0.0},
                "hedges_per_1k": {"mean": 2.1, "sd": 1.0},
            }
        },
    }
    report = {
        "chunk_count": 1,
        "word_count": 400,
        "settings": {"syntax": None},
        "summary": {"voice": {"llm_markers_per_1k": {"mean": 5.0}, "hedges_per_1k": {"mean": 2.0}}},
        "chunks": [
            {
                "id": "c1",
                "reference": {
                    "delta": 0.5,
                    "z": {"voice": {"llm_markers_per_1k": 3.0, "hedges_per_1k": -0.1}},
                },
            }
        ],
        "reference": {
            "path": "ref.json",
            "delta_mean": 0.5,
            "delta_by_group_mean": {"voice": 0.6},
            "pooled_divergence": {},
        },
        "warnings": [],
    }
    return report, reference


def _paragraph(count: int) -> str:
    return " ".join(["word"] * (count - 1)) + " end."


def test_window_respects_the_size_bound_and_keeps_code_whole() -> None:
    text = "\n\n".join(_paragraph(n) for n in (499, 400, 249))
    sizes = [len(piece.text.split()) for piece in window([Chunk("p", "s", text)], 500)]
    assert sizes == [499, 649]

    def sizes_for(*paragraphs: int) -> list[int]:
        text = "\n\n".join(_paragraph(n) for n in paragraphs)
        return [len(piece.text.split()) for piece in window([Chunk("p", "s", text)], 500)]

    assert sizes_for(200, 600) == [800]
    assert sizes_for(740, 200) == [940]

    listed = "- item one\n- item two\n\n    " + _paragraph(600) + "\n\n" + _paragraph(100)
    first = window([Chunk("p", "s", listed)], 300)[0]
    assert first.text.startswith("- item one") and "    word" in first.text

    fenced = _paragraph(90) + "\n\n```python\nx = compute(a, b)\n\ny = compute(c, d)\n```\n\n"
    fenced += _paragraph(90)
    pieces = window([Chunk("p", "s", fenced)], 100)
    assert all(piece.text.count("```") in (0, 2) for piece in pieces)
    assert "compute" not in " ".join(prose(piece.text).text for piece in pieces)


def test_prose_drops_front_matter_and_every_kind_of_code() -> None:
    parsed = prose(
        "---\ntitle: Post\ndate: 2024-01-01\n---\nIntro line.\n\n"
        "    def main():\n        return 1\n\n"
        "1. Install it:\n\n   ```bash\n   pip install thing\n\n   thing --run\n   ```\n\n"
        "See ![a diagram](d.png) and [a link](x).\n\n```python\nunclosed = True\n\nmore = 1"
    )
    assert parsed.blocks == ["Intro line.", "Install it:", "See a diagram and a link."]
    assert parsed.links == 1


def test_sentence_and_voice_edge_cases() -> None:
    assert len(sentences("The answer was no. We moved on. Apples, pears, etc. Then we left.")) == 4

    apostrophe = "\N{RIGHT SINGLE QUOTATION MARK}"
    curly = surface_metrics(
        f"It{apostrophe}s worth noting that I{apostrophe}m not sure. It's fine."
    )
    straight = surface_metrics("It's worth noting that I'm not sure. It's fine.")
    assert curly["voice"] == straight["voice"]
    assert (curly["voice"]["llm_markers_per_1k"] or 0) > 0

    possessive = surface_metrics("The author's book and John's car were here.")
    assert possessive["voice"]["contractions_per_1k"] == 0


def test_chunks_without_prose_are_skipped() -> None:
    assert surface_metrics("```\ncode only\n```")["punctuation"]["commas_per_1k"] is None

    report = build_profile(
        [Chunk("code", "s", "```\nx = 1\n```"), Chunk("text", "s", AUTHOR)], parser=None
    )
    assert [row["id"] for row in report["chunks"]] == ["text"]
    assert any("skipped 1 chunk(s) with no prose" in warning for warning in report["warnings"])

    with pytest.raises(StyleProfileError, match="1 had no prose"):
        build_profile([Chunk("code", "s", "| a | b |")], parser=None)


def test_never_varying_metrics_count_toward_delta() -> None:
    reference = build_profile(
        [
            Chunk("a1", "s", AUTHOR),
            Chunk("a2", "s", "I don't know. But we tried it anyway, and it mostly worked."),
            Chunk("a3", "s", "You can't plan everything. We shipped it, and I think it was right."),
        ],
        parser=None,
    )
    plain = build_profile([Chunk("a", "s", AUTHOR)], parser=None, reference=reference)
    marked = build_profile(
        [Chunk("a", "s", AUTHOR + " Moreover, this is crucial.")], parser=None, reference=reference
    )
    plain_z = plain["chunks"][0]["reference"]["z"]["voice"]
    marked_z = marked["chunks"][0]["reference"]["z"]["voice"]

    assert plain_z["llm_markers_per_1k"] == 0
    # The reference never uses them, so the spread is the resolution floor (half of one
    # occurrence per chunk) and the score is graded rather than a fixed constant.
    assert marked_z["llm_markers_per_1k"] > 1
    assert marked["reference"]["delta_mean"] > plain["reference"]["delta_mean"]


def test_reference_syntax_mismatch_is_warned() -> None:
    reference = build_profile([Chunk("a", "s", AUTHOR)] * 2, parser=None)
    reference["settings"]["syntax"] = {"model": "en_core_web_sm", "model_version": "3.8.0"}
    report = build_profile([Chunk("a", "s", AUTHOR)], parser=None, reference=reference)

    assert any("this run does not" in warning for warning in report["warnings"])


def test_style_profile_cli_rejects_non_positive_top_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "essay.md"
    source.write_text(AUTHOR, encoding="utf-8")
    command = ["build", str(source), "--no-syntax", "--top-k", "0", "-o", str(tmp_path / "o.json")]

    assert main(command) == 1
    assert "--top-k must be positive" in capsys.readouterr().err


def test_rules_inline_fences_quotes_and_number_abbreviations() -> None:
    rule = prose("---\n\nIntro para.\n\n---\n\nBody para.")
    assert "Intro para." in rule.blocks and "Body para." in rule.blocks
    assert prose("---\ntitle: Post\n---\nBody.").blocks == ["Body."]

    assert prose("```npm i``` is all you need.\n\nThe rest.").blocks == [
        "is all you need.",
        "The rest.",
    ]
    assert prose("> First para here.\n>\n> Second para here.").paragraphs == [
        "First para here.",
        "Second para here.",
    ]
    after_heading = prose("- item one\n- item two\n\n## Heading\n\n    x = compute(a, b)")
    assert after_heading.blocks == ["item one", "item two"]

    assert sentences("See No. 5 in the list. The answer was no. We moved on.") == [
        "See No. 5 in the list.",
        "The answer was no.",
        "We moved on.",
    ]


def test_min_words_and_rounded_references() -> None:
    report = build_profile(
        [Chunk("short", "s", "Too short."), Chunk("long", "s", AUTHOR)], parser=None, min_words=10
    )
    assert [row["id"] for row in report["chunks"]] == ["long"]
    assert any("fewer than 10 prose words" in warning for warning in report["warnings"])

    third = 100 / 3
    reference = {
        "summary": {"sentence_shape": {"one_sentence_paragraphs_pct": {
            "n": 5, "mean": round(third, 6), "sd": 0.0}}},
        "distributions": {},
    }  # fmt: skip
    metrics: Metrics = {"sentence_shape": {"one_sentence_paragraphs_pct": third}}
    z = _score(metrics, {}, reference)["z"]["sentence_shape"]["one_sentence_paragraphs_pct"]
    assert z == 0


def test_all_metrics_table_marks_never_varying_metrics() -> None:
    from styleprofile import format_summary

    report, reference = _handmade_comparison()
    full = format_summary(report, reference, full=True)
    llm_row = next(line for line in full.split("All metrics")[1].splitlines() if "LLM" in line)
    assert llm_row.endswith("▲▲▲")


def test_front_matter_indented_blocks_and_unclosed_fences() -> None:
    unclosed = "---\ntitle: x\n" + "a b c d e f g h\n" * 40 + "\nBody."
    assert "Body." in prose(unclosed).text

    front = "---\ntitle: My post\nauthor: Jane\n\ndescription: x\n---\n\nBody."
    assert prose(front).blocks == ["Body."]
    assert prose("\tFirst paragraph here.\n\n\tSecond one, too.").blocks == [
        "First paragraph here.",
        "Second one, too.",
    ]
    assert prose("Text.\n\n    x = compute(a, b)\n    return x").blocks == ["Text."]
    assert prose("One.\n\n~~~\n\nTwo.\n\nThree.").blocks == ["One.", "Two.", "Three."]
    assert prose("Intro.\n\n```python\nx = 1\n\ny = 2").blocks == ["Intro."]
    assert sentences("I said no. 5 people came. See No. 5 here.") == [
        "I said no.",
        "5 people came.",
        "See No. 5 here.",
    ]


def test_delta_weights_areas_equally_and_noisy_metrics_less() -> None:
    reference = {
        "version": 1,
        "chunk_count": 5,
        "summary": {
            "punctuation": {
                "semicolons_per_1k": {"n": 5, "mean": 0.1, "sd": 0.01},
                "commas_per_1k": {"n": 5, "mean": 40.0, "sd": 10.0},
            },
            "voice": {"hedges_per_1k": {"n": 5, "mean": 2.0, "sd": 1.0}},
        },
        "reliability": {"punctuation": {"semicolons_per_1k": 10.0, "commas_per_1k": 1.0}},
        "distributions": {},
    }
    metrics: Metrics = {
        "punctuation": {"semicolons_per_1k": 0.2, "commas_per_1k": 40.0},
        "voice": {"hedges_per_1k": 3.0},
    }
    scored = _score(metrics, {}, reference)

    # Semicolons: z = 10 at weight 1/100; commas: z = 0 at weight 1 -> 0.1 / 1.01.
    assert scored["delta_by_group"]["punctuation"] == pytest.approx(0.1 / 1.01)
    assert scored["delta_by_group"]["voice"] == pytest.approx(1.0)
    assert scored["delta"] == pytest.approx((0.1 / 1.01 + 1.0) / 2)
    assert "likeness" not in scored

    report = build_profile([Chunk("a", "s", AUTHOR)], parser=None, reference=reference)
    assert any("report version 1" in warning for warning in report["warnings"])


def test_never_varying_differences_do_not_cancel() -> None:
    from styleprofile import format_summary

    report, reference = _handmade_comparison()
    report["chunk_count"] = 2
    report["summary"]["voice"]["llm_markers_per_1k"]["mean"] = 0.0
    report["chunks"] = [
        {"id": "up", "reference": {"delta": 0.5, "z": {"voice": {"llm_markers_per_1k": 3.0}}}},
        {"id": "down", "reference": {"delta": 0.5, "z": {"voice": {"llm_markers_per_1k": -3.0}}}},
    ]
    differences = format_summary(report, reference).split("Biggest differences")[1]

    assert "reference never varies; 2 of 2 chunks differ" in differences


def test_held_out_z_matches_brute_force() -> None:
    import statistics

    from styleprofile.weighting import held_out_z

    values = [1.0, 2.0, 4.0, 3.0, 8.0]
    sources = ["a", "a", "b", "c", "c"]
    metrics: list[Metrics] = [{"punctuation": {"commas_per_1k": v}} for v in values]
    held = held_out_z(metrics, sources, {})
    for index, (own, source) in enumerate(zip(values, sources, strict=True)):
        others = [v for v, s in zip(values, sources, strict=True) if s != source]
        expected = (own - statistics.fmean(others)) / statistics.stdev(others)
        assert held[index][("punctuation", "commas_per_1k")] == pytest.approx(expected)


def test_likeness_counts_only_the_contrast_direction_with_squared_weights() -> None:
    from styleprofile.weighting import delta_weights, likeness

    effects = {("p", "em"): 2.0, ("s", "length"): -1.0}
    toward, signals = likeness({("p", "em"): 1.0, ("s", "length"): 1.0}, effects, {})
    assert toward == pytest.approx(4 / 5)
    assert [signal["metric"] for signal in signals] == ["p.em"]
    assert likeness({("p", "em"): -2.0, ("s", "length"): 3.0}, effects, {})[0] == 0
    # Only metrics the sample has count: here just the length metric, fully in the direction.
    assert likeness({("s", "length"): -2.0}, effects, {})[0] == pytest.approx(2.0)
    # A metric that swings in the reference's own writing counts in its held-out units.
    wild = likeness({("p", "em"): 400.0}, effects, {("p", "em"): 100.0})[0]
    assert wild == pytest.approx(400 / 100)
    assert delta_weights({("p", "em"): 0.5, ("p", "semi"): 4.0}) == {
        ("p", "em"): 1.0,
        ("p", "semi"): 1 / 16,
    }


def _score(metrics: Metrics, distributions: dict[str, Any], reference: dict[str, Any]) -> Any:
    return _score_with(metrics, distributions, reference, _prepare(reference))


def _author_docs() -> list[Chunk]:
    return [
        Chunk("d1", "s", AUTHOR),
        Chunk(
            "d2", "s", "I don't know. But we tried it anyway, and it mostly worked (for a while)."
        ),
        Chunk("d3", "s", "You can't plan everything. We shipped it, and I think that was right."),
        Chunk("d4", "s", "It's a small thing. So I wrote it down, because I'll forget otherwise."),
    ]


def _llm_docs() -> list[Chunk]:
    return [
        Chunk("g1", "s", GENERIC),
        Chunk(
            "g2",
            "s",
            "Moreover, robust teams leverage crucial insights. This is not just speed, "
            "but a transformation. Additionally, stakeholders must foster alignment.",
        ),
    ]


def test_contrast_reference_learns_weights_and_scores_likeness() -> None:
    reference = build_profile(_author_docs(), parser=None, contrast=_llm_docs())

    assert reference["calibration"]["sources"] == 4
    contrast = reference["contrast"]
    assert contrast["label"] == "LLM" and contrast["sources"] == 2
    assert contrast["effects"]["voice"]["llm_markers_per_1k"] > 0
    assert contrast["calibration"]["cross_validated"] is True

    author = build_profile([Chunk("a", "s", AUTHOR)], parser=None, reference=reference)
    generic = build_profile([Chunk("g", "s", GENERIC)], parser=None, reference=reference)
    assert generic["reference"]["likeness_mean"] > author["reference"]["likeness_mean"]
    assert generic["chunks"][0]["reference"]["likeness_signals"]

    with pytest.raises(StyleProfileError, match="builds a reference"):
        build_profile(_author_docs(), parser=None, reference=reference, contrast=_llm_docs())
    with pytest.raises(StyleProfileError, match="at least two documents"):
        build_profile([Chunk("one", "s", AUTHOR)], parser=None, contrast=_llm_docs())


def test_contrast_views_show_likeness(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    author_dir = tmp_path / "author"
    author_dir.mkdir()
    for chunk in _author_docs():
        (author_dir / f"{chunk.id}.md").write_text(chunk.text, encoding="utf-8")
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    for chunk in _llm_docs():
        (llm_dir / f"{chunk.id}.md").write_text(chunk.text, encoding="utf-8")
    reference = tmp_path / "reference.json"
    sample = tmp_path / "sample.md"
    sample.write_text(GENERIC, encoding="utf-8")

    build = ["build", str(author_dir), "--contrast", str(llm_dir), "-o", str(reference)]
    assert main([*build, "--no-syntax"]) == 0
    assert "Contrast: LLM drafts" in capsys.readouterr().out

    score = ["score", str(sample), str(reference), "-o", str(tmp_path / "o.json")]
    assert main([*score, "--no-syntax"]) == 0
    output = capsys.readouterr().out
    assert "LLM-likeness: " in output
    assert "The reference's own writing scores" in output


def test_weighting_without_calibration_and_level_thresholds() -> None:
    from styleprofile.display import delta_level, likeness_level
    from styleprofile.profile import document_of
    from styleprofile.weighting import _auc

    # Two single-chunk documents: no held-out spread, so no calibration and no crash.
    report = build_profile(
        [Chunk("a.md", "x/a.md", AUTHOR), Chunk("b.md", "x/b.md", GENERIC)], parser=None
    )
    assert "calibration" not in report and "reliability" not in report

    # Scoring against a reference without reliability falls back to a capped Delta.
    reference = build_profile([Chunk("one", "s", AUTHOR)] * 3, parser=None)
    scored = build_profile([Chunk("g", "s", GENERIC)], parser=None, reference=reference)
    assert any("caps each metric at 3" in warning for warning in scored["warnings"])
    assert (scored["reference"]["delta_mean"] or 0) <= 3

    assert document_of("2023/index.md", "index.md") != document_of("2024/index.md", "index.md")
    assert document_of("p.jsonl", "post#w2") == document_of("p.jsonl", "post#w7")

    assert delta_level(1.0, ceiling=1.2) == 0
    assert delta_level(2.0, ceiling=1.2) == 2
    weak = {"reference": {"p95": 0.9}, "contrast": {"median": 0.7}}
    assert likeness_level(5.0, weak) == 1

    assert _auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert _auc([1.0], [1.0]) == 0.5
    assert _auc([2.0, 0.0], [1.0]) == 0.5


def test_near_zero_held_out_range_does_not_inflate_verdicts() -> None:
    from styleprofile.display import delta_level, likeness_level, mean_ceiling

    # A writer who never uses Markdown: the formatting area barely varies on held-out text.
    formatting = {"median": 0.006, "p95": 0.024}
    assert delta_level(0.07, mean_ceiling(formatting, 1)) == 0
    assert delta_level(0.4, mean_ceiling(formatting, 16)) == 0
    assert delta_level(0.9, mean_ceiling(formatting, 1)) == 2
    assert delta_level(3.0, mean_ceiling(formatting, 1)) == 3
    assert mean_ceiling({"median": 0.0, "p95": 0.0}, 1) == 0.5
    # A calibrated range above the floor is unchanged.
    assert mean_ceiling({"median": 0.78, "p95": 0.99}, 1) == 0.99

    drafts = {"contrast": {"median": 12.0}}
    quiet = {"reference": {"median": 0.001, "p95": 0.01}, **drafts}
    assert likeness_level(0.07, quiet) == 0
    assert likeness_level(0.07, quiet, count=9) == 0
    assert likeness_level(0.3, quiet) == 1
    assert likeness_level(4.0, quiet) == 1
    assert likeness_level(10.0, quiet) == 2
    assert likeness_level(0.3, {"reference": {"median": 0.0, "p95": 0.0}, **drafts}) == 1
    # Likeness counts only the part of each z toward the contrast set, so its floor is half
    # the Delta one: a long sample averaging above the writer's own range still shows traits.
    usual = {"reference": {"median": 0.2, "p95": 0.86}, **drafts}
    assert likeness_level(0.4, usual, count=64) == 1
    assert likeness_level(0.4, usual, count=1) == 0


def test_rare_habit_cannot_dominate_delta() -> None:
    """One chunk in ten documents uses semicolons; a sample full of them stays bounded."""
    base = "We built it slowly and carefully over many months. " * 12
    docs = [Chunk(f"d{i}#w{j}", f"doc{i}", base) for i in range(10) for j in range(5)]
    docs[0] = Chunk("d0#w0", "doc0", base.replace("months.", "months; really.", 1))
    reference = build_profile(docs, parser=None)
    sample = build_profile(
        [Chunk("s", "s", base.replace("months.", "months; really.", 5))],
        parser=None,
        reference=reference,
    )
    scored = sample["chunks"][0]["reference"]
    # Five semicolons in ~120 words, against a reference that used one once: clearly
    # unusual, but measured against half an occurrence per chunk rather than a near-zero sd.
    assert 3 < scored["z"]["punctuation"]["semicolons_per_1k"] < 15
    assert scored["delta_by_group"]["punctuation"] < 2

    from styleprofile.profile import document_of

    assert document_of("f.jsonl", "faq#what") != document_of("f.jsonl", "faq#why")


def test_list_continuations_percentage_floors_and_upper_case_extensions(tmp_path: Path) -> None:
    from styleprofile.weighting import resolution

    # Indented text under a list item extends the item; it is not a paragraph.
    listed = prose("- Point one.\n\n    More detail on point one. And more.")
    assert listed.blocks == ["Point one. More detail on point one. And more."]
    assert listed.paragraphs == []

    # An intro line directly above a list stays before the items.
    assert prose("Here are the steps I use:\n- First do a.\n- Then do b.").blocks == [
        "Here are the steps I use:",
        "First do a.",
        "Then do b.",
    ]

    # Each percentage's floor comes from what it is a share of.
    counts = {"words": 480.0, "sentences": 25.0, "paragraphs": 6.0, "apostrophes": 10.0}
    assert resolution("one_sentence_paragraphs_pct", counts) == pytest.approx(50 / 6)
    assert resolution("curly_apostrophe_pct", counts) == pytest.approx(5.0)
    assert resolution("short_sentences_pct", counts) == pytest.approx(2.0)
    assert resolution("commas_per_1k", counts) == pytest.approx(500 / 480)
    assert surface_metrics("It's here, isn't it.")["size"]["apostrophes"] == 2

    # Upper-case extensions are read, in directories and as JSONL.
    (tmp_path / "README.MD").write_text(AUTHOR, encoding="utf-8")
    (tmp_path / "notes.TXT").write_text(GENERIC, encoding="utf-8")
    assert [chunk.id for chunk in load_chunks([str(tmp_path)])] == ["README.MD", "notes.TXT"]
    records = tmp_path / "posts.JSONL"
    records.write_text(json.dumps({"id": "p", "text": AUTHOR}) + "\n", encoding="utf-8")
    assert [chunk.id for chunk in load_chunks([str(records)])] == ["p"]


def _words(count: int) -> str:
    return " ".join(["word"] * (count - 1)) + " end."


def test_contrast_auc_has_a_document_bootstrap_interval() -> None:
    first = build_profile(_author_docs(), parser=None, contrast=_llm_docs())
    second = build_profile(_author_docs(), parser=None, contrast=_llm_docs())
    calibration = first["contrast"]["calibration"]

    low, high = calibration["auc_ci"]
    assert low <= calibration["auc"] <= high
    assert calibration["bootstrap"] == {"resamples": 2000, "unit": "document"}
    assert second["contrast"]["calibration"]["auc_ci"] == [low, high]

    single = build_profile(_author_docs(), parser=None, contrast=_llm_docs()[:1])
    assert single["contrast"]["calibration"]["auc_ci"] is None
    assert any("no confidence interval" in warning for warning in single["warnings"])


def test_bootstrap_resamples_whole_documents() -> None:
    from styleprofile.weighting import _auc, bootstrap_auc

    # One contrast document contributes 100 high chunks, the other a single low one. By
    # chunk, the AUC would stay near 0.99; by document, drawing the low one twice gives 0.
    reference = [[0.0], [1.0]]
    contrast = [[2.0] * 100, [-1.0]]
    aucs = bootstrap_auc(reference, contrast, resamples=400)
    possible = set()
    for draws in ([0, 0], [0, 1], [1, 1]):
        for ref_draws in ([0, 0], [0, 1], [1, 1]):
            positives = [s for d in draws for s in contrast[d]]
            negatives = [s for d in ref_draws for s in reference[d]]
            possible.add(round(_auc(positives, negatives) or 0.0, 9))
    assert {round(auc, 9) for auc in aucs} <= possible
    assert 0.0 in aucs and 1.0 in aucs
    assert bootstrap_auc(reference, contrast, resamples=50) == aucs[:50]

    # With every document drawn once, the reweighted pass equals the plain rank AUC.
    uneven = [[0.3, 0.5, 0.5], [0.1], [0.9, 0.2]]
    drafts = [[0.5, 0.8], [0.4]]
    flat = [s for scores in uneven for s in scores]
    flat_drafts = [s for scores in drafts for s in scores]
    assert _auc(flat_drafts, flat) in bootstrap_auc(uneven, drafts, resamples=200)


def test_length_baseline_detects_length_differences() -> None:
    from styleprofile.weighting import length_baseline

    matched = length_baseline([100, 120, 140, 160], [110, 130, 150, 170])
    assert matched is not None and matched["auc"] <= 0.65

    short = length_baseline([400, 420, 440], [100, 120, 140])
    assert short == {
        "auc": 1.0,
        "direction": "contrast shorter",
        "reference_median_words": 420,
        "contrast_median_words": 120,
    }

    reference = [Chunk(f"r{i}", "s", _words(30 + i)) for i in range(4)]
    long_drafts = [Chunk(f"l{i}", "s", GENERIC + " " + _words(200 + i)) for i in range(3)]
    report = build_profile(
        reference, parser=None, contrast=long_drafts, contrast_label="Editor", min_words=1
    )
    length = report["contrast"]["calibration"]["length_baseline"]
    assert length["auc"] >= 0.75 and length["direction"] == "contrast longer"
    assert any("Editor-likeness may partly reflect length" in w for w in report["warnings"])

    # Word counts 43, 45, 47, 49 on both sides.
    reference = [Chunk(f"r{i}", "s", _words(40 + 2 * i) + " I think so.") for i in range(4)]
    matched_drafts = [
        Chunk(f"m{i}", "s", _words(41 + 2 * i) + " Moreover, robust.") for i in range(4)
    ]
    report = build_profile(reference, parser=None, contrast=matched_drafts, min_words=1)
    assert report["contrast"]["calibration"]["length_baseline"]["auc"] == 0.5
    assert not any("differs strongly in length" in warning for warning in report["warnings"])


def test_contrast_summary_shows_interval_and_length_baseline() -> None:
    from styleprofile.display import _contrast_summary, _Style

    contrast = {
        "label": "LLM",
        "chunk_count": 40,
        "sources": 8,
        "effects": {"voice": {"llm_markers_per_1k": 2.0}},
        "calibration": {
            "reference": {"median": 0.1, "p95": 0.4, "max": 0.6},
            "contrast": {"median": 0.7, "min": 0.3},
            "auc": 0.96,
            "auc_ci": [0.9, 1.0],
            "length_baseline": {
                "auc": 0.54,
                "direction": "contrast longer",
                "reference_median_words": 525,
                "contrast_median_words": 526,
            },
        },
    }
    text = "\n".join(_contrast_summary(contrast, _Style(False)))
    assert "AUC 0.96 (95% CI 0.90–1.00, resampling whole documents)" in text  # noqa: RUF001
    assert (
        "Length alone: AUC 0.54 (reference 525 words per chunk, LLM drafts 526): "
        "not a length effect"
    ) in text


def test_output_cannot_overwrite_the_reference(tmp_path: Path, capsys: Any) -> None:
    source = tmp_path / "a.md"
    source.write_text(AUTHOR, encoding="utf-8")
    reference = tmp_path / "ref.json"
    assert main(["build", str(source), "--no-syntax", "-o", str(reference)]) == 0
    saved = reference.read_text(encoding="utf-8")
    capsys.readouterr()

    same = str(tmp_path / "." / "ref.json")
    assert main(["score", str(source), str(reference), "--no-syntax", "-o", same]) == 1
    assert "--output is the reference file" in capsys.readouterr().err
    assert reference.read_text(encoding="utf-8") == saved


def test_stdin_can_be_read_only_once(tmp_path: Path, capsys: Any) -> None:
    assert main(["build", "-", "-", "--no-syntax", "-o", str(tmp_path / "out.json")]) == 1
    assert "can be given only once" in capsys.readouterr().err


def test_indented_prose_with_code_characters_is_kept() -> None:
    assert prose("\tIt was late; the train had gone.\n\n\tThe next morning was worse.").blocks == [
        "It was late; the train had gone.",
        "The next morning was worse.",
    ]
    wrapped = "    We waited {a while} at the gate, and\n    for the rest of the day it rained."
    assert prose(wrapped).blocks == [
        "We waited {a while} at the gate, and for the rest of the day it rained."
    ]
    assert prose("Text.\n\n    if (ready) { start(); }").blocks == ["Text."]
    assert prose('Text.\n\n    greeting = "Hello there."').blocks == ["Text."]
    # A sentence-like comment does not make code read as prose.
    commented = "Text.\n\n    x = 1  # This is where we set the initial value for the counter."
    assert prose(commented).blocks == ["Text."]
    assert prose("Text.\n\n    import os\n    // Now we are done with the setup here.").blocks == [
        "Text."
    ]


def test_unreadable_encodings_are_reported_and_a_bom_is_accepted(
    tmp_path: Path, capsys: Any
) -> None:
    binary = tmp_path / "image.md"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00")
    with pytest.raises(StyleProfileError, match="is not UTF-8 text"):
        load_chunks([str(binary)])
    records = tmp_path / "latin.jsonl"
    records.write_bytes(b'{"text": "caf\xe9"}\n')
    with pytest.raises(StyleProfileError, match="is not UTF-8 text"):
        load_chunks([str(records)])
    assert main(["build", str(binary), "--no-syntax", "-o", str(tmp_path / "o.json")]) == 1
    assert "is not UTF-8 text" in capsys.readouterr().err

    marked = tmp_path / "bom.jsonl"
    marked.write_bytes(b"\xef\xbb\xbf" + json.dumps({"id": "b", "text": AUTHOR}).encode() + b"\n")
    assert [chunk.id for chunk in load_chunks([str(marked)])] == ["b"]
    text = tmp_path / "bom.md"
    text.write_bytes(b"\xef\xbb\xbfHello there.")
    assert load_chunks([str(text)])[0].text == "Hello there."


def test_directories_read_jsonl_and_skip_hidden_and_vendored_folders(tmp_path: Path) -> None:
    posts = tmp_path / "posts"
    posts.mkdir()
    (posts / "all.jsonl").write_text(json.dumps({"id": "p1", "text": AUTHOR}) + "\n", "utf-8")
    assert [chunk.id for chunk in load_chunks([str(posts)])] == ["p1"]

    for hidden in (".git", "node_modules", ".venv/lib"):
        (posts / hidden).mkdir(parents=True)
        (posts / hidden / "README.md").write_text(GENERIC, encoding="utf-8")
    (posts / "notes.md").write_text(AUTHOR, encoding="utf-8")
    assert [chunk.id for chunk in load_chunks([str(posts)])] == ["p1", "notes.md"]

    empty = tmp_path / "empty"
    (empty / ".git").mkdir(parents=True)
    (empty / ".git" / "HEAD.md").write_text(AUTHOR, encoding="utf-8")
    with pytest.raises(StyleProfileError, match=r"\.txt, or \.jsonl files"):
        load_chunks([str(empty)])


def test_code_only_chunks_are_counted_after_windowing() -> None:
    chunks = [Chunk("code", "s", "```python\nx = 1\n```"), Chunk("text", "s", AUTHOR)]
    windows = window(chunks, 20)
    assert windows[0].id == "code#w1"
    report = build_profile(windows, parser=None)
    assert any("skipped 1 chunk(s) with no prose" in warning for warning in report["warnings"])


def test_jsonl_ids_of_zero_are_kept(tmp_path: Path) -> None:
    records = tmp_path / "ids.jsonl"
    lines = [{"id": 0, "text": AUTHOR}, {"id": "", "text": AUTHOR}, {"id": None, "text": AUTHOR}]
    records.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    ids = [chunk.id for chunk in load_chunks([str(records)])]
    assert ids == ["0", "ids.jsonl:2", "ids.jsonl:3"]


def test_stdin_is_decoded_like_files(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"\xef\xbb\xbfHi.\r\nBye.")))
    assert load_chunks(["-"])[0].text == "Hi.\nBye."
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"caf\xe9")))
    with pytest.raises(StyleProfileError, match="stdin is not UTF-8 text"):
        load_chunks(["-"])


def _metric_keys(metrics: Metrics) -> list[tuple[str, str]]:
    return [(group, name) for group, values in metrics.items() for name in values]


def test_computed_metrics_match_the_registry() -> None:
    from styleprofile.metrics import METRICS, grouped

    surface = [(metric.group, metric.name) for metric in METRICS if not metric.syntax]
    assert _metric_keys(surface_metrics(AUTHOR + "\n\n" + MARKDOWN)) == surface
    assert _metric_keys(surface_metrics("")) == surface

    # A computed metric without a definition, or a definition without a computation, fails.
    values = {name: 1.0 for _, name in surface}
    with pytest.raises(ValueError, match="new_metric"):
        grouped({**values, "new_metric": 1.0}, syntax=False)
    del values["commas_per_1k"]
    with pytest.raises(ValueError, match="commas_per_1k"):
        grouped(values, syntax=False)

    pytest.importorskip("spacy")
    from styleprofile import SyntaxUnavailableError, load_parser

    try:
        parser = load_parser()
    except SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")
    (parsed, _), (empty, _) = parser.parse([AUTHOR, ""])
    syntax = [(metric.group, metric.name) for metric in METRICS if metric.syntax]
    assert _metric_keys(parsed) == syntax
    assert _metric_keys(empty) == syntax


def test_registry_definitions_are_complete() -> None:
    from styleprofile.metrics import (
        FUNCTION_WORDS,
        GROUPS,
        KEY_VIEW,
        METRIC_BY_NAME,
        METRICS,
        Count,
        Unit,
        describe,
    )

    assert len(METRIC_BY_NAME) == len(METRICS), "metric names must be unique"
    order = [group.name for group in GROUPS]
    groups = [metric.group for metric in METRICS]
    assert groups == sorted(groups, key=order.index), "metrics follow the group order"
    assert list(dict.fromkeys(groups)) == order, "every group has metrics"
    for metric in METRICS:
        assert metric.label and metric.about and isinstance(metric.unit, Unit), metric
        if metric.name.endswith("_per_1k"):
            assert (metric.unit, metric.share_of) == (Unit.PER_1K, Count.WORDS), metric
        if metric.name.endswith("_pct"):
            assert metric.unit == Unit.PCT and metric.share_of is not None, metric
    assert all(f"fw_{word}_per_1k" in METRIC_BY_NAME for word in FUNCTION_WORDS)
    for _, keys in KEY_VIEW:
        assert all(METRIC_BY_NAME[name].group == group for group, name in keys)

    rows = describe()
    assert len(rows) == len(METRICS)
    assert rows[0] == ("Size", "Words", "", METRIC_BY_NAME["words"].about)
    assert ("Punctuation", "Em dashes", "/1k") in [row[:3] for row in rows]
    assert len(describe(syntax=False)) == sum(not metric.syntax for metric in METRICS)


# Command line: build once, score many times.


def _write_docs(folder: Path, chunks: list[Chunk]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        (folder / f"{chunk.id}.md").write_text(chunk.text, encoding="utf-8")
    return folder


def _built(tmp_path: Path, *flags: str) -> Path:
    """A no-syntax reference built from four short documents with 100-word windows."""
    posts = _write_docs(tmp_path / "posts", _author_docs())
    reference = tmp_path / "writer.json"
    command = ["build", str(posts), "-o", str(reference), "--no-syntax", "--window-words", "100"]
    assert main([*command, *flags]) == 0
    return reference


def _sample(tmp_path: Path, text: str = GENERIC) -> Path:
    sample = tmp_path / "draft.md"
    sample.write_text(text, encoding="utf-8")
    return sample


def test_build_and_score_split_the_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = _built(tmp_path)
    sample = _sample(tmp_path)
    capsys.readouterr()
    before = sorted(tmp_path.iterdir())

    assert main(["score", str(sample), str(reference)]) == 0
    scored = capsys.readouterr()
    assert "STYLE COMPARISON" in scored.out
    assert "wrote" not in scored.out and not scored.err
    assert sorted(tmp_path.iterdir()) == before, "score writes no file without -o"

    output = tmp_path / "out" / "draft.json"
    # Options may sit between the sample and the reference.
    assert main(["score", str(sample), "-o", str(output), str(reference)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["kind"] == "score"
    # Inherited from the reference: 100-word windows, min words and no syntax.
    assert report["settings"]["window_words"] == 100
    assert report["settings"]["min_words"] == 1
    assert report["settings"]["syntax"] is None
    assert not any("window sizes differ" in warning for warning in report["warnings"])
    assert f"wrote {output}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flags", "note", "window_words", "min_words"),
    [
        ([], None, 100, 1),
        (["--window-words", "100"], None, 100, 1),
        (["--window-words", "50"], "window words 100 -> 50", 50, 1),
        (["--no-window"], "window words 100 -> off", None, 1),
        (["--window-words", "0"], "window words 100 -> off", None, 1),
        (["--min-words", "5"], "min words 1 -> 5", 100, 5),
    ],
)
def test_score_overrides_win_with_a_note(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    note: str | None,
    window_words: int | None,
    min_words: int,
) -> None:
    reference = _built(tmp_path)
    capsys.readouterr()
    output = tmp_path / "draft.json"

    command = ["score", str(_sample(tmp_path, AUTHOR)), str(reference), "-o", str(output)]
    assert main([*command, *flags]) == 0
    err = capsys.readouterr().err
    settings = json.loads(output.read_text(encoding="utf-8"))["settings"]
    assert (settings["window_words"], settings["min_words"]) == (window_words, min_words)
    if note is None:
        assert "overriding" not in err
    else:
        assert note in err and "may not be comparable" in err


def test_json_output_is_the_only_thing_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = _built(tmp_path)
    capsys.readouterr()
    output = tmp_path / "draft.json"
    sample = str(_sample(tmp_path))

    command = ["score", sample, str(reference), "--json", "-o", str(output), "--window-words", "50"]
    assert main(command) == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed == json.loads(output.read_text(encoding="utf-8"))
    assert printed["kind"] == "score" and printed["reference"]["delta_mean"] > 0
    assert "overriding" in captured.err and f"wrote {output}" in captured.err

    assert main(["score", sample, str(reference), "--quiet", "--window-words", "50"]) == 0
    quiet = capsys.readouterr()
    assert quiet.out.count("\n") == 1 and quiet.out.startswith(f"{sample}: ")
    assert "Delta" in quiet.out and not quiet.err


def test_score_reports_cannot_be_used_as_references(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = _built(tmp_path)
    scored = tmp_path / "draft.json"
    sample = str(_sample(tmp_path))
    assert main(["score", sample, str(reference), "-o", str(scored)]) == 0
    capsys.readouterr()

    assert main(["score", sample, str(scored)]) == 1
    err = capsys.readouterr().err
    assert "is a score report" in err and "hint: score against the reference profile" in err

    # Reports written before `kind` existed are recognized by their reference section.
    legacy = json.loads(scored.read_text(encoding="utf-8"))
    del legacy["kind"]
    scored.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(StyleProfileError, match="is a score report"):
        load_reference(scored)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["score", "{sample}"], "score needs a sample and then a reference profile"),
        (["score", "{reference}", "{sample}"], "the reference profile goes last"),
        (["score", "{sample}", "-"], "must be a file"),
        (["score", "{sample}", "missing.json"], "missing.json not found; it goes last"),
        (["score", "{sample}", "--reference", "{reference}"], "not --reference"),
        (["score", "{sample}", "{reference}", "-o", "{reference}"], "--output is the reference"),
        (["score", "-", "-", "{reference}"], "can be given only once"),
        (["build", "{sample}", "-o", "x.json", "--window-words", "-1"], "0 (no windowing)"),
        (["build", "{sample}", "--contrast", "{sample}", "-o", "x.json"], "at least two"),
    ],
)
def test_cli_errors_say_how_to_fix_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    reference = _built(tmp_path)
    sample = _sample(tmp_path)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    paths = {"sample": str(sample), "reference": str(reference)}

    assert main([argument.format(**paths) for argument in arguments]) == 1
    assert message in capsys.readouterr().err


def test_library_errors_do_not_name_flags() -> None:
    with pytest.raises(StyleProfileError) as error:
        build_reference([Chunk("one", "s", AUTHOR)], parser=None, contrast=_llm_docs())
    assert error.value.code == "contrast_needs_documents" and "--" not in str(error.value)


def test_repeatable_contrast_does_not_swallow_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    posts = _write_docs(tmp_path / "posts", _author_docs())
    first = _write_docs(tmp_path / "llm-a", _llm_docs()[:1])
    second = _write_docs(tmp_path / "llm-b", _llm_docs()[1:])
    reference = tmp_path / "writer.json"

    command = ["build", "--contrast", str(first), str(posts), "--contrast", str(second)]
    assert main([*command, "-o", str(reference), "--no-syntax", "--contrast-label", "GPT"]) == 0
    report = json.loads(reference.read_text(encoding="utf-8"))
    assert report["kind"] == "reference"
    assert report["settings"]["inputs"] == [str(posts)]
    assert report["settings"]["contrast"] == [str(first), str(second)]
    assert (report["contrast"]["label"], report["contrast"]["sources"]) == ("GPT", 2)
    assert "Contrast: GPT drafts" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("documents", "repeats", "window", "expected"),
    [
        (1, 1, "100", ["from 1 document", "chunks; aim for 15", "words; aim for 20,000"]),
        (4, 1, "0", ["by adding documents or using --window-words 500", "words; aim for"]),
        (16, 22, "0", []),
    ],
)
def test_build_warns_about_thin_references_and_names_the_next_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    documents: int,
    repeats: int,
    window: str,
    expected: list[str],
) -> None:
    text = "\n\n".join([AUTHOR] * repeats)
    _write_docs(tmp_path / "posts", [Chunk(f"d{i}", "s", text) for i in range(documents)])
    monkeypatch.chdir(tmp_path)

    command = ["build", "posts", "-o", "out/writer.json", "--no-syntax", "--window-words", window]
    assert main(command) == 0
    out = capsys.readouterr().out
    thin = [line for line in out.splitlines() if line.startswith("Thin reference:")]
    assert len(thin) == len(expected)
    assert all(any(phrase in line for line in thin) for phrase in expected)
    assert "\nwrote out/writer.json\n" in out
    assert out.rstrip().endswith("styleprofile score <draft> out/writer.json")


def test_missing_spacy_falls_back_with_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from styleprofile import SyntaxUnavailableError, cli

    def unavailable() -> None:
        raise SyntaxUnavailableError("no spaCy")

    monkeypatch.setattr(cli, "load_parser", unavailable)
    posts = _write_docs(tmp_path / "posts", _author_docs())
    reference = tmp_path / "writer.json"

    assert main(["build", str(posts), "-o", str(reference)]) == 0
    assert "surface metrics only" in capsys.readouterr().err
    report = json.loads(reference.read_text(encoding="utf-8"))
    assert report["settings"]["syntax"] is None

    # A reference with syntax: score follows it, and leaves syntax out when spaCy is missing.
    report["settings"]["syntax"] = {"model": "en_core_web_sm", "model_version": "3.8.0"}
    reference.write_text(json.dumps(report), encoding="utf-8")
    sample = str(_sample(tmp_path))
    assert main(["score", sample, str(reference)]) == 0
    captured = capsys.readouterr()
    assert "syntax is left out" in captured.err and "overriding" not in captured.err
    assert main(["score", sample, str(reference), "--no-syntax"]) == 0
    assert "syntax on -> off" in capsys.readouterr().err


def test_show_renders_saved_reports_without_recomputing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = _built(tmp_path)
    scored = tmp_path / "draft.json"
    assert main(["score", str(_sample(tmp_path)), str(reference), "-o", str(scored)]) == 0
    live = capsys.readouterr().out

    assert main(["show", str(reference)]) == 0
    assert capsys.readouterr().out.startswith("STYLE PROFILE")
    assert main(["show", str(scored), "--all"]) == 0
    shown = capsys.readouterr().out
    assert shown.startswith("STYLE COMPARISON") and "All metrics" in shown
    overall = next(line for line in live.splitlines() if line.startswith("Overall"))
    assert overall in shown

    # A score report from before baselines were saved falls back to its reference file.
    report = json.loads(scored.read_text(encoding="utf-8"))
    del report["reference"]["baseline"]
    scored.write_text(json.dumps(report), encoding="utf-8")
    assert main(["show", str(scored)]) == 0
    captured = capsys.readouterr()
    assert overall in captured.out and "no saved copy of its reference" in captured.err
    reference.unlink()
    assert main(["show", str(scored)]) == 1
    assert "score the sample again" in capsys.readouterr().err


def test_show_renders_a_score_against_an_uncalibrated_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # One document in several windows: a spread, but no held-out calibration.
    essay = tmp_path / "essay.md"
    essay.write_text("\n\n".join([AUTHOR, GENERIC] * 3), encoding="utf-8")
    reference = tmp_path / "writer.json"
    command = ["build", str(essay), "-o", str(reference), "--no-syntax", "--window-words", "100"]
    assert main(command) == 0
    scored = tmp_path / "draft.json"
    assert main(["score", str(_sample(tmp_path)), str(reference), "-o", str(scored)]) == 0
    live = capsys.readouterr().out
    assert "calibration" not in json.loads(reference.read_text(encoding="utf-8"))

    assert main(["show", str(scored)]) == 0
    shown = capsys.readouterr().out
    overall = next(line for line in live.splitlines() if line.startswith("Overall"))
    assert overall in shown


@pytest.mark.parametrize(
    ("flags", "syntax"), [([], None), (["--no-syntax"], False), (["--syntax"], True)]
)
def test_metrics_lists_the_registry_by_area(
    capsys: pytest.CaptureFixture[str], flags: list[str], syntax: bool | None
) -> None:
    from styleprofile.metrics import describe

    assert main(["metrics", *flags]) == 0
    out = capsys.readouterr().out
    rows = describe(syntax=syntax is not False)
    if syntax:
        rows = [row for row in rows if row not in describe(syntax=False)]
    assert out.startswith(f"{len(rows)} metrics.")
    assert all(row.about in out for row in rows)
    assert ("\nSyntax\n" in out) is (syntax is not False)
    assert ("\nPunctuation\n" in out) is (syntax is not True)


@pytest.mark.parametrize("command", ["build", "score", "show", "metrics"])
def test_each_command_has_short_help_with_an_example(
    capsys: pytest.CaptureFixture[str], command: str
) -> None:
    with pytest.raises(SystemExit) as exit_:
        main([command, "--help"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert f"example:\n  styleprofile {command}" in out
    assert len(out.splitlines()) < 40


def test_version_and_unknown_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from styleprofile import __version__
    from styleprofile.profile import VERSION

    with pytest.raises(SystemExit) as exit_:
        main(["--version"])
    assert exit_.value.code == 0
    assert (
        capsys.readouterr().out.strip() == f"styleprofile {__version__} (report schema {VERSION})"
    )
    with pytest.raises(SystemExit) as exit_:
        main(["scor", "draft.md"])
    assert exit_.value.code == 2
    assert "invalid choice: 'scor'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("environment", "tty", "expected"),
    [
        ({}, False, False),
        ({}, True, True),
        ({"NO_COLOR": "1"}, True, False),
        ({"FORCE_COLOR": "1"}, False, True),
        ({"FORCE_COLOR": "0"}, True, True),
        ({"FORCE_COLOR": "0"}, False, False),
        ({"FORCE_COLOR": "1", "NO_COLOR": "1"}, False, False),
    ],
)
def test_color_honors_no_color_and_force_color(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], tty: bool, expected: bool
) -> None:
    import sys

    from styleprofile import cli

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    for name, setting in environment.items():
        monkeypatch.setenv(name, setting)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
    assert cli._color() is expected


def test_build_profile_wraps_build_reference_and_score() -> None:
    from styleprofile import report_kind, score

    reference = build_reference(_author_docs(), parser=None)
    assert reference == build_profile(_author_docs(), parser=None)
    assert report_kind(reference) == "reference"
    sample = [Chunk("g", "s", GENERIC)]
    scored = score(sample, reference, parser=None)
    assert scored == build_profile(sample, parser=None, reference=reference)
    assert report_kind(scored) == "score"
    baseline = scored["reference"]["baseline"]
    assert baseline["summary"]["voice"]["contractions_per_1k"].keys() == {"mean", "sd"}
