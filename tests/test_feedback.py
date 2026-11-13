"""Personal feedback must preserve evidence, target separation and grouped holdouts."""

import json

import pytest

import styleprofile as sp
from styleprofile.cli import main
from styleprofile.feedback import FeedbackProfile, rhetoric


@pytest.fixture
def reference():
    writer = [
        sp.Text(
            (
                f"I tested this tool on project {i}. "
                "The measurements helped us find a specific defect in our worker queue. "
            )
            * 12,
            f"writer-{i}",
        )
        for i in range(4)
    ]
    drafts = [
        sp.Text(
            (
                "Imagine two teams. The first team needs a reliable way to inspect "
                "the changes before releasing their application to a larger audience. "
                f"This is the lesson from example {i}. "
            )
            * 10,
            f"draft-{i}",
        )
        for i in range(4)
    ]
    return sp.build(writer, sp.Settings(syntax=False, pool=False), contrast=drafts)


def examples():
    records = []
    for group in range(3):
        for variant in range(2):
            text = (
                "I ran the test yesterday. The measurements helped us find a specific "
                "defect in our worker queue before the application reached customers. "
            ) * 8
            if variant:
                text = (
                    "Picture two engineers. Each engineer needs a dependable way to "
                    "inspect every change before sending the new application to customers. "
                    "That is the point of the example. "
                ) * 8
            records.append(
                {
                    "id": f"{group}-{variant}",
                    "source_post_id": f"post-{group}-{variant}",
                    "topic_family": f"family-{group}",
                    "text": text,
                    "voice": 1 if variant else 5,
                    "revision": "light",
                    "notes": "The imaginary example sounds unlike me."
                    if variant
                    else "The specific observation sounds like me.",
                }
            )
    return records


def test_rhetoric_catches_reported_patterns_with_exact_evidence():
    text = (
        "There's a pattern I keep noticing. Picture two engineers with the same tools. "
        "Codebase shape. Sprawling legacy systems demand more context just to orient. "
        "That is the point of personalization."
    )
    report = rhetoric(text)
    assert report["counts"]["hypothetical_prompts"] == 1
    assert report["counts"]["observation_prefaces"] == 1
    assert report["counts"]["lesson_statements"] == 1
    assert report["counts"]["short_leadin_pairs"] >= 1
    assert all(report["text"][e["start"] : e["end"]] == e["text"] for e in report["evidence"])
    assert (
        rhetoric("There\N{RIGHT SINGLE QUOTATION MARK}s a pattern I keep noticing.")["counts"][
            "observation_prefaces"
        ]
        == 1
    )


def test_cues_do_not_match_ordinary_mentions_or_code_or_cross_paragraph_pairs():
    report = rhetoric(
        "The picture fell. I can imagine a queue. Consider error handling.\n\n"
        "```text\nPicture two engineers.\n```"
    )
    assert report["counts"]["hypothetical_prompts"] == 0
    assert (
        rhetoric(
            "A short line.\n\nA much longer independent paragraph with enough words "
            "to explain a completely different subject."
        )["counts"]["short_leadin_pairs"]
        == 0
    )
    with pytest.raises(sp.StyleProfileError, match="nonempty prose"):
        rhetoric("```python\nprint(1)\n```")


def test_learn_keeps_notes_and_groups_related_posts_out_together(reference):
    profile = FeedbackProfile.learn(examples(), reference)
    assert profile.report["examples"][0]["notes"] == examples()[0]["notes"]
    assert profile.report["diagnostics"]["groups"] == 3
    by_id = {r["id"]: r["topic_family"] for r in examples()}
    for fold in profile.report["diagnostics"]["folds"]:
        assert {by_id[i] for i in fold["train_ids"]}.isdisjoint(
            {by_id[i] for i in fold["test_ids"]}
        )
    # A constant revision label gives no evidence for a revision model.
    assert profile.report["deployment"]["revision"] is False
    result = profile.score(examples()[1]["text"])
    assert result["revision_estimate_0_to_2"] is None
    assert result["rhetoric"]["evidence"]
    liked = profile.score(examples()[0]["text"])
    assert liked["voice_estimate_1_to_5"] > result["voice_estimate_1_to_5"]
    short = profile.score("Picture two engineers.")
    assert short["voice_estimate_1_to_5"] is None
    assert short["rhetoric"]["evidence"]
    assert "Under 75 words" in short["warning"]


def test_profile_roundtrip_and_bad_scaling(reference, tmp_path):
    profile = FeedbackProfile.learn(examples(), reference)
    path = tmp_path / "feedback.json"
    profile.save(path)
    assert FeedbackProfile.load(path).score(examples()[0]["text"]) == profile.score(
        examples()[0]["text"]
    )
    report = json.loads(path.read_text())
    report["models"]["voice"]["scales"][0] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(sp.StyleProfileError, match="scaling"):
        FeedbackProfile.load(path)


def test_learning_rejects_duplicate_ids_and_missing_groups(reference):
    records = examples()
    records[1]["id"] = records[0]["id"]
    with pytest.raises(sp.StyleProfileError, match="unique"):
        FeedbackProfile.learn(records, reference)
    records = examples()
    del records[0]["source_post_id"]
    with pytest.raises(sp.StyleProfileError, match="source_post_id"):
        FeedbackProfile.learn(records, reference)
    records = examples()
    records[0]["voice"] = True
    with pytest.raises(sp.StyleProfileError, match="integers"):
        FeedbackProfile.learn(records, reference)


def test_feedback_cli_build_and_score(reference, tmp_path, capsys):
    base, data, output, draft = [
        tmp_path / name for name in ["base.json", "ratings.jsonl", "personal.json", "draft.md"]
    ]
    reference.save(base)
    data.write_text("\n".join(json.dumps(r) for r in examples()))
    draft.write_text(examples()[1]["text"])
    assert main(["feedback-build", str(data), "--reference", str(base), "-o", str(output)]) == 0
    capsys.readouterr()
    assert main(["feedback-score", str(draft), str(output), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rated_examples"] == 6
    assert result["rhetoric"]["counts"]["hypothetical_prompts"] > 0
    assert result["revision_estimate_0_to_2"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_missing_topic_metadata_is_resolved_per_source(reference, reverse):
    records = examples()
    records[1]["source_post_id"] = records[0]["source_post_id"]
    del records[1]["topic_family"]
    if reverse:
        records.reverse()
    profile = FeedbackProfile.learn(records, reference)
    source_by_id = {r["id"]: r["source_post_id"] for r in records}
    history = {r["id"]: r for r in profile.report["examples"]}
    assert history["0-0"]["holdout_group"] == history["0-1"]["holdout_group"]
    assert profile.report["diagnostics"]["groups"] == 3
    for fold in profile.report["diagnostics"]["folds"]:
        assert {source_by_id[i] for i in fold["train_ids"]}.isdisjoint(
            {source_by_id[i] for i in fold["test_ids"]}
        )


def test_conflicting_topics_for_same_source_are_rejected_before_scoring(reference, monkeypatch):
    records = examples()
    records[2]["source_post_id"] = records[0]["source_post_id"]
    monkeypatch.setattr(
        reference, "score", lambda *a, **kw: pytest.fail("must validate groups first")
    )
    with pytest.raises(sp.StyleProfileError, match="conflicting topic_family"):
        FeedbackProfile.learn(records, reference)


def test_topic_labels_cannot_collide_with_unrelated_source_ids(reference):
    records = examples()
    for record in records[:2]:
        record["source_post_id"] = "family-1"
        del record["topic_family"]
    profile = FeedbackProfile.learn(records, reference)
    groups = {fold["held_out_group"] for fold in profile.report["diagnostics"]["folds"]}
    assert groups == {"source:family-1", "topic:family-1", "topic:family-2"}


def test_markdown_baseline_matches_ordinary_scoring(reference):
    profile = FeedbackProfile.learn(examples(), reference)
    text = "# Draft\n\n" + "\n\n".join(
        f"- {line}" for line in examples()[1]["text"].split(". ") if line
    )
    expected = reference.score(sp.Text(text, "feedback-input"))
    result = profile.score(text)
    baseline = result["rhetoric"]["baseline"]
    assert baseline["distance"] == expected.delta
    assert baseline["contrast"] == expected.likeness
    assert baseline["distance_label"] == expected.verdict.value
    assert all(
        result["rhetoric"]["text"][e["start"] : e["end"]] == e["text"]
        for e in result["rhetoric"]["evidence"]
    )


@pytest.mark.parametrize("payload", ["[]", "null", "42", '"profile"', "true"])
def test_nonobject_profile_reports_a_cli_error(tmp_path, capsys, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(sp.StyleProfileError, match="JSON object"):
        FeedbackProfile.load(path)
    assert main(["feedback-score", str(tmp_path / "draft.md"), str(path)]) == 1
    error = capsys.readouterr().err
    assert "JSON object" in error and "Traceback" not in error


def test_feedback_file_io_uses_utf8_with_non_utf8_locale(reference, tmp_path, monkeypatch, capsys):
    from pathlib import Path

    records = examples()
    note = "Author \u540d\u524d \u2014 concise wording."
    records[0]["notes"] = note
    profile = FeedbackProfile.learn(records, reference)
    base, data, saved, draft, output = [
        tmp_path / name
        for name in ["base.json", "ratings.jsonl", "profile.json", "draft.md", "report.json"]
    ]
    reference.save(base)
    data.write_bytes("\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8"))
    text = records[1]["text"] + "\n\n" + note
    draft.write_bytes(text.encode("utf-8"))
    original_open = Path.open

    def locale_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding in [None, "locale"]:
            encoding = "cp1252"
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", locale_open)
    profile.save(saved)
    assert json.loads(saved.read_bytes().decode("utf-8"))["examples"][0]["notes"] == note
    assert FeedbackProfile.load(saved).report["examples"][0]["notes"] == note
    assert main(["feedback-build", str(data), "--reference", str(base), "-o", str(saved)]) == 0
    assert FeedbackProfile.load(saved).report["examples"][0]["notes"] == note
    capsys.readouterr()
    assert main(["feedback-score", str(draft), str(saved), "--json", "-o", str(output)]) == 0
    result = json.loads(output.read_bytes().decode("utf-8"))
    assert note in result["rhetoric"]["text"]
    assert json.loads(capsys.readouterr().out) == result


def test_earlier_experimental_profile_requires_rebuilding(reference, tmp_path):
    profile = FeedbackProfile.learn(examples(), reference)
    profile.report["version"] = 1
    path = tmp_path / "old.json"
    profile.save(path)
    with pytest.raises(sp.StyleProfileError, match="rebuild from saved ratings"):
        FeedbackProfile.load(path)
