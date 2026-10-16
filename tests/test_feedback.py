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
