# Contrast drafts from your own briefs

LLM-likeness describes resemblance to the contrast set you supplied. Make that comparison
useful by keeping topics, audience, genre and length close to the writer's texts. A change
in subject or document length can separate the sets even when writing habits do not.

For each independent writer text, prepare a brief stating what it is about, who it is for,
and what it should cover. Do not give the model the writer's actual wording or ask it to
imitate the writer. Copy this prompt and fill in the brackets:

```text
Write an original [GENRE] for [AUDIENCE].
The brief is: [SUBJECT, PURPOSE, AND POINTS TO COVER].
Aim for [WORD COUNT] words, comparable to the writer's text for this brief.
Use your ordinary writing voice rather than imitating a named writer.
Return only the finished draft, without an introduction about the task.
```

Use several models or versions and several briefs rather than many rewrites of one prompt.
A first set might contain 10 drafts; aim for 20–30 varied, independent drafts when checking
held-out behavior. These are practical targets, not a guarantee of detection. Retain
model/prompt provenance outside the folder of texts, so it does not become measured prose.

Check actual lengths after generation. Match the writer's range instead of making every
draft the same round number. Use the same genre where possible, remove refusals and prompt
echoes, and save only draft prose. Do not pick drafts because they receive a desired score.

With writer texts in `posts/` and model drafts in `llm-drafts/`:

```bash
styleprofile build posts/ --contrast llm-drafts/ -o writer.json
styleprofile score new-draft.md writer.json
```

Keep fresh writer documents and fresh model drafts outside both training folders, then
check all of them. A high training AUC does not ensure that unseen drafts reach the same
verdicts. Edited model drafts can also lose the habits that the score measures; see
[stress-testing likeness](usage.md#stress-testing-llm-likeness). A like-reference score is
not evidence that a human wrote the text.
