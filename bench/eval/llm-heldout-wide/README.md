# A second held-out set: short emails and posts from two other models

200 LLM drafts for testing WP-10b (generic contrast), in addition to the 16 drafts in
[`../llm-heldout/`](../llm-heldout/README.md). Like that set, **this is for evaluation
only.** Don't train, tune or choose a bundled contrast corpus with it, and don't read it
while writing one.

## What's here

| File | Generator | Emails | Posts | Words (min / median / max) |
|---|---|---|---|---|
| `codex.jsonl` | OpenAI Codex CLI (codex-cli 0.166.0) | 80 | 20 | 20 / 100 / 310 |
| `muse.jsonl` | Meta Muse Code CLI (`muse exec`) | 80 | 20 | 22 / 140 / 355 |

Each line is one draft:

```json
{"id": "v1-codex-01", "generator": "codex", "register": "email", "prompt": "v1-p01", "text": "..."}
```

- `register` is `email` (cold sales, follow-ups, introductions, recruiter and investor
  emails, thank-you notes) or `post` (LinkedIn-style announcements and reflections).
- `prompt` names the request. Both generators answered the same 100 prompts, so the two
  drafts that share a `prompt` are about the same thing. Group by it, never split a pair.
- The drafts were written between 2026-11-04 and 2026-11-06 for a separate rule-based
  AI-writing checker's evaluation set, before WP-10b's corpus exists. Neither model is a
  Claude model, so this set is independent of a bundled corpus written by Claude.
- The texts are exactly as generated. Nothing was edited or picked afterwards.

Score them with records grouped by prompt:

```bash
styleprofile score bench/eval/llm-heldout-wide/codex.jsonl writer.profile.json \
    --text-field text --group-field prompt
```

## Things to know before using it

- **They are short.** Most are under 200 words, and about a quarter (48 of 200) are under 75 words,
  where `score` gives a single record no verdict. Report results by length band, and report
  how many drafts got no verdict, rather than leaving them out silently.
- **Most emails start with a `Subject:` line** (123 of 160), and **most use bracketed
  placeholders** such as `[First Name]` or `[Company]` (119 of 160). Both are strong tells
  that no writer's own prose has. On the checker's set, removing placeholders cost about a
  tenth of its catches. Report results with and without the first line, and with
  placeholders replaced by a plain name, so a pass doesn't rest on them.
- **Report each generator separately.** On the checker's set, a model left out of tuning
  was caught much less often (66% for Muse against 87% for Codex), and Muse emails were the
  hardest case at 47 of 80.
- **Different register from the sample writer.** The `examples/writer` essays are long
  personal essays. These drafts say how generic contrast does on business writing, which
  is the register WP-10b's "newsletter or email" folder covers.

## License

Original model output, released under this repository's license, for evaluation,
comparison and analysis.
