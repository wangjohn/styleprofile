# A second held-out set: short emails and posts from two other models

200 LLM drafts for testing WP-10b (generic contrast), in addition to the 16 drafts in
[`../llm-heldout/`](../llm-heldout/README.md). Like that set, **this is for evaluation
only.** Don't train, tune or choose a bundled contrast corpus with it, and don't read it
while writing one.

## What's here

| File | Generator | Emails | Posts | Words (min / median / max) |
|---|---|---|---|---|
| `codex.jsonl` | OpenAI Codex CLI (codex-cli 0.166.0) | 80 | 20 | 19 / 95.5 / 299 |
| `muse.jsonl` | Meta Muse Code CLI (`muse exec`) | 80 | 20 | 20 / 132 / 344 |

Words are counted as the scorer counts them: `len(words(prose(text).text))`.

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
- The texts are as generated, and none were picked or dropped afterwards. The only edit:
  five model-invented contact details in four Muse drafts (`v1-muse-14`, `v1-muse-16`,
  `v2-muse-p06`, `v2-muse-p44`) had their email domains and a LinkedIn path changed to
  `example.com` and `linkedin.com/in/example`, so no real address appears. The `(555)` phone
  numbers are fictional and unchanged.
- **Not independent of the two new metrics.** These 200 drafts are part of the checker's
  1,124-draft set, and `corrective_contrasts_per_1k` and `parallel_triads_per_1k` were added
  because they separated LLM from human drafts on that set. A WP-10b result on this set is
  therefore not a clean held-out test of those two metrics. Report it with that caveat, and
  for comparison also report the likeness with those two metrics left out.

Score them with records grouped by prompt:

```bash
styleprofile score bench/eval/llm-heldout-wide/codex.jsonl writer.profile.json \
    --text-field text --group-field prompt
```

## Things to know before using it

- **They are short.** Most are under 200 words, and about a quarter (48 of 200) are under 75 words,
  where `score` gives a single record no verdict (27 Codex, 21 Muse). Report results by
  length band, and report how many drafts got no verdict, rather than leaving them out
  silently.
- **Most emails start with a `Subject:` line** (123 of 160), and **most use bracketed
  placeholders** such as `[First Name]` or `[Company]` (119 of 160). Both are strong tells
  that no writer's own prose has. On the checker's set, removing placeholders cost about a
  tenth of its catches. So report results twice: as given, and
  cleaned, where a first line that starts with `Subject:` is removed (other first lines
  stay; 16 drafts are a single line) and each `[...]` placeholder is replaced by `Alex`.
  Cleaning pushes some drafts under 75 words, so compare the two as pairs: only drafts that
  get a verdict in both versions, plus the abstention count for each version.
- **Report each generator separately.** On the checker's set, a model left out of tuning
  was caught much less often (66% for Muse against 87% for Codex), and Muse emails were the
  hardest case at 47 of 80.
- **Different register from the sample writer.** The `examples/writer` essays are long
  personal essays. These drafts say how generic contrast does on business writing, which
  is the register WP-10b's "newsletter or email" folder covers.

## License

Original model output, released under this repository's license, for evaluation,
comparison and analysis.
