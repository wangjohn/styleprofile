# Generic assistant contrast drafts

Sixty original English drafts for `styleprofile build --generic-contrast`, written for
onboarding plan WP-10b by Claude (Sonnet 5.5 and subagents of the same model), each as an
assistant answers a request in the register named by its folder. No published text, sample
essay or sample LLM draft was copied, and the held-out evaluation set
(`bench/eval/llm-heldout/`) was not read while they were written.

| Folder | Register | Drafts |
|---|---|---|
| `explainer/` | a headed or bulleted explainer | 10 |
| `blog/` | a plain-prose blog post | 10 |
| `essay/` | a personal essay written on request | 10 |
| `newsletter/` | a newsletter or email | 10 |
| `opinion/` | an opinion piece | 10 |
| `humanlike/` | a "make it sound human" rewrite style: no em dashes, varied sentence lengths, no stock phrases | 10 |

Each register has drafts in all four length bands (150–300, 300–600, 600–1,000 and
1,000–1,500 prose words; 15 drafts per band overall). No two drafts share a topic. Names of
people and organizations are invented. The drafts are illustrations, not advice.

All files here are original project contributions under the repository's MIT licence. They
are a convenience contrast set, not a survey of any model and not proof of where a text came
from. Drafts made from the writer's own briefs are a better comparison; see
`docs/contrast.md`.

The profile learns which habits separate the writer from these drafts, and its likeness
range for them is computed leaving out one whole register at a time. See "Generic contrast,
second attempt" in `docs/method.md`.
