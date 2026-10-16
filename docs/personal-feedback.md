# Personal feedback from rated passages

Personal feedback is an optional layer over an existing contrast reference. It adds
sentence evidence and learns associations with a writer's ratings; ordinary `build`
and `score` retain their existing measurements and verdicts.

## Rhetorical measurements

`styleprofile.feedback.rhetoric(text)` returns counts, rates per 1,000 prose words,
and matching sentences for four cues:

| Cue | Definition |
|---|---|
| Hypothetical reader prompts | Sentence-opening “picture”, “imagine”, or “consider … hypothetical”. |
| Observation prefaces | Openers such as “There's a pattern” and “I keep noticing”. |
| Short/long sentence pairs | A sentence of at most eight words followed in the same paragraph by one of at least nine words. |
| Explicit lesson statements | Constructions such as “That is the point” or “This is the lesson”. |

These are observations, not universal AI markers or automatic writing faults. Code
blocks and ordinary references to pictures do not trigger hypothetical-reader cues.
Evidence offsets refer to the normalized prose returned as `text`; they are not offsets
into the original Markdown. A short/long pair measures structure, not whether the longer
sentence explains the shorter one. The rules do not measure interestingness, semantic
redundancy, a strong hook, or flowery language generally.

## Build a personal profile

Supply JSONL records with unique IDs and these fields:

```json
{"id":"review-1/passage-1","source_post_id":"post-1","topic_family":"infrastructure","text":"The complete passage...","voice":4,"revision":"light","notes":"Tighten the ending."}
```

`voice` is an integer from 1 to 5. `revision` is `keep`, `light`, or `substantial`.
The profile requires at least six rated passages across three groups. Optional
`topic_family` groups related posts for diagnostics; otherwise `source_post_id` is used.
Duplicate IDs and invalid ratings are rejected. The example above illustrates the
schema; use actual rated prose, rather than a placeholder, for learning.

```bash
styleprofile feedback-build ratings.jsonl --reference writer.json -o personal.json
styleprofile feedback-score draft.md personal.json
styleprofile feedback-score draft.md personal.json --json -o draft-feedback.json
```

The reference must contain a learned contrast. Fixed ridge regularization learns
separate voice (1–5) and revision-effort (0–2) targets from existing Delta/contrast scores
and the four rhetorical rates. Source origins and guessed AI labels are not targets.
Each target's numeric estimate is disabled unless its grouped development error beats
a mean-only prediction. Prose under 75 words still gets cue evidence, but no estimate.

The diagnostic holds out entire groups and fits scaling only on the remaining groups.
It compares a training-mean baseline, a regression on existing scores, and a regression
with the new rhetorical measurements, using group-weighted mean absolute error. These
are post-hoc development diagnostics, not independent validation: feedback used to
choose features cannot also establish their generalization. Estimates are personal
associations, not probabilities of AI authorship or editing instructions.

## Reuse existing judgments

An existing blind review is sufficient; an additional feedback pass is not required.
Keep its ratings and notes as training records. Append later judgments as new records
with unique IDs, then rebuild from both files:

```bash
styleprofile feedback-build earlier.jsonl later.jsonl \
  --reference writer.json -o updated-personal.json
```

The profile embeds its base reference and retains IDs, notes, rating targets, grouping
and text hashes. It does not depend on a machine-specific reference path. Keep the
training JSONL for rebuilding: the profile stores hashes rather than the full passages.
Profiles contain personal notes and should be kept private as appropriate.

Python callers can use `FeedbackProfile.learn(records, reference)`, `.score(text)`,
`.save(path)` and `.load(path)` from `styleprofile.feedback`. Scores return descriptive
baseline scores, rhetorical evidence, experimental estimates, and enablement flags for
both targets. Notes are preserved for interpretation; the learner does not automatically
turn arbitrary written comments into new metric definitions.
