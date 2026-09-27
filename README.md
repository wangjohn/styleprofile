# styleprofile

Build a stylometric profile from a writer's texts, see how far a draft drifts from it, and
learn which habits separate that writer from LLM output.

## Quick start

```bash
# A reference profile from a folder of the writer's posts, contrasted with LLM drafts of
# the same material.
styleprofile posts/ --window-words 500 --contrast llm-drafts/ --output profiles/writer.json

# Score a draft against it.
styleprofile draft.md --window-words 500 --reference profiles/writer.json \
  --output profiles/draft.json
```

The terminal shows a short summary; pass `--all` to print every metric. The JSON report
always has everything. `python -m styleprofile` works too.

Inputs can be JSONL (`--text-field`, default `text`/`body_markdown`/`output`/...), Markdown
or text files, directories of them, or `-` for stdin. Pass `--no-syntax` to skip spaCy.

## What it measures

About 150 interpretable metrics: sentence shape, rhythm, vocabulary richness, punctuation,
voice (hedges, openers, LLM marker words), Markdown habits, function-word rates, and (with
spaCy) parse depth, clause density, part-of-speech mix and how sentences open. It also keeps
three distributions, compared with Jensen-Shannon divergence: function-word transitions with
content words masked, character trigrams, and part-of-speech trigrams.

A reference profile stores, per window, every metric; each metric's mean and spread; and how
much each metric varies in the writer's own held-out writing (each document scored against
the others). With `--contrast`, it also learns which metrics separate the writer from the
contrast drafts.

## Two scores

Two scores answer two questions:

- **Delta: how far is this from the writer overall?** The weighted mean |z|. Every area
  (sentence shape, punctuation, voice, ...) counts equally, and within an area each metric
  is weighted by 1 / rms², where rms is its root-mean-square z on the writer's held-out
  windows (floored at 1). A habit the writer uses only now and then, like semicolons,
  therefore counts little. The comparison prints the writer's own held-out range.
- **LLM-likeness: does this look like the contrast drafts?** Each metric's weight is its
  effect size squared: (contrast mean z − writer held-out mean z) / max(rms, 1), squared.
  Only deviations in the contrast direction count, so more em dashes than the writer uses
  raise the score and fewer do not lower it. The reference stores a leave-one-document-out
  range (each document scored with weights learned without it), which sets the verdict
  words. Rename the contrast set with `--contrast-label`.
- **How sure is the separation?** The profile reports the AUC (the chance a held-out
  contrast chunk scores above a held-out reference chunk) with a 95% confidence interval
  from 2,000 bootstrap resamples of whole documents, so windows cut from one post move
  together. With few documents the interval is wide: an AUC of 0.96 from 8 drafts is less
  certain than it looks.
- **Is it just length?** The profile also reports how well chunk word count alone separates
  the two sets. If length alone reaches an AUC of 0.75 or more, it warns: the likeness score
  may partly reflect length, so match lengths or window the drafts with `--window-words`.

## Resolution floors

Every z uses a spread of at least half of one occurrence per chunk for counts (5% of the
mean for other metrics), so a habit that is almost always absent cannot turn one use into
dozens of standard deviations. Verdicts compare an average over n chunks with a band
1/sqrt(n) as wide as a single chunk's, since averages vary less.

## Getting useful results

- **One genre per profile.** Blog posts, fiction and email have different habits; mixing
  them widens every spread and blurs every score.
- **Enough text.** Aim for 15 or more documents and 20,000 or more words in the reference.
- **The same `--window-words` everywhere.** Profile the reference, the contrast set and every
  sample with the same value, because each metric's spread depends on chunk length.
- **Contrast drafts from the writer's own briefs.** Have LLMs write from the same briefs or
  outlines the writer worked from, and use several models; the weights only know the drafts
  they were learned from.
- **Short texts are noisy.** Under about 150 words, most metrics rest on a handful of
  sentences.
- **A verdict means "unlike this reference", not proof of authorship.** A human can drift
  from their own profile, and a model can be prompted toward it.

Keep corpora, drafts and generated profiles out of version control; the `.gitignore`
excludes `profiles/`, `corpora/` and `data/`.

## Install

```bash
pip install "styleprofile[syntax] @ git+https://github.com/wangjohn/styleprofile"
```

Or, from a clone, `uv sync --extra syntax` and run `uv run styleprofile`. The `syntax` extra
installs spaCy and `en_core_web_sm`; without it, run with `--no-syntax`.

## Development

```bash
make sync
make check   # pytest, ruff check, ruff format --check, pyright
```

The package began as the stylometry module of GoodProse.

## License

MIT
