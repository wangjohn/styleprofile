# Method

How styleprofile measures a text, builds a reference profile, and turns the comparison into
two scores. For installation and usage, see the [README](../README.md).

## What it measures

About 150 interpretable metrics: sentence shape, rhythm, vocabulary richness, punctuation,
voice (hedges, openers, LLM marker words), Markdown habits, function-word rates, and (with
spaCy) parse depth, clause density, part-of-speech mix and how sentences open. It also keeps
three distributions, compared with Jensen-Shannon divergence: function-word transitions with
content words masked, character trigrams, and part-of-speech trigrams.

### The reference profile

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

### How reliable the contrast is

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
1/sqrt(n) as wide as a single chunk's, since averages vary less. That band never narrows
below 0.5 for Delta and its areas (half a standard deviation per metric), so an area the
writer never varies in, like Markdown in plain essays, cannot turn a trace into "very
different". Likeness counts only the part of each z toward the contrast drafts, about half
of |z| for noise, so its band never narrows below 0.25.
