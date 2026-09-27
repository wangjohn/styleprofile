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

`styleprofile build` makes a reference profile. It measures every metric per window (500
words by default, `--window-words`) and stores each metric's mean and spread over the windows,
and how much each metric varies in the writer's own held-out writing (each document scored
against the others). With `--contrast`, it also learns which metrics separate the writer from
the contrast drafts. `styleprofile score` reads the window size and other settings back from
the profile, so a draft is always cut the way the reference was.

The profile keeps only these summaries, not each window's metrics, so it stays well under a
megabyte however large the corpus; `build --keep-chunks` also saves the per-window rows, for
debugging. Score reports do keep a row per window of the draft, which the comparison view
ranks.

Reports save each input by the final part of its normalized path, plus the path of each file
inside it: `posts`, `./posts/`, `../x/posts` and `/home/me/posts` all save as `posts`, and a
file in it as `posts/2024/a.md`. Nothing above an input is saved, so a shared profile or
report doesn't reveal where its files lived, and the same corpus gives the same names from
any directory. The working directory (`.`) saves only the paths inside it; the home
directory, `/` and any other input without a telling name save as `input`; standard input
is `stdin`. When two inputs would give the same name to different files, the later one is
numbered: `posts (2)`, or `notes (2).md` for a file, and `settings.inputs` records the names
chosen. Documents are told apart by the files they were read from, not by these names.

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
  from bootstrap resamples of whole documents, so windows cut from one post move together.
  Resampling checks the interval after 250, 500, 1,000 and 2,000 resamples. It stops at
  the first check where both ends moved less than a fiftieth of the interval's width (at
  least 0.001, at most 0.005) since the previous check, and also did at the check before.
  Consecutive checks share most of their resamples, so one small move understates the
  remaining error; asking for two keeps the early stop about as accurate as always drawing
  2,000 (its 90th-percentile error from a 100,000-resample interval was within 5% of
  2,000's on 200 seeds of a 15-against-12-document sample). Wide intervals from few
  documents usually run to 2,000, which is cheap for them; narrow ones from thousands of
  documents stop at 1,000.

  When every contrast chunk scores above every reference chunk (AUC 1.0), or below (0.0),
  every resample gives the same AUC, so the interval is exact and takes none; the display
  then says "perfect separation on N reference and M contrast documents" rather than a
  degenerate interval, since with few documents that is far from certainty.
  `calibration.bootstrap` records which: `method` is `"bootstrap"` (with `resamples`, how
  many it took), `"exact"` (none needed), or null when either side has fewer than 2
  documents and there is no interval, with each side's document count. `evaluate` reports
  the same `bootstrap` beside each of its AUC intervals. With few documents the interval is
  wide: an AUC of 0.96 from 8 drafts is less certain than it looks.
- **Is it just length?** The profile also reports how well chunk word count alone separates
  the two sets. If length alone reaches an AUC of 0.75 or more, it warns: the likeness score
  may partly reflect length, so match lengths, or keep windowing on (`build` windows the
  writer and the drafts alike unless given `--no-window`).

## Resolution floors

Every z uses a spread of at least half of one occurrence per chunk for counts (5% of the
mean for other metrics), so a habit that is almost always absent cannot turn one use into
dozens of standard deviations. Verdicts compare an average over n chunks with a band
1/sqrt(n) as wide as a single chunk's, since averages vary less. That band never narrows
below 0.5 for Delta and its areas (half a standard deviation per metric), so an area the
writer never varies in, like Markdown in plain essays, cannot turn a trace into "very
different". Likeness counts only the part of each z toward the contrast drafts, about half
of |z| for noise, so its band never narrows below 0.25.
