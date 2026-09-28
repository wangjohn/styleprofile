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

## Length-aware verdicts

Every z-score is measured against the spread of the reference's windows (about 500 words). A
shorter text varies far more than a window by chance alone: in a 75-word paragraph one
semicolon is 13 per 1,000 words, and one long sentence moves the average sentence length a
long way. Judged against the windows' range, the writer's own paragraphs would read as "very
different". So every verdict is read against the writer's own range *at the text's length*.

- **Calibration pieces.** `build` also cuts the reference's windows into pieces of about 75,
  150 and 300 words. Each window is divided into round(words / length) equal parts, so no
  text is left over, and cut twice: once at the nearest sentence or block boundary with a
  paragraph break preferred when it is within a quarter of a piece (like a paragraph quoted
  whole), and once at the nearest sentence wherever it falls (like an excerpt lifted from
  mid-paragraph, which swings the paragraph metrics more). Code, headings, lists, quotes
  and tables stay whole. A length is only cut from windows that hold one and a half pieces
  of it. At most 30,000 words of pieces are measured per length (10,000 of the contrast's):
  every window is cut and every k-th piece kept, so every document contributes in proportion
  to its text however few and large the windows are (`--no-window` on a few books, say).
  With spaCy, each window is parsed once and a piece's syntax metrics come from its span of
  that parse, with sentences clipped to the span (spaCy's `Span.sents` returns whole
  sentences, running past a span that starts or ends mid-sentence).
- **Held out, like the windows.** Each piece is z-scored against the windows of every other
  document, exactly as a draft of that length is scored against the whole reference. Per
  length the profile stores (`calibration.by_length`, about 20 KB): each metric's rms; the
  Delta range (median and 95% bound) overall and per area, with each document's pieces
  weighted by reliabilities learned without that document; the overall 99% bound, for
  picking out one passage among many; and the likeness range of reference and contrast
  pieces cut the same way. Likeness keeps the effects learned on whole windows (each piece
  scored with the fold that left its document out), and scales each z by the pieces' own
  rms. Every entry records how many pieces and documents it rests on.
- **An honest 95% bound.** Pieces come several to a document and pieces of one document are
  alike, so the plain 95th percentile of a few hundred pieces is a noisy estimate that
  held-out text exceeds more often than 5%. The stored bound is an upper confidence bound
  instead: the quantile at 0.95 + 1.28 × sqrt(0.95 × 0.05 / n_eff), where n_eff, the
  *effective* number of pieces, is n / (1 + (n0 − 1) × ICC), with the intraclass correlation
  of the pieces' Deltas within documents from a one-way analysis of variance (n0 is the
  adjusted mean number of pieces per document). Few documents or alike pieces give a small
  n_eff and a higher bound. The windows' own range (a text as long as a window) is the
  plain percentile, as before.
- **Scoring at the text's length.** Each scored chunk reads the calibration for its own word
  count, interpolated linearly in log word count between the stored lengths; the windows
  themselves (their median word count) are the longest anchor, and anything longer uses the
  windows' values. The chunk's Delta weights (1 / rms²) and likeness scaling use the
  length-matched rms, and "Biggest differences" divides each z by how many times more that
  metric swings at the chunk's length than in a window, so a short text's arrows count
  standard deviations of the writer's own text at that length. The verdict bounds for Delta,
  each area and likeness use the length-matched ranges.
- **Too short to judge.** Under 75 words there is no verdict: the headline, `-q`, the JSON
  report (`reference.verdict`) and the library say "too short to judge (N words)", the By
  area block drops its verdict words, and the numbers and traits are shown as indicative.
  The same happens when a text is shorter than every length the reference is calibrated
  for, with the reason and its fix. A length is calibrated only with pieces from 3 or more
  documents worth at least 20 independent pieces (n_eff ≥ 20): with 20, one lies above a
  95th percentile, so it is observed rather than set by the largest piece. The contrast's
  likeness range at a length is a median and needs 5 pieces. `build` prints which lengths
  are calibrated, on how many pieces, and why the others are not. A reference with no
  held-out calibration at all (from one document) has no range for anything shorter than
  its windows, so texts under half a window get no verdict and are told to add documents.
- **Several chunks.** Each chunk is judged at its own length. The headline covers the chunks
  long enough to judge; the rest are left out of the verdict, the means, the differences,
  the arrows and the signals, and a warning names them and says why. If none is long
  enough, it is "too short to judge" and its numbers are indicative. The mean is read
  against a pooled bound, which the output prints: the mean of the chunks' medians plus the
  root sum of squares of their (95% bound − median) over n, which for n chunks of one
  length is the usual median + (bound − median) / sqrt(n).
- **`evaluate` stays on window calibration.** It builds its reference without the shorter
  lengths: its drafts and edited drafts are windowed exactly like the reference, its AUCs
  compare chunk scores directly and need no verdict bound, and its per-draft counts judge
  window-sized chunks. An edited set much shorter than its originals is visible in its
  length ratio.

### How well it holds

Measured out of sample (the reference never saw the pieces), without spaCy, cutting
held-out text three ways: the calibration's own cutter, whole paragraphs, and greedy runs
of sentences that split paragraphs.

- **Synthetic corpus** (`bench/gen.py`, 150 reference documents, 50 held out): the writer's
  own pieces land above the 95% bound 1.7–7.2% of the time across cutters and lengths (it
  was 2.5–16.9% with a plain percentile from paragraph-aligned pieces alone), read "clearly
  different" at most 0.3%, and LLM drafts cut to 150 words are flagged 97–100%. This corpus
  is **optimistic**: every document is copied or remixed from the same seven essays, so
  held-out pieces share paragraphs verbatim with the reference and documents differ from
  each other far less than a real writer's topics do.
- **Sample corpus** (`examples/`, seven essays, one left out at a time): no piece reads
  "clearly different" at any length (0 of 27–35 judged pieces at 75 and 150 words, 0 of 14
  at 300), and LLM drafts at 150 words are flagged 87–100%. With so few pieces the
  evidence is thin: the 95% confidence intervals on a 0% rate reach 12% (28 pieces) and 23%
  (14 pieces). A second real corpus should be tracked before the bound is trusted further.

## Resolution floors

Every z uses a spread of at least half of one occurrence per chunk for counts (5% of the
mean for other metrics), so a habit that is almost always absent cannot turn one use into
dozens of standard deviations. Verdicts compare an average over n chunks with a band
1/sqrt(n) as wide as a single chunk's, since averages vary less. That band never narrows
below 0.5 for Delta and its areas (half a standard deviation per metric), so an area the
writer never varies in, like Markdown in plain essays, cannot turn a trace into "very
different". Likeness counts only the part of each z toward the contrast drafts, about half
of |z| for noise, so its band never narrows below 0.25.
