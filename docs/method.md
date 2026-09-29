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
  n_eff and a higher bound; below about 31 effective pieces the bound is simply the sample
  maximum, which is conservative. The correlation is estimated once, pooled over the three
  lengths (weighted by pieces), since one length's estimate from a few documents swings
  widely; with fewer than 10 documents it is taken as at least 0.2. That floor comes from a
  simulation of 3 to 9 documents with true correlations of 0 to 0.4: among references that
  pass as calibrated, the share whose bound is exceeded more than 7.5% of the time falls
  from 7.5% with each length's own estimate to 2.6% (worst case 49% to 12%). A floor of 0.3
  would reach 0.9% but calibrate half as many small references. The windows' own range (a
  text as long as a window) is the plain percentile, as before.
- **Scoring at the text's length.** Each scored chunk reads the calibration for its own word
  count, interpolated linearly in log word count between the stored lengths; the windows
  themselves (their median word count) are the longest anchor, and anything longer uses the
  windows' values. The chunk's Delta weights (1 / rms²) and likeness scaling use the
  length-matched rms, and "Biggest differences" divides each z by how many times more that
  metric swings at the chunk's length than in a window, so a short text's arrows count
  standard deviations of the writer's own text at that length. The verdict bounds for Delta,
  each area and likeness use the length-matched ranges. Below the shortest calibrated length
  (with none calibrated, anything shorter than a window) a range is widened by
  sqrt(anchor words / words), as the chance variation of a rate grows when a text shrinks;
  a reference without calibration reads its fixed steps that much wider for texts shorter
  than its windows. Without that, the writer's own 300-word passages read "somewhat
  different" up to 35% of the time against two-document references; with it, at most 10%.
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
  against a bound for a mean, which the output prints (see below).
- **`evaluate` stays on window calibration.** It builds its reference without the shorter
  lengths: its drafts and edited drafts are windowed exactly like the reference, its AUCs
  compare chunk scores directly and need no verdict bound, and its per-draft counts judge
  window-sized chunks. An edited set much shorter than its originals is visible in its
  length ratio.

### A mean over many chunks

The headline, each area and likeness read the mean of the n judged chunks against a bound
for that mean, built from each chunk's range at its own length:

- **Centred on the mean, not the median.** Held-out Deltas are skewed to the right, so their
  median sits below their mean, and a mean over many chunks settles on the mean. Every
  stored range (the windows', each length's, overall, per area and for likeness) therefore
  also stores `mean`: an upper confidence bound on its pieces' mean, the mean plus 1.28
  standard errors with n_eff as for the 95% bound, so a mean from few documents is set
  higher rather than trusted as exact. A reference built before ranges stored it (or a
  score report made against one) is refused as outdated and must be rebuilt, since
  reading its median instead would bring the bug back.
- **Narrowing with n, but not to nothing.** Each chunk's spread is s = 95% bound − centre.
  The chunks of one run share whatever sets that text apart from the calibration pieces
  (its documents, its format, how it was cut), and that part does not average away. So they
  are taken to share r = 0.05 of their variation, and the bound is the mean of the centres
  plus sqrt((1 − r) × Σ s² + r × (Σ s)²) / n. For n chunks of one length that is
  centre + s × sqrt(r + (1 − r) / n): 0.23 s at 200 chunks rather than 0.07 s.
- **One chunk is unchanged**: its bound is its own 95% bound. The floors of 0.5 (Delta and
  areas) and 0.25 (likeness) still apply.

Before, the bound was the mean of the chunks' medians plus the root sum of squares of their
(95% bound − median) over n, which closes in on the median as n grows. On the synthetic
corpus below, the writer's own held-out chunks (439 of them, 75 to 300 words, cut as runs
of sentences) have a mean Delta of 0.855 and a median-centred bound of 0.837 over all of
them: the median is 0.810, 0.045 below the mean, and the spread term, 0.027, is smaller than
that gap. Batches of 200 of them read "somewhat different" 66% of the time and at least one
area did in every batch; the gap is there at every length (0.015–0.065), so it is the skew,
not the mixing of lengths. The value and the bound average the same chunks with the same
weights, and the conservative 95% bound only widens the spread term. Centring on the mean
alone (0.869 + 0.024 = 0.893) fixes the headline there, but not the areas or a run whose text
differs a little in kind: cutting held-out text as whole paragraphs rather than runs of
sentences moves an area's mean by up to about a tenth of its spread, and whole comments
scored against pieces of joined comments sit 0.13 of a spread above the overall centre. r =
0.05 (sqrt 0.22) covers such shifts with room for the noise of a mean over a few hundred
chunks; the writer's held-out pieces are alike within a document by only about 0.01 there.
r is **tuned, not measured**: it comes from the synthetic corpus and the seven sample
essays, where documents differ far less than a real writer's topics do, so a real corpus
may share more than 0.05 within a document and read a long run of one document too
harshly. The planned refinement is to take r as the larger of 0.05 and the reference's own
measured within-document similarity, once that measure is reliable (today it is floored
at 0.2 below 10 documents and inflated by cutting each window twice, as paragraphs and
as excerpts, which overlap).

Batches drawn from the writer's own held-out chunks (80 documents, cut by the tests' own
cutters to lengths drawn from 75 to 300 words), against a reference of 80 other documents:

| batch | "somewhat different" or worse, before → after | any area "somewhat" or worse, before → after |
|---|---|---|
| 1 | 2.5% → 2.5% | 13.5–17.0% → 13.5–17.0% |
| 5 | 1.5–3.5% → 0.0–1.5% | 14.5–21.5% → 7.0–10.0% |
| 20 | 0.5–6.5% → 0.0% | 14.0–33.0% → 1.0–3.5% |
| 50 | 0.5–15.5% → 0.0% | 28.5–84.5% → 0.0–1.0% |
| 200 | 0.0–65.5% → 0.0% | 63.5–100% → 0.0% |

Batches of held-out LLM chunks cut the same way still read "clearly different" or lean LLM
99–100% of the time at every size, and a batch of the writer's chunks with a few LLM chunks
mixed in is still flagged: 10% of LLM chunks read "somewhat different" or worse in 81% of
batches of 20 and 98% of batches of 50 (it was 98–100%). Against 3,000 generated comments
joined ten to a document, the 38 of 300 held-out comments long enough to judge (35 of them
close alone) read "somewhat different" together before (Delta 1.17 against a bound of 1.12)
and close now (bound 1.28). Their sentence shape still reads "somewhat different": a
comment is one paragraph while the pieces cut from joined comments span several, a shift
of 0.4 of that area's spread, which is the calibration's to match rather than the bound's
to absorb.

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
  at 300), and LLM drafts at 150 words are flagged 83–100%. Detection at 75 words is the
  price of honest bounds: only 44–61% of the judged 75-word LLM pieces are flagged (86–94%
  on the synthetic corpus), since a range wide enough for the writer's own 75-word pieces
  leaves little room above it. With so few pieces the
  evidence is thin: the 95% confidence intervals on a 0% rate reach 12% (28 pieces) and 23%
  (14 pieces). A second real corpus should be tracked before the bound is trusted further.

## Resolution floors

Every z uses a spread of at least half of one occurrence per chunk for counts (5% of the
mean for other metrics), so a habit that is almost always absent cannot turn one use into
dozens of standard deviations. Verdicts compare an average over n chunks with a band that
narrows as n grows, since averages vary less (see "A mean over many chunks"). That band
never narrows
below 0.5 for Delta and its areas (half a standard deviation per metric), so an area the
writer never varies in, like Markdown in plain essays, cannot turn a trace into "very
different". Likeness counts only the part of each z toward the contrast drafts, about half
of |z| for noise, so its band never narrows below 0.25.
