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

### Documents, windows and pooling

A *document* is the unit held-out calibration leaves out: each of its windows is scored
against a reference built without it, so what counts as one decides how honest the
held-out ranges are. Each chunk carries its document (`Chunk.document`), set when it is
read and kept through windowing and pooling; ids such as `post#w2` only name chunks for
people. The documents are:

- **a file** (Markdown, text or HTML), or stdin;
- **a JSONL record**, when no group field is given;
- **a group of records**, with `--group-field thread`: the records of one input (a file,
  or every file of a folder) that share a value. Records with no value (missing, null or
  blank) are one more group, `thread=(none)`. Values compare as text;
- **a pooled window**, when short texts without a group field are joined: see below.

Windowing works within a document: a long text is split into windows at paragraph breaks.
Short texts are *pooled*, joined in order into windows of about the same size:

- A group's records are joined into windows `thread=t1#w1`, `thread=t1#w2`, ... of that
  group's document, wherever the records sit in its input.
- Without a group field, consecutive short records of one JSONL file, or consecutive short
  files of one folder input (in sorted order), or `Text` inputs, are joined, and each
  window (`c0012..c0019`, its first and last text) becomes a document of its own. Nothing
  is joined across two inputs you typed.

Length calibration (above) cuts its shorter pieces from a pooled window along the texts it
was joined from, since those are what people score. At each length it takes single records
(or files) of at least that length, whole or cut if they hold one and a half pieces, the
way the texts judged at that length are selected; only when there are fewer than 20 such
pieces from 3 documents does it fall back to runs of whole records up to the length. Pieces
cut across record boundaries, or runs of shorter records, would stand in for one long
comment with several short ones, and shift the paragraph measures of sentence shape: runs put
batches of the writer's own comments, judged one by one, in "somewhat different" sentence
shape 8.7% of the time over 20 references; single records, 2.6% over 32.

Pooled windows as documents are honest only when neighbouring texts are no more alike than
any two. When records from different threads are interleaved, each window mixes threads,
the between-thread variation averages out, and the reference's spread shrinks: a draft
from one thread then looks far away. In a test of 20 threads of 40 comments by one writer,
33–85% of a new thread's windows landed above the calibrated 95th percentile with
interleaved records (3–14% when contiguous), against 2–8% with `--group-field thread`. The
`build` note says so and suggests fields named like a source (a thread, conversation,
channel, subject, reply or parent; else an author, user or sender) with 2 or more values,
and at most a third as many as records.

Pooling is on by default when the median text has under a quarter of a window of prose
words and pooling leaves at least 15 windows; below that, `build` notes that it held back.
`--pool` and `--no-pool` force it either way. Word-for-word duplicates are dropped first:
whole files, and JSONL records one by one even when a group is the document. The contrast
set pools when the writer's texts do, so the two stay alike in length. `score` inherits
the group field but judges each text on its own, at its own length (so a record under 75
words abstains), unless given `--pool`; pooled drafts against a reference that was not
pooled read too close, and it warns. `evaluate` joins each edited set into the same
windows as its originals, record by record, pairing a record with its original by its id,
with its file's path when it is read from a folder (`2024/a.jsonl:17`), since ids often
restart in each file; grouped windows pair by their group (`thread=t1`), whichever files
hold its records.

Contrast drafts and their edits need not carry the writer's group field: an input where
no record has it is read ungrouped, each record a document of its own, with a note (the
writer's own texts must have it, since its absence there is likely a typo). `build` also
flags group fields that cannot work: one group (nothing to hold out), groups too short to
pool, a group found in several inputs typed one by one (each input's records are separate
documents, so name the folder), and one document holding most windows, so that the
others' few windows carry the calibration.

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
  also stores `mean`: an upper confidence bound on its held-out mean, the mean plus t
  standard errors, with the standard error from n_eff as for the 95% bound and t the 90%
  quantile of Student's t with one degree of freedom fewer than there are documents (1.89
  for 3 documents, 1.53 for 5, 1.28 for many), so a mean from few documents is set higher
  rather than trusted as exact. n_eff uses the within-document similarity of the range's
  documents, floored at 0.2 below 10 documents: the pooled pieces' for a length's overall
  range, the windows' own for theirs, and each area's own for an area. The centre is not capped at the 95%
  bound: a heavy-tailed range can have its mean above its 95th percentile, and a long run
  still converges to the mean.
- **Narrowing with n, but not to nothing.** Each chunk's spread is s = 95% bound − centre.
  The chunks of one run share part of their variation, which does not average away: a run
  from one document, or on one topic, sits at that document's or topic's offset, and every
  run shares whatever sets its text apart from the calibration pieces (its format, how it
  was cut). Each range stores that share, r (`similarity`): the within-document similarity
  (intraclass correlation) of its own held-out values, at least 0.2 below 10 documents,
  and never below 0.05. The bound is the mean of the centres plus
  sqrt(Σ (1 − r) s² + (Σ sqrt(r) s)²) / n, the spread of a mean of values that share r of
  their variation; for n chunks of one range that is centre + s × sqrt(r + (1 − r) / n).
- **One chunk is unchanged**: its bound is its own 95% bound. The floors of 0.5 (Delta and
  areas) and 0.25 (likeness) still apply.
- **Refused when missing.** A reference built before ranges stored `mean` and
  `similarity` (or a score report made against one) is refused as outdated and must be
  rebuilt, since reading its median instead would bring the bug back.

Before, the bound was the mean of the chunks' medians plus the root sum of squares of their
(95% bound − median) over n, which closes in on the median as n grows. On the synthetic
corpus, the writer's own held-out chunks (439 of them, 75 to 300 words, cut as runs of
sentences) have a mean Delta of 0.855 and a median-centred bound of 0.837 over all of them:
the median is 0.810, 0.045 below the mean, and the spread term, 0.027, is smaller than that
gap. The gap is there at every length (0.015–0.065), so it is the skew, not the mixing of
lengths; the value and the bound average the same chunks with the same weights, and the
conservative 95% bound only widens the spread term. Centring on the mean alone
(0.869 + 0.024 = 0.893) fixes the headline there, but not the areas or a run whose text
differs a little in kind: cutting held-out text as whole paragraphs rather than runs of
sentences moves an area's mean by up to about a tenth of its spread (which the least r,
0.05, sqrt 0.22, covers), and a writer whose documents each keep to one topic has runs on
one topic whose punctuation, function words and voice sit up to a quarter of a spread
from the centre. With one r of 0.05 for every range, 10–14% of such runs of 20–50 chunks
read "somewhat different" in some area; with each range's own r, 4.5–5% (0.5–7.2% in an
independent review's harness, whose cutters also start spans mid-sentence). The cost is
detection of a small minority of LLM chunks (below): in that harness a batch of 20 with 10%
LLM chunks read "somewhat different" or worse in 47–97% of batches, against 79–97% with one
r of 0.05 and 98–100% before.

The tests' own acceptance numbers, against a synthetic reference of 80 documents: disjoint
batches (none sharing a chunk) of the writer's own held-out chunks from 300 other
documents, cut by the tests' own cutters to lengths drawn from 75 to 300 words (runs of
sentences / whole paragraphs); before is the median-centred bound on the same chunks:

| batch | batches | "somewhat different" or worse, before → after | any area "somewhat" or worse, before → after |
|---|---|---|---|
| 1 | 1,646 / 1,454 | 2.2 / 1.5% → unchanged | 13.2 / 10.4% → unchanged |
| 5 | 329 / 290 | 2.1 / 0.3% → 0.3 / 0.0% | 18.2 / 12.4% → 7.3 / 4.8% |
| 20 | 82 / 72 | 3.7 / 0.0% → 0.0 / 0.0% | 32.9 / 15.3% → 1.2 / 0.0% |
| 50 | 32 / 29 | 12.5 / 0.0% → 0.0 / 0.0% | 81.2 / 27.6% → 0.0 / 0.0% |
| 200 | 8 / 7 | 87.5 / 0.0% → 0.0 / 0.0% | 100 / 42.9% → 0.0 / 0.0% |

Batches of held-out LLM chunks cut the same way read "clearly different" or lean LLM in
every batch of 5, 20 and 50, before and after. Against 3,000 generated comments joined ten
to a document, the 38 of 300 held-out comments long enough to judge (35 of them close
alone) read "somewhat different" together before (Delta 1.17 against a bound of 1.12) and
close now. Their sentence shape still reads "somewhat different": a comment is one
paragraph while the pieces cut from joined comments span several, a shift of 0.4 of that
area's spread, which is the calibration's to match rather than the bound's to absorb.

#### The trade-off: a few very different chunks among many

The pooled headline answers whether the run as a whole is like the writer. Once a bound
for a mean no longer closes in on the median, a small minority of very different chunks
moves the mean too little to cross it. In the review's harness (400 random batches of
held-out chunks with 10% LLM chunks mixed in), the headline read "somewhat different" or
worse, before → at the revision with one r of 0.05 for every range:

| chunks | n = 20 | n = 50 | n = 200 |
|---|---|---|---|
| spans that keep paragraph breaks | 97.5% → 78.8% | 100% → 97.8% | 100% → 100% |
| ~100-word records, scored one by one | 70.2% → 14.8% | 99.0% → 11.8% | 100% → 5.0% |
| single paragraphs | 62.0% → 8.8% | 97.5% → 7.8% | 100% → 1.5% |
| against a 4-document reference | 87.2% → 25.5% | 100% → 22.0% | — |

Each range's own r is never less than 0.05, so these are upper bounds now: at this
revision the records row reads 2.5%, 0.2% and 0.0%. With 30% LLM chunks the headline
still reads "somewhat different" or worse in 99–100% of batches, though "clearly different"
or "leans LLM" far less often (35% of 30% batches on one topic-clustered corpus, and about
1% against 4- or 5-document references). This is intended: the headline is not the place
to catch a few chunks. The user never misses them, though:

- **Chunks flagged on their own are counted beside the headline.** A chunk is flagged
  when, judged at its own length, it reads "clearly different" or worse, or leans toward
  the contrast set or more. Whenever a run has more chunks than documents and some are
  flagged, the headline says how many ("close   Delta 1.08; 2 of 17 chunks read clearly
  different or lean LLM on their own (see below)"), `-q` ends its line with "(2 of 17
  chunks read clearly different or lean LLM)", the report's verdict and each document
  record `flagged`, and the library has `ScoreResult.flagged` and `DocumentResult.flagged`.
  With one chunk per document the per-document table and its count line already say it.
- **The chunk lists always name them.** "Most LLM-like chunks" and "Least like the
  reference" put the flagged chunks first (`--all` lists every one).
- **Exit codes.** `--fail-above` and `--fail-likeness` judge each document's verdict, as a
  whole; `--fail-flagged N` fails a document with N or more flagged chunks
  (`ScoreResult.failing(flagged=N)`), and every `failed:` line and JSON `failed` entry
  gives the count whichever check failed.
- **The By area view** still reads such a mixture as different in 81–100% of batches
  against 80-document references, but less against small ones: 64% and 76% (n = 20, 50)
  against a 4-document reference, 40% and 39% against a 5-document one, since each range's
  r is at least 0.2 below 10 documents. **Drift passages** (plan PR 12) will point at the
  passages within a document.

On the tests' own chunks, every writer batch of 20 or 50 with 10% LLM chunks named a
flagged chunk, and the headline read "somewhat different" or worse in 78–100% of them.

**Chance flags.** The writer's own chunks are flagged rarely, 0–0.14% of held-out chunks in
the review's harness, but a long run has many chances: runs of 200 of the writer's own
chunks name at least one in 15–27% of batches, and runs of 50 in 0–8%. So a flagged chunk
in a long document is a pointer to read, not proof, and `--fail-flagged 1` on documents of
a hundred windows or more will sometimes fail the writer's own text; choose N by length.
The note does not say how many to expect by chance: that rate depends on the reference and
on how the text is cut (it varies nearly tenfold across the harness's corpora), and a
reference does not yet store its own held-out flag rate to quote.

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

## Where a draft drifts

One Delta or likeness over a whole draft dilutes a paragraph or two in another register:
`examples/draft.md` has two paragraphs written like an LLM (lines 9 and 15, 93 of its 542
words) and reads "close" overall. So `score` also reads a single document in parts
(`drift`; `--by-paragraph` or `passages=True` does it for several). A paragraph that reads
unlike the writer is said to *drift*; that is a different thing from a chunk *flagged* on its
own (a whole window reading clearly different), which the headline notes.

- **Paragraphs and spans.** The document is split into paragraphs as written: each block
  with prose, measured with any heading or code block just above it and its list
  continuations; its line range is its prose's, so headings and code never widen it (for
  HTML, the lines are the Markdown conversion's, and the output says so). Every paragraph
  gets two spans of whole paragraphs holding at least 100 words: one starting at it and
  running forward, one ending at it and running back. Near the ends, where one side runs
  out, it takes the document's first or last span instead; where the two are the same span
  (a paragraph of 100 words or more is its own span both ways, and so, often, is the
  first or last), the second is that span widened by the shorter neighbouring paragraph. So
  every paragraph rests on two distinct spans, unless one span is the whole document. A
  document under 100 words has no span and abstains.
- **Measured once.** Each span's Markdown is measured once, like a chunk; with spaCy its
  syntax comes from its part of the parse the windows already have, joined into one parse of
  the document (`Doc.from_docs`), so nothing is parsed twice. Each paragraph is also measured
  on its own, for its traits.
- **The statistic.** Each span is scored like a chunk of its length (Lengths.at) and read as
  its score over the 95% bound at that length (floored as the verdicts are): LLM-likeness
  when the reference has a contrast set, else Delta. A paragraph's statistic is the lower of
  its two spans. In span scoring only (verdicts are unchanged), each z of the parser's
  metrics (syntax and sentence openers) counts at most 3 either way, the cap an unseen
  habit gets: a 100-word span has few sentences, and one rare construction in it could
  otherwise carry the span alone. The null below is scored the same way.
- **Explained by a stronger neighbour.** A span that is high only because of a neighbouring
  paragraph has the neighbour's other span beside it, without the drifting paragraph, so the
  neighbour's statistic stays low. Paragraphs over their threshold are then taken strongest
  first: one whose two spans both hold a paragraph at least as strong that still drifts,
  while it is not in both of that one's spans, is explained by it and does not drift
  (`--by-paragraph` says so). Only a paragraph still drifting can explain another, so there
  are no chains, and a weaker paragraph never hides a stronger one. A paragraph that reads
  unlike the writer on its own too (scored alone at its own length, over its bound) is never
  explained away, so two LLM paragraphs side by side, as at the start of a draft, both
  drift.
- **A threshold per document, from the writer's documents.** Checking every paragraph of a
  70-paragraph document at a 95% bound would find some paragraph of the writer's own most of
  the time. So `build` reads up to 40,000 words of its own documents the same way, each span
  scored held out (z-scores against the windows of every other document, likeness with the
  contrast fold that left the document out, both at the span's length). It splits each
  paragraph's statistic into its document's *level* (the median statistic of its paragraphs)
  and its *excess* over that level, and stores (`calibration.drift`) the spread of the
  documents' levels (median and 90th percentile) and the tail of the excess above its 90th
  percentile, modelled as exponential. A document of n paragraphs lets a paragraph drift
  only above the level of the writer's documents at the top of their spread (the 90th
  percentile) plus the 1 − 0.05 / n quantile of the excess, with the tail's scale at an upper
  90% confidence bound (the mean of k exponential excesses is a chi-square with 2k degrees
  of freedom: 1.24 times the mean for 40 values above the tail's start, 1.16 for 80). So the
  chance of any paragraph of the writer's drifting is at most about 5% whatever the
  document's length, and the margin comes from how far the writer's own documents move, and
  how well the reference pins the tail down, rather than from a constant. It is never at or
  below the 95% bound.
- **Too small, or no contrast.** The null needs paragraphs from 10 or more documents and 20
  values in the tail (about 200 paragraphs): with fewer, no paragraph drifts and the output
  says the reference is too small to set paragraph thresholds. The seven sample essays are
  such a reference. A document that is a single span (one tail value per document, too few
  to fit) is never judged to drift, and says why. Against a reference without a contrast
  set, spans are judged by Delta, which dilutes a paragraph's few LLM tells across every area
  (an earlier rule, judged by Delta, caught 12–21% of prose LLM paragraphs spliced into
  corpus A), so the output says paragraph checks need `--contrast` rather than "no paragraph
  drifts".
- **What a passage shows.** Its lines, the likeness and Delta verdicts of its lower span, the
  start of its text, and its own top three traits (its strongest likeness signals, or its
  largest z-scores, at its own length), so a neighbouring list cannot lend it "List items".
  `--by-paragraph` lists every paragraph's statistic ("x range") against the document's
  threshold. More than half the paragraphs drifting reads "drifts throughout (13 of 17
  paragraphs)", and `-q` "drifts throughout".
- **Pooled input abstains.** Records pooled with `--pool` are short texts joined into
  windows, not a document with paragraphs, so they are not read in parts: the JSON marks
  each document `pooled`, and the output says in one dim line that paragraph checks don't
  apply to pooled records (score without `--pool` for each record's own verdict).

### How well it holds

Measured with `bench/drift.py`, the review's harness: 5 folds per corpus, each with a
contrast set of 40 documents generated from 4 of the 5 LLM drafts, and inserts from the
fifth. Inserts are runs of 1, 2 or 3 consecutive blocks of that draft (headings and lists
included) at the start, middle or end of a held-out document, 9 per document, plus two
adjacent prose paragraphs of it at the start, middle or end. Documents are also scored with
every paragraph split into 1–2 sentences, and concatenated 2, 4, 6 and 10 at a time (about 25
to 160 paragraphs), 6 of each per fold.

- **A** is bench medium (seed 0): the reference is 150 documents, and 50 are held out, on
  the reference's own topics. It is optimistic: its documents are remixed from the same
  seven essays, so held-out text shares topics and sentences with the reference.
- **B** is a topic shift: the reference is remixed from 5 of the essays, and the held-out
  documents from the other 2 (`sharpening`, `walking-in-rain`, their topics and sentences
  unseen), plus those 2 real essays.
- **C** is a topic shift the contrast covers but the reference does not: the held-out topics
  are `fence-lines` and `the-woodstove`, which the LLM drafts also write about, in documents
  of about 600 words.
- **D** is a wider topic shift in longer documents: the reference is remixed from 4 essays,
  and the held-out documents (about 1,200 words) from the other 3 (`sharpening`,
  `the-woodstove`, `walking-in-rain`), plus those 3 real essays.

| | A no spaCy | A spaCy | B no spaCy | B spaCy | C no spaCy | C spaCy | D no spaCy | D spaCy |
|---|---|---|---|---|---|---|---|---|
| held-out documents with any drift | 2/50 (4%) | 1/50 (2%) | 0/32 (0%) | 0/32 (0%) | 1/32 (3%) | 0/32 (0%) | 0/33 (0%) | 7/33 (21%) |
| same, 1–2-sentence paragraphs | 0/50 (0%) | 0/50 (0%) | 0/32 (0%) | 0/32 (0%) | 2/32 (6%) | 0/32 (0%) | 0/33 (0%) | 2/33 (6%) |
| 2 concatenated | 4/30 (13%) | 1/30 (3%) | 0/30 (0%) | 0/30 (0%) | 3/30 (10%) | 0/30 (0%) | 0/30 (0%) | 3/30 (10%) |
| 4, 6 or 10 concatenated | 0/90 (0%) | 0/90 (0%) | 0/90 (0%) | 0/90 (0%) | 3/90 (3%) | 1/90 (1%) | 0/90 (0%) | 25/90 (28%) |
| real held-out essays with any drift | - | - | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 0/3 (0%) | 1/3 (33%) |
| distinct writer paragraphs behind those | 2 | 1 | 0 | 0 | 5 | 1 | 0 | 8 |
| 1 / 2 / 3 blocks found | 69% / 87% / 97% | 72% / 86% / 97% | 57% / 88% / 96% | 67% / 91% / 98% | 69% / 82% / 97% | 66% / 76% / 97% | 62% / 86% / 96% | 76% / 93% / 99% |
| insert found with nothing else drifting | 79% | 82% | 78% | 81% | 79% | 78% | 80% | 68% |
| 1 block at start / middle / end | 68% / 78% / 60% | 78% / 78% / 60% | 66% / 53% / 53% | 69% / 59% / 72% | 81% / 66% / 59% | 75% / 59% / 62% | 67% / 58% / 61% | 76% / 76% / 76% |
| adjacent LLM pair, both drift: start / middle / end | 84% / 68% / 74% | 84% / 76% / 72% | 69% / 50% / 78% | 72% / 56% / 88% | 78% / 78% / 81% | 75% / 69% / 81% | 64% / 52% / 79% | 76% / 70% / 97% |
| prose insert under 30 words | 46% | 40% | 17% | 29% | 48% | 33% | 30% | 52% |
| 1 block in a 24–164-paragraph document | 46% | 48% | 42% | 52% | 48% | 39% | 35% | 48% |
| draft.md exactly lines 9 and 15 | 5/5 (100%) | 5/5 (100%) | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 3/5 (60%) | 5/5 (100%) |

Every setup but D with spaCy keeps the writer's documents at or under about 10% false drift
(13% counting repeats in A's two-document concatenations, which are 2 distinct paragraphs of
the writer). **D with spaCy is over that:** 21% of its single held-out documents and 28% of
its long concatenations have a paragraph drift. Those are 8 distinct paragraphs of the
writer, repeated across documents, each just over its threshold (1.02–1.55 times the bound
against thresholds of about 1.0–1.2), on topics the reference never saw, and carried mostly
by vocabulary (long words, word length, nominalizations) rather than syntax; one is in a
real essay (`the-woodstove.md`, 1 of the 3). Raising the minimum threshold to 1.2 would take
D to 8% / 23% but costs detection and `draft.md` on the other setups, so it is not done.
Before capping the parser's metrics in spans and restricting explanation to stronger
paragraphs, D with spaCy drifted falsely in 67% of single documents and 86% of long ones
(one rare parser metric carrying a 100-word span), and two adjacent LLM paragraphs at a
document's start or end were never both found (now 64–97%).

Before this rule (a constant floor of 1.2 times the bound, and single spans judged against a
tail fitted to all spans), the same harness gave C 9–16% false drift in single documents and
8–27% in 48–140-paragraph ones, 28 of 29 long-document flags being paragraphs of 100 words or
more resting on one span. Now every paragraph rests on two spans (none of the 59,000 clean
paragraphs rests on one). The fitted excess tail predicts the writer's held-out paragraphs
on seen topics (at p = 0.1 / 0.02 / 0.005: 0.113 / 0.037 / 0.010 on A without spaCy, 0.087 /
0.023 / 0.007 with it) and runs heavier on unseen ones (C: 0.19 / 0.05–0.06 / 0.007–0.021),
which the margin from the documents' spread and the tail's confidence bound absorb.

What this does not catch well: a single short LLM paragraph (under 30 words, 17–52%), one
paragraph in a long document (the per-document threshold rises with its length: 35–52% at
24–164 paragraphs), and a single block (57–76%). Two or three blocks are found 76–99% of
the time. So "no paragraph drifts" is not "clean": a single short paragraph in the LLM
register is missed about as often as it is found. `draft.md`'s line 15 is such a paragraph
(33 words); the folds find both planted paragraphs 3 to 5 times in 5, and the demo
reference finds both.

**These are synthetic corpora.** Every held-out document in A–D is remixed from seven
essays by one writer, so held-out text shares sentences with other held-out text, and in A
with the reference; the null is fitted to such documents, and the inserts come from five LLM
drafts on the same few topics. Real writers vary more from piece to piece, and D shows what
a new topic does with spaCy. Expect more false drift on a real writer's new topics than
these tables show, and treat a paragraph just over its threshold as a lead to read, not a
finding.

## Resolution floors

Every z uses a spread of at least half of one occurrence per chunk for counts (5% of the
mean for other metrics), so a habit that is almost always absent cannot turn one use into
dozens of standard deviations. Verdicts compare an average over n chunks with a band that
narrows as n grows, since averages vary less (see "A mean over many chunks"). That band
never narrows below 0.5 for Delta and its areas (half a standard deviation per metric), so
an area the writer never varies in, like Markdown in plain essays, cannot turn a trace into
"very different". Likeness counts only the part of each z toward the contrast drafts, about
half of |z| for noise, so its band never narrows below 0.25.
