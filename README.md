# styleprofile

Build a stylometric profile from a writer's texts, see how far a draft drifts from it, and
learn which habits separate that writer from LLM output.

## Install

styleprofile supports Python 3.11, 3.12 and 3.13.

```bash
pip install "styleprofile[syntax] @ git+https://github.com/wangjohn/styleprofile"
```

Or, from a clone, `uv sync --extra syntax` and run `uv run styleprofile`. The `syntax` extra
installs spaCy and `en_core_web_sm` for the parser-based metrics; without it, styleprofile
uses the surface metrics only and says so. `python -m styleprofile` works too.

## Quick start

Build a reference profile from the writer's texts once, then score as many drafts against it
as you like:

```bash
# The writer's posts, contrasted with LLM drafts of the same material.
styleprofile build posts/ --contrast llm-drafts/ -o writer.json

# Score a draft. Window size and the other settings come from writer.json.
styleprofile score draft.md writer.json
```

To try it on the sample corpus in [`examples/`](examples/), run `make demo`: it scores a
draft that slips into the LLM register in two paragraphs, with the experimental paragraph
check (`--by-paragraph`), first against the seven sample essays (too few to check
paragraphs, as the output says), then against a synthetic corpus remixed from them, an
optimistic stand-in for a larger archive, which finds the two.

Inputs can be Markdown or text files, HTML (`.html`/`.htm`), JSONL (`--text-field`, default
`text`/`body_markdown`/`output`/...), directories of them, or `-` for stdin. HTML is converted
to Markdown, keeping its paragraphs, lists, headings and inline formatting and dropping
navigation, site headers and footers, scripts, form controls, comment sections and subscribe
widgets; when a page has an `<article>` or `<main>`, the one holding the post is read. A `.md`
or `.txt` file that is really HTML is read as HTML, stdin is read as JSONL when every line is
a JSON object, and both are noted on stderr; `--input-format {auto,markdown,html,jsonl}`
overrides the detection for every input, directory contents included. `build` records it in
the reference, but `score` does not inherit it, since drafts are often in a different format
from the writer's archive.

A directory walk names the documents it skipped (Word, PDF, reStructuredText, ...); convert
those first, for example with pandoc. It also leaves out static-site output and templates,
which would repeat or wrap the posts: the built HTML in `_site` and `public`, and the template
folders `_layouts`, `_includes`, `layouts`, `themes` and `resources`. Name such a folder
directly to read it. `build` and `evaluate` keep only the first of documents with
word-for-word the same text, so a post and its copy can't calibrate against each other.

`build` splits texts into ~500-word windows (`--window-words N`, or `--no-window`), and joins
short ones, such as comments, into windows of the same size (see [Comments, tweets and
emails](#comments-tweets-and-emails)). `score` uses whatever the reference used, so the two
always match. `--contrast` can be repeated.

`build` prints a short summary: how the reference scores its own held-out writing, what
separates it from the contrast drafts, and any warnings; pass `--all` to see every metric.
`score` prints the verdict and the biggest differences, and `--all` prints every metric there
too. `--by-paragraph` (experimental, off by default) also reads each document in
overlapping spans of 100 words or more and points to the paragraphs that drift from the
writer (see [Where it drifts](#where-it-drifts-experimental)).

The profile holds summaries only (each metric's mean and spread, the held-out ranges and the
contrast weights), so it stays small however large the corpus; `build --keep-chunks` also
saves every window's metrics, for debugging. Profiles and reports save each input by its
final name only (`posts`, then `posts/2024/a.md` for a file in it), however it was typed, so
they never reveal where your files live.

More commands:

- `styleprofile score draft.md writer.json -o draft.json` also saves the full JSON report;
  `--json` prints it on stdout instead of the summary, and `--quiet` prints one verdict line
  per document (with `--by-paragraph`, ending "drifts at lines 9, 15" when paragraphs
  drift).
- `styleprofile score drafts/ writer.json` scores several documents at once. The output opens
  with a table giving each document its own verdict, furthest from the writer first (past
  ten close documents, the rest are counted; `--all` lists them); the figures after it are
  pooled over all of them ("Across 5 documents"). The JSON report lists each one under
  `documents`. Add `--by-paragraph` to also read each one in parts
  ([Where it drifts](#where-it-drifts-experimental)).
- `styleprofile show writer.json` (or a saved score report) shows it again without
  recomputing anything.
- `styleprofile metrics` lists every metric, what it means and its unit.
- `styleprofile evaluate` checks whether LLM-likeness survives editing of the contrast
  drafts (see [Stress-testing LLM-likeness](#stress-testing-llm-likeness)).
- `styleprofile COMMAND --help` shows a command's options.

## Reading the output

A score has two numbers:

- **Delta: how far is this from the writer overall?** The average distance of the draft's
  metrics from the writer's, in the writer's own standard deviations, with every area
  (sentence shape, punctuation, voice, ...) counting equally. Lower is closer. The report
  prints what the writer's own held-out writing scores, so 0.8 next to a typical 0.78 means
  "about as close as the writer is to themself".
- **LLM-likeness: does this look like the contrast drafts?** Only reported when the reference
  was built with `--contrast`. It counts deviations in the directions the contrast drafts
  deviate (more em dashes, more headings, ...), weighted by how well each one separated the
  drafts from the writer. The report prints what the writer and the drafts typically score.

The verdict words place each number against the writer's own range:

| Delta | Meaning |
|---|---|
| close | within the writer's usual held-out range |
| somewhat different | up to 1.5 times that range |
| clearly different | up to twice that range |
| very different | further still |

| LLM-likeness | Meaning |
|---|---|
| like the reference | within the writer's own range |
| a few LLM traits | above it, but less than halfway to the LLM drafts |
| leans LLM | more than halfway to the LLM drafts |
| like the LLM drafts | at or beyond the drafts' typical score |

Both are read against the writer's own range **at the draft's length**. Short texts vary
far more than whole windows by chance, so `build` also measures the writer's held-out text
in pieces of about 75, 150 and 300 words, and `score` judges each chunk against the range
for its own length (see [docs/method.md](docs/method.md#length-aware-verdicts)). Under 75
words there is no verdict: the output says "too short to judge (36 words)" and shows the
numbers and traits as indicative only. When several chunks are scored, each is judged at
its own length, and chunks too short to judge are left out of the headline.

"By area" breaks Delta down the same way. Areas vary by different amounts on the writer's
own text, so each area's number is its Delta ÷ the top of that area's usual held-out range:
close up to 1x, somewhat different to 1.5x, clearly different to 2x, very different above.
Areas are listed most different first; `--all` adds the raw Delta for each area. The bars
are on a log scale, full at 32x, so an LLM draft's areas (often 2x to over 20x) still rank
against each other, while 1x, 1.5x and 2x stay apart. "Biggest differences" lists the
metrics that moved most, with one ▲ or ▼ per standard deviation. See
[docs/method.md](docs/method.md) for the metrics, the weighting math, the reliability checks
and the resolution floors.

### Where it drifts (experimental)

**Experimental, and off by default.** On writer text of a topic the reference never saw, it
found a paragraph drifting in up to about a quarter of the writer's own documents (the table
below), so it runs only when you ask for it with `--by-paragraph` (`passages=True` in the
library), and what it finds is a lead to read, not a finding.

One number over a whole draft dilutes a paragraph or two in another register: `make demo`'s
draft has two paragraphs written like an LLM and still reads "close" overall. With
`--by-paragraph`, `score` also reads each document in spans of 100 words or more, two per
paragraph (one running forward from it, one back), judges each against the writer's own
range at its length, and lists up to three paragraphs that drift. Against the demo's
reference (a synthetic corpus remixed from the sample essays; see below):

```
Where it drifts (experimental)   2 of 8 paragraphs drift, read in spans of at least 100 words
  Line 9          a few LLM traits (1.04), close (Delta 1.15)
                  "But the store is more than a place to buy things — it's a…"
                  Em dashes ▲▲▲, LLM marker words (delve, crucial) ▲▲▲, Long words (7+ letters) ▲▲▲
  Line 15         a few LLM traits (0.92), close (Delta 0.89)
                  "Ultimately, the future of the village hardware store depends…"
                  "these" ▲▲▲, Nominalizations (-tion, -ment) ▲▲▲, Long words (7+ letters) ▲▲▲
```

Each passage gives its lines, how it reads (LLM-likeness, then Delta), the start of its text
and its own strongest traits. A paragraph drifts when both its spans read unlike the writer,
above a threshold `build` sets from the writer's own documents: read the same way and held
out, they show how far a paragraph of the writer's rises above the rest of its document by
chance, and how far the writer's documents differ from each other. A document of n
paragraphs is held to the 1 − 0.05 / n point of that range, so a draft of the writer's own
should rarely have a paragraph drift however long it is. A span high only because of a
neighbour has the neighbour's other span beside it, so the neighbour does not drift. When
more than half the paragraphs drift, it says the document drifts throughout. (A paragraph
that *drifts* is not the same as a chunk *flagged* in the headline: a flagged chunk is a whole
window reading clearly different.)

Measured on synthetic corpora remixed from the sample essays (A on the reference's own
topics; B, C and D on topics it never saw, D the widest shift), the share of the writer's
own held-out documents with a paragraph drifting falsely:

| | A | B | C | D |
|---|---|---|---|---|
| single documents, without / with spaCy | 4% / 2% | 0% / 0% | 3% / 0% | 0% / **21%** |
| 4–10 documents joined, without / with spaCy | 0% / 0% | 0% / 0% | 3% / 1% | 0% / **28%** |

A run of two or three LLM blocks spliced in was found 76–99% of the time, and a single block
57–76%. **A single short paragraph is often missed** (17–52% found under 30 words), and so
is one paragraph in a long document, so "no paragraph drifts" does not mean the text is
clean. Those corpora are synthetic and optimistic: expect more false drift on a real
writer's new topics. [docs/method.md](docs/method.md#where-a-draft-drifts) has the full
tables.

`--by-paragraph` lists every paragraph with its statistic (x its range, `*` for one that
drifts) against the document's threshold, and says why a paragraph above its range does not
drift. When no
paragraph drifts it is one dim line: "no paragraph drifts (8 paragraphs checked)", or why
the check could not run or means little: a document under 100 words; a reference too small
to set thresholds; a reference without `--contrast`, whose Delta-only checks rarely catch an
LLM passage. For HTML input, line numbers are those of the converted text, and the output
says so. The JSON report has it all under `passages`, and the library as
`ScoreResult.passages`.

The thresholds need a reference big enough (see
[Getting useful results](#getting-useful-results)). With the seven sample essays alone, no
paragraph drifts and the output says the reference is too small to set paragraph
thresholds. That is why `make demo` scores the draft (with `--by-paragraph`) against the
essays first, then against a larger synthetic corpus in the same voice (see
[`examples/`](examples/README.md)).

### In a pre-commit hook or CI

`--fail-above {somewhat,clearly,very}` makes `score` exit with status 3 when any document is
at least that different from the writer, and `--fail-likeness {few,leans,like}` when any
document's LLM-likeness reaches a few traits, leans LLM, or like the LLM drafts (it needs a
reference built with `--contrast`). Each document is judged on its own, so one drifting
draft fails the run even when the rest are close:

```bash
styleprofile score -q --fail-above clearly --fail-likeness leans drafts/ writer.json
```

Those two judge each document **as a whole**: the mean over its chunks (windows), which a
few off-voice passages in a long document move little. `--fail-flagged N` catches them: it
fails a document with N or more chunks that read clearly different or lean LLM on their
own, the chunks the output names under "Least like the reference" and "Most LLM-like
chunks". The writer's own text flags such a chunk now and then by chance, about one in a
thousand chunks (0–0.14% in our measurements), so a run of 200 of the writer's own chunks
names at least one in 15–27% of cases. Choose N by length: 1 suits documents of a few
windows (blog posts, essays), and 2 or more documents of a hundred windows or more.

`-r writer.json` gives the reference before the drafts instead of last, for tools that append
file names, such as [pre-commit](https://pre-commit.com). This `.pre-commit-config.yaml`
checks every staged Markdown file:

```yaml
repos:
  - repo: local
    hooks:
      - id: styleprofile
        name: styleprofile
        entry: styleprofile score -q --fail-above clearly --fail-flagged 1 -r writer.json
        language: system
        types: [markdown]
        require_serial: true  # one run for all the files: spaCy loads once
```

A failing commit prints each file's verdict, then a `failed:` line on stderr for each file
that reached a level, in the order given, with how many of its chunks are flagged on their
own whenever it has some:

```
posts/old-maps.md: very different (Delta 4.61); LLM-likeness leans LLM (13.83)
posts/sharpening.md: close (Delta 0.57); LLM-likeness like the reference (0.08)
posts/long-essay.md: close (Delta 1.08); LLM-likeness a few LLM traits (1.59) (2 of 17 chunks read clearly different or lean LLM)
failed: posts/old-maps.md: delta very different
failed: posts/long-essay.md: 2 of 17 chunks read clearly different or lean LLM
```

| Exit status | Meaning |
|---|---|
| 0 | scored; no document reached a `--fail-above`, `--fail-likeness` or `--fail-flagged` level |
| 1 | an error, such as a missing file or an unreadable profile |
| 2 | invalid command-line usage |
| 3 | a document reached the `--fail-above`, `--fail-likeness` or `--fail-flagged` level |

Each document is named as you typed it, and a JSONL record by its file and id
(`exports/comments.jsonl:17`); standard input is `<stdin>`. A document with no verdict,
because it is too short to judge (its line says `too short to judge (36 words)`) or has no
metric in common with the reference, never fails a run. With a fail flag, the JSON report (`--json` or `-o`)
records the levels asked for under `fail`, and under `failed` each document that reached
one, with the Delta and likeness verdicts that did and how many of its judged chunks are
flagged on their own (`flagged` of `chunks_judged`); that is the form for scripts to read.

## Getting useful results

- **One genre per profile.** Blog posts, fiction and email have different habits; mixing
  them widens every spread and blurs every score.
- **Enough text.** Aim for 15 or more chunks (windows) from several documents, and 20,000
  or more words, in the reference. `build` warns when a reference is thinner than that.
- **Contrast drafts from the writer's own briefs.** Have LLMs write from the same briefs or
  outlines the writer worked from, and use several models; the weights only know the drafts
  they were learned from.
- **Short texts get wider ranges, then no verdict.** A 150-word paragraph is judged against
  how much the writer's own 150-word passages vary, which is far more than whole essays do,
  so only a larger difference counts. Under 75 words, or below the shortest length the
  reference is calibrated for (it needs 3 or more documents), the verdict is "too short to
  judge", with the reason. A reference from a single document judges nothing shorter than
  half a window. Score whole drafts, or several paragraphs together, when you can.
- **Paragraph checks need more documents than verdicts do.** "Where it drifts" (experimental) sets its
  thresholds from paragraphs of the writer's own documents: it needs 10 or more documents
  and about 200 paragraphs among the (up to 40,000) words it reads, so in practice 20 or
  more documents of a few hundred words each. With fewer, no paragraph drifts and the output
  says the reference is too small; the seven sample essays are such a reference. Build with
  `--contrast`: judged by Delta alone, paragraph checks rarely catch an LLM passage.
- **A verdict means "unlike this reference", not proof of authorship.** A human can drift
  from their own profile, and a model can be prompted toward it.

### Comments, tweets and emails

Short texts can be JSONL, one record per text, or a folder with one file per text. Each
record or file is a document; a JSONL file is read in order, a folder in sorted file order.

- **Pooling.** When the median text is under a quarter of a window (125 words, by default),
  `build` joins consecutive short records from the same JSONL file, or consecutive short
  files from the same folder you named, into ~500-word windows, and says so: `note: joined
  20,000 records into 1,913 windows of about 500 words`. It never joins across two inputs
  you typed. It holds back, with a note, when that would leave fewer than 15 windows, since
  a handful of windows can't calibrate. `--pool` always pools and `--no-pool` never does.
  Long texts are windowed on their own as usual.
- **Group records by where they came from.** Pass `--group-field thread` (or
  `conversation_id`, `channel`) when records carry one. The records sharing a value in one
  input (a file, or all the files of a folder, so a thread split across monthly exports
  stays together: name the folder, not the files, and `build` notes a group it finds in
  several inputs) are then one document: they are pooled together, never with another
  group's, and held-out calibration leaves out a whole group at a time. Values compare as
  text, so `1` and `"1"` are one group. Records without a value form one more group,
  `thread=(none)`, with a note. In the writer's texts, a field no record has is an error
  that lists the fields the records do have; contrast drafts, which rarely carry the
  writer's threads, are read ungrouped instead, each draft a document, with a note.
- **Pick a field with several values, each with several records.** Grouping a
  one-writer corpus by `author` gives one document, and nothing to hold out; grouping
  tweets by a field that is new on almost every record leaves nothing to pool. `build`
  names both problems.
- **Without a group field, calibration can be far too narrow.** Each pooled window counts
  as a document. When records from different threads or authors are mixed, windows that
  mix them average out the differences between threads, so the reference's own spread
  shrinks, while a new draft comes from one thread and looks far away. In one test (one
  writer, 20 threads of 40 comments, each thread on its own topic and, in variant B, with
  a mild habit of its own), the share of a new thread's windows above the reference's
  calibrated 95th percentile, nominally 5%, was:

  | record order | variant | no group field | `--group-field thread` |
  |---|---|---|---|
  | contiguous by thread | A: topic only | 3–12% | 5–7% |
  | contiguous by thread | B: topic and habit | 8–14% | 3–4% |
  | interleaved (sorted by time, say) | A | 33–44% | 5–8% |
  | interleaved | B | 66–85% | 2–4% |

  So pass a group field when there is one. `build` suggests fields named like one
  (`thread`, `conversation_id`, `channel`, `subject`, `in_reply_to`, `parent`, and as a
  fallback `author`, `user` or `sender`) that have a few values per many records; never
  a language, an app or a count.
- **Duplicates.** A record repeating another word for word (20 words or more) is dropped
  before pooling, inside a group or across groups, so a cross-posted comment can't
  calibrate against itself. Files are compared whole.
- **Scoring.** `score` inherits `--group-field` and `--window-words`, but judges each
  record on its own, at its own length: a record under 75 words gets no verdict, and
  `score` notes when the drafts are this short. `score --pool` joins them into windows and
  judges the batch as a whole (each group apart, with a group field). With a group field,
  the per-document results are per group, one by one or pooled
  (`comments.jsonl:thread=t3`). Scoring with `--pool` against a reference that was not
  pooled reads too close, and warns.
- **Name the folder, not the files.** Files named one by one on the command line are
  separate inputs, so they are never joined; `Text` inputs in Python pool like records.

Keep corpora, drafts and generated profiles out of version control; the `.gitignore`
excludes `profiles/`, `corpora/` and `data/`.

## Stress-testing LLM-likeness

The contrast weights learn whatever separates the writer from the drafts they were given.
When those are unedited first drafts, the strongest tells (em dashes, sentence length) may be
exactly what a light edit or a "humanizer" removes. `styleprofile evaluate` measures how much
of the separation survives editing.

Make edited copies of the contrast drafts with whatever you want to test: a person, an
editing tool, or a model asked to polish or "humanize" them. Save each set in its own folder
with the originals' file names. An edited draft is matched to its original by its path
relative to the folder you pass, so keep the same layout: a file given directly matches by
its bare file name. A JSONL record matches by its `id`; in a folder, by its file's path in
the folder and its id (`2024/a.jsonl:17`), since ids often restart in each file. A set may leave out some drafts; it is
then compared only with the drafts it covers.

Short JSONL drafts are pooled as in `build`, when the writer's texts are. Each edited set
is then joined into the same windows as its originals, record by record (matched as above),
however the edits changed their lengths, so every edited window pairs with its original
window. With `--group-field`, drafts pair by group instead: `thread=t1` with `thread=t1`,
whichever files of a folder hold its records, and even when an edited set keeps only some
of a group's records. An edited set that covers only some of a pooled window's drafts is
compared with that window rebuilt from just those drafts, so its rewrite share, length and
signal survival are like with like.

A draft dropped for repeating another draft word for word keeps its edits: an edit of it
pairs with the copy kept, since their text was the same, and a note names the pairing. Only
one edit of a text is scored. When the kept copy is edited too, its own edit is used,
otherwise the first in the set, and the others are left out with a note. An edit of a draft
dropped for repeating one of the writer's texts has no draft to pair with, so it is left
out with a note.

```bash
styleprofile evaluate posts/ --contrast llm-drafts/ \
  --edited light=edits/light humanize=edits/humanize --retrain -o stress.json
```

Two edit levels are worth comparing. **Light editing** simulates ordinary polishing: each
few sentences edited for naturalness and flow, with meaning, facts and length kept.
**Humanizing** simulates a deliberate evasion attempt: text rewritten to read as if a person
wrote it, with varied sentence lengths, no em dashes and no stock phrasing.

The evaluation builds the reference with the original drafts as contrast. It then scores
every draft with weights learned without that draft. Each edited draft is matched to its
original by file name and scored with the weights that left out the original, so no draft
is judged by weights its own original shaped. It reports:

- the AUC against the reference's held-out chunks, with a 95% document-bootstrap interval
  and how it was found (`bootstrap.method`, as in the profile; see
  [docs/method.md](docs/method.md)), for the original drafts and for each edited set;
- the median likeness over chunks (as in the reference's stored range), and how many
  drafts still read "leans LLM" or "like the LLM drafts";
- **signal survival**: for the ten strongest contrast metrics, the drafts' mean z before
  and after editing, next to the reference's. "Em dashes: +22.5 → +0.0, 100% gone" means
  the edit closed the whole gap to the writer on that habit, and "0% gone" means the habit
  survived untouched. An edit can also push a habit further, shown as "% stronger". The
  metrics that survive editing are the ones a verdict on edited text can still rest on;
- how much each edit changed the text: the share of each original's 13-word sequences that
  no longer appear verbatim, and the change in length;
- with `--retrain`, the cross-validated AUC with the edited drafts added to the contrast
  set. Each original and its edits are held out together. The result shows whether weights
  that have seen edited drafts recover the separation. Every edited set adds a full copy of
  the drafts, so with two sets the retrained weights lean two to one toward edited text.

Drafts too short to score (under `--min-words`) are reported and skipped rather than
failing the run. Pass `-o report.json` to save the full report, and `styleprofile show
report.json` to see it again.

Verdicts on edited text are weaker evidence than verdicts on raw drafts. If the AUC falls
after editing, a "like the reference" reading only shows that the edits removed the habits
the score measures, not that a person wrote the text.

## Using it from Python

The library runs the same pipeline as the CLI, so it gives the same numbers:

```python
from pathlib import Path
import styleprofile as sp

profile = sp.build(Path("posts/"), contrast=Path("llm-drafts/"))
result = profile.score(sp.Text("A draft to check against the writer."))
print(result.verdict, result.delta, result.likeness_verdict.words(result.contrast_label))
```

See [docs/library.md](docs/library.md) for inputs, settings, notes and saving.

## Development

```bash
make sync
make check   # pytest, ruff check, ruff format --check, pyright
make demo    # build, score and show on the sample corpus
make bench          # time build and score on generated corpora; compare with the targets
make bench-quick    # only the case CI runs: 200k words without spaCy
make snapshots      # regenerate tests/snapshots/ after an intended change to CLI output
```

`make bench` generates seeded synthetic corpora from `examples/` (`bench/gen.py`, written to
the git-ignored `bench/corpora/`), then measures wall time, CPU time, peak memory and profile
size for each case in `bench/targets.py`. It prints a table beside the plan's targets and saves
`bench/results.json`. Pick cases with `make bench BENCH="--case big --repeat 3"`. To compare
with another revision as CI does, add `--against`:
`make bench-quick BENCH="--against origin/main --repeat 5"` also benchmarks `src/` as of
`origin/main`, alternating its runs with yours, and prints each metric's ratio to it.

### The benchmark gate in CI

CI's benchmark job runs `bench/run.py --quick --repeat 5 --against HEAD^ --check`. It
benchmarks your change and its base (the tip of `main` it merges into) in the same job, on the
same runner, and fails when your change is worse than the base by more than a margin set in
`bench/targets.py`: CPU time 1.15x, peak memory 1.1x, reference profile size 1.05x. Time is
judged on CPU time because wall time on shared runners swings by up to 75% between runs. Wall
time and the plan's targets are reported in the job summary but never fail the job.

If your change is meant to cost more, declare it in `bench/accepted.toml` in the same PR, so
reviewers approve it:

```toml
[[regression]]
pr = 15                    # your pull request's number
case = "medium-nosyntax"   # the case, from bench/targets.py
metric = "build_cpu_s"     # the metric, from MARGINS in bench/targets.py
ratio = 1.9                # the most it may cost, as change / base, with room for noise
reason = "Length-aware verdicts bootstrap each length bucket separately."
```

Open the PR first to get its number. Take the ratio from the failing job's summary (or from
`make bench-quick BENCH="--against origin/main --repeat 5"`) and add about 0.1 for noise. Add
one entry per metric that fails. An entry applies only to the PR it names, on its own CI runs
and on the push to `main` that merges it. After that it is inert, and CI warns until someone
deletes it (do so in any later PR). CI also warns when your own PR turns out not to need its
entry. To try an entry locally, pass the PR number:
`make bench-quick BENCH="--against origin/main --repeat 5 --check --pr 15"`.

### Changing CLI output

`tests/test_snapshots.py` compares the exact output of `build`, `score`, `show`, `metrics` and
`evaluate` on `examples/` with the files in `tests/snapshots/`. When you change what the CLI
prints, or rebase onto a change that did, run `make snapshots`. It installs spaCy (the
`syntax` extra) first, so the syntax snapshots refresh too. Review the diff in `tests/snapshots/` and commit it
with the change, so reviewers see the output change.

The package began as the stylometry module of GoodProse.

## License

MIT
