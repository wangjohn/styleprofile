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

To try it on the sample corpus in [`examples/`](examples/), run `make demo`.

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

`build` splits texts into ~500-word windows (`--window-words N`, or `--no-window`), and `score`
uses whatever the reference used, so the two always match. `--contrast` can be repeated.

`build` prints a short summary: how the reference scores its own held-out writing, what
separates it from the contrast drafts, and any warnings; pass `--all` to see every metric.
`score` prints the verdict and the biggest differences, and `--all` prints every metric there
too.

The profile holds summaries only (each metric's mean and spread, the held-out ranges and the
contrast weights), so it stays small however large the corpus; `build --keep-chunks` also
saves every window's metrics, for debugging. Profiles and reports save each input by its
final name only (`posts`, then `posts/2024/a.md` for a file in it), however it was typed, so
they never reveal where your files live.

More commands:

- `styleprofile score draft.md writer.json -o draft.json` also saves the full JSON report;
  `--json` prints it on stdout instead of the summary, and `--quiet` prints one verdict line
  per document.
- `styleprofile score drafts/ writer.json` scores several documents at once. The output opens
  with a table giving each document its own verdict, furthest from the writer first (past
  ten close documents, the rest are counted; `--all` lists them); the figures after it are
  pooled over all of them ("Across 5 documents"). The JSON report lists each one under
  `documents`.
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
Areas are listed most different first; `--all` adds the raw Delta for each area. "Biggest
differences" lists the metrics that moved most, with one ▲ or ▼ per standard deviation. See
[docs/method.md](docs/method.md) for the metrics, the weighting math, the reliability checks
and the resolution floors.

### In a pre-commit hook or CI

`--fail-above {somewhat,clearly,very}` makes `score` exit with status 3 when any document is
at least that different from the writer, and `--fail-likeness {few,leans,like}` when any
document's LLM-likeness reaches a few traits, leans LLM, or like the LLM drafts (it needs a
reference built with `--contrast`). Each document is judged on its own, so one drifting
draft fails the run even when the rest are close:

```bash
styleprofile score -q --fail-above clearly --fail-likeness leans drafts/ writer.json
```

`-r writer.json` gives the reference before the drafts instead of last, for tools that append
file names, such as [pre-commit](https://pre-commit.com). This `.pre-commit-config.yaml`
checks every staged Markdown file:

```yaml
repos:
  - repo: local
    hooks:
      - id: styleprofile
        name: styleprofile
        entry: styleprofile score -q --fail-above clearly -r writer.json
        language: system
        types: [markdown]
        require_serial: true  # one run for all the files: spaCy loads once
```

A failing commit prints each file's verdict, then a `failed:` line on stderr for each file
that reached a level, in the order given:

```
posts/old-maps.md: very different (Delta 4.61); LLM-likeness leans LLM (13.83)
posts/sharpening.md: close (Delta 0.57); LLM-likeness like the reference (0.08)
failed: posts/old-maps.md: delta very different
```

| Exit status | Meaning |
|---|---|
| 0 | scored; no document reached a `--fail-above` or `--fail-likeness` level |
| 1 | an error, such as a missing file or an unreadable profile |
| 2 | invalid command-line usage |
| 3 | a document reached the `--fail-above` or `--fail-likeness` level |

Each document is named as you typed it, and a JSONL record by its file and id
(`exports/comments.jsonl:17`); standard input is `<stdin>`. A document with no verdict,
because it is too short to judge (its line says `too short to judge (36 words)`) or has no
metric in common with the reference, never fails a run. With a fail flag, the JSON report (`--json` or `-o`)
records the levels asked for under `fail`, and under `failed` each document that reached
one, with the Delta and likeness verdicts that did; that is the form for scripts to read.

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
- **A verdict means "unlike this reference", not proof of authorship.** A human can drift
  from their own profile, and a model can be prompted toward it.

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
relative to the folder you pass (for JSONL, by record `id`), so keep the same layout: a
file given directly matches by its bare file name. A set may leave out some drafts; it is
then compared only with the drafts it covers.

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
