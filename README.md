# styleprofile

Build a stylometric profile from a writer's texts, see how far a draft drifts from it, and
learn which habits separate that writer from LLM output.

## Install

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

Inputs can be Markdown or text files, JSONL (`--text-field`, default
`text`/`body_markdown`/`output`/...), directories of them, or `-` for stdin. `build` splits
texts into ~500-word windows (`--window-words N`, or `--no-window`), and `score` uses whatever
the reference used, so the two always match. `--contrast` can be repeated.

`build` prints a short summary: how the reference scores its own held-out writing, what
separates it from the contrast drafts, and any warnings; pass `--all` to see every metric.
`score` prints the verdict and the biggest differences, and `--all` prints every metric there
too. More commands:

- `styleprofile score draft.md writer.json -o draft.json` also saves the full JSON report;
  `--json` prints it on stdout instead of the summary, and `--quiet` prints one verdict line.
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

"By area" breaks Delta down the same way, and "Biggest differences" lists the metrics that
moved most, with one ▲ or ▼ per standard deviation. See [docs/method.md](docs/method.md) for
the metrics, the weighting math, the reliability checks and the resolution floors.

## Getting useful results

- **One genre per profile.** Blog posts, fiction and email have different habits; mixing
  them widens every spread and blurs every score.
- **Enough text.** Aim for 15 or more chunks (windows) from several documents, and 20,000
  or more words, in the reference. `build` warns when a reference is thinner than that.
- **Contrast drafts from the writer's own briefs.** Have LLMs write from the same briefs or
  outlines the writer worked from, and use several models; the weights only know the drafts
  they were learned from.
- **Short texts are noisy.** Under about 150 words, most metrics rest on a handful of
  sentences.
- **A verdict means "unlike this reference", not proof of authorship.** A human can drift
  from their own profile, and a model can be prompted toward it.

Keep corpora, drafts and generated profiles out of version control; the `.gitignore`
excludes `profiles/`, `corpora/` and `data/`.

The flat form of earlier releases (`styleprofile posts/ --output writer.json`, with
`--reference` to score) still works in this release and prints the equivalent new command.

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
styleprofile evaluate --reference-inputs posts/ --contrast llm-drafts/ \
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

- the AUC against the reference's held-out chunks, with a 95% document-bootstrap interval,
  for the original drafts and for each edited set;
- the median likeness, and how many drafts still read "leans LLM" or "like the LLM drafts";
- **signal survival**: for the ten strongest contrast metrics, the drafts' mean z before
  and after editing, next to the reference's. "Em dashes: +22.5 → +0.0, 100% gone" means
  the edit closed the whole gap to the writer on that habit, and "0% gone" means the habit
  survived untouched. An edit can also push a habit further, shown as "% stronger". The
  metrics that survive editing are the ones a verdict on edited text can still rest on;
- how much each edit changed the text: the share of each original's 13-word sequences that
  no longer appear verbatim, and the change in length;
- with `--retrain`, the cross-validated AUC with the edited drafts added to the contrast
  set. Each original and its edits are held out together. The result shows whether weights
  that have seen edited drafts recover the separation.

Verdicts on edited text are weaker evidence than verdicts on raw drafts. If the AUC falls
after editing, a "like the reference" reading only shows that the edits removed the habits
the score measures, not that a person wrote the text.

## Development

```bash
make sync
make check   # pytest, ruff check, ruff format --check, pyright
make demo    # build, score and show on the sample corpus
```

The package began as the stylometry module of GoodProse.

## License

MIT
