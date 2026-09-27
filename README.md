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

## Development

```bash
make sync
make check   # pytest, ruff check, ruff format --check, pyright
make demo    # build, score and show on the sample corpus
```

The package began as the stylometry module of GoodProse.

## License

MIT
