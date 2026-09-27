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

To try it on the sample corpus in [`examples/`](examples/), run `make demo`.

The terminal shows a short summary; pass `--all` to print every metric. The JSON report
always has everything. `python -m styleprofile` works too.

Inputs can be JSONL (`--text-field`, default `text`/`body_markdown`/`output`/...), Markdown
or text files, directories of them, or `-` for stdin. Pass `--no-syntax` to skip spaCy.

## How it works

styleprofile measures about 150 interpretable metrics per window of text (sentence shape,
rhythm, vocabulary, punctuation, voice, Markdown habits, function words and, with spaCy,
syntax) and stores each one's mean, spread and held-out variation as a reference profile.
A draft then gets two scores: Delta, how far it sits from the writer overall, and
LLM-likeness, how much it deviates in the directions the contrast drafts do. See
[docs/method.md](docs/method.md) for the metrics, the weighting math, the reliability
checks and the resolution floors.

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

## Stress-testing LLM-likeness

The contrast weights learn whatever separates the writer from the drafts they were given.
When those are unedited first drafts, the strongest tells (em dashes, sentence length) may be
exactly what a light edit or a "humanizer" removes. `styleprofile evaluate` measures how much
of the separation survives editing.

Make edited copies of the contrast drafts with whatever you want to test: a person, an
editing tool, or a model asked to polish or "humanize" them. Save each set in its own folder
with the originals' file names:

```bash
styleprofile evaluate --reference-inputs posts/ --contrast llm-drafts/ \
  --edited light=edits/light humanize=edits/humanize \
  --window-words 500 --retrain --output profiles/stress.json
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
