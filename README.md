# styleprofile

Compare a draft with a writer's usual style, then see which habits make it different.
With LLM drafts of the same briefs as a contrast set, also measure resemblance to those
drafts. Scores describe this comparison; they do not prove who wrote a text.

## Install

Python 3.11+; CI tests 3.11–3.14. The first public release,
[0.2.0](https://pypi.org/project/styleprofile/0.2.0/), is available on PyPI.
Choose pip in an activated virtual environment, or uv:

```bash
python -m pip install "styleprofile[syntax]"
styleprofile setup
```

```bash
uv tool install --python 3.11 "styleprofile[syntax]"
styleprofile setup
```

`setup` installs spaCy's English model once. Make sure uv's tool executable directory is
on your PATH. For virtual-environment activation, Windows, and a standard-library-only
install, see [installation details](https://github.com/wangjohn/styleprofile/blob/main/docs/usage.md#installation-details).

## Try it in 60 seconds

```bash
styleprofile demo
```

Expected verdict: **`Overall: close`**. The command copies bundled, original samples into
`styleprofile-demo/`, builds a profile and scores a draft; it needs no checkout. The draft
contains two off-voice paragraphs that its whole-document verdict can hide. The samples
are deliberately small, so the command also warns about a thin reference. See the
[sample guide](https://github.com/wangjohn/styleprofile/blob/main/examples/README.md) for the experimental paragraph demonstration.

## Your own texts

Use one genre per profile. In these recipes, replace the example paths with your own:
`book.md` is a long manuscript, `posts/` holds independent essays, `draft.md` and `drafts/`
are new writing. Inputs are UTF-8 Markdown, text, HTML or JSONL. Profiles are summaries,
not copies of the text; keep both your source texts and generated reports out of version control.

**One long file.** Build from a book or archive, then score a new draft. The file is split
at suitable chapter headings or issue boundaries; automatic stand-in sections are less
reliable when it has no such divisions.

```bash
styleprofile build book.md
styleprofile score draft.md book.profile.json
```

**A folder of essays or posts.** Build once, then reuse the profile. Without `-o`, the
output is `<folder-name>.profile.json` in the current directory.

```bash
styleprofile build posts/
styleprofile score drafts/ posts.profile.json
```

**Tweets, comments or emails.** Use one JSON object per line with a text field and a
thread or conversation ID. Group independent conversations and pool short records:

```bash
styleprofile build comments.jsonl --text-field text --group-field thread --pool
styleprofile score new-comments.jsonl comments.profile.json --pool
```

Records in different groups are never pooled together. Without `--pool` at score time,
each record is judged separately, and a record under 75 words gets no verdict. See
[grouping and pooling](https://github.com/wangjohn/styleprofile/blob/main/docs/usage.md#comments-tweets-and-emails) for data requirements and pitfalls.

**One-shot checks.** Build the reference in memory and score immediately:

```bash
styleprofile score draft.md --against posts/
```

**LLM-likeness.** First make model drafts from the writer's own briefs, keeping topics,
genre and lengths similar and using several models. Save those drafts in `llm-drafts/`:

```bash
styleprofile build posts/ --contrast llm-drafts/ -o writer.json
styleprofile score draft.md writer.json
```

The contrast is optional: without it, you get Delta but no LLM-likeness. The weights know
only the drafts they learned from. The [contrast recipe](https://github.com/wangjohn/styleprofile/blob/main/docs/contrast.md)
includes a copyable prompt and advice on counts and independent held-out checks.

## How much text you need

| Feature | Minimum |
|---|---|
| A useful Delta reference | 15+ windows, 20,000+ words, from independent documents |
| Calibrated verdicts on short texts | 3+ documents in the reference |
| Paragraph drift checks | 10+ documents, about 200 paragraphs; build with `--by-paragraph` |
| A scorable text | 75 prose words, and a reference calibrated at that length |

The first row is a recommended size, not a hard cutoff: the small demo still returns a
verdict with a warning. Three documents alone do not guarantee a sound reference. Aim for
20 or more substantial documents for paragraph checks; check whether the profile actually
has thresholds. Short drafts get wider ranges, and texts below the shortest calibrated
length can remain “too short to judge” even above 75 words.

## Reading the output

**Delta** measures distance from the writer's reference in the writer's own standard
deviations, with each area (sentence shape, vocabulary, punctuation, and so on) counting
equally. Lower is closer. The verdict compares it with the writer's own held-out range.

| Delta | Meaning |
|---|---|
| close | within the writer's usual held-out range |
| somewhat different | up to 1.5 times that range |
| clearly different | up to twice that range |
| very different | further still |

**LLM-likeness** measures deviations in the directions the contrast drafts differ, weighted
by how well those habits separate the two sets. It appears only with a contrast set.

| LLM-likeness | Meaning |
|---|---|
| like the reference | within the writer's own range |
| a few LLM traits | above it, but less than halfway to the LLM drafts |
| leans LLM | more than halfway to the LLM drafts |
| like the LLM drafts | at or beyond the drafts' typical score |

Both scores use the reference range **at the draft's length**. Under 75 words there is no
verdict; numbers and traits are indicative only. Chunks too short to judge are left out
of the headline. “By area” divides each area's Delta by the top of its own held-out range:
close up to 1x, somewhat different to 1.5x, clearly different to 2x, very different above.
Bars use a log scale, full at 32x; biggest-difference arrows show direction, one per
standard deviation.

Human output uses short notes by default. `--verbose` on build, score or evaluate restores
full explanations; `--all` shows every metric. JSON always keeps full warnings. Use
`styleprofile show writer.json` for a saved report and `styleprofile metrics` for metric definitions.
The [command reference](https://github.com/wangjohn/styleprofile/blob/main/docs/usage.md) covers formats,
splitting, cache, worker processes and `evaluate`; [Method](https://github.com/wangjohn/styleprofile/blob/main/docs/method.md)
explains the mathematics, reliability checks and resolution floors.

## What a verdict can't tell you

A whole-document average can hide one or two paragraphs in another voice: the demo draft
reads “close” despite its two LLM-style paragraphs. English is the supported language.
A human can drift from their reference, and a model can be prompted toward it; resemblance
is not proof of authorship.

Paragraph checks are **experimental and off by default**. To try them, build with
`--by-paragraph`, then score with the same flag. On synthetic writer texts with new topics,
a paragraph falsely drifted in up to about a third of documents. These are historical
measurements, not a promise about your writing:

| Writer heldouts with false paragraph drift | A | B | C | D |
|---|---|---|---|---|
| single documents, without / with spaCy | 4% / 2% | 0% / 6% | 3% / 0% | 0% / **33%** |
| 4–10 documents joined, without / with spaCy | 0% / 0% | 0% / 0% | 3% / 0% | 0% / **30%** |

A uses reference topics; B–D use unseen topics, D the widest shift. Runs of two or three
LLM blocks were found 76–100% of the time; single blocks 57–74%. Short paragraphs under
30 words were often missed (17–48% found), as was one paragraph in a long document.
“No paragraph drifts” does not mean a text is clean. The synthetic corpora share sample
material and are optimistic; expect more uncertainty on real new topics. Treat a drift
as a lead to read. See [the full research limits](https://github.com/wangjohn/styleprofile/blob/main/docs/method.md#where-a-draft-drifts).

## Troubleshooting

Match the message below; identifiers name existing library `StyleProfileError.code` or
`Note.code` values. Pip and uv installation errors have no styleprofile code.

| What happened | What to do |
|---|---|
| “No matching distribution” | Use Python 3.11+ and check that pip is using PyPI; upgrade pip in your virtual environment. |
| “Externally managed environment” (PEP 668) | Use an activated virtual environment or `uv tool install`; don't replace system Python packages. |
| spaCy/model versions differ (`code=setup_spacy_version`, `code=syntax_model_mismatch`) | Run `styleprofile setup` in the same environment, then rebuild the profile. If spaCy itself is outside the required range, reinstall the syntax extra there. |
| Syntax metrics left out (`code=no_syntax`, `code=syntax_unavailable`) | Install the syntax extra and run `styleprofile setup`, or choose `--no-syntax`. Scoring a surface-only reference stays surface-only; rebuild to add syntax. |
| Cache unavailable (`code=cache_unavailable`) | Check `styleprofile cache`; choose a writable absolute `XDG_CACHE_HOME`, or use `--no-cache`. The run continues without the cache. |
| “Rebuild your profile” (`code=outdated`) | Rebuild a reference from its original texts, rescore a saved score, or reevaluate an evaluation. Different report majors are incompatible. |
| Newer minor report (`code=newer_report_version`) | Matching majors still load; upgrade if you need the added fields. Missing paragraph calibration requires rebuilding with `--by-paragraph`. |
| Windows console symbols look different | Symbols fall back to ASCII when the stream cannot encode them. UTF-8 inputs are still required. Use a UTF-8 terminal or `python -X utf8 -m styleprofile demo` for redirected output. |
| No usable prose (`code=no_chunks`) | Convert unsupported formats and check that the input contains prose, not only code or lists. |

CLI build/evaluate caching is on by default and stores counts that can reveal wording;
treat the cache like your texts. Library calls leave it off unless you opt in. See
[cache and workers](https://github.com/wangjohn/styleprofile/blob/main/docs/usage.md#speed),
[report compatibility](https://github.com/wangjohn/styleprofile/blob/main/docs/library.md#saved-report-compatibility),
and the [changelog](https://github.com/wangjohn/styleprofile/blob/main/CHANGELOG.md) for details.

## Use it in CI

Start by reading ordinary scores before setting a failure threshold. Each document is
judged separately; one drifting document can fail a batch. `--fail-above clearly` checks
whole-document Delta and `--fail-likeness leans` checks resemblance to the supplied contrast:

```bash
styleprofile score -q --fail-above clearly --fail-likeness leans drafts/ writer.json
```

A threshold hit returns **3**, which is an expected gate failure. Whole-document scores
can mask a few off-voice chunks; `--fail-flagged N` catches N chunks that individually read
clearly different or lean LLM. These occur by chance too: about 0–0.14% of writer chunks
in the measured corpora, with at least one in 15–27% of runs of 200 chunks. Choose N for
the document length; 1 suits a few windows, while long documents may need 2 or more.

For [pre-commit](https://pre-commit.com), save this as `.pre-commit-config.yaml`. The installed
command and `writer.json` must be available in the repository:

```yaml
repos:
  - repo: local
    hooks:
      - id: styleprofile
        name: styleprofile
        entry: styleprofile score -q --fail-above clearly --fail-flagged 1 -r writer.json
        language: system
        types: [markdown]
        require_serial: true
```

| Exit status | Meaning |
|---|---|
| 0 | scored; no document reached a requested fail level |
| 1 | an error, such as a missing file or unreadable profile |
| 2 | invalid command-line usage |
| 3 | a document reached a fail level, or could not be compared when a fail flag was given |

A text too short to judge never fails a run. An incomparable document fails when a fail
flag is given. JSON records requested thresholds under `fail` and failures under `failed`;
see [automation details](https://github.com/wangjohn/styleprofile/blob/main/docs/usage.md#fail-flags-and-automation).

## Using it from Python

```python
import styleprofile as sp

profile = sp.build("posts/", contrast="llm-drafts/")
result = profile.score_text("A draft to check against the writer.")
print(result.verdict, result.delta)
```

The tiny placeholder draft abstains; supply at least 75 prose words for a verdict.
For the string-first API, settings, saving/loading and notes, see the
[library guide](https://github.com/wangjohn/styleprofile/blob/main/docs/library.md).

Development, benchmark gates, snapshots and release checks are in
[CONTRIBUTING](https://github.com/wangjohn/styleprofile/blob/main/CONTRIBUTING.md).
The package began as the stylometry module of GoodProse.

## License

MIT
