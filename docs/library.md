# Using styleprofile from Python

The library runs the same pipeline as the `styleprofile` command: it reads the same inputs,
cuts them into the same windows, loads spaCy the same way, and a score inherits its
profile's settings. The CLI and the library can't give different numbers.

The examples on this page run as a test (`tests/test_api.py`) from the repository root, on
the sample corpus in [`examples/`](../examples/).

## Build a profile, then score a draft

```python
>>> from pathlib import Path
>>> import styleprofile as sp
>>> profile = sp.build(Path("examples/writer"), contrast=Path("examples/llm-drafts"))
>>> draft = Path("examples/draft.md").read_text(encoding="utf-8")
>>> result = profile.score(sp.Text(draft))
>>> result.verdict
<Verdict.CLOSE: 'close'>
>>> print(f"Delta {result.delta:.2f}, {result.likeness_verdict.words('LLM')}")
Delta 0..., like the reference

```

`result.verdict` is a `Verdict` and `result.likeness_verdict` a `LikenessVerdict`
(`.words(label)` names the contrast set). Both are string enums with the same words as the
CLI (see [Reading the output](../README.md#reading-the-output)). A score with no metric in
common with the profile is `Verdict.NOT_COMPARABLE`. Delta and the verdicts are pooled over
every input you score at once, each chunk read against the writer's range at its own length
(see [Several documents at once](#several-documents-at-once) for each document's own).
`result.to_text()` is what `styleprofile score` prints, and `result.report` is the full JSON
report: the live dict, not a copy. Its keys are typed by the TypedDicts in
`styleprofile.schema` (`ReferenceReport`, `ScoreReport`, `EvaluationReport`), so a type
checker catches a misspelled key.

A text too short to judge gets no verdict: `judged` is False, both verdicts are `TOO_SHORT`,
and `reason` says why (see [Length-aware verdicts](method.md#length-aware-verdicts)).

```python
>>> short = profile.score(sp.Text("The stones shift; the posts do not. My father kept his."))
>>> short.verdict, short.likeness_verdict, short.judged
(<Verdict.TOO_SHORT: 'too short to judge'>, <LikenessVerdict.TOO_SHORT: 'too short to judge'>, False)
>>> short.reason
"under 75 words, the writer's own text varies too much by chance to judge"

```

**Experimental:** `passages=True` also reads each document in spans of 100 words or more to
show where it drifts (see [Where it drifts](../README.md#where-it-drifts-experimental)). It is
off by default: on writer text of topics the reference never saw, it found a paragraph
drifting in up to about a third of the writer's own documents (see
[method.md](method.md#where-a-draft-drifts)). `result.passages` holds one `Passage` per
paragraph, in order: its `lines` (first and last line of its prose), `words`, `excerpt`, the
figures of the lower of its two spans (`delta`, `verdict`, `likeness`, `likeness_verdict`),
its own `traits`, each a `Trait(metric, z, value, reference)`, and whether it `drifts`; each
names its `document` as `result.documents` does. Without `passages=True` it is empty. With a
reference as small as the seven sample essays no paragraph drifts: it is too small to set
paragraph thresholds (`report["passages"][0]["sensitive"]` is False), though the planted
paragraph reads highest.

```python
>>> result = profile.score(sp.Text(draft), passages=True)
>>> passage = result.passages[3]
>>> passage.lines, passage.words, passage.excerpt[:32], passage.drifts
((9, 9), 60, 'But the store is more than a pla', False)
>>> max(result.passages, key=lambda p: p.likeness).lines
(9, 9)
>>> result.to_text(by_paragraph=True)  # doctest: +ELLIPSIS
'STYLE COMPARISON...By paragraph...'

```

## Several documents at once

`result.delta`, `result.verdict` and the likeness figures are pooled over every chunk of
every document scored, so one very different draft can make the pooled verdict "very
different" while the rest are close. `result.documents` judges each document on its own
chunks, the same way, as a tuple of `DocumentResult` in input order:

```python
>>> mixed = profile.score([Path("examples/writer/sharpening.md"), Path("examples/llm-drafts/old-maps.md")])
>>> mixed.verdict
<Verdict.CLEARLY_DIFFERENT: 'clearly different'>
>>> for document in mixed.documents:
...     print(document.name, document.words, document.verdict, document.judged)
sharpening.md 637 close True
old-maps.md 620 very different True
>>> [document.name for document in mixed.failing(sp.Verdict.CLEARLY_DIFFERENT)]
['old-maps.md']

```

Each `DocumentResult` has:

- `name`, as reports list it: a file by its saved path (`posts/2024/a.md`), a JSONL record
  by its file and id (`comments.jsonl:17`), a `Text` by its name. Records that share an id
  in one file stay separate documents, named by line (`same@3`), with a note. Ids are
  shown as given, `x#w2` included.
- `path`, the saved path of the file it came from, and `location`, the file as you gave it
  (`drafts/2024/a.md`; `<stdin>` for standard input). `location` is None for a `Text`, and
  for a report loaded from disk, since reports never save the paths you typed. `shown` is
  what the command line prints: `location`, plus a JSONL record's id
  (`exports/comments.jsonl:17`), or else `name`.
- `words`, `chunks`, `delta`, `verdict`, `likeness` and `likeness_verdict`, judged at the
  document's own chunks' lengths by the same function as the pooled verdict;
- `judged`, `chunks_judged` and `reason`: a document too short to judge has `judged`
  False, both verdicts `TOO_SHORT`, and `reason` says why, as for a whole score; one
  judged on only some of its chunks has `chunks_judged` below `chunks`;
- up to three `differences`, as (metric, mean z), and `signals`, as (metric, mean z, share
  of the likeness). The z values are uncapped, so a metric the reference never varies on
  can show a large one; the shares are approximate, from each chunk's five strongest
  signals.

The JSON report has the same under `documents` (typed by `schema.DocumentEntry`), apart
from `location` and `shown`.
`result.failing(above, likeness, flagged)` gives the documents that reach a Delta or
likeness verdict, or that have at least `flagged` chunks reading clearly different or
leaning toward the contrast set on their own (`document.flagged`), as `styleprofile score
--fail-above`, `--fail-likeness` and `--fail-flagged` check; a document with no verdict
(too short to judge, or not comparable) never does. The verdicts judge each document as a
whole, which a few very different chunks among many close ones move little: `flagged`
catches those.

## Inputs

`build`, `Profile.score` and `evaluate` take one input or a list of them:

- **a path**, as a `str` or `Path`: a Markdown, text, HTML or JSONL file, a folder of
  them, or `"-"` for stdin, exactly as on the command line (see the README for how HTML is
  converted, formats are detected with `input_format="auto"`, and folders are walked);
- **`sp.Text("...")`** for raw text. Its optional `name` identifies it in reports and
  pairs an edited text with its original in `evaluate`. Unnamed texts are `text1`,
  `text2`, ... in order, and two texts with the same name are an error;
- **`sp.Chunk(id, source, text)`**.

A plain `str` is always a path, never text, so a typo in a folder name fails instead of
being profiled as a two-word text. A missing `str` that reads like text (it has spaces,
say) fails with a message pointing to `Text`.

Every input is cut into windows, `Chunk`s included. Re-windowing chunks you already cut
is harmless (they keep their documents); pass `Settings(window_words=0)` to use them as
they are. Short texts are pooled as on the command line: the `Text` inputs of one call, like
the records of one JSONL file, the files of one folder (or chunks sharing a source), are
joined in order into windows when most are short (see `pool` below). Files and records are the documents;
`Settings(group_field="thread")` groups JSONL records into documents by that field, and each chunk carries its document in `Chunk.document`.

## Settings

```python
>>> settings = sp.Settings(window_words=300, syntax=False)
>>> small = sp.build(Path("examples/writer"), settings)
>>> small.settings.window_words
300
>>> small.score(Path("examples/draft.md")).report["settings"]["window_words"]
300
>>> small.score(Path("examples/draft.md"), window_words=0).warnings
('window sizes differ from the reference (off vs 300); z-scores assume equal-sized chunks',)

```

`Settings` holds `window_words` (0 turns windowing off), `min_words`, `text_field` for
JSONL, `syntax`, `top_k`, `input_format`, `group_field` (the JSONL field that groups
records into documents), `pool` (`"auto"`, `True` or `False`: join short texts into
windows; `"auto"` does when the median text is under a quarter of a window) and `split_on`
(`"auto"`, `"heading"`, `"heading:N"`, `"rule"` or `"none"`: split long texts into
documents at their headings or rules; `"auto"` does when the texts are fewer than 10 documents, too few to
calibrate well, as `sp.build(sp.Text(manuscript, "book"))` is, and cuts a lone text with
neither into stand-ins). A profile records them verbatim, plus `pool_used`, `split_used` and
`contrast_split_used`, and `profile.settings`
reads them back. `Profile.score(inputs, settings=None, **overrides)` inherits them, except
that it uses spaCy only when the profile has syntax metrics, reads drafts with
`input_format="auto"`, scores each draft on its own (`pool=False`; pass `pool=True` to
judge short drafts as one batch), and keeps each draft whole (`split_on="none"`; pass
`split_on="heading"`, or `"heading:2"` for chapters under parts, for a verdict per
chapter). Pass whole `Settings` to replace them, or keyword
overrides (`window_words=0`) to change single fields, as `styleprofile score` takes flags.
Leave a keyword out to inherit it; the keywords are typed (`api.SettingsOverrides`), so a
type checker catches a misspelled one. A different window size or syntax setting is warned
about in the report, and a different `min_words` gets a note.

`syntax="auto"`, the default, uses spaCy when it and its English model are installed
(`pip install "styleprofile[syntax]"`, then `styleprofile setup` for the model). Without
them, it runs with the surface metrics only and adds a note saying which is missing. `syntax=True` raises `SyntaxUnavailableError`
instead, and `syntax=False` skips spaCy.

## Notes, warnings and errors

The library never prints.

- **`notes`**: a tuple of `Note(message, code, setting)` on `Profile`, `ScoreResult` and
  `Evaluation`, describing the run. `code` is a `NoteCode`: syntax metrics left out, an
  input given twice, an overridden setting, a reference too thin to trust (one note per
  reason), how inputs were read (HTML or JSONL detected, HTML with no text, documents or
  static-site folders a walk skipped), duplicate documents dropped, short
  texts pooled into windows, records with no group, or short drafts scored one by one. The CLI prints
  them as `note:` lines. They aren't saved.
- **`warnings`**: a tuple of strings saved in the report, about the text itself, such as
  short chunks or mismatched settings.
- **Errors**: problems raise `StyleProfileError` (`SyntaxUnavailableError` is one), whose
  `code` names the kind of problem. Messages never mention command-line flags; when one
  is about a setting, `setting` names it. The notes collected before the error are in its
  `notes`, and a file that can't be read is a `StyleProfileError` too, with the `OSError`
  as its cause.

A `progress` callback, if given, is called with a `Progress` at the start of each `Phase`:
`READ`, `LOAD_PARSER` (only when spaCy is used), then `BUILD`, `SCORE` or `EVALUATE`, then
`DONE`. Within the work, `MEASURE` is reported again after each chunk (`done` of `total`
chunks, `words` so far, and `parsing` when spaCy reads them), then a reference reports
`CALIBRATE` and, with a contrast set, `MEASURE_CONTRAST` (per chunk) and `CONTRAST`.

Two more options of `build`, `Profile.score` and `evaluate` change how fast a run goes, never
its numbers, so they are not `Settings` and are never saved:

- `jobs`: how many processes run the spaCy parser. The default, 0, uses one per CPU, at most
  4 and no more than fit in a quarter of the machine's memory, once there are 50,000 words to
  parse; 1 parses in the calling process. Workers are spawned, so each imports your main
  script again: they start only when the call runs inside an `if __name__ == "__main__":`
  block of the script (read from its syntax tree), or from a notebook or interactive
  session. Otherwise everything is parsed in one process, whatever `jobs` says.
- `cache`: with `True` (the default for `build` and `evaluate`), chunks measured by an
  earlier run come from the measurement cache (see the README) and new ones are added to
  it; `False` neither reads nor writes it. `Profile.score` uses it only with `cache=True`.
  `STYLEPROFILE_NO_CACHE=1` turns it off everywhere. When it cannot be used, the run goes
  on without it and adds a `NoteCode.CACHE_UNAVAILABLE` note.

## Saving, loading and evaluating

- `profile.save("writer.json")` and `sp.Profile.load("writer.json")` read and write the
  same files as `styleprofile build` and `styleprofile score`. `save` also sets
  `profile.path`, whose file name later scores record. `result.save(path)` writes a score
  report. Reports never save paths: inputs are recorded by their final name (`posts`,
  `posts/2024/a.md`; see [method.md](method.md)), while `result.sources` lists the real
  files read, unsaved. A profile keeps summaries only; `sp.build(..., keep_chunks=True)`
  also saves every chunk's metrics, for debugging.
  A report saved by another version of styleprofile, or one lacking or garbling a part this
  version needs, is refused with a message naming the part and saying to build (or score)
  it again; nothing is migrated before 0.2.0.
- `sp.evaluate(inputs, contrast, {"light": Path("edits/light")})` runs the rewording stress
  test that `styleprofile evaluate` runs.
- `build_reference(chunks)` and `score(chunks, reference)` in `styleprofile.profile` are
  the lower-level steps. They measure chunks exactly as given: no windowing, no inherited
  settings, and no syntax metrics unless you pass `parser=load_parser()` (from
  `styleprofile.syntax`). They measure in the calling process without the cache unless you
  pass `measurer=Measurer(cache=MeasurementCache(), jobs=0)` (from `styleprofile.measure`
  and `styleprofile.cache`); close it, or use it in a `with` block, to stop its workers and
  write the cache.
