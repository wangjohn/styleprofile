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
every input you score at once; per-document results arrive with plan PR 7.
`result.to_text()` is what `styleprofile score` prints, and `result.report` is the full JSON
report: the live dict, not a copy.

## Inputs

`build`, `Profile.score` and `evaluate` take one input or a list of them:

- **a path**, as a `str` or `Path`: a Markdown, text or JSONL file, a folder of them, or
  `"-"` for stdin, exactly as on the command line;
- **`sp.Text("...")`** for raw text. Its optional `name` identifies it in reports and
  pairs an edited text with its original in `evaluate`. Unnamed texts are `text1`,
  `text2`, ... in order, and two texts with the same name are an error;
- **`sp.Chunk(id, source, text)`**.

A plain `str` is always a path, never text, so a typo in a folder name fails instead of
being profiled as a two-word text. A missing `str` that reads like text (it has spaces,
say) fails with a message pointing to `Text`.

Every input is cut into windows, `Chunk`s included. Re-windowing chunks you already cut
is harmless (they keep their documents); pass `Settings(window_words=0)` to use them as
they are.

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
JSONL, `syntax`, `top_k` and `input_format`. A profile records them verbatim, and
`profile.settings` reads them back. `Profile.score(inputs, settings=None, **overrides)`
inherits them, except that it uses spaCy only when the profile has syntax metrics and
reads drafts with `input_format="auto"`. Pass whole `Settings` to replace them, or keyword
overrides (`window_words=0`) to change single fields, as `styleprofile score` takes flags.
Leave a keyword out to inherit it; the keywords are typed (`api.SettingsOverrides`), so a
type checker catches a misspelled one. A different window size or syntax setting is warned about in the report, and a different
`min_words` gets a note.

`syntax="auto"`, the default, uses spaCy when it's installed. Without spaCy, it runs with
the surface metrics only and adds a note. `syntax=True` raises `SyntaxUnavailableError`
instead, and `syntax=False` skips spaCy.

## Notes, warnings and errors

The library never prints.

- **`notes`**: a tuple of `Note(message, code, setting)` on `Profile`, `ScoreResult` and
  `Evaluation`, describing the run. `code` is a `NoteCode`: syntax metrics left out, an
  input given twice, an overridden setting, or a reference too thin to trust (one note per
  reason). The CLI prints them as `note:` lines. They aren't saved.
- **`warnings`**: a tuple of strings saved in the report, about the text itself, such as
  short chunks or mismatched settings.
- **Errors**: problems raise `StyleProfileError` (`SyntaxUnavailableError` is one), whose
  `code` names the kind of problem. Messages never mention command-line flags; when one
  is about a setting, `setting` names it. The notes collected before the error are in its
  `notes`, and a file that can't be read is a `StyleProfileError` too, with the `OSError`
  as its cause.

A `progress` callback, if given, is called with a `Progress` at the start of each `Phase`:
`READ`, `LOAD_PARSER` (only when spaCy is used), then `BUILD`, `SCORE` or `EVALUATE`, then
`DONE`.

## Saving, loading and evaluating

- `profile.save("writer.json")` and `sp.Profile.load("writer.json")` read and write the
  same files as `styleprofile build` and `styleprofile score`. `save` also sets
  `profile.path`, which later scores record. `result.save(path)` writes a score report.
  A report saved by another version of styleprofile is refused with a message saying to
  build (or score) it again; nothing is migrated before 0.2.0.
- `sp.evaluate(inputs, contrast, {"light": Path("edits/light")})` runs the rewording stress
  test that `styleprofile evaluate` runs.
- `build_reference(chunks)` and `score(chunks, reference)` in `styleprofile.profile` are
  the lower-level steps. They measure chunks exactly as given: no windowing, no inherited
  settings, and no syntax metrics unless you pass `parser=load_parser()` (from
  `styleprofile.syntax`).
