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
'close'
>>> print(f"Delta {result.delta:.2f}, LLM-likeness {result.likeness_verdict}")
Delta 0..., LLM-likeness like the reference

```

`result.verdict` and `result.likeness_verdict` use the same words as the CLI (see
[Reading the output](../README.md#reading-the-output)). `result.to_text()` is what
`styleprofile score` prints, and `result.report` is the full JSON report.

## Inputs

`build`, `Profile.score` and `evaluate` take one input or a list of them:

- **a path**, as a `str` or `Path`: a Markdown, text or JSONL file, a folder of them, or
  `"-"` for stdin, exactly as on the command line;
- **`sp.Text("...")`** for raw text. Its optional `name` identifies it in reports;
- **`sp.Chunk(id, source, text)`**, used as given.

A plain `str` is always a path, never text, so a typo in a folder name fails instead of
being profiled as a two-word text. A `str` that can't be a path (it spans lines, say)
fails with a message pointing to `Text`.

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
>>> small.score(Path("examples/draft.md"), window_words=0).warnings[0]
'window sizes differ from the reference (off vs 300); z-scores assume equal-sized chunks'

```

`Settings` holds `window_words` (0 turns windowing off), `min_words`, `text_field` for
JSONL, `syntax`, `top_k` and `input_format`. `Profile.score` inherits all of them except
`input_format` from the profile, and takes keyword overrides, as `styleprofile score` takes
flags. A different window size or syntax setting is warned about in the report, and a
different `min_words` gets a note.

`syntax="auto"`, the default, uses spaCy when it's installed. Without spaCy, it runs with
the surface metrics only and adds a note. `syntax=True` raises `SyntaxUnavailableError`
instead, and `syntax=False` skips spaCy.

## Notes, warnings and errors

The library never prints.

- **`notes`**: `Note(message, code)` objects on `Profile`, `ScoreResult` and `Evaluation`
  that describe the run, such as syntax metrics left out or an input given twice. The CLI
  prints them as `note:` lines. They aren't saved.
- **`warnings`**: strings saved in the report, about the text itself, such as short chunks
  or mismatched settings.
- **Errors**: problems raise `StyleProfileError`, whose `code` names the kind of problem.
  Messages never mention command-line flags. The notes collected before the error are in
  its `notes`.

A `progress` callback, if given, is called with a `Progress` at the start of each phase.

## Saving, loading and evaluating

- `profile.save("writer.json")` and `sp.Profile.load("writer.json")` read and write the
  same files as `styleprofile build` and `styleprofile score`. `result.save(path)` writes a
  score report.
- `sp.evaluate(inputs, contrast, {"light": Path("edits/light")})` runs the rewording stress
  test that `styleprofile evaluate` runs.
- `sp.build_reference(chunks)` and `sp.score(chunks, reference)` are the lower-level steps.
  They measure chunks exactly as given: no windowing, no inherited settings, and no syntax
  metrics unless you pass `parser=sp.load_parser()`.
