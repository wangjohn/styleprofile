# Changelog

All notable changes to styleprofile. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers follow
[Semantic Versioning](https://semver.org/); before 1.0, a minor version may break things.

## [0.2.0] - Unreleased

The first release on PyPI: `pip install styleprofile`, or `pip install "styleprofile[syntax]"`
then `styleprofile setup` for the parser-based metrics.

> [!IMPORTANT]
> **Rebuild your profiles.** Reports now carry report version 7, and styleprofile refuses
> any other version with a message saying to rebuild. Run `styleprofile build` again on the
> writer's texts, and `styleprofile score` again for saved score reports. There is no
> migration: several metrics, the calibration and the report layout all changed.

### Breaking changes

- **Report version 7.** Profiles and score reports from earlier versions are refused with a
  "rebuild it" message rather than migrated (#9, #13).
- **The flat command form is gone.** `styleprofile posts/ --output x.json` now says which
  command to run; use `styleprofile build` and `styleprofile score` (#9).
- **The `syntax` extra installs spaCy 3.8 only.** PyPI refuses packages that depend on a
  URL, so spaCy's English model is installed by `styleprofile setup` instead (see Added).
- **Nominalizations are counted more strictly**, so `nominalizations_per_1k` values differ
  from earlier profiles (see Fixed).
- **Reference profiles leave out per-chunk rows** unless built with `--keep-chunks`, and
  save each input by its name, never as a path (#14).

### Added

- **Build a reference once, then score drafts against it**: the `build`, `score`, `show`
  and `metrics` commands. `score` reads the window size and every other setting from the
  reference, so a draft is always cut the way the reference was; `show` displays a saved
  profile or report without recomputing it, and `metrics` lists every metric by area (#7).
- `styleprofile setup` installs the spaCy English model the package is tested with
  (`en_core_web_sm` 3.8.0), checking the file's hash, and refuses a spaCy other than the
  3.8 series the model is built for. When spaCy is installed but the model is not, the
  note and the error that say syntax metrics are left out now suggest it.
- **A Python library that runs the CLI's pipeline**: `styleprofile.build`,
  `Profile.score`, `styleprofile.evaluate`, `Settings`, `Text`, notes and typed errors, so
  the library and the command line give the same numbers (#13). See
  [docs/library.md](https://github.com/wangjohn/styleprofile/blob/main/docs/library.md).
- **Length-aware verdicts.** `build` calibrates the writer's held-out range at about 75, 150
  and 300 words, and `score` judges each chunk at its own length. Text under 75 words, or
  shorter than the reference is calibrated for, gets "too short to judge" instead of a
  confident verdict (#15).
- **A verdict per document** when scoring several files, a table of them furthest first,
  and `--fail-above` and `--fail-likeness` for pre-commit hooks and CI, with exit status 3
  (#16). `--fail-flagged N` fails a document with N or more chunks that read clearly
  different or lean LLM on their own (#20).
- **Input formats**: HTML (converted to Markdown, keeping paragraphs and dropping page
  furniture), `--input-format`, detection of HTML in `.md`/`.txt` files and of JSONL on
  stdin, and a report of the files a directory walk skipped (#12).
- **How far to trust the contrast**: the contrast AUC has a 95% interval from a
  document bootstrap, and a length-only baseline shows how much of the separation text
  length alone explains (#1).
- **A typed report schema** (`styleprofile.schema`) and strict validation of saved reports
  (#17).
- `styleprofile evaluate`, a rewording stress test that measures how much of the LLM-likeness
  signal survives editing of the contrast drafts (#2).
- Python 3.11 support; 3.11, 3.12 and 3.13 are tested (#8).
- `py.typed`, so type checkers read the library's annotations.
- **Comments, tweets and other short texts.** When the median text is under a quarter of a
  window, `build` joins consecutive short records or files into ~500-word windows, and says
  so; `--pool` and `--no-pool` override it. `--group-field thread` (or another field) makes
  the records sharing a value one document: pooled together, never with another group's,
  and held out together in calibration. `build` suggests a group field when records carry
  one and warns when pooling without one may make calibration too narrow. `score` inherits
  the group field and judges each record at its own length, or pools a batch with `--pool`;
  `evaluate` pools and pairs edited drafts the same way (#18).
<!-- Plan PR 11, splitting a single large file: add here. -->
<!-- Plan PR 12 (#21), showing where a draft drifts: add here. -->

### Changed

- **Faster and smaller.** The contrast AUC's bootstrap is about 5x faster at the same
  accuracy, and a reference profile holds summaries only: 0.05 MB instead of 105 MB for 20k
  comments (#14).
- **Display.** "By area" shows each area's Delta as a multiple of its usual range, sorted by
  verdict; raw area Deltas move to `--all`. Saved profiles get normal file permissions (#10).
- `build` and `evaluate` keep only the first of documents with word-for-word the same text
  (#12).
<!-- Plan PR 13 (#22), progress reporting, spaCy throughput and the measurement cache: add here. -->

### Fixed

- **Nominalizations** count nouns that name the action, state or quality of a different
  verb or adjective (decision, motion, darkness, distance), and no longer count fence,
  city, sentence, science, moment and similar words: a word needs two or more letters
  before the suffix and must not be on a list of words whose ending is part of the root,
  that have no English base (quality, community), or whose only related verb is the same
  word (question, document). Plurals in -ities and -nesses, which were missed, now count,
  and US spellings count as British ones (defense, defence). On the sample corpus the
  writer's rate falls from 3.1 to 1.8 per 1k words and the LLM drafts' from 35.4 to 27.8.
  The metric is still somewhat topic-sensitive (see docs/method.md).
- A pooled verdict over many short chunks was harsher than almost every chunk in it; the
  bound a pooled mean is read against is fixed, and a few very different chunks inside a
  close run are always named (#20).
- HTML saved line by line in `<p>` tags lost its paragraph breaks and read as one paragraph
  (#12).
- The library and the command line gave different numbers: the library did not window texts
  or inherit a reference's settings (#13).
- Calibrated verdict ceilings are floored, so near-zero held-out ranges no longer inflate
  verdicts (#6); input handling and output safety fixes (#3).

### Development

- A benchmark harness (`make bench`) and a CI gate that compares CPU time, peak memory and
  profile size with the base revision in the same job (#11, #19); snapshot tests of the
  CLI's exact output (#11).
- CI builds the sdist and wheel, installs the wheel into a clean Python 3.11 environment,
  and runs `build`, `score` and the README's library example on `examples/`, then runs the
  test suite from the unpacked sdist. A release workflow publishes the files it checked to
  PyPI with trusted publishing when a `v*` tag is pushed.
- Every metric's definition lives in `metrics.py` (#4); `docs/method.md` explains the
  metrics and the weighting; `examples/` holds a sample corpus for `make demo` (#5).

[0.2.0]: https://github.com/wangjohn/styleprofile/tree/v0.2.0
