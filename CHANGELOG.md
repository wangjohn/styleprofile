# Changelog

All notable changes to styleprofile. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers follow
[Semantic Versioning](https://semver.org/); before 1.0, a minor version may break things.

## [0.2.0] - Unreleased

The first release on PyPI: `pip install styleprofile`, or `pip install "styleprofile[syntax]"`
then `styleprofile setup` for the parser-based metrics.

> [!IMPORTANT]
> **Rebuild your profiles.** Reports now carry report version 8, and styleprofile refuses
> any other major with a message saying to rebuild. Existing version-8 reports remain readable. Run `styleprofile build` again on the
> writer's texts, and `styleprofile score` again for saved score reports. There is no
> migration: several metrics, the calibration and the report layout all changed.

### Changed

- **Additive report compatibility.** Reports keep the integer `version` as their major and
  add `minor_version`. Existing integers mean minor zero; matching majors load across
  minors, with a note for newer minors. Major changes still require rebuilding; optional
  fields use documented defaults or explain how to enable their features.

- **Paragraph calibration runs on request.** Use `build --by-paragraph` or
  `build(..., passages=True)` to prepare experimental paragraph checks. Ordinary builds
  skip this work; scores without it explain how to rebuild.
- **Large builds measure surface metrics in workers.** Duplicate detection stores compact
  text digests, grouped JSONL fallbacks reuse records, and workers respect CPU and memory limits.

### Breaking changes

- **The library leaves the measurement cache off by default.** `build` and `evaluate`
  now require `cache=True` to store or reuse text pattern counts. CLI caching is unchanged.
- **Settings objects on `Profile.score` are deprecated.** They still replace every
  inherited setting; use keyword overrides to change only the fields you choose.

- **Report version 8.** Rebuild profiles for paragraph metrics omitted on long single
  paragraphs and sentence-boundary windowing of long prose blocks.

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

- **One-command draft checks.** `score draft.md --against posts/` builds a cached reference
  in memory, with optional contrast texts and a command to save it for reuse. `build`
  defaults to the first input’s name plus `.profile.json`; stdin still needs `-o`.
  Short-only scoring runs explain the reference’s actual minimum length and suggest `--pool`.

- **Try the bundled samples.** `styleprofile demo` copies the sample corpus, builds a
  profile and scores a draft. The samples ship in the wheel; `--dir` chooses the folder.

- **Start the Python library with strings.** `build_texts` and `Profile.score_text` accept
  raw text, `load` reads a saved profile, and `build` accepts individual setting keywords.
  Path strings keep their meaning and missing text-like paths point to the new functions.

- **Faster CI and safer tests.** Test runs use up to four parallel workers, Python 3.14
  joins the syntax matrix, and permission tests skip under root. Release smoke tests also
  install syntax and run setup; Dependabot checks Python dependencies.
- **Windows and macOS support.** Output uses readable ASCII bars and arrows when the
  stream cannot encode them, command hints use the platform shell, and the measurement
  cache uses the platform cache directory. CI tests both systems, including syntax on macOS.

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
- **One big file, or a few manuscripts, just work.** `build` splits a manuscript or an
  archive in one file into documents where it divides: at its top headings (or `Chapter 12`
  lines in a `.txt` file), at rules before headings as a newsletter's issues have, or
  between an HTML page's `<article>`s, and says so. With fewer than 10 documents, long texts
  are split at their headings the same way when their parts average a window or more. A
  file with no such markers is cut into 8 stand-in documents, and the profile keeps a
  warning that calibration from them is less sensitive. `--split-on` (`auto`, `heading`,
  `heading:N`, `rule` or `none`) overrides it; `score` keeps drafts whole unless asked, and
  `score --split-on heading` (or `heading:2` for the chapters under `# Part`s) gives each
  chapter its own verdict (#25).
- **Where a draft drifts (experimental, off by default).** `score --by-paragraph`
  (`passages=True` in the library) also reads each document in spans of 100 words or more
  around each paragraph, and lists up to three paragraphs that read unlike the writer,
  against thresholds `build` sets from the writer's own documents (#21). It is opt-in
  because it raises false alarms: on synthetic corpora remixed from the sample essays, a
  paragraph of the writer's own drifted falsely in up to 4% of held-out documents on the
  reference's own topics and up to 6% under narrower topic shifts, but **33% (30% of long
  ones) under a wide topic shift with spaCy**, and more is to be expected on a real
  writer's new topics. It also misses things: a run of two or three spliced LLM blocks was
  found 76–100% of the time and a single block 57–74%, but **a single short paragraph often
  goes unnoticed** (17–48% found under 30 words), so "no paragraph drifts" does not mean the
  text is clean. Treat what it lists as a
  lead to read, not a finding. It needs a reference big enough to set thresholds, and
  without `--contrast` it rarely catches an LLM passage.

### Changed

- **Faster and smaller.** The contrast AUC's bootstrap is about 5x faster at the same
  accuracy, and a reference profile holds summaries only: 0.05 MB instead of 105 MB for 20k
  comments (#14).
- **Display.** "By area" shows each area's Delta as a multiple of its usual range, sorted by
  verdict; raw area Deltas move to `--all`. Saved profiles get normal file permissions (#10).
- `build` and `evaluate` keep only the first of documents with word-for-word the same text
  (#12).
- **Progress and speed** (#22). On a terminal, `build`, `score` and `evaluate` keep one
  line on stderr up to date (phase, chunks, words a second, time left); nothing is printed
  when stderr is piped. spaCy loads only the parts the metrics use, and from 50,000 words
  it parses in worker processes (one per CPU, at most 4, fewer on small machines; `--jobs
  N` sets how many). Held-out calibration scales to thousands of one-chunk documents.
  Building 200,000 words with spaCy went from about 22 s to about 10 s on the benchmark
  machine, and profiles are byte-identical to before.
- **The measurement cache** (#22). `build` and `evaluate` save what they measure in
  `~/.cache/styleprofile` (`$XDG_CACHE_HOME/styleprofile` when set), so rebuilding an
  unchanged corpus measures nothing and adding a few documents measures only those. What
  it stores and who can read it:
  - **Numbers derived from your texts, not the texts**, keyed by a hash of each text. But
    they include character- and word-pattern counts from which much of the wording can
    be recovered, so **treat the cache like the texts**. The file is readable only by you.
  - **`score` never uses it** (from Python, only with `cache=True`), so drafts you score
    leave nothing behind.
  - It holds about 512 MB at most, dropping the least recently used entries first. A
    change to the metrics' code, the package version or the spaCy model changes every key,
    so stale numbers are never read.
  - `--no-cache`, or `STYLEPROFILE_NO_CACHE=1` for every run, turns it off;
    `styleprofile cache` shows where it is and how large, and `styleprofile cache --clear`
    deletes it. When it can't be used (a read-only folder, a full disk), the run goes on
    without it and says so.

### Fixed

- **References need at least two chunks.** Build refuses smaller references and suggests
  adding documents or reducing the window size. Incomparable library results are not
  judged, and any CLI fail flag rejects them with exit status 3. Too-short texts still
  never fail a run.

- **Robust text input.** Long prose blocks split at sentence boundaries. Paragraph
  structure is omitted on long single paragraphs, with notes for missing breaks, oversized
  windows and text unlikely to be English. Empty-input errors explain what was removed.

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
- `evaluate` failed, with a misleading "no original" error, when a contrast draft was
  dropped as a word-for-word duplicate and an edited set had an edit of it; the edit now
  pairs with the kept copy, with a note (#24).
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
