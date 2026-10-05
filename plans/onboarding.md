# Plan: onboarding, robustness and speed

This plan turns the September 2026 repo review into work packages (WPs) that cloud agents can
implement in parallel. Each WP is one agent and one pull request. The plan is the source of
truth: every agent reads **Rules for every work package** and then its own WP, and implements
only that.

## Release status — 2026-10-04

Twelve of the fourteen work packages are complete, independently reviewed and merged.
WP-1 and WP-10 were explicitly deferred; their draft PRs and failed acceptance evidence
remain available. No metric z-score cap or bundled generic contrast set is in 0.2.0.
LLM-likeness still requires contrast drafts supplied by the caller; see the
[own-brief recipe](../docs/contrast.md).

| WP | Delivered scope | PR | Status |
|---|---|---|---|
| 1 | Metric z-score cap | [#30](https://github.com/wangjohn/styleprofile/pull/30) | Deferred; not shipped |
| 2 | Unjudgeable inputs fail requested CI gates | [#29](https://github.com/wangjohn/styleprofile/pull/29) | Complete |
| 3 | Encoding, shell hints and platform paths | [#28](https://github.com/wangjohn/styleprofile/pull/28) | Complete |
| 4 | CI and test hygiene | [#31](https://github.com/wangjohn/styleprofile/pull/31) | Complete |
| 5 | Input robustness and flattened paragraphs | [#32](https://github.com/wangjohn/styleprofile/pull/32) | Complete |
| 6 | Installed demo and packaged original samples | [#34](https://github.com/wangjohn/styleprofile/pull/34) | Complete |
| 7 | Default output paths and one-shot scoring | [#36](https://github.com/wangjohn/styleprofile/pull/36) | Complete |
| 8 | String-first library and cache opt-in | [#35](https://github.com/wangjohn/styleprofile/pull/35) | Complete |
| 9 | Build performance and opt-in paragraph calibration | [#38](https://github.com/wangjohn/styleprofile/pull/38) | Complete |
| 10 | Bundled generic contrast drafts | [#37](https://github.com/wangjohn/styleprofile/pull/37) | Deferred; empirical acceptance failed |
| 11 | Short default notes, verbose explanations | [#40](https://github.com/wangjohn/styleprofile/pull/40) | Complete |
| 12 | Additive report compatibility | [#39](https://github.com/wangjohn/styleprofile/pull/39) | Complete |
| 13 | Domain modules behind the public API | [#41](https://github.com/wangjohn/styleprofile/pull/41) | Complete |
| 14 | First-screen README and reference guides | [#42](https://github.com/wangjohn/styleprofile/pull/42) | Complete |

The completed onboarding source at `cbae1295c757e22c20369aaf2b192f8d3fa1f225`
passed all ten required CI jobs in [run 37188793902](https://github.com/wangjohn/styleprofile/actions/runs/37188793902).
Its advisory full Windows suite still had fourteen known failures; that suite is not
represented as passing. A separate release-readiness audit owns those failures. Paragraph
checks remain experimental, with the existing false-drift and synthetic-corpus limitations.
Performance completion preserves the measured timing and memory caveats, not a guarantee
for every machine or corpus.

The specifications below preserve the original plan and pre-refactor source locations;
completed and deferred WPs are not instructions to rerun or to advertise unshipped work.
The first public 0.2.0 release follows the completed work through wave 4. Index publication
and installation are verified separately by the maintainer; a prepared release date is not
proof that an upload succeeded.

## Goals

A new user should get from `pip install` to a verdict they trust, quickly, on whatever texts
they have: one essay, a folder of posts, a book, or a pile of comments. Success means:

| # | Outcome | Measured by |
|---|---|---|
| 1 | A first verdict within 60 seconds of installing, with no files of your own | `pip install styleprofile && styleprofile demo` |
| 2 | A first verdict on your own texts in one command, without managing a profile file | `styleprofile score draft.md --against posts/` |
| 3 | Deferred: LLM-likeness without caller-provided contrast (WP-10) | Not shipped; current `--contrast` uses your own drafts |
| 4 | Formatting alone never flips a verdict | the writer's own essays, blank lines removed, read "close" |
| 5 | A CI gate never passes when nothing could be judged | `--fail-*` exits 3 on a document that can't be compared |
| 6 | Works on Windows and macOS, including piped output | CI jobs on both |
| 7 | Large corpora build within the repo's own targets | 1M words without spaCy in under 8 s, 200k words with spaCy in under 12 s (`make bench`) |
| 8 | The README gets a newcomer running in one screen | the quickstart is at most 40 lines; reference material moves to `docs/` |

## Original execution sequence

The WPs run in four waves. Start a wave once the previous wave's PRs are merged. WPs in the
same wave touch different code, so they merge with few conflicts; the conflicts that remain are
in `CHANGELOG.md` and `tests/snapshots/`, and the rules below say how to handle both.

| Wave | Work packages (run in parallel) | Why this wave |
|---|---|---|
| 1 | WP-1 z-cap, WP-2 unjudgeable, WP-3 cross-platform, WP-4 CI hygiene, WP-5 input robustness | Correctness first: the fixes later waves' output depends on |
| 2 | WP-6 demo, WP-7 one-command paths, WP-8 library, WP-9 speed, WP-10 generic contrast | The onboarding features, on a correct base |
| 3 | WP-11 concise output, WP-12 profile compatibility | Output wording settles after the features that print things |
| 4 | WP-13 refactor, WP-14 README and docs | The refactor last, so it doesn't conflict with everything; the docs last, so they describe what shipped |
| — | **Release 0.2.0** (maintainer; see below) | Publish the twelve completed WPs, with WP-1 and WP-10 deferred |

Size: S is under a day of agent work, M one to two, L more.

### Launching an agent

Start one cloud session per WP on `wangjohn/styleprofile`, with the model set to **Sonnet 5.5**
(`claude-sonnet-5-5`), and this prompt, with `WP-N` filled in:

> Implement work package WP-N of the styleprofile onboarding plan. Read `plans/onboarding.md`
> from `main`; if it isn't on `main` yet, read it with
> `git fetch origin claude/style-profile-repo-review-i441nw && git show origin/claude/style-profile-repo-review-i441nw:plans/onboarding.md`.
> Follow its "Rules for every work package" and implement WP-N only. When it's done and
> validated, open a draft PR against `main`, then keep driving it until CI is green.

You can also ask the Claude session that wrote this plan to launch a wave for you.

### Release 0.2.0 (maintainer)

The package remains version 0.2.0 for its first public release. PyPI had no project on
2026-10-04; the README therefore retains a verified GitHub installation fallback. Follow
[docs/releasing.md](../docs/releasing.md) after release preparation and the Windows audit
are reviewed, merged and validated: verify a TestPyPI rehearsal, then tag the exact release
commit `v0.2.0` and verify the PyPI artifacts and a fresh index installation. Do not bump to
0.3.0 before 0.2.0 is published. Switch the README to verified index instructions and start
the next development version in a follow-up after publication.

## Rules for every work package

1. **Scope.** Implement your WP, and nothing else. If you find another problem, list it
   under "Follow-ups" in your PR body; don't fix it. Where the WP names a decision (a flag
   name, a threshold), follow it. Where it leaves one open, choose, and say what you chose
   and why in the PR body.
2. **Reproduce first.** Where the WP gives a "Reproduce" block, run it on `main` before you
   change anything, and put the before and after output in the PR body.
3. **Setup.** Run `uv sync --extra syntax`; `make check` runs pytest, ruff and pyright. Permission tests detect and skip a root process where the
   permission contract cannot be exercised; do not ignore new failures.
4. **Validate before every push:**
   - `make check` passes;
   - `make demo` runs;
   - `make dist-check` passes when you touched packaging, `pyproject.toml` or anything the
     wheel ships;
   - `make bench-quick BENCH="--against origin/main --repeat 5"` when you touched
     measuring, calibration or scoring. If your change is meant to cost more, declare it
     in `bench/accepted.toml` as CONTRIBUTING.md's "The benchmark gate in CI" describes.
5. **Snapshots.** When the CLI's output changes, run `make snapshots` and review the diff in
   `tests/snapshots/`. When you merge `main` and snapshot files conflict, never resolve them
   by hand: take either side, then run `make snapshots` again.
6. **Report version.** `VERSION` and `MINOR_VERSION` in `src/styleprofile/reports.py`
   are currently 8 and 1 for references and scores; evaluation reports use 2 and 1.
   Saved integer majors without `minor_version` mean minor zero. Matching majors load
   across additive minor changes; different majors require rebuilding. If measurement,
   calibration or an incompatible layout changes, compare with the latest public release
   and bump the major once for that development release, updating the rebuild note.
   Additive compatible fields use a minor bump and documented defaults. Do not bump report
   versions for release bookkeeping, docs or movement of unchanged code.
7. **CHANGELOG.** Add your entry to the topmost section that has no release date: under
   Added, Changed, Fixed or Breaking changes. If the top section is released, start
   `## [Unreleased]` above it. Match the existing entries' style: a bold lead-in, then plain
   sentences.
8. **House style.** Match the code around you: its comment density, naming, and the
   prose style of messages (plain words, full sentences, no jargon). User-facing messages
   say what happened and what to do next. The package has no runtime dependencies
   (`dependencies = []`); keep it that way. Links in `README.md` stay absolute
   (`https://github.com/wangjohn/styleprofile/...`), because PyPI renders it.
9. **PR.** Open a draft PR against `main` titled `<what it does> (onboarding plan WP-N)`,
   with these sections:
   - **What and why**, linking this WP;
   - **Evidence**: the before and after of each Reproduce block, plus benchmark numbers if
     you touched speed;
   - **Tests**;
   - **Decisions**;
   - **Follow-ups**.

   Keep it mergeable: when `main` moves and conflicts, merge `main` in.

---

## Wave 1: correctness

### WP-1: Cap each metric's z-score in Delta and LLM-likeness (S)

**Deferred; not shipped.** PR #30 failed the statistical acceptance controls. The original
specification below is retained for future research, without a cap in the public release.

**Why.** One metric that is far outside the writer's range decides the whole verdict. When
the writer's own sample essays have their blank lines removed, they read "very different"
(Delta about 4) against their own profile. `paragraph_sentences_mean` then has z about 100
and `paragraph_words_mean` about 85. `delta()` caps |z| only for metrics with no weight
(around `weighting.py:393-407`).

**Reproduce.**
```bash
uv run styleprofile build examples/writer --contrast examples/llm-drafts -o profiles/w.json
mkdir -p /tmp/flat && for f in examples/writer/*.md; do grep -v '^$' "$f" > /tmp/flat/$(basename "$f"); done
uv run styleprofile score -q /tmp/flat profiles/w.json     # every essay: very different, Delta ~4
```

**Do.**
- Add `Z_CAP = 5.0` in `weighting.py`, and cap |z| at it for every metric wherever z feeds
  Delta, area Deltas or LLM-likeness.
- Apply the cap everywhere those numbers are computed, the held-out calibration included,
  so a draft's number and the writer's stored range are computed the same way.
- "Biggest differences" and the report's per-metric z values stay uncapped: they explain,
  and don't judge.
- Document the cap and why in `docs/method.md`.
- This changes the stored ranges, so follow rule 6.

**Done when.**
- The flat essays no longer read "very different". Report their new verdicts. WP-5 makes
  them read "close".
- `examples/draft.md` and the LLM drafts keep verdicts in the same bands as before.
- A regression test covers the flat essays.
- The bench gate passes.

### WP-2: Never report success when nothing could be judged (S)

**Why.** A profile built from one 640-word file has one chunk. `z_score` returns None
whenever n < 2 (`weighting.py:131`), so every score against it is `NOT_COMPARABLE`, and
`--fail-above` exits 0: the CI gate passes having checked nothing. In the library the same
case returns `judged=True` with `reason=None` (around `api.py:691-718`).

**Reproduce.**
```bash
uv run styleprofile build examples/writer/old-maps.md -o profiles/one.json      # succeeds
uv run styleprofile score -q --fail-above somewhat examples/draft.md profiles/one.json; echo "exit $?"   # 0
```

**Do.**
- **`build` refuses a reference with fewer than 2 chunks**, exiting 1 with an error that
  says why and what to do: add documents, or pass a smaller `--window-words` (give the
  number that would yield at least 2 chunks). In the library, raise `StyleProfileError`
  with the same message.
- **Library:** a `NOT_COMPARABLE` result has `judged=False` and a `reason`.
- **CLI:** when any `--fail-*` flag is given, a document that could not be compared fails
  the run (exit 3), with a `failed: <doc>: could not be compared with the reference` line.
  "Too short to judge" still never fails a run, as the README documents; say so next to the
  new rule.
- Update the README's exit-status table, and the JSON `failed` entries (`schema.py`).

**Done when.**
- The Reproduce block builds nothing (exit 1, with a clear message).
- An uncomparable document fails a `--fail-*` run. You can reach that state with a
  hand-edited or older profile, or through the library; test whichever you can.
- There are tests for the CLI, the library and the JSON schema.

### WP-3: Windows and macOS: encoding, shell hints, platform paths (M)

**Why.**
- Output that is not a terminal uses the locale's encoding, which is cp1252 on Windows, and
  the bars and arrows (`█ ▲ ▼ ÷`) can't be encoded. With `PYTHONIOENCODING=cp1252`, `score`
  crashes with `UnicodeEncodeError` and exits 1, which breaks the pre-commit and CI use the
  README advertises.
- The hints use `shlex` quoting (`'…'`), which cmd.exe doesn't understand (`cli.py:631`,
  `:1022`).
- `os.sysconf` doesn't exist on Windows (`measure.py:80`).
- The cache lives at `~/.cache` on every platform.

**Reproduce.**
```bash
PYTHONIOENCODING=cp1252 uv run styleprofile score examples/draft.md profiles/w.json > /tmp/o.txt; echo "exit $?"   # 1
```

**Do.**
- **Encoding.** At the start of `cli.main`, reconfigure stdout and stderr with
  `errors="replace"` when their encoding can't encode the output's symbols. Better still,
  pick ASCII fallbacks (`#`, `^`, `v`, `/`) when the stream's encoding can't encode the
  Unicode ones. The display code should ask one helper which glyphs to use.
- **Shell hints.** Quote the commands in hints for the platform: `subprocess.list2cmdline`
  on Windows, `shlex.join` elsewhere.
- **Memory probe.** Guard `os.sysconf`; on Windows, read memory with `ctypes`
  (`GlobalMemoryStatusEx`) or fall back as the code already does.
- **Cache location.**
  - Windows: `%LOCALAPPDATA%\styleprofile\Cache`.
  - macOS: `~/Library/Caches/styleprofile`.
  - Keep `$XDG_CACHE_HOME` and `~/.cache` elsewhere, and let `$XDG_CACHE_HOME` win when it
    is set.
  - Update the README and the `cache` command's output.
- **CI.** Add macOS and Windows jobs to `.github/workflows/ci.yml`: Python 3.11 without
  spaCy on both, and one macOS job with spaCy. Fix what fails. If Windows fails in ways
  outside this WP, mark that job `continue-on-error: true` and list each failure under
  Follow-ups.

**Done when.**
- The Reproduce block exits 0 with readable output.
- A test runs the CLI under `PYTHONIOENCODING=cp1252`.
- The macOS and Windows jobs run in CI.

### WP-4: CI and test hygiene (S)

**Why.**
- The 4-minute suite runs serially 5 times per PR; `pytest-xdist -n 4` measured 2.8x
  faster.
- Wall-clock asserts will flake on shared runners (`tests/test_formats.py:554,772`,
  `tests/test_cache.py:477`).
- Two permission tests fail when run as root.
- dependabot covers only GitHub Actions.
- Python 3.14 isn't tested.
- `__main__.py` has no coverage.
- The release job never smoke-tests the uploaded wheel's `setup` path.

**Do.**
- Add `pytest-xdist` to the `dev` group and run `pytest -n auto` in `make test` and in CI.
  Make the suite xdist-safe, since shared fixtures or files may clash.
- Profile the 45 s module fixture in `tests/test_pooling.py` (around `:591` and `:659`) and
  make it cheaper, for example with a smaller corpus that still exercises the same paths.
- In `ci.yml`, add `concurrency: {group: ${{ github.workflow }}-${{ github.ref }},
  cancel-in-progress: true}` and a `timeout-minutes` on every job.
- Loosen the wall-clock asserts about 3x, or better, assert scaling: twice the input takes
  at most about three times the time.
- Mark `tests/test_cache.py:363` and `:438` `skipif(hasattr(os, "geteuid") and os.geteuid() == 0)`.
- Add the `uv` ecosystem to `.github/dependabot.yml`.
- Add Python 3.14 to the CI matrix and the classifiers, if spaCy 3.8 installs on it
  (check). Otherwise run 3.14 without spaCy and say so.
- Add a subprocess test of `python -m styleprofile --version`.
- In `release.yml`, run `make dist-check` with `--syntax`.
- Leave the macOS and Windows jobs to WP-3. If both PRs edit `ci.yml`, whichever merges
  second merges `main` in.

**Done when.** CI's wall time per PR drops (report before and after), and the suite passes
as root and as a normal user.

### WP-5: Robust input: paragraph-less text, long blocks, non-English, empty results (M)

**Why.**
- **Text with no blank lines.** Plain-text exports, transcripts and pasted text often have
  none. Their paragraph metrics are then meaningless and dominate the verdict (see WP-1's
  Reproduce).
- **A block over a window is never split.** Windows split only at Markdown block
  boundaries (`_pack`/`_pieces`, around `profile.py:941-980`). A 4,400-word draft with no
  blank lines became one chunk against 500-word windows, scored Delta 29 with no warning,
  and went to spaCy as one huge document.
- **Non-English text.** It is judged silently: Japanese scored Delta 10.8 and German 7.98,
  both `judged=True` with no note. `WORD` (`surface.py:40`) counts a Japanese sentence as
  one word.
- **An unhelpful error.** A text whose only line is a heading is dropped as having no
  prose, and the `no_chunks` error (`measure.py:558`) doesn't say why.

**Do.**
- **Paragraph-less text.** When a chunk has 300 or more words in a single paragraph, don't
  measure the paragraph-structure metrics for it; treat them as missing, the way metrics
  that can't be measured already are. List which metrics those are in `docs/method.md`.
  Check how missing metrics flow through `z_score`, Delta and the calibration, and add a
  note (a `NoteCode`) saying paragraph metrics were left out because the text has no
  paragraph breaks.
- **Long blocks.** Split any block longer than 1.5 windows at sentence boundaries before
  packing, and add a note when a chunk is still over twice `window_words`.
- **Non-English text.** Add a cheap English check: the share of the 50 most common English
  function words among tokens, plus the share of letters that are ASCII. Test it on the
  sample corpus and a few other languages, set a conservative threshold, and add a note
  (`NoteCode`) on `build` and `score` when a text fails it. Warn; don't refuse.
- **The error.** Make the `no_chunks` error name the cause (only headings, only code, empty
  after conversion, and so on).
- This changes what profiles measure, so follow rule 6.

**Done when.**
- WP-1's flat essays read "close" against their own profile.
- A 4,000-word paragraph-less draft is scored in window-sized chunks.
- Japanese, German and Spanish samples get the note.
- Each fix has a test.

---

## Wave 2: onboarding and speed

### WP-6: `styleprofile demo`, with the sample corpus in the wheel (M)

**Why.** The first thing a new user wants is to see it work. `examples/` isn't in the wheel,
and `make demo` needs a clone plus uv, so a `pip install` user has nothing to try it on.

**Do.**
- **Ship the samples.** Put `examples/` in the wheel as package data at
  `styleprofile/_examples` (hatch `force-include`), and keep `examples/` at the repository
  root, where the tests read it. Find the samples with `importlib.resources`, and fall back
  to the repository's `examples/` in an editable install.
- **Add `styleprofile demo [--dir DIR]`.** It:
  - copies the samples to `DIR` (default `./styleprofile-demo`; refuse to overwrite a
    non-empty folder that isn't a previous demo);
  - builds a profile from `writer/` with `llm-drafts/` as contrast;
  - scores `draft.md` and prints the usual output;
  - ends with a short "Next: try it on your own texts" block of three commands (build,
    score, and the `--against` form from WP-7; if WP-7 hasn't merged, leave that one out
    and list it under Follow-ups).
- It runs in under 5 s without spaCy, and uses spaCy when it's installed.
- Add `styleprofile demo` to the root `--help` epilog.
- Add a snapshot test of `demo`, and add `styleprofile demo` to `scripts/check_dist.py`'s
  wheel smoke test, run from outside the repository.

**Done when.** In a fresh venv, `pip install <wheel> && styleprofile demo` prints a verdict,
and `make dist-check` passes.

### WP-7: One-command paths for the CLI (M)

**Why.**
- Checking one draft takes two commands and a profile file to name.
- `build` requires `-o`.
- A single tweet never gets a verdict (the floor is 75 words), and the user only learns
  that after trying.

**Do.**
- **Default output.** `build` without `-o` writes `<name>.profile.json` in the current
  folder. `<name>` is the stem of the first input (a folder's name, or a file's name without
  its extension). Stdin still requires `-o`. Print `wrote posts.profile.json`, as `-o` does
  today.
- **One-shot scoring.** Add `score DRAFT [DRAFT ...] --against CORPUS [CORPUS ...]
  [--contrast DIR ...]`, which builds the reference in memory, with the measurement cache as
  `build` uses it, then scores. It can't be combined with a reference argument or `-r`. The
  output opens with one line, for example: `Reference: built from 7 documents (4,496 words)
  in 0.7 s, not saved. To reuse it: styleprofile build posts/ -o posts.profile.json`.
  `build`'s warnings about a thin reference still show.
- **Short texts, said up front.** When every draft is under the shortest calibrated length,
  add one line after the verdicts: `styleprofile judges 75 words or more; score several short
  texts together with --pool`. Use the reference's actual shortest length.
- Update `--help` texts and the root epilog.
- Add snapshot tests for `--against` and the default output name.

**Done when.**
- `styleprofile score examples/draft.md --against examples/writer --contrast
  examples/llm-drafts` prints the same verdict and numbers as building then scoring.
- `styleprofile build examples/writer` writes `writer.profile.json`.

### WP-8: A library that is easy to start with (M)

**Why.**
- Every `str` is treated as a path (around `api.py:1316-1338`), so the obvious first lines,
  `sp.build(["text", ...])` and `profile.score("draft text")`, both fail.
- `build` takes only a `Settings` object, while `score` takes keyword overrides.
- There is no top-level `sp.load`.
- `build` writes the measurement cache under `~/.cache` by default (`cache=True`), which
  can recover much of the texts' wording. That is surprising in a web server or notebook.
- `profile.score(x, sp.Settings(syntax=False))` resets every other setting to its default
  instead of the profile's (around `api.py:431-441`).

**Do.**
- **Build from strings:** `sp.build_texts(texts: Iterable[str], *, contrast: Iterable[str] |
  None = None, **overrides)`. Each string is one document.
- **Score a string:** `Profile.score_text(text: str | Iterable[str], **overrides) ->
  ScoreResult`. Keep `str` meaning a path in `build` and `score`, but when such a path
  doesn't exist and the string contains whitespace or a newline, name the new functions in
  the error.
- **Load:** export `sp.load` (`Profile.load`).
- **Keyword overrides for `build`:** `build` accepts `**overrides: Unpack[SettingsOverrides]`
  as `score` does. A `Settings` argument still works.
- **No cache by default:** the library's `build` and `evaluate` default to `cache=False`.
  The CLI passes `cache=True` explicitly, so its behavior doesn't change. Document this in
  `docs/library.md` and the CHANGELOG (Breaking changes).
- **Fix the `Settings` bug:** a `Settings` passed to `score` overrides only the fields the
  caller set. Or deprecate `settings=` on `score` in favor of overrides; pick one and say why.
- Rewrite the top of `docs/library.md` (a doctest run by `tests/test_api.py`) so it opens
  with a five-line example from strings, then the `Path` form.

**Done when.** This runs as a doctest:
```python
>>> import styleprofile as sp
>>> profile = sp.build_texts(essays, contrast=drafts)
>>> result = profile.score_text(draft)
>>> result.verdict
```

### WP-9: Faster builds on large corpora (M)

**Why.** On a 4-core machine, 1M words without spaCy took 19 s to build (the target is under
8 s) and 200k words with spaCy took 20 s (the target is under 12 s).
- **One process without spaCy.** Workers start only for spaCy parsing (`measure.py:343`),
  and surface metrics were about 20 of 33 s under a profiler.
- **The paragraph calibration always runs.** `_calibrate_drift` runs on every build (around
  `profile.py:2398`), although `--by-paragraph` is experimental and off by default. It costs
  18–25% of build time, plus memory for the spaCy documents it keeps (`keep_docs`, around
  `profile.py:2365`).

**Do.**
- **Surface metrics in parallel.** Measure them in worker processes too when the corpus is
  large. Measure where parallelism starts to pay without spaCy (worker start-up is far
  cheaper without the model) and set that threshold; don't reuse `PARALLEL_WORDS` blindly.
  Profiles must stay byte-identical.
- **Paragraph calibration on request.** Make the drift calibration opt-in at build time:
  - `build --by-paragraph` in the CLI, `build(..., passages=True)` in the library;
  - `score --by-paragraph` against a profile built without it prints one line naming the
    `build` command that adds it, and scores as usual;
  - update `make demo`, the README's "Where it drifts" section and the CHANGELOG;
  - this changes what profiles store, so follow rule 6.
- **Duplicate detection:** key `drop_duplicates` (around `profile.py:915`) on a blake2b
  digest of the normalized text instead of the text itself.
- **Grouped JSONL:** don't read grouped JSONL inputs twice (around `api.py:1411-1419`);
  re-group the chunks already loaded.
- **CPU and memory limits:** `cpus()` respects cgroup CPU quotas on Python below 3.13
  (read `/sys/fs/cgroup/cpu.max`), and an explicit `--jobs N` is capped by
  `memory_jobs()`, with a note when it is lowered.

**Done when.**
- `make bench` meets the build targets, or the PR shows how close it gets and why.
- Profiles built without `--by-paragraph` are byte-identical to before, apart from the
  drift section.
- The bench gate passes.

### WP-10: LLM-likeness without writing your own contrast drafts (L)

**Deferred; not shipped.** PR #37's frozen original corpus failed held-out acceptance.
No generic flag or corpus ships in 0.2.0. The commands below are the historical proposal,
not available commands. Keep the bars and failed evidence; use caller-provided contrast
drafts and the own-brief recipe instead.

**Why.** LLM-likeness, the headline feature, needs a folder of LLM drafts, and a new user
has none. A small bundled generic set, clearly labelled as weaker than drafts made from the
writer's own briefs, gets people to a first LLM-likeness score. A recipe gets them to the
better version.

**Do.**
- **Write the drafts.** Write 24 original drafts of 400–900 words in the generic
  assistant register:
  - a spread of genres (personal essay, blog post, how-to, opinion piece, product or book
    review, newsletter, short story), 3–4 of each;
  - varied topics, none overlapping the sample essays' topics;
  - a range of LLM habits, not only the extremes: some with headings and lists, some plain
    prose with the register's vocabulary and rhythm.
  - Put them in `src/styleprofile/data/generic-contrast/` (package data in the wheel, under
    150 KB in total), with a README saying what they are and that they are original and MIT
    licensed.
- **Add `build --generic-contrast`** (library: `generic_contrast=True`), which uses the
  bundled set as the contrast. It can be combined with `--contrast` to add your own drafts.
  The report labels the contrast "generic LLM drafts", and the build summary says in one
  line that drafts from the writer's own briefs are better, linking the recipe.
- **Point people to it.** When `build` runs without any contrast, the summary's last line
  suggests `--generic-contrast` for LLM-likeness.
- **Validate it** on `examples/writer`, and on the synthetic `medium` corpus from
  `bench/gen.py`, with the bundled set as contrast, and write the numbers into
  `docs/method.md`:
  - the writer's held-out documents should read "like the reference";
  - `examples/llm-drafts`, which isn't in the bundled set, should read "leans LLM" or
    beyond;
  - report the AUC and `evaluate`'s numbers.
  - If the generic set fails this, stop and report it in the PR rather than weakening the
    bar.
- **Write the recipe** in `docs/contrast.md`: how to produce contrast drafts from your own
  briefs, with a copy-paste prompt template, advice to use several models and to keep
  lengths similar, and how many drafts are enough.

**Done when.**
- `styleprofile build examples/writer --generic-contrast -o w.json && styleprofile score
  examples/llm-drafts w.json` scores every draft "leans LLM" or beyond.
- The writer's essays read "like the reference".
- The wheel includes the set, and `make dist-check` passes.

---

## Wave 3: polish

### WP-11: Concise output: one line per note (M)

**Why.** Notes and warnings are paragraphs. The pooling note alone is four lines, and a
first run's output is mostly caveats. People read the first line of each.

**Do.**
- **Short and long text.** Give every note and warning a short form (one line, at most
  about 100 characters, saying what happened and what to do next) and a long form (today's
  text).
  - Human output prints the short form.
  - `--verbose` (on `build`, `score` and `evaluate`) prints the long form.
  - JSON reports keep the long form.
  - Put both on the `NoteCode` definitions, so every message lives in one place.
- **Merge the thin-reference warnings.** The two or three "Thin reference" lines on `build`
  become one: `Thin reference: 7 chunks, 4,496 words (aim for 15+ chunks and 20,000+
  words); add more of the writer's documents`.
- Keep the Reproduce outputs from WP-1, WP-2 and WP-7 readable, and update the snapshots.

**Done when.**
- No default-mode output line is longer than 120 characters, apart from verdict tables and
  quoted text. A test checks this over the snapshots.
- `--verbose` shows today's full text.

### WP-12: Saved profiles survive additive changes (S)

**Why.** `check_version` requires an exact match (around `profile.py:1888-1916`), so any
change to the report format makes users rebuild from the source texts, which they may no
longer have.

**Do.**
- Split the report version into a major and a minor number. A profile whose major matches
  and whose minor is older loads; fields it lacks get documented defaults, or the features
  that need them say so (for example "rebuild with `--by-paragraph`").
- A newer minor loads, with a note that some fields may be ignored.
- A different major is refused as today.
- Write the policy (what counts as major or minor) at the top of `schema.py` and in
  `docs/library.md`.
- Keep the version numbers of reports already written readable.

**Done when.** There are tests for an older minor, a newer minor and a different major, and
the CHANGELOG explains the policy.

---

## Wave 4: structure and docs

### WP-13: Split `profile.py` and slim `api.py` (L)

**Why.**
- `profile.py` is 2,857 lines. It holds file, JSONL and folder reading (around 210–940),
  windowing and pooling (941–1380), z-scores and scoring (1388–1760, 2465–2790), report
  versions and I/O (1764–1938, 2790–2857), and build and calibration (1939–2465).
- `api.py` adds about 570 lines of corpus preparation (`_split`, `_stand_ins`, `_pooled`,
  `_edits_of_repeats`) and imports about 35 names from `profile`.
- `display.py` imports domain logic (`summarize`, `average_z`).

**Do.**
- **Move code, don't change it:**
  - a `corpus/` package (reading, ids, windowing, pooling, splitting, duplicates);
  - `reference.py` (build and calibration);
  - `scoring.py` (z-scores, Delta, verdicts);
  - `reports.py` (versions, load and save, validation);
  - leave `api.py` as a thin façade of about 500 lines;
  - move the domain helpers `display.py` uses into the modules they belong to.
- `styleprofile/__init__.py`'s public names don't change.
- Update the tests' imports.
- Add a short `docs/architecture.md`: a module map and how data flows from files to
  report.

**Done when.**
- Every snapshot is byte-identical.
- A profile built from `examples/` and from the `medium` corpus is byte-identical to one
  built on `main` (compare hashes).
- The bench gate passes, and `make check` is clean.

### WP-14: A README that gets a newcomer running in one screen (M)

**Why.** The README is about 600 lines, and "Quick start" is about 125 lines of dense prose
covering formats, splitting, the cache and workers. Guidance on corpus size is scattered
over four places, there is no troubleshooting section, and contributor material (the
benchmark gate) sits in the user README.

**Do.**
- **Restructure `README.md`.** Keep every link absolute. In order:
  1. **What it is**: three lines.
  2. **Install**: pip and uv, plus `styleprofile setup` for syntax.
  3. **Try it in 60 seconds**: `styleprofile demo`, with its expected verdict shown.
  4. **Your own texts**: one short recipe per situation:
     - one long file (a book or an archive);
     - a folder of essays or posts;
     - tweets, comments or emails (JSONL, `--group-field`, `--pool`);
     - one-shot checks with `--against`;
     - LLM-likeness with your own `--contrast` drafts (WP-10 is deferred).
  5. **How much text you need**: one table, merging `README.md:338-381` and the drift
     guidance.

     | Feature | Minimum |
     |---|---|
     | a Delta verdict | 15+ windows, 20k+ words |
     | verdicts on short texts | 3+ documents |
     | paragraph drift | 10+ documents, about 200 paragraphs |
     | a scorable text | 75 words |

  6. **Reading the output**: keep today's tables.
  7. **What a verdict can't tell you**, stated plainly:
     - a whole-document verdict can hide one or two off-voice paragraphs (the demo draft
       reads "close");
     - English only;
     - not proof of authorship.
  8. **Troubleshooting**:
     - no matching distribution;
     - PEP 668 (externally managed Python);
     - spaCy version mismatch;
     - syntax metrics left out;
     - cache unavailable;
     - "rebuild your profile";
     - Windows console output.

     Use the existing `code=` error ids.
  9. **Use it in CI**: pre-commit and `--fail-*`, with the exit-status table.
  10. **Python**: five lines linking `docs/library.md`.
- **Move reference material** to `docs/usage.md`: input formats, splitting, pooling and
  grouping details, the cache and worker processes, `evaluate`.
- **Move contributor material** (Development, the benchmark gate, snapshots, packaging) to
  `CONTRIBUTING.md`.
- Update `examples/README.md` to start with `styleprofile demo`.
- Check every command in the README by running it.
- Check that `scripts/check_dist.py`'s required and forbidden lists still hold.

**Done when.**
- "Try it in 60 seconds" is within the first 40 lines.
- Every code block runs as written.
- No section of the old README is lost: each moved or merged section is listed in the PR
  body.

---

## Not in this plan

- **Making `--by-paragraph` good enough to turn on by default.** It is research: its false
  drift rate on new topics is up to a third of the writer's documents. WP-14 states the
  limit instead.
- **Languages other than English.** WP-5 detects them and warns.
- **A single-tweet verdict.** Below 75 words the writer's own text varies too much by
  chance; WP-7 says so up front and points to `--pool`.
