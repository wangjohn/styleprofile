# Contributing

Start with a clone and Python 3.11 or newer. CI tests Python 3.11–3.14 with syntax,
Python 3.11 without spaCy, macOS with and without syntax, and targeted Windows regressions.
The full Windows suite is advisory while documented pre-existing failures remain; new
in-scope Windows failures are still defects. The package uses only the standard library
at runtime unless the syntax extra is installed.

## Development

```bash
make sync
make check   # pytest, ruff check, ruff format --check, pyright
make demo    # build, score and show on the sample corpus
make bench          # time build and score on generated corpora; compare with the targets
make bench-quick    # only the case CI runs: 200k words without spaCy
make snapshots      # regenerate tests/snapshots/ after an intended change to CLI output
make dist-check     # build the sdist and wheel, check them, smoke-test the wheel on 3.11
```

`make bench` generates seeded synthetic corpora from `examples/` (`bench/gen.py`, written to
the git-ignored `bench/corpora/`), then measures wall time, CPU time, peak memory and profile
size for each case in `bench/targets.py`. It prints a table beside the plan's targets and saves
`bench/results.json`. Pick cases with `make bench BENCH="--case big --repeat 3"`. To compare
with another revision as CI does, add `--against`:
`make bench-quick BENCH="--against origin/main --repeat 5"` also benchmarks `src/` as of
`origin/main`, alternating its runs with yours, and prints each metric's ratio to it. CPU time
and peak memory include spaCy's worker processes. Every run gets an empty measurement cache,
except the `medium-warm` case, which times a rebuild after an untimed build has filled it.

## The benchmark gate in CI

CI's benchmark job runs `bench/run.py --quick --repeat 5 --against HEAD^ --check`. It
benchmarks your change and its base (the tip of `main` it merges into) in the same job, on the
same runner, and fails when your change is worse than the base by more than a margin set in
`bench/targets.py`: CPU time 1.15x, peak memory 1.1x, reference profile size 1.05x. Time is
judged on CPU time because wall time on shared runners swings by up to 75% between runs. Wall
time and the plan's targets are reported in the job summary but never fail the job.

If your change is meant to cost more, declare it in `bench/accepted.toml` in the same PR, so
reviewers approve it:

```toml
[[regression]]
pr = 15                    # your pull request's number
case = "medium-nosyntax"   # the case, from bench/targets.py
metric = "build_cpu_s"     # the metric, from MARGINS in bench/targets.py
ratio = 1.9                # the most it may cost, as change / base, with room for noise
reason = "Length-aware verdicts bootstrap each length bucket separately."
```

Open the PR first to get its number. Take the ratio from the failing job's summary (or from
`make bench-quick BENCH="--against origin/main --repeat 5"`) and add about 0.1 for noise. Add
one entry per metric that fails. An entry applies only to the PR it names, on its own CI runs
and on the push to `main` that merges it. After that it is inert, and CI warns until someone
deletes it (do so in any later PR). CI also warns when your own PR turns out not to need its
entry. To try an entry locally, pass the PR number:
`make bench-quick BENCH="--against origin/main --repeat 5 --check --pr 15"`.

## Changing CLI output

`tests/test_snapshots.py` compares the exact output of `build`, `score`, `show`, `metrics` and
`evaluate` on `examples/` with the files in `tests/snapshots/`. When you change what the CLI
prints, or rebase onto a change that did, run `make snapshots`. It installs spaCy (the
`syntax` extra) first, so the syntax snapshots refresh too. Review the diff in `tests/snapshots/` and commit it
with the change, so reviewers see the output change.

## Packaging and releases

`make dist-check` builds the sdist and the wheel, checks what each contains, installs the wheel into a new Python 3.11 environment and runs `build`,
`score` and the README library example on `examples/` from outside the repository.
`make dist-check DIST="--syntax --sdist-tests"` also installs the `syntax` extra and runs
`styleprofile setup`, which downloads spaCy's model, then runs the test suite from the
unpacked sdist, as CI does. Releases are published to PyPI by pushing a `v*` tag;
see [docs/releasing.md](https://github.com/wangjohn/styleprofile/blob/main/docs/releasing.md).


