# Releasing

Releases go to [PyPI](https://pypi.org/p/styleprofile) from
[`.github/workflows/release.yml`](../.github/workflows/release.yml), with PyPI's
[trusted publishing](https://docs.pypi.org/trusted-publishers/): GitHub Actions proves to
PyPI which repository, workflow and environment is uploading, so no API token is stored
anywhere. The workflow runs only when a `v*` tag is pushed (a real release) or when it is
started by hand (a dry run to TestPyPI), never on ordinary pushes or pull requests.

## First public release: 0.2.0

Published on 2026-10-08 (America/Los_Angeles; the index timestamps are October 9 UTC).
The immutable tag [v0.2.0](https://github.com/wangjohn/styleprofile/releases/tag/v0.2.0)
points to source commit `687da439b90ad2b07450623e1c9ce7beeb26a20a`.
[Source CI](https://github.com/wangjohn/styleprofile/actions/runs/37885395568) passed all
eleven jobs, including the full Windows suite and native regressions without spaCy.
[Release run 37887071239](https://github.com/wangjohn/styleprofile/actions/runs/37887071239)
published both checked archives to [PyPI](https://pypi.org/project/styleprofile/0.2.0/).

Both actual index downloads matched the checked workflow artifacts:

| Archive | SHA-256 |
|---|---|
| `styleprofile-0.2.0-py3-none-any.whl` | `f0b0bf651faf736c7c9b75ce8a0a6160788f0d46b9015b706553d793b941f125` |
| `styleprofile-0.2.0.tar.gz` | `64918d815c992e90fa2c1b79c9ccbe303335a4ebd470b7e618ed0c3c097b12dc` |

A fresh Python 3.11.16 environment installed `styleprofile[syntax]==0.2.0` from PyPI.
Model setup and repeated setup, version, demo and explicit scoring all exited 0.
The demo and score read `Overall: close` (Delta 0.81) with the thin-reference warning;
reports carried 8.1 and used `en_core_web_sm` 3.8.0 with spaCy 3.8.16.
TestPyPI rehearsal was skipped because its separate account was unavailable. No TestPyPI
upload or index install is claimed. This production index verification is separate from
local wheel checks. The next development version has not been selected by this follow-up.

## Publisher setup

PyPI now has a project: inspect its normal trusted publisher. A *pending* publisher is
only for an index where the project does not yet exist; it creates the project on first
upload and does not reserve the name. Earlier pre-publication 404 checks are historical.
Recheck each index before choosing a pending or normal publisher.

1. **PyPI.** Sign in at <https://pypi.org> (with two-factor authentication on), open
   the project's *Publishing* settings, and verify the GitHub publisher:

   | Field | Value |
   |---|---|
   | PyPI project name | `styleprofile` |
   | Owner | `wangjohn` |
   | Repository name | `styleprofile` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

2. **TestPyPI (optional rehearsal).** TestPyPI is a separate site with separate accounts.
   If rehearsing there, configure its publisher at
   <https://test.pypi.org/manage/account/publishing/>, with environment name `testpypi`.
   Use a pending publisher only if the project still does not exist there.

3. **GitHub environments.** In the repository's *Settings > Environments*, verify `pypi`
   and `testpypi`. For `pypi`, add yourself under *Required reviewers*, so every release
   waits for your approval before it uploads, and under *Deployment branches and tags*
   allow only tags matching `v*`. For `testpypi`, allow only the `main` branch, so rehearsal
   artifacts come from the reviewed release source. Verify the publisher environment
   names exactly match the workflow; the two indexes have separate registrations.

## Optional dry run on TestPyPI

*Actions > Release > Run workflow* on `main`. The workflow runs the tests, builds the sdist
and wheel as `<version>.dev<run number>`, checks those exact archives, smoke-tests the
wheel on Python 3.11 with syntax and setup, and uploads them to TestPyPI. Each new dispatch
has a new run number. A rerun has the same number; if an upload already happened, start a
new dispatch instead of trying to replace that version.

Record the workflow run, source commit, development version, artifact names and SHA-256
hashes. Confirm both wheel and sdist appear on TestPyPI and match the checked workflow
artifacts. Then use a new directory and fresh Python 3.11+ virtual environment, replacing
`<development-version>` below with the actual uploaded version. These commands require
the completed upload; a local wheel smoke test does not prove index publication:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ "styleprofile[syntax]==<development-version>"
styleprofile setup
styleprofile --version
styleprofile demo
styleprofile score styleprofile-demo/draft.md styleprofile-demo/writer.profile.json
```

Expect the requested development package version and its documented report version.
The demo should say `Overall: close`, with its thin-reference warning; the explicit score
should agree. Record installed location/version, parser use and exit statuses. If using
the rehearsal, complete it before tagging; otherwise record it as skipped, never as
passing. See the README's installation details for Windows activation.

## Cutting future releases

Version 0.2.0 is already published; do not recut its tag or replace its artifacts.
It includes twelve completed onboarding WPs. WP-1's z-score cap
and WP-10's generic contrast are deferred. Paragraph checks remain experimental; do not
turn the release into a claim of authorship detection or stronger statistical guarantees.
Required platform checks must pass for the exact future release commit.

1. Select a new package version in a separate change; update `pyproject.toml`, `uv.lock`
   and its CHANGELOG section together. On reviewed, validated `main`, required CI and
   package smoke checks must pass for the exact selected commit. Date the new CHANGELOG
   section on release day. If using TestPyPI, complete and record the rehearsal above.
   A dated changelog is preparation, not evidence that an upload succeeded.
2. Tag that commit with its new version and push the tag:

   ```bash
   release_version="$(uv version --short)"
   git tag -a "v${release_version}" -m "styleprofile ${release_version}"
   git push origin "v${release_version}"
   ```

3. The workflow checks that the tag matches the version, runs the tests and the package
   checks, and waits for your approval of the `pypi` environment. Approve it, and it
   uploads the exact checked artifacts to PyPI. No token is stored in the repository.
   Wait for a successful upload, then verify the new version, both archive hashes and a
   fresh installation from PyPI before calling the release complete.
4. After verified publication, optionally create a GitHub release from the tag with the
   CHANGELOG section as its notes
   (`gh release create <new-tag> --notes-file <file>`). The CHANGELOG links each version to
   its tag, not to a GitHub release, so this step is never required.
5. After verified publication, update installation/status docs as needed. Selecting the
   next development version (for example 0.3.0) is separate maintenance work; update
   package/lock versions and the matching CHANGELOG heading/link together.
   `tests/test_packaging.py` checks the heading against the package version. Release
   bookkeeping does not change report versions.

For the public index check, use another new directory and fresh environment, replacing
`<release-version>` with the new version:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple/ "styleprofile[syntax]==<release-version>"
styleprofile setup
styleprofile --version
styleprofile demo
styleprofile score styleprofile-demo/draft.md styleprofile-demo/writer.profile.json
```

Check the requested package/report versions, installed location, parser use and exit
statuses. Keep the TestPyPI and PyPI evidence separate; do not report index installation
as passing merely because the local wheel worked.

A release can't be replaced: PyPI never accepts the same file name twice, even after a
delete. Fix a bad release with a new version (0.2.1), and yank the bad one on PyPI.

## The spaCy model and spaCy's version

spaCy's English model is not on PyPI, and PyPI refuses packages that depend on a URL, so the
`syntax` extra installs spaCy only, and `styleprofile setup` installs the model version in
`MODEL_VERSION` (`src/styleprofile/syntax.py`), checking the wheel's SHA-256
(`MODEL_SHA256` in `src/styleprofile/spacy_model.py`).

A model works only with the spaCy minor version it was built for (`en_core_web_sm` 3.8.0
with spaCy 3.8.x), so the `syntax` extra is bound to that series (`spacy>=3.8,<3.9`) and
`setup` refuses to install next to any other. **Bumping spaCy to a new minor version means
bumping the model with it**, all in one change:

1. `MODEL_VERSION` in `src/styleprofile/syntax.py`;
2. the `syntax` extra's bound, the `spacy-model` dependency group and its URL in
   `[tool.uv.sources]` (`pyproject.toml`);
3. `uv lock`, then `MODEL_SHA256` in `src/styleprofile/spacy_model.py` from the model's
   hash in `uv.lock`;
4. `make snapshots`, since a new parser changes the syntax metrics, and report `VERSION` in
   `src/styleprofile/reports.py` if saved profiles' values would change.

`tests/test_packaging.py` checks that the version, bound, URL and hash agree.
