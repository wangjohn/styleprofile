# Releasing

Releases go to [PyPI](https://pypi.org/p/styleprofile) from
[`.github/workflows/release.yml`](../.github/workflows/release.yml), with PyPI's
[trusted publishing](https://docs.pypi.org/trusted-publishers/): GitHub Actions proves to
PyPI which repository, workflow and environment is uploading, so no API token is stored
anywhere. The workflow runs only when a `v*` tag is pushed (a real release) or when it is
started by hand (a dry run to TestPyPI), never on ordinary pushes or pull requests.

## One-time setup

For the first upload, each index needs a *pending* publisher, which creates the project
on that upload. Both project endpoints returned 404 on 2026-10-04. Recheck before
publishing: a pending publisher does not reserve the name. Once a project exists, inspect
its normal trusted publisher instead of adding another pending one.

1. **PyPI.** Sign in at <https://pypi.org> (with two-factor authentication on), open
   *Your account > Publishing* (<https://pypi.org/manage/account/publishing/>), and add a
   pending GitHub publisher:

   | Field | Value |
   |---|---|
   | PyPI project name | `styleprofile` |
   | Owner | `wangjohn` |
   | Repository name | `styleprofile` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

2. **TestPyPI.** TestPyPI is a separate site with separate accounts. Do the same at
   <https://test.pypi.org/manage/account/publishing/>, with environment name `testpypi`.

3. **GitHub environments.** In the repository's *Settings > Environments*, create `pypi`
   and `testpypi`. For `pypi`, add yourself under *Required reviewers*, so every release
   waits for your approval before it uploads, and under *Deployment branches and tags*
   allow only tags matching `v*`. For `testpypi`, allow only the `main` branch, so rehearsal
   artifacts come from the reviewed release source. Verify the publisher environment
   names exactly match the workflow; the two indexes have separate registrations.

## Dry run on TestPyPI

*Actions > Release > Run workflow* on `main`. The workflow runs the tests, builds the sdist
and wheel as `<version>.dev<run number>`, checks those exact archives, smoke-tests the
wheel on Python 3.11 with syntax and setup, and uploads them to TestPyPI. Each new dispatch
has a new run number. A rerun has the same number; if an upload already happened, start a
new dispatch instead of trying to replace that version.

Record the workflow run, source commit, development version, artifact names and SHA-256
hashes. Confirm both wheel and sdist appear on TestPyPI and match the checked workflow
artifacts. Then use a new directory and fresh Python 3.11+ virtual environment, replacing
`<N>` below with the actual run number. These commands require the completed upload; a
local wheel smoke test does not prove index publication:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ "styleprofile[syntax]==0.2.0.dev<N>"
styleprofile setup
styleprofile --version
styleprofile demo
styleprofile score styleprofile-demo/draft.md styleprofile-demo/writer.profile.json
```

Expect the requested development package version and report version 8.1. The demo should
say `Overall: close`, with its thin-reference warning; the explicit score should agree.
Record the installed location/version and parser use as well as exit statuses. Do not
proceed to a release tag until this index-install rehearsal passes. See the README's
installation details for Windows activation.

## Cutting a release

The first public release includes twelve completed onboarding WPs. WP-1's z-score cap
and WP-10's generic contrast are deferred. Paragraph checks remain experimental; do not
turn the release into a claim of authorship detection or stronger statistical guarantees.
The separate Windows release-readiness audit must have its reviewed disposition before
selecting the release commit.

1. On reviewed, validated `main`, keep `version` in `pyproject.toml` at 0.2.0 and
   `CHANGELOG.md` headed `## [0.2.0] - 2026-10-05`. Required CI and the current package
   smoke checks must pass for the exact selected commit. Complete the TestPyPI rehearsal
   above and record its index-install evidence. A dated changelog is preparation, not
   evidence that the public release already exists.
2. Tag that commit and push the tag:

   ```bash
   git tag -a v0.2.0 -m "styleprofile 0.2.0"
   git push origin v0.2.0
   ```

3. The workflow checks that the tag matches the version, runs the tests and the package
   checks, and waits for your approval of the `pypi` environment. Approve it, and it
   uploads the exact checked artifacts to PyPI. No token is stored in the repository.
   Wait for a successful upload, then verify version 0.2.0, both archive hashes and a
   fresh installation from PyPI before calling the release complete.
4. After verified publication, optionally create a GitHub release from the tag with the
   CHANGELOG section as its notes
   (`gh release create v0.2.0 --notes-file <file>`). The CHANGELOG links each version to
   its tag, not to a GitHub release, so this step is never required.
5. Only after 0.2.0 is published and its index installation verified, replace the README
   GitHub fallback with tested PyPI instructions in a follow-up. Start the next version:
   bump `version` in `pyproject.toml` (with `uv version --bump
   minor`, which updates `uv.lock` too) and add its section to the top of the CHANGELOG,
   headed exactly `## [0.3.0] - Unreleased`: `tests/test_packaging.py` checks the CHANGELOG
   has a `## [<version>]` heading for the version in `pyproject.toml`, so the tests fail
   until it does. Add its link at the bottom too
   (`[0.3.0]: https://github.com/wangjohn/styleprofile/tree/v0.3.0`).

For the public index check, use another new directory and fresh environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple/ "styleprofile[syntax]==0.2.0"
styleprofile setup
styleprofile --version
styleprofile demo
styleprofile score styleprofile-demo/draft.md styleprofile-demo/writer.profile.json
```

The expected package version is 0.2.0, with reports 8.1. Keep the TestPyPI and PyPI evidence
separate. Until this succeeds, retain the README's GitHub fallback; do not report index
installation as passing merely because the local wheel worked.

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
