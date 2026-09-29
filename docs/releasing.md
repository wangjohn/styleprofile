# Releasing

Releases go to [PyPI](https://pypi.org/p/styleprofile) from
[`.github/workflows/release.yml`](../.github/workflows/release.yml), with PyPI's
[trusted publishing](https://docs.pypi.org/trusted-publishers/): GitHub Actions proves to
PyPI which repository, workflow and environment is uploading, so no API token is stored
anywhere. The workflow runs only when a `v*` tag is pushed (a real release) or when it is
started by hand (a dry run to TestPyPI), never on ordinary pushes or pull requests.

## One-time setup

The project doesn't exist on PyPI yet, so each index gets a *pending* publisher, which
creates the project on its first upload. The name `styleprofile` was free on both indexes
on 2026-09-29; a pending publisher doesn't reserve it, so do this shortly before the first
release.

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
   allow only tags matching `v*`.

## Dry run on TestPyPI

*Actions > Release > Run workflow* on `main`. The workflow runs the tests, builds the sdist
and wheel as `<version>.dev<run number>` (so the dry run can be repeated; an index never
accepts the same version twice), checks them, smoke-tests the wheel on Python 3.11, and
uploads to TestPyPI. To try the result:

```bash
python -m venv /tmp/sp && . /tmp/sp/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ "styleprofile[syntax]==0.2.0.dev<N>"
styleprofile setup
styleprofile --version
```

## Cutting a release

1. On `main`, make sure `version` in `pyproject.toml` is the version to release, and
   `CHANGELOG.md` has its section. Replace "Unreleased" in the heading with today's date
   (`## [0.2.0] - 2026-10-01`) and commit that ("Release 0.2.0").
2. Tag that commit and push the tag:

   ```bash
   git tag -a v0.2.0 -m "styleprofile 0.2.0"
   git push origin v0.2.0
   ```

3. The workflow checks that the tag matches the version, runs the tests and the package
   checks, and waits for your approval of the `pypi` environment. Approve it, and it
   uploads to PyPI.
4. Optionally, create a GitHub release from the tag with the CHANGELOG section as its notes
   (`gh release create v0.2.0 --notes-file <file>`). The CHANGELOG links each version to
   its tag, not to a GitHub release, so this step is never required.
5. Start the next version: bump `version` in `pyproject.toml` (with `uv version --bump
   minor`, which updates `uv.lock` too) and add its section to the top of the CHANGELOG,
   headed exactly `## [0.3.0] - Unreleased`: `tests/test_packaging.py` checks the CHANGELOG
   has a `## [<version>]` heading for the version in `pyproject.toml`, so the tests fail
   until it does. Add its link at the bottom too
   (`[0.3.0]: https://github.com/wangjohn/styleprofile/tree/v0.3.0`).

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
   `src/styleprofile/profile.py` if saved profiles' values would change.

`tests/test_packaging.py` checks that the version, bound, URL and hash agree.
