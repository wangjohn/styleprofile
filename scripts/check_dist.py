"""Build the sdist and wheel, check what they contain, and smoke-test the wheel.

    uv run python scripts/check_dist.py                  # build, check, smoke-test
    uv run python scripts/check_dist.py --syntax         # also the syntax extra and `setup`
    uv run python scripts/check_dist.py --sdist-tests    # also the sdist's own test suite
    uv run python scripts/check_dist.py --dist dist/     # check files already built

It builds with ``uv build`` (the wheel from the sdist, so a file missing from the sdist
fails the build), or with ``--dist`` takes the sdist and wheel already in a folder, so the
files checked are exactly the files a release uploads. It checks both archives' contents
and metadata and runs ``twine check``. It installs the wheel
with pip into a new virtual environment on Python 3.11, the oldest supported version, and
runs everything from a temporary directory, so the repository is never on ``sys.path``:
``styleprofile build`` and ``score`` on ``examples/``, and the README's library example
with the example corpus in place of ``posts/`` and ``llm-drafts/``. With ``--syntax`` it
then installs the ``syntax`` extra, runs ``styleprofile setup`` (which downloads spaCy's
English model) and checks that ``build`` uses the parser.

With ``--sdist-tests`` it unpacks the sdist, installs it with the ``syntax`` extra, pytest
and (through ``styleprofile setup``) the model into another new 3.11 environment, and runs
the test suite from the unpacked folder, as a downstream packager would.

Everything goes to a temporary directory, removed afterwards unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "3.11"

# Paths (relative to the sdist's top folder) that must be there: enough to run the tests.
SDIST_REQUIRED = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "PKG-INFO",
    "src/styleprofile/__init__.py",
    "src/styleprofile/py.typed",
    "tests/test_snapshots.py",
    "tests/snapshots/surface/build.txt",
    "examples/draft.md",
    "examples/writer/old-maps.md",
    "examples/llm-drafts/old-maps.md",
    "docs/library.md",
    "docs/method.md",
    "bench/run.py",
]
# Patterns that must match nothing in the sdist: generated corpora and results, profiles and
# personal corpora, build and tool output, CI config, the lock file and local files.
SDIST_FORBIDDEN = [
    r"bench/corpora(/|$)",
    r"bench/work(/|$)",
    r"bench/results\.json$",
    r"(^|/)profiles/",
    r"^corpora/",
    r"^data/",
    r"^dist/",
    r"^scripts/",
    r"^\.github/",
    r"^\.claude/",
    r"^\.venv/",
    r"^uv\.lock$",
    r"^Makefile$",
    r"^\.python-version$",
    r"__pycache__/",
    r"\.py[cod]$",
    r"\.DS_Store$",
    r"\.egg-info/",
]


def _run(command: Sequence[str | Path], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    shown = " ".join(str(part) for part in command)
    print(f"$ {shown}", flush=True)
    done = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        raise SystemExit(f"failed with exit status {done.returncode}: {shown}")
    return done.stdout + done.stderr


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"check_dist: {message}")


def check_sdist(sdist: Path, version: str) -> None:
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    top = f"styleprofile-{version}/"
    if not all(name == top.rstrip("/") or name.startswith(top) for name in names):
        _fail(f"sdist entries outside {top}")
    inside = {name.removeprefix(top) for name in names}
    missing = [path for path in SDIST_REQUIRED if path not in inside]
    if missing:
        _fail(f"sdist lacks {missing}")
    unwanted = sorted(
        name for name in inside if any(re.search(pattern, name) for pattern in SDIST_FORBIDDEN)
    )
    if unwanted:
        _fail(f"sdist has files it should not: {unwanted}")
    print(f"sdist: {len(inside)} entries, none unwanted")


def check_wheel(wheel: Path, version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = archive.read(f"styleprofile-{version}.dist-info/METADATA").decode()
    info = f"styleprofile-{version}.dist-info/"
    stray = [name for name in names if not name.startswith(("styleprofile/", info))]
    if stray:
        _fail(f"wheel has files outside the package: {stray}")
    if "styleprofile/py.typed" not in names:
        _fail("wheel lacks styleprofile/py.typed")
    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    if any("@" in line or "://" in line for line in requires):
        _fail(f"a requirement names a URL, which PyPI refuses: {requires}")
    for field in ("License-Expression: MIT", "Requires-Python: >=3.11", "Project-URL: "):
        if field not in metadata:
            _fail(f"wheel metadata lacks {field!r}")
    print(f"wheel: {len(names)} files; requirements {requires}")


def _readme_example() -> str:
    """The README's library example, on the sample corpus."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Using it from Python", 1)[1]
    match = re.search(r"```python\n(.*?)```", section, re.S)
    if not match:
        _fail("README.md has no Python example under 'Using it from Python'")
    code = match[1]
    for typed, example in (("posts/", "examples/writer/"), ("llm-drafts/", "examples/llm-drafts/")):
        if f'"{typed}"' not in code:
            _fail(f"the README example no longer reads {typed!r}; update check_dist.py")
        code = code.replace(f'"{typed}"', f'"{example}"')
    return code


def _clean_env() -> dict[str, str]:
    """The environment for commands in a new venv: nothing from the repository or the
    caller's virtualenv."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")
    }


def _venv(path: Path) -> tuple[Path, Path]:
    """A new virtual environment with pip on the oldest supported Python: its python and
    styleprofile executables."""
    _run(["uv", "venv", "--seed", "--no-project", "--python", PYTHON, path], cwd=path.parent)
    bindir = path / ("Scripts" if os.name == "nt" else "bin")
    return bindir / "python", bindir / "styleprofile"


def _dev_requirements() -> list[str]:
    """The dev group's pytest requirement, which the sdist's tests need."""
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [
        requirement
        for requirement in groups["dependency-groups"]["dev"]
        if requirement.startswith("pytest")
    ]


def sdist_tests(sdist: Path, version: str, work: Path) -> None:
    """Run the test suite from the unpacked sdist, with the package installed from it."""
    unpacked = work / "sdist"
    unpacked.mkdir()
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    source = unpacked / f"styleprofile-{version}"
    python, cli = _venv(work / "sdist-venv")
    env = _clean_env()
    _run(
        [python, "-m", "pip", "install", "--quiet", f"{source}[syntax]", *_dev_requirements()],
        cwd=source,
        env=env,
    )
    print(_run([cli, "setup"], cwd=source, env=env).strip().splitlines()[-1])
    out = _run([python, "-m", "pytest", "-p", "no:cacheprovider"], cwd=source, env=env)
    summary = out.strip().splitlines()[-1]
    if " failed" in summary or " error" in summary or " passed" not in summary:
        _fail(f"the sdist's tests: {summary}")
    print(f"sdist tests: {summary}")


def smoke_test(wheel: Path, work: Path, *, syntax: bool) -> None:
    venv = work / "venv"
    python, cli = _venv(venv)
    env = _clean_env()
    run = work / "run"
    run.mkdir()
    shutil.copytree(ROOT / "examples", run / "examples")
    _run([python, "-m", "pip", "install", "--quiet", wheel], cwd=run, env=env)

    where = _run(
        [python, "-c", "import styleprofile, sys; print(styleprofile.__file__); print(sys.path)"],
        cwd=run,
        env=env,
    )
    location = Path(where.splitlines()[0]).resolve()
    if venv.resolve() not in location.parents:
        _fail(f"styleprofile imported from {location}, not the clean environment")
    if str(ROOT) in where:
        _fail("the repository is on sys.path")

    print(_run([cli, "--version"], cwd=run, env=env).strip())
    out = _run(
        [
            cli,
            "build",
            "examples/writer",
            "--contrast",
            "examples/llm-drafts",
            "-o",
            "writer.json",
        ],
        cwd=run,
        env=env,
    )
    if "spaCy is not installed" not in out:
        _fail("build without spaCy did not say syntax metrics are left out")
    out = _run(
        [cli, "score", "examples/draft.md", "writer.json", "-o", "draft.json"], cwd=run, env=env
    )
    print("\n".join(out.splitlines()[:5]))
    _run([cli, "show", "draft.json"], cwd=run, env=env)
    out = _run([python, "-c", _readme_example()], cwd=run, env=env)
    print(f"README library example: {out.strip()}")

    if not syntax:
        return
    _run([python, "-m", "pip", "install", "--quiet", f"{wheel}[syntax]"], cwd=run, env=env)
    # spaCy without its model: the note must point at `styleprofile setup`.
    out = _run([cli, "build", "examples/writer", "-o", "no-model.json"], cwd=run, env=env)
    if "run `styleprofile setup`" not in out:
        _fail(f"build with spaCy but no model did not suggest `styleprofile setup`:\n{out}")
    out = _run([cli, "setup"], cwd=run, env=env)
    print(out.strip())
    out = _run([cli, "setup"], cwd=run, env=env)
    if "already installed" not in out:
        _fail("a second `styleprofile setup` did not find the model installed")
    out = _run([cli, "build", "examples/writer", "-o", "syntax.json"], cwd=run, env=env)
    if "not installed" in out:
        _fail(f"build after setup still left syntax out:\n{out}")
    report = json.loads((run / "syntax.json").read_text(encoding="utf-8"))
    used = report["settings"]["syntax_used"]
    if not used or used["model"] != "en_core_web_sm":
        _fail(f"build after setup did not use the parser: {used}")
    print(f"syntax after setup: {used}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--syntax", action="store_true", help="also test the syntax extra")
    parser.add_argument(
        "--sdist-tests", action="store_true", help="also run the test suite from the sdist"
    )
    parser.add_argument(
        "--dist",
        type=Path,
        metavar="DIR",
        help="check the sdist and wheel already in DIR instead of building them",
    )
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args(argv)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    work = Path(tempfile.mkdtemp(prefix="styleprofile-dist-"))
    try:
        dist = args.dist.resolve() if args.dist else work / "dist"
        if not args.dist:
            _run(["uv", "build", "--out-dir", dist], cwd=ROOT)
        sdist = dist / f"styleprofile-{version}.tar.gz"
        wheel = dist / f"styleprofile-{version}-py3-none-any.whl"
        built = sorted(path.name for path in dist.iterdir() if not path.name.startswith("."))
        if built != sorted([sdist.name, wheel.name]):
            _fail(f"expected exactly {sdist.name} and {wheel.name} in {dist}, found {built}")
        check_sdist(sdist, version)
        check_wheel(wheel, version)
        # What PyPI checks on upload, the README's rendering included.
        _run(["uvx", "twine", "check", "--strict", sdist, wheel], cwd=ROOT)
        smoke_test(wheel, work, syntax=args.syntax)
        if args.sdist_tests:
            sdist_tests(sdist, version, work)
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print("check_dist: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
