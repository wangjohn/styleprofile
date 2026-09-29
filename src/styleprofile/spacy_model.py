"""Installing spaCy's English model, which `styleprofile setup` does.

The model is not on PyPI, and PyPI refuses a package that depends on a URL, so the `syntax`
extra installs spaCy only. ``install_model`` then installs the model version the package is
tested with (``MODEL_VERSION``) from spaCy's releases, as ``python -m spacy download
en_core_web_sm-3.8.0 --direct`` would, but pinned to the file's hash. It uses pip, or ``uv
pip`` in an environment that has no pip (one made by uv).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from styleprofile.core import StyleProfileError
from styleprofile.syntax import DEFAULT_MODEL, MODEL_VERSION

# The model wheel and its hash, which the installer checks. Keep them equal to the
# spacy-model source and its hash in uv.lock (tests check).
MODEL_WHEEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{DEFAULT_MODEL}-{MODEL_VERSION}/{DEFAULT_MODEL}-{MODEL_VERSION}-py3-none-any.whl"
)
MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
MODEL_URL = f"{MODEL_WHEEL_URL}#sha256={MODEL_SHA256}"
# A model works only with the spaCy minor version it was built for (3.8.x for 3.8.0). The
# `syntax` extra's bound in pyproject.toml matches; bump them together.
SPACY_SERIES = ".".join(MODEL_VERSION.split(".")[:2])


def _series(version: str) -> str:
    return ".".join(version.split(".")[:2])


@dataclass(frozen=True)
class ModelStatus:
    """What is installed: spaCy's version and the model's, each None when missing."""

    spacy_version: str | None
    model_version: str | None

    @property
    def spacy_matches(self) -> bool:
        """spaCy is installed, in the series the model is built for."""
        return self.spacy_version is not None and _series(self.spacy_version) == SPACY_SERIES

    @property
    def ready(self) -> bool:
        """spaCy of the right series and the tested model version are both installed."""
        return self.spacy_matches and self.model_version == MODEL_VERSION


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_status() -> ModelStatus:
    importlib.invalidate_caches()
    return ModelStatus(_installed_version("spacy"), _installed_version(DEFAULT_MODEL))


def _has_pip() -> bool:
    return importlib.util.find_spec("pip") is not None


def uses_uv() -> bool:
    """This environment has no pip but uv is on the PATH: an environment made by uv."""
    return not _has_pip() and shutil.which("uv") is not None


def syntax_install() -> str:
    """How to install the `syntax` extra here, then the model: with uv in an environment
    made by uv, which has no pip, and with pip otherwise."""
    if uses_uv():
        return (
            "uv pip install 'styleprofile[syntax]' then run `styleprofile setup` (in a clone, "
            "uv sync --extra syntax installs both)"
        )
    return "pip install 'styleprofile[syntax]' then run `styleprofile setup`"


def _externally_managed() -> bool:
    """pip would refuse to install here (PEP 668): a system Python that marks itself as
    managed by the operating system, outside any virtual environment."""
    if sys.prefix != sys.base_prefix:
        return False
    return (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists()


def install_command(url: str = MODEL_URL) -> list[str]:
    """The command that installs ``url`` into this Python's environment: pip when it has
    pip, otherwise ``uv pip`` aimed at this interpreter."""
    if _has_pip():
        return [sys.executable, "-m", "pip", "install", url]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, url]
    raise StyleProfileError(
        f"this Python environment has neither pip nor uv, so {DEFAULT_MODEL} cannot be "
        "installed automatically",
        code="setup_no_installer",
    )


def check_spacy(status: ModelStatus) -> None:
    """Refuse to install the model without spaCy, or next to a spaCy it wasn't built for."""
    if status.spacy_version is None:
        raise StyleProfileError(
            f"spaCy is not installed, and {DEFAULT_MODEL} needs it; install the `syntax` "
            f"extra first: {syntax_install().split(' then ')[0]}",
            code="setup_needs_spacy",
        )
    if not status.spacy_matches:
        raise StyleProfileError(
            f"spaCy {status.spacy_version} is installed, but {DEFAULT_MODEL} "
            f"{MODEL_VERSION}, the model styleprofile is tested with, needs spaCy "
            f"{SPACY_SERIES}.x",
            code="setup_spacy_version",
        )


def install_model(
    run: Callable[[Sequence[str]], int] | None = None,
) -> list[str]:
    """Install the tested model version, and return the command that did. ``run`` runs a
    command and returns its exit status (the default runs it with ``subprocess``; tests
    replace it). Raises ``StyleProfileError`` when spaCy is missing or of another series,
    when there is no way to install, or when the install fails."""
    check_spacy(model_status())
    command = install_command()
    if command[1:3] == ["-m", "pip"] and _externally_managed():
        raise StyleProfileError(
            f"this Python is managed by the operating system (PEP 668), so pip won't install "
            f"{DEFAULT_MODEL} into it",
            code="setup_externally_managed",
        )
    status = (run or _run)(command)
    if status != 0:
        raise StyleProfileError(
            f"installing {DEFAULT_MODEL} {MODEL_VERSION} failed (exit status {status}); "
            f"the command was: {' '.join(command)}",
            code="setup_failed",
        )
    installed = model_status().model_version
    if installed != MODEL_VERSION:
        raise StyleProfileError(
            f"the install finished, but {DEFAULT_MODEL} {MODEL_VERSION} is not installed in "
            f"this environment ({sys.executable}); found {installed or 'no version'}",
            code="setup_failed",
        )
    return command


def _run(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode
