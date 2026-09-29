"""Packaging and release: the package metadata PyPI sees, `styleprofile setup`, the hint
toward it when spaCy's model is missing, and the nominalization metric fixed for 0.2.0.

The built sdist and wheel themselves are checked by ``scripts/check_dist.py`` (``make
dist-check``), which CI runs in its own job. These tests also run from an unpacked sdist,
which has no ``.python-version`` or ``uv.lock``; the tests of those skip there.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

import styleprofile as sp
from styleprofile import api, spacy_model, syntax
from styleprofile.cli import main
from styleprofile.spacy_model import (
    MODEL_SHA256,
    MODEL_URL,
    MODEL_WHEEL_URL,
    SPACY_SERIES,
    ModelStatus,
)
from styleprofile.syntax import DEFAULT_MODEL, MODEL_VERSION, is_nominalization

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WRITER = ROOT / "examples" / "writer"


def _repository_file(name: str) -> Path:
    """A file of the repository that the sdist leaves out; skip without it."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not here (running from an sdist)")
    return path


# The package metadata.


def test_published_requirements_name_no_urls() -> None:
    """PyPI refuses a package whose requirements point at a URL, so neither the dependencies
    nor any extra may; the model's URL lives in [tool.uv.sources], for development only."""
    project = PYPROJECT["project"]
    requirements = [*project["dependencies"]]
    for extra in project["optional-dependencies"].values():
        requirements += extra
    assert not any("@" in requirement or "://" in requirement for requirement in requirements)
    assert "allow-direct-references" not in str(PYPROJECT.get("tool", {}).get("hatch", {}))


def test_the_syntax_extra_is_bound_to_the_models_spacy_series() -> None:
    """A model works only with the spaCy minor version it was built for."""
    major, minor = (int(part) for part in SPACY_SERIES.split("."))
    assert PYPROJECT["project"]["optional-dependencies"]["syntax"] == [
        f"spacy>={major}.{minor},<{major}.{minor + 1}"
    ]


def test_the_development_model_is_the_version_setup_installs() -> None:
    assert PYPROJECT["dependency-groups"]["spacy-model"] == [f"en-core-web-sm=={MODEL_VERSION}"]
    assert PYPROJECT["tool"]["uv"]["sources"]["en-core-web-sm"] == {"url": MODEL_WHEEL_URL}
    assert "spacy-model" in PYPROJECT["tool"]["uv"]["default-groups"]
    assert MODEL_WHEEL_URL.endswith(f"/{DEFAULT_MODEL}-{MODEL_VERSION}-py3-none-any.whl")
    assert f"{MODEL_WHEEL_URL}#sha256={MODEL_SHA256}" == MODEL_URL


def test_setup_pins_the_hash_uv_lock_recorded() -> None:
    lock = tomllib.loads(_repository_file("uv.lock").read_text(encoding="utf-8"))
    [model] = [package for package in lock["package"] if package["name"] == "en-core-web-sm"]
    assert model["version"] == MODEL_VERSION
    assert [wheel["hash"] for wheel in model["wheels"]] == [f"sha256:{MODEL_SHA256}"]
    assert [wheel["url"] for wheel in model["wheels"]] == [MODEL_WHEEL_URL]


def test_metadata_for_pypi() -> None:
    project = PYPROJECT["project"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"])
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE"]
    # A license expression replaces the License :: classifiers (PEP 639).
    assert not any(line.startswith("License ::") for line in project["classifiers"])
    floor = re.fullmatch(r">=3\.(\d+)", project["requires-python"])
    assert floor
    versions = [
        line.rsplit(" ", 1)[1]
        for line in project["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", line)
    ]
    assert versions[0] == f"3.{floor[1]}"
    assert set(project["urls"]) >= {"Homepage", "Source", "Issues", "Changelog"}
    assert project["keywords"]
    assert (ROOT / "src" / "styleprofile" / "py.typed").exists()


def test_the_installed_version_is_the_project_version() -> None:
    assert sp.__version__ == importlib.metadata.version("styleprofile")
    assert sp.__version__ == PYPROJECT["project"]["version"]


def test_the_changelog_covers_this_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{PYPROJECT['project']['version']}]" in changelog


def test_the_python_version_for_development_is_supported() -> None:
    chosen = _repository_file(".python-version").read_text(encoding="utf-8").strip()
    supported = [
        line.rsplit(" ", 1)[1]
        for line in PYPROJECT["project"]["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", line)
    ]
    assert chosen in supported


# `styleprofile setup`, with the install replaced: these tests never touch the network.


class _Installer:
    """Stands in for the environment: what is installed, and a fake ``run`` that records
    the command and installs the model when it succeeds."""

    def __init__(self, spacy: str | None, model: str | None, *, status: int = 0) -> None:
        self.status = ModelStatus(spacy, model)
        self.exit = status
        self.commands: list[list[str]] = []

    def run(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        if self.exit == 0:
            self.status = ModelStatus(self.status.spacy_version, MODEL_VERSION)
        return self.exit


@pytest.fixture
def installer(monkeypatch: pytest.MonkeyPatch) -> _Installer:
    """A virtual environment with pip and spaCy 3.8, and no model."""
    fake = _Installer("3.8.16", None)
    monkeypatch.setattr(spacy_model, "model_status", lambda: fake.status)
    monkeypatch.setattr(spacy_model, "_run", fake.run)
    monkeypatch.setattr(spacy_model, "_has_pip", lambda: True)
    monkeypatch.setattr(spacy_model, "_externally_managed", lambda: False)
    return fake


def test_setup_installs_the_tested_model_with_pip(
    installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup"]) == 0
    assert installer.commands == [[sys.executable, "-m", "pip", "install", MODEL_URL]]
    assert MODEL_URL.endswith(f"#sha256={MODEL_SHA256}")  # the installer checks the file
    out, err = capsys.readouterr()
    assert f"Installing {DEFAULT_MODEL} {MODEL_VERSION}" in out
    assert f"Installed {DEFAULT_MODEL} {MODEL_VERSION}." in out
    assert "note: ran: " in err and MODEL_URL in err


def test_setup_does_nothing_when_the_model_is_installed(
    installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    installer.status = ModelStatus("3.8.16", MODEL_VERSION)
    assert main(["setup"]) == 0
    assert installer.commands == []
    assert "already installed" in capsys.readouterr().out


def test_setup_replaces_another_model_version(
    installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    installer.status = ModelStatus("3.8.16", "3.7.1")
    assert main(["setup"]) == 0
    assert len(installer.commands) == 1
    assert "replacing 3.7.1" in capsys.readouterr().out


def test_setup_without_spacy_says_to_install_the_extra(
    installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    installer.status = ModelStatus(None, None)
    assert main(["setup"]) == 1
    assert installer.commands == []
    err = capsys.readouterr().err
    assert "install the `syntax` extra first: pip install 'styleprofile[syntax]'\n" in err
    assert "hint: then run `styleprofile setup` again" in err


@pytest.mark.parametrize("model", [None, MODEL_VERSION])
def test_setup_refuses_a_spacy_the_model_was_not_built_for(
    installer: _Installer, capsys: pytest.CaptureFixture[str], model: str | None
) -> None:
    installer.status = ModelStatus("3.9.1", model)
    assert not installer.status.ready
    assert main(["setup"]) == 1
    assert installer.commands == []
    err = capsys.readouterr().err
    assert f"spaCy 3.9.1 is installed, but {DEFAULT_MODEL} {MODEL_VERSION}" in err
    assert f"needs spaCy {SPACY_SERIES}.x" in err
    assert f"hint: install spaCy {SPACY_SERIES}.x" in err


def test_setup_reports_a_failed_install_without_guessing_why(
    installer: _Installer, capsys: pytest.CaptureFixture[str]
) -> None:
    installer.exit = 1
    assert main(["setup"]) == 1
    err = capsys.readouterr().err
    assert "failed (exit status 1)" in err
    assert "hint: the installer's own output above says why" in err
    assert f"pip install '{MODEL_URL}'" in err
    assert "network" not in err


def test_setup_refuses_an_externally_managed_python(
    installer: _Installer, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spacy_model, "_externally_managed", lambda: True)
    assert main(["setup"]) == 1
    assert installer.commands == []
    err = capsys.readouterr().err
    assert "managed by the operating system (PEP 668)" in err
    assert "hint: install styleprofile in a virtual environment" in err
    assert "network" not in err


def test_setup_without_an_installer_says_so(
    installer: _Installer, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(spacy_model, "_has_pip", lambda: False)
    monkeypatch.setattr(spacy_model.shutil, "which", lambda name: None)
    assert main(["setup"]) == 1
    assert installer.commands == []
    err = capsys.readouterr().err
    assert "neither pip nor uv" in err
    assert "hint: add pip to this environment (python -m ensurepip)" in err
    assert "network" not in err


def test_setup_checks_the_model_really_arrived(
    installer: _Installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install into some other environment (a pip that isn't this Python's) succeeds
    but leaves this one without the model; setup must say so."""
    monkeypatch.setattr(spacy_model, "_run", lambda command: 0)
    with pytest.raises(sp.StyleProfileError, match="is not installed in this environment"):
        spacy_model.install_model()


def test_install_command_falls_back_to_uv_then_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spacy_model, "_has_pip", lambda: False)
    monkeypatch.setattr(spacy_model.shutil, "which", lambda name: "/bin/uv")
    assert spacy_model.install_command() == [
        "/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        MODEL_URL,
    ]
    monkeypatch.setattr(spacy_model.shutil, "which", lambda name: None)
    with pytest.raises(sp.StyleProfileError, match="neither pip nor uv") as error:
        spacy_model.install_command()
    assert error.value.code == "setup_no_installer"


def test_model_status_reads_installed_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"spacy": "3.8.16", DEFAULT_MODEL: MODEL_VERSION}

    def version(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(spacy_model.importlib.metadata, "version", version)
    assert spacy_model.model_status().ready
    versions["spacy"] = "3.9.0"
    assert not spacy_model.model_status().ready
    versions["spacy"] = "3.8.16"
    del versions[DEFAULT_MODEL]
    assert spacy_model.model_status() == ModelStatus("3.8.16", None)
    assert not spacy_model.model_status().ready


# spaCy installed but its model missing: every message points at `styleprofile setup`.


@pytest.fixture
def model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """spaCy imports, but loading its English model fails as it does when not installed."""

    def load(name: str, **kwargs: object) -> None:
        raise OSError(f"[E050] Can't find model '{name}'.")

    fake_spacy = SimpleNamespace(load=load, __version__="3.8.16")
    monkeypatch.setattr(syntax, "import_module", lambda name: fake_spacy)
    # The real loader, uncached, so the fake is what it sees.
    monkeypatch.setattr(api, "_default_parser", syntax.load_parser)


@pytest.mark.usefixtures("model_missing")
def test_a_missing_model_suggests_setup() -> None:
    with pytest.raises(sp.SyntaxUnavailableError, match="run `styleprofile setup`") as error:
        syntax.load_parser()
    assert error.value.model_missing and error.value.code == "syntax_unavailable"

    profile = sp.build(WRITER)
    [note] = [note for note in profile.notes if note.code == sp.NoteCode.NO_SYNTAX]
    assert note.message == (
        f"spaCy's English model ({DEFAULT_MODEL}) is not installed, so this profile has "
        "surface metrics only; for syntax metrics, run `styleprofile setup` and build again"
    )


@pytest.mark.usefixtures("model_missing")
def test_the_cli_notes_a_missing_model_with_setup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["build", str(WRITER), "-o", str(tmp_path / "writer.json")]) == 0
    assert "run `styleprofile setup` and build again" in capsys.readouterr().err


@pytest.fixture
def no_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_module(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(syntax, "import_module", import_module)
    monkeypatch.setattr(api, "_default_parser", syntax.load_parser)


@pytest.mark.usefixtures("no_spacy")
def test_missing_spacy_suggests_the_extra_with_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spacy_model, "_has_pip", lambda: True)
    with pytest.raises(sp.SyntaxUnavailableError) as error:
        syntax.load_parser()
    assert not error.value.model_missing
    assert "pip install 'styleprofile[syntax]'), then run `styleprofile setup`" in str(error.value)
    [note] = [note for note in sp.build(WRITER).notes if note.code == sp.NoteCode.NO_SYNTAX]
    assert note.message == (
        "spaCy is not installed, so this profile has surface metrics only; for syntax "
        "metrics, pip install 'styleprofile[syntax]' then run `styleprofile setup` and "
        "build again"
    )


@pytest.mark.usefixtures("no_spacy")
def test_missing_spacy_suggests_uv_in_an_environment_made_by_uv(
    monkeypatch: pytest.MonkeyPatch, installer: _Installer
) -> None:
    monkeypatch.setattr(spacy_model, "_has_pip", lambda: False)
    monkeypatch.setattr(spacy_model.shutil, "which", lambda name: "/bin/uv")
    [note] = [note for note in sp.build(WRITER).notes if note.code == sp.NoteCode.NO_SYNTAX]
    assert "uv pip install 'styleprofile[syntax]' then run `styleprofile setup`" in note.message
    assert "uv sync --extra syntax" in note.message
    assert "; for syntax metrics, pip install" not in note.message
    installer.status = ModelStatus(None, None)
    with pytest.raises(sp.StyleProfileError, match="first: uv pip install"):
        spacy_model.install_model()


# The nominalization metric.


@pytest.mark.parametrize(
    "word",
    [
        "decision",
        "argument",
        "darkness",
        "sadness",
        "payment",
        "ability",
        "performance",
        "difference",
        "distance",
        "silence",  # silent
        "information",
        "Government",
        # Short stems that are real nominalizations.
        "motion",
        "vision",
        "fusion",
        "option",
        "action",
        "unity",
        # US and British spellings alike.
        "defence",
        "defense",
        "offenses",
        # Plurals count as their singular, -ities and -nesses included.
        "decisions",
        "abilities",
        "priorities",
        "weaknesses",
        "differences",
    ],
)
def test_nominalizations(word: str) -> None:
    assert is_nominalization(word)


@pytest.mark.parametrize(
    "word",
    [
        # The ending is part of a one-syllable root.
        "fence",
        "fences",
        "city",
        "cities",
        "pity",
        "dance",
        # The exception list: the ending is part of the root ...
        "chance",
        "moment",
        "moments",
        "nation",
        "station",
        # ... there is no English base ...
        "science",
        "quality",
        "community",
        "communities",
        "university",
        "tradition",
        "instance",
        # ... or the only related verb is the same word.
        "sentence",
        "sentences",
        "question",
        "comment",
        "document",
        "influence",
        "experience",
        "business",
        "witnesses",
        # No nominalizing suffix at all.
        "table",
        "princess",
        "license",
    ],
)
def test_not_nominalizations(word: str) -> None:
    assert not is_nominalization(word)


def test_every_exception_would_otherwise_count() -> None:
    """An exception that the stem rule already rules out, or that has no suffix, is dead
    weight in the list."""
    for word in syntax._NOT_NOMINALIZATIONS:
        match = syntax._NOMINALIZATION.match(word)
        assert match and len(match["stem"]) >= syntax._MIN_NOMINALIZATION_STEM, word
    for us, british in syntax._US_SPELLINGS.items():
        assert is_nominalization(us) == is_nominalization(british), us


def test_the_parser_counts_only_real_nominalizations() -> None:
    try:
        parser = syntax.load_parser()
    except sp.SyntaxUnavailableError:
        pytest.skip("spaCy English model is not installed")
    text = (
        "The fence by the station kept the city out of sight. "
        "Her decision to paint it showed real kindness and patience."
    )
    [(metrics, _)] = list(parser.parse([text]))
    words = sum(token.is_alpha for token in parser.nlp(text))
    value = metrics["syntax"]["nominalizations_per_1k"]
    assert value == pytest.approx(3 * 1000 / words)  # decision, kindness, patience
