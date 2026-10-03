"""Bundled sample texts and a directory that can be safely reused by the demo."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from styleprofile.core import StyleProfileError

MARKER = ".styleprofile-demo.json"
PROFILE = "writer.profile.json"


def _samples() -> dict[str, bytes]:
    source = files("styleprofile").joinpath("_examples")
    if not source.is_dir():
        source = Path(__file__).resolve().parents[2] / "examples"
    if not source.is_dir():
        raise StyleProfileError("the demo samples are missing; reinstall styleprofile")
    samples: dict[str, bytes] = {}

    def read(folder: Traversable, prefix: str = "") -> None:
        for child in sorted(folder.iterdir(), key=lambda item: item.name):
            name = prefix + child.name
            if child.is_dir():
                read(child, name + "/")
            else:
                samples[name] = child.read_bytes()

    read(source)
    return samples


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(directory: Path, samples: dict[str, bytes]) -> None:
    """Record exactly the files this run wrote, including its profile when available."""
    names = list(samples)
    if (directory / PROFILE).is_file():
        names.append(PROFILE)
    hashes = {name: _digest((directory / name).read_bytes()) for name in names}
    (directory / MARKER).write_text(
        json.dumps({"demo": 1, "files": hashes}, indent=2) + "\n", encoding="utf-8"
    )


def prepare(typed: str) -> tuple[Path, dict[str, bytes]]:
    """Copy the samples, refusing unmarked, edited or symlinked destinations."""
    directory = Path(typed).expanduser().absolute()
    samples = _samples()
    refusal = (
        f"{typed} is not an unchanged previous demo or an empty folder; "
        "choose another folder with --dir"
    )
    # Refuse symlinks in the destination and its parents before following any of them.
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise StyleProfileError(refusal)
    if directory.exists():
        if not directory.is_dir():
            raise StyleProfileError(refusal)
        entries = list(directory.rglob("*"))
        if any(path.is_symlink() for path in entries):
            raise StyleProfileError(refusal)
        if entries:
            try:
                marker = json.loads((directory / MARKER).read_text(encoding="utf-8"))
                hashes = marker["files"]
                allowed = set(samples) | {PROFILE}
                actual = {p.relative_to(directory).as_posix() for p in entries if p.is_file()}
                valid = (
                    marker["demo"] == 1
                    and isinstance(hashes, dict)
                    and set(samples) <= set(hashes) <= allowed
                    and actual == set(hashes) | {MARKER}
                    and all(
                        _digest((directory / name).read_bytes()) == digest
                        for name, digest in hashes.items()
                    )
                )
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise StyleProfileError(refusal)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in samples.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record(directory, samples)
    return directory, samples
