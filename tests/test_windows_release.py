"""Release regressions for native Windows paths and private measurement caches."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path, PureWindowsPath

import pytest

import styleprofile as sp
from styleprofile import cache as caching
from styleprofile.cache_acl import _WindowsACL
from styleprofile.reports import dumps_report

ROOT = Path(__file__).resolve().parent.parent


def test_cache_protection_failure_leaves_measurements_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from styleprofile import cache_acl

    expected = sp.build(ROOT / "examples/writer", sp.Settings(syntax=False), cache=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(caching.sys, "platform", "win32")

    def denied(path: Path, *, owned_directory: bool) -> None:
        raise PermissionError("cannot protect the cache ACL")

    monkeypatch.setattr(cache_acl, "protect_cache", denied)
    actual = sp.build(ROOT / "examples/writer", sp.Settings(syntax=False), cache=True)
    assert dumps_report(actual.report) == dumps_report(expected.report)
    assert any(note.code is sp.NoteCode.CACHE_UNAVAILABLE for note in actual.notes)
    assert not (tmp_path / "styleprofile" / caching.FILENAME).exists()


@pytest.mark.parametrize("sid", ["WD", "BU", "S-1-5-21-123-456-789-1000"])
def test_acl_validation_rejects_other_principals(sid: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the actual validation independently of host OS and localized account names.
    acl = object.__new__(_WindowsACL)
    acl.user = "S-1-5-21-1-2-3-1000"
    monkeypatch.setattr(
        acl,
        "_entries",
        lambda path: (acl.user, [(0, 0, 0x1F01FF, acl.user), (0, 0, 0x120089, sid)]),
    )
    with pytest.raises(PermissionError, match="beyond this user"):
        acl.verify(Path("cache"))


def test_acl_validation_rejects_denied_access(monkeypatch: pytest.MonkeyPatch) -> None:
    acl = object.__new__(_WindowsACL)
    acl.user = "S-1-5-21-1-2-3-1000"
    monkeypatch.setattr(
        acl,
        "_entries",
        lambda path: (acl.user, [(1, 0, 0x1F01FF, acl.user), (0, 0, 0x1F01FF, acl.user)]),
    )
    with pytest.raises(PermissionError):
        acl.verify(Path("cache"))


def test_explicit_cache_parent_requires_private_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acl = object.__new__(_WindowsACL)
    acl.user = "S-1-5-21-1-2-3-1000"
    monkeypatch.setattr(acl, "_entries", lambda path: (acl.user, [(0, 0, 0x1F01FF, acl.user)]))
    acl.verify(Path("cache"))
    with pytest.raises(PermissionError):
        acl.verify(Path("cache"), directory=True)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows ACL integration")
@pytest.mark.parametrize("environment", ["XDG_CACHE_HOME", "LOCALAPPDATA"])
def test_windows_cache_override_is_private_without_changing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment: str
) -> None:
    # Explicitly grant Everyone access to the caller's override. The app must only
    # protect its own child, not rewrite these unrelated parent permissions.
    acl = _WindowsACL()
    descriptor = ctypes.c_void_p()
    acl._check(
        acl.api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:P(A;OICI;FA;;;{acl.user})(A;OICI;FA;;;WD)", 1, ctypes.byref(descriptor), None
        )
    )
    try:
        acl._check(acl.api.SetFileSecurityW(str(tmp_path), 0x80000004, descriptor))
    finally:
        acl.kernel.LocalFree(descriptor)
    before = acl.read(tmp_path)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv(environment, str(tmp_path))
    settings = sp.Settings(syntax=False)
    cold = sp.build(ROOT / "examples/writer", settings, cache=True)
    warm = sp.build(ROOT / "examples/writer", settings, cache=True)
    assert dumps_report(cold.report) == dumps_report(warm.report)
    assert (
        "\n".join(
            note.message
            for note in (*cold.notes, *warm.notes)
            if note.code is sp.NoteCode.CACHE_UNAVAILABLE
        )
        == ""
    )
    path = caching.cache_dir() / caching.FILENAME
    acl.verify(path.parent)
    if environment == "LOCALAPPDATA":
        acl.verify(path.parent.parent)
    acl.verify(path)
    journal = Path(f"{path}-journal")
    if journal.exists():
        acl.verify(journal)
    assert acl.read(tmp_path) == before


def test_cache_description_handles_uri_characters_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("cache #1?.sqlite3") if sys.platform != "win32" else Path("cache #1.sqlite3")
    with caching.MeasurementCache(path) as store:
        store.put(b"one", 3, {"value": 1})
    assert store.problem is None, store.problem
    assert caching.describe(path)[2:] == (1, None)
    assert caching.clear(path)


def test_cache_description_closes_a_connection_when_a_query_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    path = tmp_path / "empty.sqlite3"
    path.touch()
    closed = []

    class BrokenQuery:
        def execute(self, query: str) -> None:
            raise sqlite3.OperationalError("no entries table")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: BrokenQuery())
    assert caching.describe(path)[2:] == (0, "no entries table")
    assert closed == [True]


@pytest.mark.parametrize("linked", ["directory", "database", "journal"])
def test_cache_acl_refuses_links_before_touching_unrelated_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: str
) -> None:
    from styleprofile import cache_acl

    target = tmp_path / "unrelated"
    target.mkdir()
    original = target / "private.txt"
    original.write_text("unrelated data", encoding="utf-8")
    folder = tmp_path / "styleprofile"
    if linked == "directory":
        folder.symlink_to(target, target_is_directory=True)
    else:
        folder.mkdir()
        suffix = "-journal" if linked == "journal" else ""
        Path(f"{folder / caching.FILENAME}{suffix}").symlink_to(original)

    def unexpected_acl() -> None:
        pytest.fail("refusal must precede any Windows ACL access or mutation")

    monkeypatch.setattr(cache_acl, "_WindowsACL", unexpected_acl)
    with pytest.raises(PermissionError, match="link or reparse point"):
        cache_acl.protect_cache(folder / caching.FILENAME, owned_directory=True)
    assert original.read_text(encoding="utf-8") == "unrelated data"


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows junction integration")
def test_windows_cache_junction_is_refused_and_scoring_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    target = tmp_path / "unrelated"
    target.mkdir()
    acl = _WindowsACL()
    before = acl.read(target)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(tmp_path / "styleprofile"), str(target)],
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    settings = sp.Settings(syntax=False)
    expected = sp.build(ROOT / "examples/writer", settings, cache=False)
    actual = sp.build(ROOT / "examples/writer", settings, cache=True)
    assert dumps_report(actual.report) == dumps_report(expected.report)
    assert any(note.code is sp.NoteCode.CACHE_UNAVAILABLE for note in actual.notes)
    assert not (target / caching.FILENAME).exists()
    assert acl.read(target) == before


def test_windows_owned_root_link_is_refused_before_creating_cache_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "unrelated"
    target.mkdir()
    (tmp_path / "styleprofile").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(caching.sys, "platform", "win32")
    # The native default has a Cache child below the owned styleprofile root.
    store = caching.MeasurementCache(tmp_path / "styleprofile/Cache" / caching.FILENAME)
    store._owned_directory = True
    assert store.fetch(b"one") is None
    store.close()
    assert store.problem and "link or reparse point" in store.problem
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("home", ["C:/Users/alice", "D:/Users/bob", "D:/home/bob", "C:/root"])
def test_home_identity_does_not_depend_on_current_drive(home: str) -> None:
    from styleprofile.corpus.reading import _is_home

    assert _is_home(PureWindowsPath(home))
    assert not _is_home(PureWindowsPath(home) / "posts")


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_cache_hard_links_are_refused_before_acl_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    import os

    from styleprofile import cache_acl

    original = tmp_path / "unrelated.txt"
    original.write_text("unrelated data", encoding="utf-8")
    folder = tmp_path / "styleprofile"
    folder.mkdir()
    path = folder / caching.FILENAME
    os.link(original, Path(f"{path}{suffix}"))
    monkeypatch.setattr(cache_acl, "_WindowsACL", lambda: pytest.fail("must refuse before ACL"))
    with pytest.raises(PermissionError, match="multiple hard links"):
        cache_acl.protect_cache(path, owned_directory=True)
    assert original.read_text(encoding="utf-8") == "unrelated data"


def test_cache_linked_ancestor_is_refused_before_creating_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "unrelated"
    target.mkdir()
    (tmp_path / "override").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "override"))
    monkeypatch.setattr(caching.sys, "platform", "win32")
    with caching.MeasurementCache() as store:
        assert store.fetch(b"one") is None
    assert store.problem and "link or reparse point" in store.problem
    assert list(target.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows ACL integration")
def test_native_acl_preserves_unrelated_children_and_reads_numeric_user(tmp_path: Path) -> None:
    acl = _WindowsACL()
    folder = tmp_path / "styleprofile"
    folder.mkdir()
    child = folder / "unrelated.txt"
    child.write_text("unrelated data", encoding="utf-8")
    original_owner = acl.read(folder)[0]
    before = acl.read(child)
    acl.protect(folder, directory=True)
    assert acl.read(child) == before
    owner, entries = acl._entries(folder)
    assert owner == original_owner  # Elevated Windows creation can choose Administrators.
    assert {entry[3] for entry in entries} == {acl.user, "S-1-5-18", "S-1-5-32-544"}
    acl.verify(folder, directory=True)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows ACL integration")
def test_native_denied_acl_is_not_reported_as_usable(tmp_path: Path) -> None:
    acl = _WindowsACL()
    path = tmp_path / "denied.txt"
    path.touch()
    descriptor = ctypes.c_void_p()
    acl._check(
        acl.api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:P(D;;FW;;;{acl.user})(A;;FA;;;{acl.user})", 1, ctypes.byref(descriptor), None
        )
    )
    try:
        acl._check(acl.api.SetFileSecurityW(str(path), 0x80000004, descriptor))
    finally:
        acl.kernel.LocalFree(descriptor)
    try:
        with pytest.raises(PermissionError):
            acl.verify(path)
    finally:
        acl.protect(path, directory=False)
