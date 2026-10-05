"""Windows cache privacy, using native ACLs without an optional dependency.

Only the app's cache directory and its own database files are changed. A caller's
parent directory is inspected, never rewritten. Failure disables caching upstream.
"""

from __future__ import annotations

import ctypes
import os
import re
import stat
from pathlib import Path
from typing import Any

_native: Any = ctypes


class _WindowsACL:
    def __init__(self) -> None:
        # Imported lazily: scores and non-Windows runs do not load Windows libraries.
        from ctypes import wintypes

        self.api: Any = _native.WinDLL("advapi32", use_last_error=True)
        self.kernel: Any = _native.WinDLL("kernel32", use_last_error=True)
        pointer = ctypes.c_void_p
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.LocalFree.argtypes = [pointer]
        self.kernel.LocalFree.restype = pointer
        signatures = {
            "OpenProcessToken": (
                [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(pointer)],
                wintypes.BOOL,
            ),
            "GetTokenInformation": (
                [pointer, ctypes.c_int, pointer, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)],
                wintypes.BOOL,
            ),
            "ConvertSidToStringSidW": ([pointer, ctypes.POINTER(pointer)], wintypes.BOOL),
            "ConvertStringSecurityDescriptorToSecurityDescriptorW": (
                [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(pointer), pointer],
                wintypes.BOOL,
            ),
            "ConvertSecurityDescriptorToStringSecurityDescriptorW": (
                [pointer, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(pointer), pointer],
                wintypes.BOOL,
            ),
            "GetNamedSecurityInfoW": (
                [
                    wintypes.LPCWSTR,
                    ctypes.c_int,
                    wintypes.DWORD,
                    ctypes.POINTER(pointer),
                    pointer,
                    pointer,
                    pointer,
                    ctypes.POINTER(pointer),
                ],
                wintypes.DWORD,
            ),
            "SetFileSecurityW": ([wintypes.LPCWSTR, wintypes.DWORD, pointer], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes = arguments
            function.restype = result
        token = pointer()
        self._check(
            self.api.OpenProcessToken(self.kernel.GetCurrentProcess(), 8, ctypes.byref(token))
        )
        try:
            size = wintypes.DWORD()
            self.api.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
            if not size.value:
                self._check(False)
            data = ctypes.create_string_buffer(size.value)
            self._check(self.api.GetTokenInformation(token, 1, data, size, ctypes.byref(size)))
            self.user = self._sid(pointer.from_buffer(data).value)
        finally:
            self.kernel.CloseHandle(token)

    @staticmethod
    def _check(success: Any) -> None:
        if not success:
            raise _native.WinError(_native.get_last_error())

    def _sid(self, value: Any) -> str:
        text = ctypes.c_void_p()
        self._check(self.api.ConvertSidToStringSidW(value, ctypes.byref(text)))
        try:
            return ctypes.wstring_at(text)
        finally:
            self.kernel.LocalFree(text)

    def read(self, path: Path) -> tuple[str, str]:
        owner, descriptor = ctypes.c_void_p(), ctypes.c_void_p()
        status = self.api.GetNamedSecurityInfoW(
            str(path), 1, 5, ctypes.byref(owner), None, None, None, ctypes.byref(descriptor)
        )
        if status:
            raise _native.WinError(status)
        text = ctypes.c_void_p()
        try:
            self._check(
                self.api.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                    descriptor, 1, 4, ctypes.byref(text), None
                )
            )
            return self._sid(owner), ctypes.wstring_at(text)
        finally:
            if text.value:
                self.kernel.LocalFree(text)
            self.kernel.LocalFree(descriptor)

    def protect(self, path: Path, *, directory: bool) -> None:
        owner, _ = self.read(path)
        if owner not in {self.user, "S-1-5-18", "S-1-5-32-544"}:
            raise PermissionError(f"the cache path belongs to another user: {path}")
        inherit = "OICI" if directory else ""
        dacl = "D:P" + "".join(f"(A;{inherit};FA;;;{sid})" for sid in (self.user, "SY", "BA"))
        descriptor = ctypes.c_void_p()
        self._check(
            self.api.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                dacl, 1, ctypes.byref(descriptor), None
            )
        )
        try:
            # SetFileSecurity changes this object; unlike SetNamedSecurityInfo, it
            # does not propagate new permissions to unrelated existing children.
            self._check(self.api.SetFileSecurityW(str(path), 0x80000004, descriptor))
        finally:
            self.kernel.LocalFree(descriptor)
        self.verify(path)

    def verify(self, path: Path) -> None:
        owner, dacl = self.read(path)
        if owner not in {self.user, "S-1-5-18", "S-1-5-32-544"}:
            raise PermissionError(f"the cache path belongs to another user: {path}")
        allowed = {self.user, "SY", "BA", "S-1-5-18", "S-1-5-32-544"}
        entries = re.findall(r"\(([^()]*)\)", dacl)
        user_access = False
        for entry in entries:
            fields = entry.split(";")
            if len(fields) != 6:
                raise PermissionError(f"the cache ACL cannot be verified: {path}")
            kind, flags, rights, _, _, sid = fields
            if kind == "D":
                continue
            if kind != "A" or sid not in allowed:
                raise PermissionError(f"the cache ACL grants access beyond this user: {path}")
            if sid == self.user and "IO" not in flags and rights in {"FA", "GA", "0x1f01ff"}:
                user_access = True
        if not entries or not user_access:
            raise PermissionError(f"the cache ACL does not grant this user full access: {path}")


def check_cache_paths(path: Path, *, owned_directory: bool) -> list[Path]:
    """Refuse existing reparse points before creating directories or changing ACLs."""
    directories = [path.parent]
    if (
        owned_directory
        and path.parent.name == "Cache"
        and path.parent.parent.name == "styleprofile"
    ):
        directories.insert(0, path.parent.parent)
    files = [
        Path(f"{path}{suffix}")
        for suffix in ("", "-journal", "-wal", "-shm")
        if os.path.lexists(f"{path}{suffix}")
    ]
    for item in [*directories, *files]:
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PermissionError(f"the cache path is a link or reparse point: {item}")
    return files


def protect_cache(path: Path, *, owned_directory: bool) -> None:
    """Protect the owned directory and existing database/sidecars, or fail closed."""
    files = check_cache_paths(path, owned_directory=owned_directory)
    acl = _WindowsACL()
    if owned_directory:
        if path.parent.name == "Cache" and path.parent.parent.name == "styleprofile":
            acl.protect(path.parent.parent, directory=True)
        acl.protect(path.parent, directory=True)
    else:
        acl.verify(path.parent)
    for item in files:
        acl.protect(item, directory=False)
