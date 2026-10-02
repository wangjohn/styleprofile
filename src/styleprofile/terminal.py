"""Output symbols and command hints that fit the user's terminal and shell."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence
from typing import TextIO

UNICODE_GLYPHS = {"bar": "█", "empty": "░", "up": "▲", "down": "▼", "divide": "÷"}
ASCII_GLYPHS = {"bar": "#", "empty": ".", "up": "^", "down": "v", "divide": "/"}


def glyphs(stream: TextIO | None = None) -> dict[str, str]:
    """Choose one consistent symbol set for the stream's encoding."""
    output = sys.stdout if stream is None else stream
    encoding = getattr(output, "encoding", None) or "utf-8"
    try:
        "".join(UNICODE_GLYPHS.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return ASCII_GLYPHS
    return UNICODE_GLYPHS


def prepare_output() -> None:
    """Keep arbitrary input names and excerpts printable on legacy encoded streams too."""
    for stream in (sys.stdout, sys.stderr):
        if glyphs(stream) is ASCII_GLYPHS:
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(errors="replace")


def shell_join(command: Sequence[str]) -> str:
    """Quote command arguments for cmd.exe on Windows and the POSIX shell elsewhere."""
    return subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command)
