"""Stylometric profiles of arbitrary prose.

Measure a writer's style from their texts, score other prose against that reference with
Delta, and, given contrast drafts, learn which habits separate the writer from LLM output.
Only the standard library is required; spaCy adds parser-based syntax metrics.
"""

from styleprofile.display import format_summary
from styleprofile.profile import (
    Chunk,
    StyleProfileError,
    build_profile,
    load_chunks,
    load_reference,
    window,
    write_report,
)
from styleprofile.surface import surface_metrics
from styleprofile.syntax import SyntaxUnavailableError, load_parser

__all__ = [
    "Chunk",
    "StyleProfileError",
    "SyntaxUnavailableError",
    "build_profile",
    "format_summary",
    "load_chunks",
    "load_parser",
    "load_reference",
    "surface_metrics",
    "window",
    "write_report",
]
