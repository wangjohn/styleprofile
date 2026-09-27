"""Stylometric profiles of arbitrary prose.

Measure a writer's style from their texts, score other prose against that reference with
Delta, and, given contrast drafts, learn which habits separate the writer from LLM output.
Only the standard library is required; spaCy adds parser-based syntax metrics.

``build``, ``Profile.score`` and ``evaluate`` run the same pipeline as the command line; see
``docs/library.md``. ``build_reference`` and ``score`` are the lower-level steps under them,
taking chunks exactly as given.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from styleprofile.api import (
    Evaluation,
    Profile,
    Progress,
    ScoreResult,
    Settings,
    Text,
    build,
    evaluate,
)
from styleprofile.display import format_summary
from styleprofile.profile import (
    Chunk,
    Note,
    StyleProfileError,
    build_reference,
    load_chunks,
    load_reference,
    load_report,
    report_kind,
    score,
    window,
    write_report,
)
from styleprofile.surface import surface_metrics
from styleprofile.syntax import SyntaxUnavailableError, load_parser

try:
    __version__ = _version("styleprofile")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Chunk",
    "Evaluation",
    "Note",
    "Profile",
    "Progress",
    "ScoreResult",
    "Settings",
    "StyleProfileError",
    "SyntaxUnavailableError",
    "Text",
    "__version__",
    "build",
    "build_reference",
    "evaluate",
    "format_summary",
    "load_chunks",
    "load_parser",
    "load_reference",
    "load_report",
    "report_kind",
    "score",
    "surface_metrics",
    "window",
    "write_report",
]
