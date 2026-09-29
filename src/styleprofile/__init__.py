"""Stylometric profiles of arbitrary prose.

Measure a writer's style from their texts, score other prose against that reference with
Delta, and, given contrast drafts, learn which habits separate the writer from LLM output.
Only the standard library is required; spaCy adds parser-based syntax metrics.

``build``, ``Profile.score`` and ``evaluate`` run the same pipeline as the command line; see
``docs/library.md``. The lower-level steps under them, which take chunks exactly as given,
are in ``styleprofile.profile`` (``build_reference``, ``score``).
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from styleprofile.api import (
    DocumentResult,
    Evaluation,
    Profile,
    ScoreResult,
    Settings,
    Text,
    build,
    evaluate,
)
from styleprofile.core import (
    LikenessVerdict,
    Note,
    NoteCode,
    Phase,
    Progress,
    StyleProfileError,
    Verdict,
)
from styleprofile.drift import Passage, Trait
from styleprofile.profile import Chunk
from styleprofile.syntax import SyntaxUnavailableError

try:
    __version__ = _version("styleprofile")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Chunk",
    "DocumentResult",
    "Evaluation",
    "LikenessVerdict",
    "Note",
    "NoteCode",
    "Passage",
    "Phase",
    "Profile",
    "Progress",
    "ScoreResult",
    "Settings",
    "StyleProfileError",
    "SyntaxUnavailableError",
    "Text",
    "Trait",
    "Verdict",
    "__version__",
    "build",
    "evaluate",
]
