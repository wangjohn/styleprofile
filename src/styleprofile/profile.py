"""Compatibility entry points for lower-level chunk profiling and scoring.

Implementations live in corpus, reference, scoring and reports; new internal imports
should use those modules directly.
"""

from styleprofile.corpus.types import Chunk as Chunk
from styleprofile.reference import build_reference as build_reference
from styleprofile.scoring import score as score
