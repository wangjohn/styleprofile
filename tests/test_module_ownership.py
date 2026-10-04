"""Public imports saved by callers still resolve after the domain module split."""

import pickle

import pytest

import styleprofile
from styleprofile import api, profile, reference, scoring


@pytest.mark.parametrize(
    "name",
    ["Text", "Settings", "Profile", "ScoreResult", "DocumentResult", "Evaluation"],
)
def test_legacy_public_pickle_globals(name):
    # Protocol zero records the defining module and name. Old public objects lived
    # in api.py; loading those globals must still find the identical public class.
    old_global = f"cstyleprofile.api\n{name}\n.".encode("ascii")
    assert pickle.loads(old_global) is getattr(styleprofile, name)
    assert getattr(api, name) is getattr(styleprofile, name)


def test_lower_level_compatibility_imports():
    assert pickle.loads(b"cstyleprofile.profile\nChunk\n.") is styleprofile.Chunk
    assert profile.Chunk is styleprofile.Chunk
    assert profile.build_reference is reference.build_reference
    assert profile.score is scoring.score
