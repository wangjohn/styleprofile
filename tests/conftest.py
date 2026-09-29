"""Test-wide setup: the measurement cache lives in a temporary directory, never the user's."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _private_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    # One cache for the session, so later tests also read what earlier ones measured: every
    # comparison between two runs then covers a cache hit too.
    saved = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(tmp_path_factory.mktemp("cache"))
    yield
    if saved is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = saved
