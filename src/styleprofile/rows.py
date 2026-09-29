"""Chunks' metrics held compactly, for corpora of thousands of chunks.

A chunk's metrics are nested dicts (group, then metric name, to a value or None): about 8 KB
each, so 20,000 comments took 170 MB. ``MetricRows`` keeps each chunk as one array of doubles
over a layout the chunks share (1.2 KB), and reads each back as exactly the dicts it was made
from. Code that goes through every chunk reads what it needs directly (``column``,
``scored_values``) rather than rebuilding dicts.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Callable, Iterator, Sequence
from operator import itemgetter
from typing import Any, overload

from styleprofile.metrics import UNSCORED_GROUPS
from styleprofile.surface import Metrics

Key = tuple[str, str]
Layout = tuple[tuple[str, tuple[str, ...]], ...]
_TYPES = frozenset({float, type(None)})

# One key object per metric, shared by every chunk's values and z-scores: with thousands of
# chunks, a key tuple of their own each was over a hundred megabytes.
_KEYS: dict[Key, Key] = {}


def key(group: str, name: str) -> Key:
    """The shared ``(group, name)`` key of a metric."""
    found = (group, name)
    return _KEYS.setdefault(found, found)


def _layout(metrics: Metrics) -> Layout:
    return tuple((group, tuple(values)) for group, values in metrics.items())


class MetricRows(Sequence[Metrics]):
    """Chunks' metrics as arrays of doubles over one layout (NaN for None). A chunk laid out
    otherwise, or with a NaN value of its own, is kept as its dicts."""

    def __init__(self) -> None:
        self._layout: Layout | None = None
        self._rows: list[array[float] | Metrics] = []
        self._columns: dict[Key, int] = {}
        self._scored_keys: list[Key] = []
        self._scored: Callable[[array[float]], Any] | None = None
        self.uniform = True  # every row follows the layout

    def append(self, metrics: Metrics) -> None:
        if self._layout is None:
            self._layout = _layout(metrics)
            flat = [(group, name) for group, names in self._layout for name in names]
            self._columns = {pair: position for position, pair in enumerate(flat)}
            scored = [
                (position, key(group, name))
                for position, (group, name) in enumerate(flat)
                if group not in UNSCORED_GROUPS
            ]
            self._scored_keys = [pair for _, pair in scored]
            self._scored = itemgetter(*(position for position, _ in scored)) if scored else None
        values = [value for group in metrics.values() for value in group.values()]
        row = (
            array("d", [math.nan if value is None else value for value in values])
            # Only floats and None go in: an int would come back as a float...
            if (_TYPES.issuperset(map(type, values)) and _layout(metrics) == self._layout)
            else None
        )
        # ... and a NaN of its own as None.
        if row is None or sum(map(math.isnan, row)) != values.count(None):
            self.uniform = False
            self._rows.append(metrics)
            return
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Metrics]:
        return map(self.__getitem__, range(len(self._rows)))

    @overload
    def __getitem__(self, index: int) -> Metrics: ...
    @overload
    def __getitem__(self, index: slice) -> list[Metrics]: ...
    def __getitem__(self, index: int | slice) -> Metrics | list[Metrics]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        row = self._rows[index]
        if not isinstance(row, array):
            return row
        assert self._layout is not None
        metrics: Metrics = {}
        start = 0
        for group, names in self._layout:
            end = start + len(names)
            metrics[group] = {
                name: None if value != value else value
                for name, value in zip(names, row[start:end], strict=True)
            }
            start = end
        return metrics

    @property
    def layout(self) -> Layout:
        return self._layout or ()

    def column(self, group: str, name: str) -> list[float | None]:
        """One metric's value in every chunk, in order (None where a chunk has none)."""
        position = self._columns.get((group, name))
        if self.uniform and position is not None:
            return [
                None if value != value else value for value in map(itemgetter(position), self._rows)
            ]  # type: ignore[arg-type]
        values: list[float | None] = []
        for row in self._rows:
            if isinstance(row, array):
                value = row[position] if position is not None else math.nan
                values.append(None if value != value else value)
            else:
                values.append(row.get(group, {}).get(name))
        return values

    def present(self, group: str, name: str) -> list[float]:
        """``column`` without the chunks that have no value."""
        position = self._columns.get((group, name))
        if self.uniform and position is not None:
            return [value for value in map(itemgetter(position), self._rows) if value == value]
        return [value for value in self.column(group, name) if value is not None]

    def scored_values(self, index: int) -> dict[Key, float] | None:
        """A chunk's scored metrics that have a value, in order, keyed as ``key`` makes
        them; None for a chunk kept as its dicts."""
        row = self._rows[index]
        if not isinstance(row, array) or self._scored is None:
            return None
        picked = self._scored(row)
        if len(self._scored_keys) == 1:
            picked = (picked,)
        return {
            found: value
            for found, value in zip(self._scored_keys, picked, strict=True)
            if value == value
        }
