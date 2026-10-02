"""Extreme metrics stay explanatory without deciding scores or calibration alone."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from styleprofile.core import Verdict
from styleprofile.profile import Chunk, build_reference, load_chunks, score, window
from styleprofile.weighting import (
    UNSEEN_Z,
    Z_CAP,
    ZRows,
    ZScores,
    cross_validate,
    delta,
    delta_weights,
    held_out_deltas,
    likeness,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("z", [-100.0, 100.0])
def test_scores_cap_each_metric_but_keep_explanatory_z(z: float) -> None:
    key = ("sentence_shape", "paragraph_words_mean")
    overall, areas = delta({key: z}, {key: 0.25})
    assert overall == Z_CAP
    assert areas == {key[0]: Z_CAP}
    assert delta({key: z}, {})[0] == UNSEEN_Z
    liked, signals = likeness({key: z}, {key: z}, {key: 2.0})
    assert liked == Z_CAP / 2
    assert signals[0]["z"] == z
    assert likeness({key: -z}, {key: z}, {}) == (0.0, [])


@pytest.mark.parametrize("compact", [False, True])
def test_held_out_calibration_uses_the_same_cap(compact: bool) -> None:
    keys = [("sentence_shape", "paragraph_words_mean"), ("punctuation", "commas_per_1k")]
    reference = [
        dict(zip(keys, values, strict=True))
        for values in [(100.0, -100.0), (1.0, -1.0), (2.0, -2.0)]
    ]
    contrast = [dict(zip(keys, values, strict=True)) for values in [(80.0, -80.0), (90.0, -90.0)]]
    rows: Sequence[ZScores] = reference
    if compact:
        stored = ZRows(keys)
        for row in reference:
            stored.append(row)
        rows = stored
    sources = ["a", "b", "c"]
    deltas = held_out_deltas(rows, sources)
    fitted = cross_validate(rows, sources, contrast, ["x", "y"])
    for index, row in enumerate(reference):
        others = [other for position, other in enumerate(reference) if position != index]
        rms = {key: (sum(other[key] ** 2 for other in others) / len(others)) ** 0.5 for key in keys}
        effects = {
            key: (
                sum(other[key] for other in contrast) / len(contrast)
                - sum(other[key] for other in others) / len(others)
            )
            / max(rms[key], 1.0)
            for key in keys
        }
        assert deltas.overall[index] == pytest.approx(delta(row, delta_weights(rms))[0])
        assert fitted.reference_scores[index] == pytest.approx(likeness(row, effects, rms)[0])
    assert deltas.overall[0] == Z_CAP


def test_flat_writer_essays_do_not_read_very_different() -> None:
    writer = load_chunks([str(ROOT / "examples" / "writer")])
    reference = build_reference(window(writer, 500), parser=None, settings={"window_words": 500})
    flat = [
        Chunk(chunk.id, chunk.source, "\n".join(line for line in chunk.text.splitlines() if line))
        for chunk in writer
    ]
    report = score(window(flat, 500), reference, parser=None)
    assert len(report["chunks"]) == 7
    assert all(document["verdict"] != Verdict.VERY_DIFFERENT for document in report["documents"])
    for chunk in report["chunks"]:
        scored = chunk["reference"]
        assert scored["z"]["sentence_shape"]["paragraph_words_mean"] > Z_CAP
        assert scored["delta_by_group"]["sentence_shape"] <= Z_CAP
