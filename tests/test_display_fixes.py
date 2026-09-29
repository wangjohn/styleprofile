"""Display and output fixes: the "By area" block, color-only wording, layout, file modes."""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from styleprofile.calibration import verdict
from styleprofile.display import (
    BAR_SCALE,
    BAR_WIDTH,
    LABEL_WIDTH,
    format_evaluation,
    format_summary,
)
from styleprofile.metrics import METRICS
from styleprofile.profile import write_report

# The demo's areas: sentence shape's raw Delta is higher, but it is well inside its own wide
# held-out range, while voice is just past a narrow one.
AREAS = {
    "sentence_shape": (1.45, {"median": 0.73, "p95": 2.24}),
    "voice": (1.34, {"median": 0.80, "p95": 0.95}),
    "punctuation": (0.68, {"median": 0.46, "p95": 0.88}),
    "markdown": (0.07, {"median": 0.006, "p95": 0.024}),
}


def _comparison(calibrated: bool = True, chunks: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    """A scored report and its reference, with the areas above and one metric per area."""
    reference: dict[str, Any] = {
        "chunk_count": 7,
        "summary": {
            "voice": {
                "llm_markers_per_1k": {"mean": 0.0, "sd": 0.0},
                "hedges_per_1k": {"mean": 2.1, "sd": 1.0},
            }
        },
    }
    if calibrated:
        reference["calibration"] = {
            "sources": 7,
            "delta": {
                "median": 0.78,
                "p95": 0.99,
                "by_group": {group: stats for group, (_, stats) in AREAS.items()},
            },
        }
    report = {
        "chunk_count": chunks,
        "word_count": 400 * chunks,
        "settings": {"syntax": None},
        "summary": {"voice": {"llm_markers_per_1k": {"mean": 5.0}, "hedges_per_1k": {"mean": 2.0}}},
        "chunks": [
            {
                "id": f"c{index}",
                "metrics": {"size": {"words": 400.0}},
                "reference": {
                    "delta": 0.8,
                    "delta_by_group": {group: amount for group, (amount, _) in AREAS.items()},
                    "z": {"voice": {"llm_markers_per_1k": 3.0, "hedges_per_1k": -0.1}},
                    # As a chunk of the reference's window length is judged.
                    "calibration": {
                        "judged": True,
                        "reason": None,
                        "delta": {"median": 0.78, "p95": 0.99} if calibrated else None,
                        "delta_by_group": (
                            {group: stats for group, (_, stats) in AREAS.items()}
                            if calibrated
                            else {}
                        ),
                        "likeness": None,
                        "length_scale": {},
                    },
                },
            }
            for index in range(chunks)
        ],
        "reference": {
            "path": "ref.json",
            "delta_mean": 0.8,
            "delta_by_group_mean": {group: amount for group, (amount, _) in AREAS.items()},
            "pooled_divergence": {},
        },
        "warnings": [],
    }
    _judge(report)
    return report, reference


def _judge(report: dict[str, Any], areas: dict[str, float] | None = None) -> None:
    """Set every chunk's area Deltas (when given) and the verdict read from the chunks."""
    for chunk in report["chunks"]:
        if areas is not None:
            chunk["reference"]["delta_by_group"] = areas
            chunk["reference"]["calibration"]["delta_by_group"] = {
                group: stats
                for group, stats in chunk["reference"]["calibration"]["delta_by_group"].items()
                if group in areas
            }
    report["reference"]["verdict"] = verdict(report["chunks"], None)


def _area_block(text: str) -> list[str]:
    """The rows of "By area", without its one or two header lines."""
    block = text.split("By area", 1)[1].split("\n\n", 1)[0]
    return [line for line in block.splitlines()[1:] if not line.startswith("  close up to")]


def test_areas_sort_by_verdict_and_show_distance_relative_to_their_range() -> None:
    rows = _area_block(format_summary(*_comparison()))
    titles = [row.split("  ")[1].strip() for row in rows]
    assert titles == ["Voice and stance", "Punctuation", "Sentence shape", "Formatting"]
    # Voice is past its usual range; sentence shape is inside its own, so it reads lower.
    assert "1.41x" in rows[0] and rows[0].endswith("somewhat different")
    assert "0.65x" in rows[2] and rows[2].endswith("close")
    # Formatting's near-zero range is floored, so a tiny Delta stays a small multiple.
    assert "0.14x" in rows[3]
    numbers = [float(re.search(r"([\d.]+)x", row).group(1)) for row in rows]  # type: ignore[union-attr]
    assert numbers == sorted(numbers, reverse=True)
    for row, number in zip(rows, numbers, strict=True):
        assert row.count("█") == round(number / BAR_SCALE * BAR_WIDTH)


def test_areas_fall_back_to_raw_delta_without_calibration() -> None:
    text = format_summary(*_comparison(calibrated=False))
    assert "Delta in each area" in text
    rows = _area_block(text)
    assert "Delta 1.45" in rows[0] and "Delta 0.07" in rows[-1]
    assert "close up to 1x" not in text
    # The raw table would only repeat these rows with dashes.
    assert "Delta by area" not in format_summary(*_comparison(calibrated=False), full=True)


def test_area_verdict_follows_the_rounded_multiple() -> None:
    report, reference = _comparison()
    # 0.953 / 0.95 is 1.003: shown as 1.00x, so it must read close.
    _judge(report, {"voice": 0.953})
    (row,) = _area_block(format_summary(report, reference))
    assert "1.00x" in row and row.endswith("close")
    _judge(report, {"voice": 0.96})
    (row,) = _area_block(format_summary(report, reference))
    assert "1.01x" in row and row.endswith("somewhat different")


def test_by_area_header_says_what_the_multiple_is() -> None:
    text = format_summary(*_comparison())
    assert "Delta ÷ the top of the reference's usual range" in text
    assert (
        "close up to 1x, somewhat different to 1.5x, clearly different to 2x, "
        "very different above" in text
    )


def test_all_keeps_the_raw_area_deltas() -> None:
    report, reference = _comparison()
    assert "Delta by area" not in format_summary(report, reference)
    full = format_summary(report, reference, full=True)
    raw = full.split("Delta by area", 1)[1].split("\n\n", 1)[0]
    assert re.search(r"Sentence shape\s+1\.45\s+0\.73\s+2\.24", raw)
    assert re.search(r"Voice and stance\s+1\.34\s+0\.80\s+0\.95", raw)
    assert re.search(r"this text\s+median\s+range top", raw)


def test_shading_is_only_described_when_colored() -> None:
    report, reference = _comparison()
    assert "orange" not in format_summary(report, reference, color=False)
    assert "Darker orange is further away." in format_summary(report, reference, color=True)


def test_rows_have_no_trailing_spaces_and_labels_fit() -> None:
    assert max(len(metric.label) for metric in METRICS) < LABEL_WIDTH
    report, reference = _comparison()
    full = format_summary(report, reference, full=True)
    assert all(line == line.rstrip() for line in full.splitlines())


def test_single_chunk_note_skips_the_chunk_count() -> None:
    one = format_summary(*_comparison())
    assert "reference never varies" in one and "1 of 1 chunks" not in one
    assert "2 of 2 chunks differ" in format_summary(*_comparison(chunks=2))


def test_openers_name_themselves_outside_their_heading() -> None:
    labels = {metric.name: metric.label for metric in METRICS}
    assert labels["opens_verb_pct"].startswith("Opener: ")
    assert labels["opens_pronoun_pct"].startswith("Opener: ")


def test_evaluation_rows_show_whole_labels() -> None:
    longest = max((metric for metric in METRICS), key=lambda metric: len(metric.label))
    result = {
        "label": "LLM",
        "contrast": {"drafts": 2},
        "reference": {"documents": 3},
        "sets": {"original": {"likeness_median_chunks": 1.0, "flagged": 1, "drafts": 2}},
        "signals": [
            {
                "metric": f"{longest.group}.{longest.name}",
                "reference_z": 0.0,
                "original_z": 2.0,
                "edited": {},
            }
        ],
        "warnings": [],
    }
    assert longest.label in format_evaluation(result)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
@pytest.mark.parametrize(("umask", "expected"), [(0o022, 0o644), (0o027, 0o640), (0o077, 0o600)])
def test_saved_reports_get_normal_permissions(tmp_path: Path, umask: int, expected: int) -> None:
    previous = os.umask(umask)
    try:
        write_report({"a": 1.0}, tmp_path / "out" / "report.json")
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "out" / "report.json").stat().st_mode) == expected
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["report.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_overwriting_a_report_keeps_its_mode(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}")
    path.chmod(0o604)
    write_report({"a": 1.0}, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o604
    assert '"a": 1.0' in path.read_text()


def _spy_on_open_mode(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record each file's mode at the moment ``write_report`` opens it for writing."""
    modes: list[int] = []
    real = os.fdopen

    def spy(descriptor: int, *args: Any, **kwargs: Any) -> Any:
        modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        return real(descriptor, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", spy)
    return modes


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_mode_is_final_before_any_bytes_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes = _spy_on_open_mode(monkeypatch)
    private = tmp_path / "private.json"
    private.write_text("{}")
    private.chmod(0o600)
    previous = os.umask(0o022)
    try:
        write_report({"a": 1.0}, private)
        write_report({"a": 1.0}, tmp_path / "new.json")
    finally:
        os.umask(previous)
    assert modes == [0o600, 0o644]
    assert stat.S_IMODE(private.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_special_mode_bits_are_not_copied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modes = _spy_on_open_mode(monkeypatch)
    path = tmp_path / "report.json"
    path.write_text("{}")
    path.chmod(0o4644)
    if not stat.S_IMODE(path.stat().st_mode) & stat.S_ISUID:
        pytest.skip("this filesystem does not keep setuid on a user's file")
    write_report({"a": 1.0}, path)
    assert modes == [0o644]
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_a_failed_write_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.json"
    path.write_text("old")

    def fail(descriptor: int, *args: Any, **kwargs: Any) -> Any:
        os.close(descriptor)
        raise OSError("disk full")

    monkeypatch.setattr(os, "fdopen", fail)
    with pytest.raises(OSError, match="disk full"):
        write_report({"a": 1.0}, path)
    assert [item.name for item in tmp_path.iterdir()] == ["report.json"]
    assert path.read_text() == "old"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and file modes")
def test_a_symlinked_report_is_replaced_by_a_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("old")
    target.chmod(0o640)
    link = tmp_path / "report.json"
    link.symlink_to(target)
    write_report({"a": 1.0}, link)
    assert not link.is_symlink()
    assert stat.S_IMODE(link.stat().st_mode) == 0o640
    assert target.read_text() == "old"
