"""Judge the bundled generic contrast drafts against the bars in docs/method.md.

    python bench/eval/generic_contrast.py dev      # examples/llm-drafts (may be iterated on)
    python bench/eval/generic_contrast.py writer   # the writer's held-out documents
    python bench/eval/generic_contrast.py test     # bench/eval/llm-heldout/ (run once)

``writer`` needs ``python bench/gen.py --corpus medium``. ``test`` is the one-shot test of
WP-10b: run it after the corpus and code are committed, never to choose between them.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import styleprofile as sp
from styleprofile.core import LikenessVerdict
from styleprofile.weighting import auc

ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "examples" / "writer"
DRAFTS = ROOT / "examples" / "llm-drafts"
HELDOUT = ROOT / "bench" / "eval" / "llm-heldout"
MEDIUM = ROOT / "bench" / "corpora" / "medium"
LEANS = (LikenessVerdict.LEANS, LikenessVerdict.LIKE_DRAFTS)


def _profile(folder: Path | list[Path]) -> sp.Profile:
    return sp.build(folder, generic_contrast=True)


def _rows(profile: sp.Profile, folder: Path) -> list[dict]:
    return [
        {
            "name": Path(d.name).name,
            "words": d.words,
            "likeness": d.likeness,
            "verdict": str(d.likeness_verdict),
            "leans": d.likeness_verdict in LEANS,
        }
        for d in profile.score(folder).documents
        if Path(d.name).name != "README.md"
    ]


def _writer_heldout() -> list[dict]:
    rows: list[dict] = []
    essays = sorted(WRITER.glob("*.md"))
    for held in essays:
        with tempfile.TemporaryDirectory() as tmp:
            for other in essays:
                if other != held:
                    (Path(tmp) / other.name).write_text(other.read_text(encoding="utf-8"))
            rows += [{**r, "set": "examples/writer"} for r in _rows(_profile(Path(tmp)), held)]
    documents = sorted(MEDIUM.joinpath("writer").glob("*.md"))
    held_out = documents[::5]
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as rest:
        for document in held_out:
            (Path(tmp) / document.name).write_text(document.read_text(encoding="utf-8"))
        for document in documents:
            if document not in held_out:
                (Path(rest) / document.name).write_text(document.read_text(encoding="utf-8"))
        rows += [{**r, "set": "medium"} for r in _rows(_profile(Path(rest)), Path(tmp))]
    return rows


def _symmetric(a: list[float], b: list[float]) -> float:
    value = auc(a, b) or 0.5
    return max(value, 1 - value)


def main(which: str) -> None:
    if which == "dev":
        rows = _rows(_profile(WRITER), DRAFTS)
        print(
            json.dumps(
                {
                    "set": "dev",
                    "leans_or_beyond": sum(r["leans"] for r in rows),
                    "of": len(rows),
                    "rows": rows,
                },
                indent=1,
            )
        )
    elif which == "writer":
        rows = _writer_heldout()
        like = sum(r["verdict"] == str(LikenessVerdict.LIKE_REFERENCE) for r in rows)
        print(
            json.dumps(
                {
                    "set": "writer",
                    "like_the_reference": like,
                    "of": len(rows),
                    "share": like / len(rows),
                    "rows": rows,
                },
                indent=1,
            )
        )
    elif which == "test":
        rows = _rows(_profile(WRITER), HELDOUT)
        writer = _writer_heldout()
        test_likeness = [r["likeness"] or 0.0 for r in rows]
        writer_likeness = [r["likeness"] or 0.0 for r in writer]
        print(
            json.dumps(
                {
                    "set": "test",
                    "leans_or_beyond": sum(r["leans"] for r in rows),
                    "of": len(rows),
                    "auc_likeness": auc(test_likeness, writer_likeness),
                    "auc_length_alone": _symmetric(
                        [r["words"] for r in rows], [r["words"] for r in writer]
                    ),
                    "rows": rows,
                },
                indent=1,
            )
        )
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
