"""Measure "Where it drifts": false flags on the writer's own documents, and how often LLM
passages spliced into them are found. Adapted from the review harness of plan PR 12.

    python bench/drift.py A|B|C|D [--syntax] [--out RESULTS.json]

Two corpora, each in 5 folds; every fold builds a reference whose contrast set (40 documents
generated from 4 of the 5 LLM drafts) leaves out the draft its inserts come from:

- ``A``: bench medium (seed 0): the reference is its first 150 documents, 50 are held out;
- ``B``: a topic shift. The reference is remixed from 5 of the 7 essays; the held-out
  documents are remixed from the other 2 (``sharpening``, ``walking-in-rain``), so their
  topics and sentences are unseen, plus those 2 real essays.
- ``C``: a topic shift the contrast set covers but the reference does not: the held-out
  topics are ``fence-lines`` and ``the-woodstove``, which LLM drafts exist for, in
  documents of about 600 words.
- ``D``: a wider topic shift in longer documents: the reference is remixed from 4 essays,
  and the held-out documents (about 1,200 words) from the other 3 (``sharpening``,
  ``the-woodstove``, ``walking-in-rain``), plus those 3 real essays.

Each held-out document is scored as it is and with 1-2-sentence paragraphs (false flags); with
runs of 1, 2 or 3 consecutive blocks of the left-out draft (headings and lists included)
inserted at its start, middle or end; with two adjacent prose paragraphs of it at its
start, middle or end; and concatenated 2, 4, 6 and 10 at a time (up to about 160
paragraphs), with and without an insert. examples/draft.md is scored in every fold. The
tables print as Markdown.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import gen

import styleprofile as sp
from styleprofile.surface import classify, prose

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DRAFTS = sorted((EXAMPLES / "llm-drafts").glob("*.md"))
SCRATCH = Path(tempfile.mkdtemp())


def _pool_from(paths: list[Path]) -> gen.Pool:
    folder = Path(tempfile.mkdtemp(dir=SCRATCH))
    for path in paths:
        (folder / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return gen._pool(folder)


def _generate(pool: gen.Pool, count: int, words: int, seed: str) -> list[str]:
    rng = random.Random(seed)
    return [gen._document(rng, pool, words) for _ in range(count)]


def _folder(texts: list[str], prefix: str = "doc") -> Path:
    folder = Path(tempfile.mkdtemp(dir=SCRATCH))
    for index, text in enumerate(texts):
        (folder / f"{prefix}-{index:04d}.md").write_text(text, encoding="utf-8")
    return folder


def _runs(path: Path) -> dict[int, list[tuple[list[str], bool]]]:
    """Runs of 1-3 consecutive blocks of an LLM draft (title dropped) ending in prose, and
    whether each has a heading or list."""
    blocks = [block for block in classify(path.read_text()) if not block.raw.startswith("# ")]
    raws = [block.raw.strip() for block in blocks]
    ends = [len(prose(raw).text.split()) >= 5 and not raw.startswith("#") for raw in raws]
    formatted = [raw.startswith(("#", "-", "*", "1.", "2.")) for raw in raws]
    return {
        size: [
            (raws[start : start + size], any(formatted[start : start + size]))
            for start in range(len(raws) - size + 1)
            if ends[start + size - 1]
        ]
        for size in (1, 2, 3)
    }


def _splice(
    text: str, chunk: list[str], where: str, rng: random.Random
) -> tuple[str, tuple[int, int]]:
    blocks = text.strip().split("\n\n")
    if where == "start":
        position = 1 if blocks[0].startswith("# ") else 0
    elif where == "end":
        position = len(blocks)
    else:
        position = rng.randint(2, max(2, len(blocks) - 2))
    first = ("\n\n".join(blocks[:position]).count("\n") + 3) if position else 1
    last = first + "\n\n".join(chunk).count("\n")
    return "\n\n".join(blocks[:position] + chunk + blocks[position:]) + "\n", (first, last)


def _shorten(text: str) -> str:
    """Every prose paragraph split into paragraphs of 1-2 sentences."""
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(("#", "-", "*", ">", "1.")) or "\n" in block:
            out.append(block)
            continue
        sentences = [part for part in gen.SENTENCE_END.split(block) if part.strip()]
        rng = random.Random(len(block))
        index = 0
        while index < len(sentences):
            size = rng.choice((1, 2))
            out.append(" ".join(sentences[index : index + size]))
            index += size
    return "\n\n".join(out) + "\n"


def _concat(texts: list[str]) -> str:
    parts = [texts[0].strip()]
    for text in texts[1:]:
        blocks = text.strip().split("\n\n")
        parts.append("\n\n".join(blocks[1:] if blocks[0].startswith("# ") else blocks))
    return "\n\n".join(parts) + "\n"


def _score(profile: sp.Profile, text: str, name: str) -> list[list[Any]] | None:
    """[first line, last line, words, spans, drifts] per paragraph, or None if not read."""
    result = profile.score(sp.Text(text, name=name), passages=True)
    document = (result.report.get("passages") or [None])[0]
    if not document or not document["judged"]:
        return None
    return [
        [entry["lines"][0], entry["lines"][1], entry["words"], entry["spans"], entry["drifts"]]
        for entry in document["paragraphs"]
    ]


def collect(corpus: str, syntax: bool) -> list[dict[str, Any]]:
    if corpus == "A":
        documents = sorted((gen.generate("medium") / "writer").glob("*.md"))
        reference: Any = documents[:150]
        held = [(path.name, path.read_text()) for path in documents[150:]]
    else:
        essays = sorted((EXAMPLES / "writer").glob("*.md"))
        pick = {
            "B": ("sharpening", "walking-in-rain"),
            "C": ("fence-lines", "the-woodstove"),
            "D": ("sharpening", "the-woodstove", "walking-in-rain"),
        }[corpus]
        unseen = [path for path in essays if path.stem in pick]
        words, seed = {"B": (900, 7), "C": (600, 11), "D": (1200, 23)}[corpus]
        seen = [path for path in essays if path not in unseen]
        reference = _folder(_generate(_pool_from(seen), 150, words, f"{corpus}:ref:{seed}"))
        held = [
            (f"unseen-{index}", text)
            for index, text in enumerate(
                _generate(_pool_from(unseen), 30, words, f"{corpus}:held:{seed}")
            )
        ]
        held += [(path.name, path.read_text()) for path in unseen]
    results: list[dict[str, Any]] = []
    for fold, left_out in enumerate(DRAFTS):
        others = [path for path in DRAFTS if path != left_out]
        contrast = _folder(
            _generate(_pool_from(others), 40, 900, f"{corpus}:contrast:{left_out.stem}"), "llm"
        )
        profile = sp.build(reference, sp.Settings(syntax=syntax), contrast=contrast)
        runs = _runs(left_out)
        prose_runs = [chunk for chunk, formatted in runs[1] if not formatted]
        rng = random.Random(f"{corpus}:{fold}")
        for name, text in held[fold::5]:
            results.append({"kind": "clean", "doc": _score(profile, text, name)})
            results.append({"kind": "clean-short", "doc": _score(profile, _shorten(text), name)})
            for size in (1, 2, 3):
                for where in ("start", "middle", "end"):
                    chunk, formatted = rng.choice(runs[size])
                    spliced, lines = _splice(text, chunk, where, rng)
                    results.append(
                        {
                            "kind": "splice",
                            "k": size,
                            "where": where,
                            "formatted": formatted,
                            "insert_words": len(prose("\n\n".join(chunk)).text.split()),
                            "lines": lines,
                            "doc": _score(profile, spliced, name),
                        }
                    )
            pairs = [chunk for chunk, formatted in runs[2] if not formatted]
            for where in ("start", "middle", "end"):
                spliced, lines = _splice(text, rng.choice(pairs), where, rng)
                results.append(
                    {
                        "kind": "pair",
                        "where": where,
                        "lines": lines,
                        "doc": _score(profile, spliced, name),
                    }
                )
        for count in (2, 4, 6, 10):
            for repeat in range(6):
                pick = random.Random(f"{corpus}:{fold}:{count}:{repeat}").sample(held, count)
                text = _concat([item[1] for item in pick])
                results.append({"kind": f"long{count}", "doc": _score(profile, text, "long")})
                spliced, lines = _splice(text, rng.choice(prose_runs), "middle", rng)
                results.append(
                    {
                        "kind": f"long{count}-splice",
                        "k": 1,
                        "lines": lines,
                        "doc": _score(profile, spliced, "long"),
                    }
                )
        draft = _score(profile, (EXAMPLES / "draft.md").read_text(), "draft.md")
        results.append({"kind": "draft", "doc": draft})
        print(f"  fold {fold + 1} of {len(DRAFTS)}", file=sys.stderr, flush=True)
    return results


def _overlaps(paragraph: list[Any], lines: tuple[int, int] | list[int]) -> bool:
    return paragraph[0] <= lines[1] and paragraph[1] >= lines[0]


def report(results: list[dict[str, Any]], title: str) -> None:
    print(f"\n### {title}\n\n| Clean documents | with any drift | paragraphs drifting |")
    print("|---|---|---|")
    for kind, label in (
        ("clean", "held out"),
        ("clean-short", "same, 1-2-sentence paragraphs"),
        ("long2", "2 concatenated (~25 paragraphs)"),
        ("long4", "4 concatenated (~50)"),
        ("long6", "6 concatenated (~75)"),
        ("long10", "10 concatenated (~130)"),
    ):
        docs = [result["doc"] for result in results if result["kind"] == kind and result["doc"]]
        drifting = sum(any(p[4] for p in doc) for doc in docs)
        paragraphs = sum(len(doc) for doc in docs)
        print(
            f"| {label} | {drifting}/{len(docs)} ({drifting / max(len(docs), 1):.0%}) | "
            f"{sum(sum(p[4] for p in doc) for doc in docs)}/{paragraphs} |"
        )
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for result in results:
        if "splice" not in result["kind"] or not result["doc"]:
            continue
        paragraphs = result["doc"]
        inside = [p for p in paragraphs if _overlaps(p, result["lines"])]
        hit = any(p[4] for p in inside)
        alone = hit and not any(p[4] for p in paragraphs if not _overlaps(p, result["lines"]))
        keys = [result["kind"]]
        if result["kind"] == "splice":
            keys += [f"{result['k']} block(s)", f"at the {result['where']}"]
            keys.append("with a heading or list" if result["formatted"] else "prose only")
            words = result["insert_words"]
            keys.append("under 30 words" if words < 30 else "30-59 words" if words < 60 else "60+")
        for key in keys:
            groups[key][0] += 1
            groups[key][1] += hit
            groups[key][2] += alone
    print("\n| Inserts | n | found | and nothing else |\n|---|---|---|---|")
    for key, (count, hit, alone) in groups.items():
        print(f"| {key} | {count} | {hit / count:.0%} | {alone / count:.0%} |")
    print("\n| Adjacent LLM pair | n | both drift | one | neither |\n|---|---|---|---|---|")
    for where in ("start", "middle", "end"):
        counts = [0, 0, 0]
        pairs = [r for r in results if r["kind"] == "pair" and r["where"] == where and r["doc"]]
        for result in pairs:
            inside = [p for p in result["doc"] if _overlaps(p, result["lines"])]
            found = sum(bool(p[4]) for p in inside)
            counts[0 if found >= 2 else 1 if found else 2] += 1
        print(f"| at the {where} | {len(pairs)} | " + " | ".join(map(str, counts)) + " |")
    drafts = [result["doc"] for result in results if result["kind"] == "draft"]
    exact = sum(bool(doc) and sorted(p[0] for p in doc if p[4]) == [9, 15] for doc in drafts if doc)
    print(f"\ndraft.md, exactly lines 9 and 15: {exact} of {len(drafts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("corpus", choices=["A", "B", "C", "D"])
    parser.add_argument("--syntax", action="store_true", help="build and score with spaCy")
    parser.add_argument("--out", type=Path, help="also save the raw results as JSON")
    args = parser.parse_args()
    results = collect(args.corpus, args.syntax)
    if args.out:
        args.out.write_text(json.dumps(results))
    report(results, f"Corpus {args.corpus}, {'with' if args.syntax else 'without'} spaCy")


if __name__ == "__main__":
    main()
