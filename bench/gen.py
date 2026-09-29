"""Generate the benchmark and demo corpora: seeded, synthetic, and built from the sample texts in
examples/.

    python bench/gen.py [--corpus NAME ...] [--out DIR]

Each corpus has a writer side (from ``examples/writer``) and a contrast side (from
``examples/llm-drafts``):

- ``medium``: 200 Markdown documents of about 1,000 words, plus 40 contrast documents;
- ``big``: 1,000 documents, plus 200 contrast documents;
- ``comments``: 20,000 JSONL records of about 50 words, plus 2,000 contrast records.
- ``demo``: 40 documents of about 700 words, plus 10 contrast documents, for ``make demo``: a
  reference the size a calibrated one needs (20,000 words or more), which the seven essays
  alone are not.

Documents mix whole paragraphs from the samples with paragraphs recombined from their
sentences, and vary in length, so windows differ from each other the way real texts do rather
than repeating a handful of paragraphs. Comments are runs of consecutive sentences or sentences
recombined from across the samples, and no two comments on one side share their words: the
samples hold only about 300 sentences, so runs repeat often, and ``build`` drops a document
that repeats another word for word (a comments corpus of 20,000 records with repeats measured
only about 11,000). The same seed always gives byte-identical output, and a corpus that is
already on disk with the same seed is not rewritten.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DEFAULT_OUT = Path(__file__).resolve().parent / "corpora"  # git-ignored
SEED = 0
# A sentence ends at . ! or ? (optionally closed by a quote or bracket) followed by a space.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"')\]])\s+")
# Bump when the output of a given seed changes, so stale corpora on disk are regenerated.
# 2: comments never repeat another comment's words.
GENERATOR_VERSION = 2
# Draws before a comment that keeps repeating earlier ones is accepted anyway.
UNIQUE_TRIES = 50
REMIX_SHARE = 0.5  # share of prose paragraphs recombined from sentences rather than copied


@dataclass(frozen=True)
class Corpus:
    name: str
    documents: int
    contrast: int
    words: int  # mean words per document or record
    jsonl: bool = False


CORPORA = {
    corpus.name: corpus
    for corpus in (
        Corpus("medium", documents=200, contrast=40, words=1_000),
        Corpus("big", documents=1_000, contrast=200, words=1_000),
        Corpus("comments", documents=20_000, contrast=2_000, words=50, jsonl=True),
        Corpus("demo", documents=40, contrast=10, words=700),
    )
}


@dataclass(frozen=True)
class Pool:
    """The material one side of a corpus is drawn from."""

    titles: list[str]
    prose: list[str]  # paragraphs of running text
    blocks: list[str]  # lists, subheadings and other non-prose blocks, kept verbatim
    sentences: list[str]


def _pool(folder: Path) -> Pool:
    titles: list[str] = []
    prose: list[str] = []
    blocks: list[str] = []
    for path in sorted(folder.glob("*.md")):  # sorted: glob order differs across filesystems
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("# "):
                titles.append(block)
            elif block.startswith(("#", "-", "*", ">", "1.")) or "\n" in block:
                blocks.append(block)
            elif len(block.split()) >= 8:
                prose.append(block)
    sentences = [s for p in prose for s in SENTENCE_END.split(p) if len(s.split()) >= 3]
    return Pool(titles, prose, blocks, sentences)


def _remixed(rng: random.Random, pool: Pool, low: int, high: int) -> str:
    """A paragraph of low to high sentences drawn from anywhere in the pool."""
    return " ".join(rng.choice(pool.sentences) for _ in range(rng.randint(low, high)))


def _document(rng: random.Random, pool: Pool, words: int) -> str:
    """A Markdown document of about ``words`` words (±30%) under one of the pool's titles."""
    target = round(words * rng.uniform(0.7, 1.3))
    parts = [rng.choice(pool.titles)] if pool.titles else []
    count = 0
    # Non-prose blocks turn up about as often as they do in the samples.
    block_share = len(pool.blocks) / (len(pool.blocks) + len(pool.prose))
    while count < target:
        if pool.blocks and rng.random() < block_share:
            part = rng.choice(pool.blocks)
        elif rng.random() < REMIX_SHARE:
            part = _remixed(rng, pool, 2, 7)
        else:
            part = rng.choice(pool.prose)
        parts.append(part)
        count += len(part.split())
    return "\n\n".join(parts) + "\n"


def _comment(rng: random.Random, pool: Pool, words: int) -> str:
    """A comment of about ``words`` words: a run of consecutive sentences, or a remix."""
    target = max(8, round(rng.gauss(words, words / 3)))
    if rng.random() < REMIX_SHARE:
        sentences = [rng.choice(pool.sentences)]
        while len(" ".join(sentences).split()) < target:
            sentences.append(rng.choice(pool.sentences))
    else:
        start = rng.randrange(len(pool.sentences))
        sentences = pool.sentences[start : start + 1]
        while len(" ".join(sentences).split()) < target:
            sentences.append(pool.sentences[(start + len(sentences)) % len(pool.sentences)])
    # Stop at whichever side of the target is closer, so the mean stays near ``words``.
    over = len(" ".join(sentences).split()) - target
    if len(sentences) > 1 and over > target - len(" ".join(sentences[:-1]).split()):
        sentences.pop()
    return " ".join(sentences)


def _words(text: str) -> str:
    """A comment's words, lowercased: what ``build`` compares to find repeated documents."""
    return " ".join(re.findall(r"\w+", text.lower()))


def _documents(rng: random.Random, pool: Pool, corpus: Corpus, count: int) -> Iterator[str]:
    seen: set[str] = set()
    for _ in range(count):
        if not corpus.jsonl:
            yield _document(rng, pool, corpus.words)
            continue
        text = _comment(rng, pool, corpus.words)
        for _ in range(UNIQUE_TRIES):
            if _words(text) not in seen:
                break
            text = _comment(rng, pool, corpus.words)
        seen.add(_words(text))
        yield text


def _write_side(folder: Path, texts: Iterator[str], jsonl: bool, author: str) -> int:
    folder.mkdir(parents=True)
    words = 0
    if jsonl:
        with (folder / "comments.jsonl").open("w", encoding="utf-8") as out:
            for index, text in enumerate(texts):
                words += len(text.split())
                record = {"id": f"{author}-{index:05d}", "author": author, "text": text}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        for index, text in enumerate(texts):
            words += len(text.split())
            (folder / f"doc-{index:04d}.md").write_text(text, encoding="utf-8")
    return words


def generate(name: str, out: Path = DEFAULT_OUT, seed: int = SEED) -> Path:
    """Write corpus ``name`` to ``out/name`` (writer/ and contrast/) and return that folder."""
    corpus = CORPORA[name]
    target = out / name
    stamp = target / "corpus.json"
    manifest = {"name": name, "seed": seed, "generator": GENERATOR_VERSION}
    if stamp.exists():
        saved = json.loads(stamp.read_text(encoding="utf-8"))
        if {key: saved.get(key) for key in manifest} == manifest:
            return target
    if target.exists():
        shutil.rmtree(target)
    rng = random.Random(f"{seed}:{name}")  # one stream per corpus, so each is reproducible alone
    writer, llm = _pool(EXAMPLES / "writer"), _pool(EXAMPLES / "llm-drafts")
    words = _write_side(
        target / "writer", _documents(rng, writer, corpus, corpus.documents), corpus.jsonl, "writer"
    )
    contrast_words = _write_side(
        target / "contrast", _documents(rng, llm, corpus, corpus.contrast), corpus.jsonl, "llm"
    )
    manifest |= {"words": words, "contrast_words": contrast_words}
    stamp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--corpus", action="append", choices=list(CORPORA), help="default: every corpus"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="default: bench/corpora")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    for name in args.corpus or list(CORPORA):
        folder = generate(name, args.out, args.seed)
        stats = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
        print(f"{folder}: {stats['words']:,} words, {stats['contrast_words']:,} contrast words")


if __name__ == "__main__":
    main()
