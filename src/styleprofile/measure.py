"""Measuring chunks: every metric of each chunk and of the shorter pieces cut from it for length
calibration, each chunk read and parsed once.

``measure`` is what ``profile`` calls for every set of chunks it profiles or scores. How it
measures is set by a ``Measurer``, the run's options: the measurement cache (``cache``), worker
processes for the spaCy parser, and a progress callback. None of them changes a number, only
how fast the numbers arrive, so they are options of a run rather than settings a report
records.

The parser is by far the slowest step (about 17,000 words a second on one core, against
about 200,000 for everything else), so on a large corpus it runs in worker processes, each
with its own copy of the model. Workers parse and return each chunk's syntax metrics, never the
parsed documents, and the main process measures everything else meanwhile.
"""

from __future__ import annotations

import ast
import os
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import repeat
from operator import add
from pathlib import Path
from typing import TYPE_CHECKING, Any

from styleprofile import cache as caching
from styleprofile.cache import MeasurementCache
from styleprofile.calibration import CALIBRATION_WORDS, plan_pieces
from styleprofile.core import Phase, Progress, StyleProfileError
from styleprofile.rows import MetricRows
from styleprofile.surface import (
    WORD,
    Metrics,
    Prose,
    char_trigrams,
    masked_bigrams,
    prose,
    surface_metrics,
    words,
)
from styleprofile.syntax import Parser, load_parser, pos_trigrams, syntax_metrics

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from styleprofile.profile import Chunk

# Automatic worker processes: one per CPU, at most AUTO_JOBS, and only when there are at least
# PARALLEL_WORDS words to parse. A worker takes 1-2 seconds to start (it loads its own spaCy
# model) and peaks near 300 MB. Measured on the medium benchmark corpus's documents (an M5
# Pro): 26,000 words build in 2.9 s either way, 53,000 in 4.8 s with one process and 3.2 s
# with four, 210,000 in 16 s and 7 s.
AUTO_JOBS = 4
PARALLEL_WORDS = 50_000
# ... and no more than fit in a quarter of the machine's memory at about 300 MB each (a
# worker's peak parsing 500-word windows): a 4 GB machine gets 3, 2 GB gets 1 (no workers).
WORKER_MEMORY_SHARE = 0.25
WORKER_BYTES = 300 * 1024 * 1024
# Texts per task sent to a worker: large enough that the model's batches stay full, small
# enough that the workers finish together.
TASK_TEXTS = 32
# Documents per task when documents are read in parts (a few windows each).
TASK_PARTS = 4

ProgressCallback = Callable[[Progress], None]
# A parsed chunk: its syntax metrics, its POS trigram counts, each requested piece's syntax
# metrics (None when the piece's prose could not be found in the chunk's), and its parse when
# asked to keep it (only ever in this process; None otherwise).
Parsed = tuple[Metrics, dict[str, int], list[Metrics | None], Any]


def memory_jobs() -> int | None:
    """How many parser workers fit in ``WORKER_MEMORY_SHARE`` of the machine's physical
    memory at ``WORKER_BYTES`` each (at least 1), or None when it cannot be read."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if total <= 0:
        return None
    return max(1, int(total * WORKER_MEMORY_SHARE // WORKER_BYTES))


def cpus() -> int:
    """The CPUs this process may use."""
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return count or 1


def check_jobs(jobs: object) -> int:
    """``jobs`` if it is 0 (automatic) or a positive number of processes; else an error."""
    if type(jobs) is not int or jobs < 0:
        raise StyleProfileError(
            f"jobs must be 0 (automatic) or a number of processes, not {jobs!r}",
            code="invalid_setting",
            setting="jobs",
        )
    return jobs


def workers_can_start() -> bool:
    """Whether worker processes can start safely. They are spawned: each imports the main
    script again, which is harmless only when the call that got here runs under an
    ``if __name__ == "__main__":`` guard. An interactive session or notebook has no script, and
    ``python -m package`` is never imported again. Otherwise the line of the main script that
    is running must sit inside such a guard, which is read from the script's syntax tree;
    when that cannot be told, no workers start (parsing then runs in this process)."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if main is None or path is None:
        return True
    spec = getattr(main, "__spec__", None)
    if spec is not None and spec.name.endswith(".__main__"):
        return True
    line = _main_line(vars(main))
    return line is not None and _guarded(Path(path), line)


def _main_line(main_globals: dict[str, Any]) -> int | None:
    """The line the main script's own code is running, from its module-level frame."""
    frame = sys._getframe(1)
    found: int | None = None
    while frame is not None:
        if frame.f_globals is main_globals and frame.f_code.co_name == "<module>":
            found = frame.f_lineno
        frame = frame.f_back
    return found


def _is_main_test(test: ast.expr) -> bool:
    """``__name__ == "__main__"``, either way round."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, test.comparators[0]]
    names = [side for side in sides if isinstance(side, ast.Name) and side.id == "__name__"]
    constants = [
        side for side in sides if isinstance(side, ast.Constant) and side.value == "__main__"
    ]
    return len(names) == 1 and len(constants) == 1


def _guarded(path: Path, line: int) -> bool:
    """Whether ``line`` of the script at ``path`` is inside an ``if __name__ == "__main__":``
    block (its body, not its ``else``); False when the script cannot be read."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_main_test(node.test) and node.body:
            first, last = node.body[0].lineno, node.body[-1].end_lineno or node.body[-1].lineno
            if first <= line <= last:
                return True
    return False


def start_pool(count: int, model: str) -> Executor:
    """``count`` spawned worker processes, each loading spaCy ``model`` as it starts."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(
        count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(model,),
    )


# The parser of a worker process, loaded once when it starts.
_worker_parser: Parser | None = None


def _start_worker(model: str) -> None:
    global _worker_parser
    _worker_parser = load_parser(model)


def _parse_in_worker(items: Sequence[tuple[str, Sequence[tuple[int, int]]]]) -> list[Parsed]:
    assert _worker_parser is not None
    return list(parse(_worker_parser, items))


def _parts_in_worker(items: Sequence[_PartsItem]) -> list[list[Metrics]]:
    assert _worker_parser is not None
    return [_read_parts(_worker_parser, item) for item in items]


def piece_offsets(text: str, pieces: Sequence[tuple[int, str]]) -> list[int | None]:
    """Where each piece's prose (``(length, prose)``, in plan order) starts in its chunk's
    prose ``text``: searching from where the previous piece of the same length ended, and
    from the start when that fails; None when it is not there (the piece gets no syntax)."""
    cursors: dict[int, int] = defaultdict(int)
    offsets: list[int | None] = []
    for length, part in pieces:
        offset = text.find(part, cursors[length])
        if offset < 0:
            offset = text.find(part)
        if offset < 0:
            offsets.append(None)
            continue
        cursors[length] = offset + len(part)
        offsets.append(offset)
    return offsets


def parse(
    parser: Parser,
    items: Sequence[tuple[str, Sequence[tuple[int, int]]]],
    keep: Collection[int] = (),
) -> Iterator[Parsed]:
    """Parse each chunk's prose once, and measure its syntax and that of its pieces' spans
    (``(start, end)`` in its prose, from ``piece_offsets``) from the one parse, keeping the
    parse of the items at the positions in ``keep``."""
    docs = parser.docs(text for text, _ in items)
    for position, ((_, spans), doc) in enumerate(zip(items, docs, strict=True)):
        pieces: list[Metrics | None] = []
        for begin, end in spans:
            span = doc.char_span(begin, end, alignment_mode="expand")
            pieces.append(None if span is None else syntax_metrics(span))
        kept = doc if position in keep else None
        yield syntax_metrics(doc), dict(pos_trigrams(doc)), pieces, kept


# Documents read in parts (``drift``).


class JoinedParse:
    """One document's windows as one parse (``Doc.from_docs``), so spans that cross a window
    edge are measured without parsing the document again.

    The document's prose is its windows' prose joined by blank lines, which ``from_docs``
    joins by one space instead, so an offset in the prose maps to the joined parse by
    dropping one character per window edge before it."""

    def __init__(self, text: str, texts: Sequence[str], docs: Sequence[Any]) -> None:
        self.text = text
        self.starts: list[int] = []
        offset = 0
        for part in texts:
            self.starts.append(offset)
            offset += len(part) + 2
        self.doc = docs[0] if len(docs) == 1 else type(docs[0]).from_docs(list(docs))

    def span(self, start: int, end: int) -> Any | None:
        def mapped(offset: int) -> int:
            return offset - (bisect_right(self.starts, offset) - 1)

        return self.doc.char_span(mapped(start), mapped(end), alignment_mode="expand")


def prose_text(markdown: str) -> str:
    """The prose of ``markdown``, as its measured text."""
    return prose(markdown).text


def parsed_document(
    text: str, windows: Sequence[tuple[str, Any | None]], parser: Parser
) -> JoinedParse:
    """The document's prose (``text``) as parsed: its windows' parses (prose text, parse or
    None to parse it here) joined when they make up the whole document in order, else a
    fresh parse (a window was dropped for ``min_words``, say)."""
    texts = [part for part, _ in windows]
    if windows and "\n\n".join(texts) == text:
        missing = [part for part, doc in windows if doc is None]
        parsed = iter(parser.docs(missing))
        docs = [doc if doc is not None else next(parsed) for _, doc in windows]
        return JoinedParse(text, texts, docs)
    return JoinedParse(text, [text], list(parser.docs([text])))


def span_markdown(found: Sequence[Any], layout: Any) -> list[str]:
    """The Markdown of each of a document's spans: its paragraphs ``found``
    (``drift.paragraphs``) as ``layout`` (``drift.plan``) groups them."""
    return ["\n\n".join(paragraph.raw for paragraph in found[a:b]) for a, b in layout.spans]


def measure_spans(parts: Sequence[str], whole: JoinedParse | None) -> list[Metrics]:
    """Each part's metrics, from its Markdown and, with the document's parse ``whole``, its
    span of it: found by searching the document's prose from where the previous part
    started."""
    measured: list[Metrics] = []
    cursor = 0
    for markdown in parts:
        parsed = prose(markdown)
        metrics = surface_metrics(markdown, parsed)
        if whole is not None:
            offset = whole.text.find(parsed.text, cursor)
            if offset >= 0:
                cursor = offset
                span = whole.span(offset, offset + len(parsed.text))
                if span is not None:
                    metrics = _merge(metrics, syntax_metrics(span))
        measured.append(metrics)
    return measured


# What a document read in parts needs parsed: its prose, its windows' prose, and its spans'
# Markdown.
_PartsItem = tuple[str, list[str], list[str]]


def _read_parts(parser: Parser, item: _PartsItem) -> list[Metrics]:
    text, windows, parts = item
    return measure_spans(parts, parsed_document(text, [(part, None) for part in windows], parser))


class Measurer:
    """How a run measures: the measurement cache, worker processes for the parser, and a
    progress callback. These decide how fast the numbers arrive, never the numbers.

    ``jobs`` is the number of processes that parse: 1 parses in this process, 0 picks
    automatically (one per CPU up to ``AUTO_JOBS``, and as many as ``memory_jobs`` allows,
    once there are ``PARALLEL_WORDS`` words to parse). Workers start only when
    ``workers_can_start``, whatever ``jobs`` says. Use it as a context manager, or ``close``
    it: that stops the workers and writes the cache.
    """

    def __init__(
        self,
        *,
        cache: MeasurementCache | None = None,
        jobs: int = 1,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.cache = cache
        self.jobs = check_jobs(jobs)
        self.progress = progress
        self._pool: Executor | None = None
        self._pool_failed = False
        self._pool_size = 1

    def report(self, progress: Progress) -> None:
        if self.progress is not None:
            self.progress(progress)

    def workers(self, words: int) -> int:
        """How many processes parse ``words`` words; 1 means this one."""
        if self.jobs == 1 or self._pool_failed:
            return 1
        if self._pool is not None:  # started for an earlier set of chunks: keep using it
            return self._pool_size
        if not workers_can_start():
            return 1
        if self.jobs > 1:
            return self.jobs
        if words < PARALLEL_WORDS:
            return 1
        return min(AUTO_JOBS, cpus(), memory_jobs() or AUTO_JOBS)

    def parse(
        self,
        parser: Parser,
        items: list[tuple[str, list[tuple[int, int]]]],
        keep: Collection[int] = (),
        *,
        here: bool = False,
    ) -> Iterator[Parsed]:
        """``parse(parser, items, keep)``, in worker processes when ``workers`` says so, and
        ``here`` does not require this process (parses are only kept when parsed here).
        Workers that fail to start (a script without a main guard, say) leave the rest to
        this process."""
        size = sum(len(text.split()) for text, _ in items)
        count = 1 if here else self.workers(size)
        done = 0
        if count > 1:
            # Imported here: most runs start no workers, and these modules cost a scoring
            # run about 7 MB.
            from concurrent.futures.process import BrokenProcessPool

            try:
                if self._pool is None:
                    self._pool_size = count
                    self._pool = start_pool(count, parser.model)
                tasks = [items[i : i + TASK_TEXTS] for i in range(0, len(items), TASK_TEXTS)]
                for results in self._pool.map(_parse_in_worker, tasks):
                    for result in results:
                        yield result
                        done += 1
                return
            except (BrokenProcessPool, OSError):
                self._pool_failed = True
                self._stop_pool()
        yield from parse(parser, items[done:], {position - done for position in keep})

    def read_parts(self, parser: Parser, items: list[_PartsItem]) -> Iterator[list[Metrics]]:
        """``_read_parts`` of each item, in worker processes when ``workers`` says so (for
        the words of their windows), as ``parse`` runs."""
        size = sum(len(text.split()) for text, _, _ in items)
        count = self.workers(size)
        done = 0
        if count > 1:
            from concurrent.futures.process import BrokenProcessPool

            try:
                if self._pool is None:
                    self._pool_size = count
                    self._pool = start_pool(count, parser.model)
                tasks = [items[i : i + TASK_PARTS] for i in range(0, len(items), TASK_PARTS)]
                for results in self._pool.map(_parts_in_worker, tasks):
                    for result in results:
                        yield result
                        done += 1
                return
            except (BrokenProcessPool, OSError):
                self._pool_failed = True
                self._stop_pool()
        for item in items[done:]:
            yield _read_parts(parser, item)

    def _stop_pool(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def close(self) -> None:
        self._stop_pool()
        if self.cache is not None:
            self.cache.close()

    def __enter__(self) -> Measurer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class Piece:
    """A shorter piece cut from a measured chunk for length calibration (``calibration``)."""

    chunk: int  # index of the chunk it was cut from, among the measured chunks
    length: int  # the length it was cut for, in prose words
    metrics: Metrics


@dataclass(frozen=True)
class Measured:
    """Chunks with enough prose, their metrics, how many were skipped, and the calibration
    pieces cut from them. ``totals`` sums the chunks' pattern counts; ``distributions`` has
    each chunk's own only when they were asked for (``keep_distributions``), since keeping
    them for a large corpus takes more memory than everything else together. Without them,
    ``metrics`` is a ``MetricRows``."""

    chunks: list[Chunk]
    metrics: Sequence[Metrics]
    distributions: list[dict[str, Counter[str]]]
    totals: dict[str, Counter[str]]
    empty: int
    below: int
    pieces: list[Piece]
    # Each chunk's prose text and parse, where asked for (``keep_docs``) and parsed in this
    # process; None without a parser or when none were asked for.
    docs: list[tuple[str, Any] | None] | None = None


@dataclass
class _Kept:
    """A chunk with enough prose, as the first pass over the chunks found it."""

    chunk: Chunk
    size: int
    key: bytes | None
    cached: bytes | None  # its cache entry, still encoded
    parsed: Prose | None = None  # its prose, when it is measured rather than read from the cache
    pieces: list[int] = field(default_factory=list)  # its pieces, by position in the plan


@dataclass
class _PlannedPiece:
    chunk: int
    length: int
    markdown: str
    key: bytes | None
    metrics: Metrics | None = None  # from the cache, or its surface metrics once measured
    parsed: Prose | None = None  # set when it is measured rather than read from the cache
    offset: int | None = None  # where its prose starts in its chunk's (with the parser)


def _merge(target: Metrics, extra: Metrics) -> Metrics:
    return {**target, **extra}


def _add(total: Counter[str], counts: Mapping[str, int]) -> None:
    """``total.update(counts)``, in C rather than a Python loop: the same sums, and new keys
    added at the end in ``counts``' order, as ``Counter.update`` adds them."""
    dict.update(
        total,
        zip(counts, map(add, map(total.get, counts, repeat(0)), counts.values()), strict=True),
    )


def _distributions(text: str, tokens: list[str] | None = None) -> dict[str, Counter[str]]:
    return {"masked_bigram": masked_bigrams(text, tokens), "char_trigram": char_trigrams(text)}


def measure(
    chunks: Sequence[Chunk],
    parser: Parser | None,
    min_words: int,
    *,
    allow_empty: bool = False,
    piece_lengths: Sequence[int] = (),
    piece_words: int = CALIBRATION_WORDS,
    keep_distributions: bool = True,
    measurer: Measurer | None = None,
    phase: Phase = Phase.MEASURE,
    documents: Sequence[str] | None = None,
    keep_docs: bool | Collection[str] = False,
    docs_required: bool = False,
) -> Measured:
    """Measure each chunk once, dropping chunks without enough prose.

    With ``piece_lengths``, also cut up to ``piece_words`` words of the chunks into pieces of
    those lengths and measure them (``calibration.plan_pieces``). A piece's syntax metrics come
    from its span of the chunk's parse, so nothing is parsed twice. Chunks and pieces measured
    before under the same fingerprint come from ``measurer.cache``. ``documents`` names each
    chunk's document, for the pieces of pooled windows. With no chunk left this is an error,
    unless ``allow_empty``.

    With a parser and ``keep_docs`` (True, or the documents to keep, by ``documents``), the
    parses of those chunks are kept (``Measured.docs``) when they are parsed in this process;
    with ``docs_required`` they always are, cache or no cache.
    """
    measurer = measurer or Measurer()
    store = measurer.cache if measurer.cache is not None and measurer.cache.enabled else None
    print_ = caching.run_fingerprint(parser.used() if parser else None) if store else b""

    # First pass: every chunk's size, from the cache or from its prose.
    empty = below = 0
    kept: list[_Kept] = []
    kept_documents: list[str] = []
    for position, chunk in enumerate(chunks):
        key = caching.key(print_, chunk.text) if store else None
        found = store.fetch(key) if store and key else None
        parsed = None
        if found is not None:
            size = found[0]
        else:
            parsed = prose(chunk.text)
            size = len(WORD.findall(parsed.text))  # as many as ``words`` gives
        if not size:
            empty += 1
        elif size < min_words:
            below += 1
        else:
            kept.append(_Kept(chunk, size, key, found[1] if found else None, parsed))
            kept_documents.append(documents[position] if documents is not None else str(position))
    if not kept:
        if allow_empty:
            return Measured([], [], [], {}, empty, below, [])
        raise StyleProfileError(
            f"no chunks with at least {max(min_words, 1)} prose word(s) to profile "
            f"({empty} had no prose, {below} were shorter)",
            code="no_chunks",
        )

    # The calibration pieces.
    planned: list[_PlannedPiece] = []
    for index, length, markdown in plan_pieces(
        [item.chunk.text for item in kept],
        [item.size for item in kept],
        piece_lengths,
        piece_words,
        [item.chunk.parts for item in kept],
        kept_documents,
    ):
        kept[index].pieces.append(len(planned))
        planned.append(_PlannedPiece(index, length, markdown, None))
    # With the parser, a piece's syntax is that of its span of the chunk's parse, found by
    # searching the chunk's prose piece by piece (``piece_offsets``). Where a piece's prose
    # repeats in the chunk, the span found depends on the pieces before it, so the offset is
    # part of what the piece's cache entry is kept under.
    texts: dict[int, str] = {}
    if parser is not None:
        for index, item in enumerate(kept):
            if not item.pieces:
                continue
            texts[index] = (item.parsed or prose(item.chunk.text)).text
            with_words: list[_PlannedPiece] = []
            for position in item.pieces:
                piece = planned[position]
                piece.parsed = prose(piece.markdown)
                if words(piece.parsed.text):
                    with_words.append(piece)
            found_at = piece_offsets(
                texts[index],
                [(piece.length, piece.parsed.text) for piece in with_words if piece.parsed],
            )
            for piece, offset in zip(with_words, found_at, strict=True):
                piece.offset = offset
    for piece in planned:
        item = kept[piece.chunk]
        if store is None:
            if piece.parsed is None:
                piece.parsed = prose(piece.markdown)
            continue
        extra = () if parser is None else (str(piece.offset),)
        piece.key = caching.key(print_, item.chunk.text, piece.markdown, *extra)
        found = store.fetch(piece.key)
        value = caching.decode_or_none(found[1]) if found is not None else None
        if value is not None:
            piece.metrics = value["metrics"]
            piece.parsed = None
        elif piece.parsed is None:
            piece.parsed = prose(piece.markdown)

    # What the parser has to read: chunks measured here, and chunks read from the cache whose
    # pieces were not, with those pieces' spans.
    to_parse: list[tuple[str, list[tuple[int, int]]]] = []
    parsed_chunks: set[int] = set()
    wanted: list[bool] = [
        parser is not None
        and bool(keep_docs)
        and (keep_docs is True or kept_documents[index] in keep_docs)  # type: ignore[operator]
        for index in range(len(kept))
    ]
    keep_positions: set[int] = set()
    if parser is not None:
        for index, item in enumerate(kept):
            spans = [
                (piece.offset, piece.offset + len(piece.parsed.text))
                for position in item.pieces
                if (piece := planned[position]).parsed is not None
                and piece.offset is not None
                and words(piece.parsed.text)
            ]
            if item.cached is None or spans or (docs_required and wanted[index]):
                text = texts.get(index)
                if text is None:
                    if item.parsed is None:
                        item.parsed = prose(item.chunk.text)
                    text = item.parsed.text
                if wanted[index]:
                    keep_positions.add(len(to_parse))
                to_parse.append((text, spans))
                parsed_chunks.add(index)
    texts.clear()
    results = (
        measurer.parse(
            parser, to_parse, keep_positions, here=docs_required and bool(keep_positions)
        )
        if parser is not None and to_parse
        else iter(())
    )
    docs: list[tuple[str, Any] | None] | None = None
    if parser is not None and keep_docs:
        docs = [None for _ in kept]

    # Main pass, in order: each chunk's metrics and counts (read, or measured and parsed), and
    # its pieces'. Counts are summed as they come, so only a score keeps each chunk's.
    # Without each chunk's counts (a reference, not a score), each chunk's metrics are kept
    # compactly too: a reference of thousands of chunks is where memory runs out.
    all_metrics: MetricRows | list[Metrics] = [] if keep_distributions else MetricRows()
    distributions: list[dict[str, Counter[str]]] = []
    totals: dict[str, Counter[str]] = {}
    parsing = bool(to_parse)
    parsed_order: list[int] = []
    total_words = 0
    measurer.report(Progress(phase, 0, len(kept), 0, parsing))
    for index, item in enumerate(kept):
        value = caching.decode_or_none(item.cached) if item.cached is not None else None
        if value is not None:
            metrics: Metrics = value["metrics"]
            counts = {name: Counter(entry) for name, entry in value["distributions"].items()}
        else:
            if item.parsed is None:
                item.parsed = prose(item.chunk.text)
            # The prose's words, found once for the metrics and the patterns.
            tokens = words(item.parsed.text)
            metrics = surface_metrics(item.chunk.text, item.parsed, tokens)
            counts = _distributions(item.parsed.text, tokens)
        pending = [
            planned[position] for position in item.pieces if planned[position].parsed is not None
        ]
        for piece in pending:
            assert piece.parsed is not None
            tokens = words(piece.parsed.text)
            if tokens:
                piece.metrics = surface_metrics(piece.markdown, piece.parsed, tokens)
        if parser is not None and (index in parsed_chunks or value is None):
            if index in parsed_chunks:
                syntax, trigrams, piece_syntax, doc = next(results)
                if doc is not None and docs is not None:
                    docs[index] = (to_parse[len(parsed_order)][0], doc)
                parsed_order.append(index)
            else:  # an entry that could not be read: parse it here
                assert item.parsed is not None
                [(syntax, trigrams, piece_syntax, doc)] = parse(
                    parser, [(item.parsed.text, [])], {0} if wanted[index] else ()
                )
                if doc is not None and docs is not None:
                    docs[index] = (item.parsed.text, doc)
            if value is None:
                metrics = _merge(metrics, syntax)
                counts["pos_trigram"] = Counter(trigrams)
            spanned = [
                piece for piece in pending if piece.metrics is not None and piece.offset is not None
            ]
            for piece, extra in zip(spanned, piece_syntax, strict=True):
                if extra is not None and piece.metrics is not None:
                    piece.metrics = _merge(piece.metrics, extra)
        if store is not None and value is None and item.key is not None:
            store.put(
                item.key,
                item.size,
                {"metrics": metrics, "distributions": {k: dict(v) for k, v in counts.items()}},
            )
        for piece in pending:
            piece.parsed = None
            if store is not None and piece.key is not None and piece.metrics is not None:
                store.put(piece.key, 0, {"metrics": piece.metrics})
        item.parsed = None
        item.cached = None
        all_metrics.append(metrics)
        for name, entry in counts.items():
            _add(totals.setdefault(name, Counter()), entry)
        if keep_distributions:
            distributions.append(counts)
        total_words += item.size
        measurer.report(Progress(phase, index + 1, len(kept), total_words, parsing))
    return Measured(
        [item.chunk for item in kept],
        all_metrics,
        distributions,
        totals,
        empty,
        below,
        [
            Piece(piece.chunk, piece.length, piece.metrics)
            for piece in planned
            if piece.metrics is not None
        ],
        docs,
    )


def measure_in_parts(
    documents: Sequence[tuple[str, list[tuple[str, Any | None]]]],
    parser: Parser | None,
    measurer: Measurer | None = None,
) -> list[tuple[Any, list[Metrics]]]:
    """Each document (its Markdown, and its windows' prose with their parses where this
    process has them) read in parts (``drift``): its layout and the metrics of each span,
    as ``profile`` reads a draft's. A document's spans come from ``measurer.cache`` when it
    was read before; otherwise their syntax is measured from the document's parse, which
    worker processes make when the windows' parses are not at hand (they were measured by
    workers, or came from the cache)."""
    from styleprofile import drift

    measurer = measurer or Measurer()
    store = measurer.cache if measurer.cache is not None and measurer.cache.enabled else None
    print_ = caching.run_fingerprint(parser.used() if parser else None) if store else b""
    read: list[tuple[Any, list[Metrics] | None]] = []
    keys: list[bytes | None] = []
    jobs: list[tuple[int, _PartsItem]] = []
    for text, windows in documents:
        found = drift.paragraphs(text)
        layout = drift.plan([paragraph.words for paragraph in found])
        key: bytes | None = None
        if not layout.spans:
            read.append((layout, []))
            keys.append(None)
            continue
        parts = span_markdown(found, layout)
        if store is not None:
            key = caching.key(print_, text, "in parts", *(part for part, _ in windows))
            entry = store.fetch(key)
            value = caching.decode_or_none(entry[1]) if entry is not None else None
            if value is not None:
                read.append((layout, value["spans"]))
                keys.append(None)
                continue
        keys.append(key)
        if parser is None:
            read.append((layout, measure_spans(parts, None)))
        elif all(doc is not None for _, doc in windows):
            whole = parsed_document(prose_text(text), windows, parser)
            read.append((layout, measure_spans(parts, whole)))
        else:
            jobs.append((len(read), (prose_text(text), [part for part, _ in windows], parts)))
            read.append((layout, None))
    if parser is not None and jobs:
        for (position, _), spans in zip(
            jobs, measurer.read_parts(parser, [item for _, item in jobs]), strict=True
        ):
            read[position] = (read[position][0], spans)
    done: list[tuple[Any, list[Metrics]]] = []
    for (layout, spans), key in zip(read, keys, strict=True):
        assert spans is not None
        if store is not None and key is not None:
            store.put(key, 0, {"spans": spans})
        done.append((layout, spans))
    return done
