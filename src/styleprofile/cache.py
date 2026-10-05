"""The measurement cache: a chunk's metrics, kept so text measured once is never measured again.

Measuring is most of a build's time, and spaCy most of that, while a corpus mostly grows by a
few documents at a time. So every measured chunk, and every calibration piece cut from it, is
saved in a SQLite file in the platform's cache directory (see ``cache_dir``), and a later
run that meets the same text takes its numbers from there.

An entry's key is a hash of the text together with a *fingerprint* of everything that decides
its numbers (``fingerprint``): the code of the metric registry and of the modules that compute
the metrics, the package and Python versions and, when syntax is measured, the spaCy model and
its versions. Changing any of them changes every key, so a stale entry is never read; it is
only left to age out. The value is what measuring gave (metrics and pattern counts), stored as
JSON, which reads back to exactly the same numbers in the same order, so a report built from
the cache is byte-for-byte the one measuring would give.

The cache is safe to share between processes (SQLite's locking, and one short transaction per
batch of writes), holds about ``MAX_BYTES`` at most (least recently used entries are dropped
first), and never fails a run: if the file cannot be opened, read or written, the run measures
everything itself and says why (``MeasurementCache.problem``). ``styleprofile cache --clear``
deletes it, and ``STYLEPROFILE_NO_CACHE=1`` turns it off.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

# sqlite3, threading and queue are imported where they are used: a run without the cache
# (every score) should not pay for them.
if TYPE_CHECKING:
    import sqlite3
    import threading

# The file's own format; bump it when the schema or the value encoding changes.
FORMAT = 1
FILENAME = f"measurements-v{FORMAT}.sqlite3"
# The cache is pruned to PRUNE_TO of this, least recently used first, once it grows past it.
MAX_BYTES = 512 * 1024 * 1024
PRUNE_TO = 0.8
# Writes are batched: one transaction per this many entries, with at most WRITE_QUEUE batches
# (about 2.5 MB each) waiting for the writer thread.
WRITE_BATCH = 256
WRITE_QUEUE = 8
PAGE_SIZE = 65536
# Marking an entry as used again is skipped when it was used within this many milliseconds.
RECENT_MS = 3_600_000
# The most the journal kept between writes may hold, in bytes.
JOURNAL_LIMIT = 16 * 1024 * 1024
# How long a writer waits for another process's transaction before giving up on the cache.
TIMEOUT_S = 10.0
# How long handing a batch to the writer thread waits between checks that it is still alive,
# and at most in all before the cache is turned off (also the longest ``close`` waits for a
# writer that finishes no batch); and the longest ``close`` waits for the writer to finish
# what is queued (at most WRITE_QUEUE batches) before dropping it.
HAND_TIMEOUT_S = 1.0
HAND_DEADLINE_S = 5.0
CLOSE_TIMEOUT_S = 30.0
# Set to anything but empty or 0, this turns the cache off for every run.
ENVIRONMENT = "STYLEPROFILE_NO_CACHE"
# The modules whose code decides a metric's value. The fingerprint hashes their source, so
# editing any of them (a lexicon, a threshold, a bug fix) invalidates every entry.
MEASURING_MODULES = (
    "styleprofile.metrics",
    "styleprofile.surface",
    "styleprofile.syntax",
    "styleprofile.measure",
)


def cache_dir() -> Path:
    """The platform's cache directory, with an absolute XDG_CACHE_HOME taking priority."""
    base = os.environ.get("XDG_CACHE_HOME", "")
    if base and os.path.isabs(base):
        return Path(base) / "styleprofile"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        root = Path(local) if local and os.path.isabs(local) else Path.home() / "AppData" / "Local"
        return root / "styleprofile" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "styleprofile"
    return Path.home() / ".cache" / "styleprofile"


def _source(module: ModuleType) -> bytes:
    """A module's source, or just its name when it has no readable source file."""
    try:
        return Path(module.__file__ or "").read_bytes()
    except OSError:
        return module.__name__.encode()


def fingerprint(
    syntax: dict[str, str] | None,
    *,
    modules: Sequence[str] = MEASURING_MODULES,
    version: str | None = None,
) -> bytes:
    """What every cached value depends on, hashed: the metric registry, the source of the
    modules that measure, the package and Python versions, and ``syntax``, the spaCy model and
    versions when the parser runs (``None`` without it)."""
    from styleprofile import __version__
    from styleprofile.metrics import DISTRIBUTION_LABELS, METRICS

    digest = hashlib.blake2b(digest_size=16)
    parts: list[bytes] = [
        f"format {FORMAT}".encode(),
        f"styleprofile {version or __version__}".encode(),
        f"python {sys.version_info.major}.{sys.version_info.minor}".encode(),
        repr(METRICS).encode(),
        repr(DISTRIBUTION_LABELS).encode(),
        *(_source(import_module(name)) for name in modules),
        json.dumps(syntax, sort_keys=True).encode(),
    ]
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


@cache
def _fingerprint(model: str | None, model_version: str | None, spacy_version: str | None) -> bytes:
    syntax = (
        None
        if model is None
        else {"model": model, "model_version": model_version, "spacy_version": spacy_version}
    )
    return fingerprint(syntax)


def run_fingerprint(syntax: dict[str, str] | None) -> bytes:
    """``fingerprint(syntax)``, computed once per process for each parser."""
    if syntax is None:
        return _fingerprint(None, None, None)
    return _fingerprint(syntax["model"], syntax["model_version"], syntax["spacy_version"])


def key(print_: bytes, text: str, *extra: str) -> bytes:
    """The cache key of ``text`` (and of the piece ``extra`` cut from it) under a fingerprint."""
    digest = hashlib.blake2b(print_, digest_size=20)
    for part in (text, *extra):
        data = part.encode("utf-8", "surrogatepass")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.digest()


def encode(value: Any) -> bytes:
    return _compress(_serialize(value))


def _serialize(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _compress(data: bytes) -> bytes:
    return zlib.compress(data, 1)


def decode(data: bytes) -> Any:
    return json.loads(zlib.decompress(data))


def decode_or_none(data: bytes) -> Any | None:
    """``decode``, or None for a value that cannot be read (measure it again instead)."""
    try:
        return decode(data)
    except (zlib.error, ValueError):
        return None


class MeasurementCache:
    """One run's connection to the cache file. Opened lazily. Nothing that goes wrong with it
    fails the run: the first problem turns it off for the rest of the run, and ``problem``
    says what it was, for the run to tell the user.

    Reads happen as the run asks. Writes go to a thread of their own in batches, so a disk
    that is slow to take them (a busy laptop's can take seconds) slows the run only by what is
    still unwritten when it ends. ``close`` waits for them (at most ``CLOSE_TIMEOUT_S``), then
    prunes the file when it has grown past about ``max_bytes``."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        max_bytes: int = MAX_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        import queue

        self.path = path if path is not None else cache_dir() / FILENAME
        self._owned_directory = path is None or self.path.parent == cache_dir()
        self.max_bytes = max_bytes
        self.clock = clock
        self.problem: str | None = None
        self._connection: sqlite3.Connection | None = None
        self._pending: list[tuple[bytes, int, bytes]] = []
        self._used: list[bytes] = []
        self._writes: queue.Queue[_Batch | None] = queue.Queue(maxsize=WRITE_QUEUE)
        self._writer: threading.Thread | None = None
        self._progress_at = 0.0  # when the writer last finished a batch (monotonic)
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.problem is None

    def _fail(self, error: BaseException | str) -> None:
        """Turn the cache off for the rest of the run, keeping the first reason."""
        if self.problem is None:
            self.problem = error if isinstance(error, str) else _reason(error)

    def _connect(self) -> sqlite3.Connection | None:
        if self.problem is not None:
            return None
        if self._connection is None:
            self._connection = self._open_or_reset()
        return self._connection

    def _open_or_reset(self) -> sqlite3.Connection | None:
        """A new connection, or None (and the cache off) when there can be none."""
        import sqlite3

        try:
            return self._open()
        except sqlite3.OperationalError as error:
            self._fail(error)  # locked for too long, read-only, ...: skip it this run
        except sqlite3.DatabaseError:
            # Not a database, or damaged (it is never synced to disk, so a crash of the
            # machine can do that): start it afresh, if it can be deleted.
            try:
                clear(self.path)
                return self._open()
            except (sqlite3.Error, OSError) as error:
                self._fail(error)
        except OSError as error:
            self._fail(error)
        return None

    def _open(self) -> sqlite3.Connection:
        import sqlite3

        if sys.platform == "win32":
            from styleprofile.cache_acl import check_cache_paths

            check_cache_paths(self.path, owned_directory=self._owned_directory)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if sys.platform == "win32":
            from styleprofile.cache_acl import protect_cache

            protect_cache(self.path, owned_directory=self._owned_directory)
        # The file holds pattern counts of the user's texts: private, as they are.
        if not self.path.exists():
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
        connection = sqlite3.connect(self.path, timeout=TIMEOUT_S, check_same_thread=False)
        try:
            # Both only take effect on a new file, and only before anything reads it (with
            # some SQLite versions), so they come first and are no-ops on an existing file:
            # pages are large, since values are about 10 KB and a file grows a page at a
            # time (4 KB at a time took several times longer on a busy macOS disk), and
            # pages freed by pruning go back to the file system.
            connection.execute(f"PRAGMA page_size={PAGE_SIZE}")
            # Set the connection's journal policy before auto_vacuum, which can write
            # even when its value is unchanged. A new connection defaults to DELETE:
            # on Windows it cannot delete a journal another PERSIST connection holds.
            connection.execute("PRAGMA journal_mode=PERSIST")
            connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
            new = connection.execute("PRAGMA user_version").fetchall()[0][0] != FORMAT
            # A rollback journal kept between writes rather than deleted: a run that only
            # reads opens one file and writes none, and writes don't create a file each time.
            # On a busy disk, creating and growing files is what is slow: a write-ahead log
            # made reading a warm cache 0.1-0.3 s slower.
            connection.execute(f"PRAGMA journal_size_limit={JOURNAL_LIMIT}")
            # A cache need not survive a power cut, so writes are never synced to disk; a
            # file damaged that way is started afresh (``_open_or_reset``).
            connection.execute("PRAGMA synchronous=OFF")
            if new:
                self._create(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    @staticmethod
    def _create(connection: sqlite3.Connection) -> None:
        with connection:
            # Values are large, so they live in an ordinary table; when each entry was last
            # used lives in a small one, so marking a hit never rewrites a value.
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries (key BLOB PRIMARY KEY, "
                "size INTEGER NOT NULL, value BLOB NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS used (key BLOB PRIMARY KEY, "
                "at INTEGER NOT NULL, bytes INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS used_at ON used (at)")
            connection.execute(f"PRAGMA user_version={FORMAT}")

    def fetch(self, entry: bytes) -> tuple[int, bytes] | None:
        """An entry's stored size (prose words) and its still encoded value (``decode``), or
        None when there is none or it cannot be read."""
        import sqlite3

        connection = self._connect()
        if connection is None:
            return None
        try:
            # fetchall, so the statement is finished and holds no lock the writer waits for.
            rows = connection.execute(
                "SELECT size, value, at FROM entries JOIN used USING (key) WHERE key = ?",
                (entry,),
            ).fetchall()
            row = rows[0] if rows else None
        except sqlite3.Error as error:
            self._fail(error)
            return None
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        # Recency only orders pruning, so it is kept to the hour: a run that repeats one an
        # hour ago writes nothing.
        if int(row[2]) < self.clock() * 1000 - RECENT_MS:
            self._used.append(entry)
        return int(row[0]), bytes(row[1])

    def put(self, entry: bytes, size: int, value: Any) -> None:
        """Save an entry (written in batches, and by ``close``)."""
        if self.problem is not None:
            return
        # Compressed by the writer thread: zlib lets go of the interpreter while it works, so
        # that half of the cost overlaps with measuring.
        self._pending.append((entry, size, _serialize(value)))
        if len(self._pending) >= WRITE_BATCH:
            self.flush()

    def flush(self) -> None:
        """Hand what is pending to the writer thread (dropped when the cache is off)."""
        import threading

        pending, self._pending = self._pending, []
        used, self._used = self._used, []
        if self.problem is not None or not (pending or used):
            return
        if self._writer is None:
            self._writer = threading.Thread(target=self._write, daemon=True)
            self._writer.start()
        self._hand(_Batch(pending, used, int(self.clock() * 1000)))

    def _hand(self, item: _Batch | None) -> bool:
        """Queue ``item`` for the writer, waiting at most ``HAND_DEADLINE_S`` for room. A
        writer that has stopped, or keeps the queue full that long (stuck, or much slower
        than measuring), turns the cache off: what is pending is dropped and the run goes on
        without it rather than wait."""
        import queue

        writer = self._writer
        deadline = time.monotonic() + HAND_DEADLINE_S
        while writer is not None and writer.is_alive():
            try:
                self._writes.put(item, timeout=min(HAND_TIMEOUT_S, HAND_DEADLINE_S))
                return True
            except queue.Full:
                if time.monotonic() >= deadline:
                    self._fail("the cache writer stopped responding")
                    return False
        self._fail("its writer stopped")
        return False

    def _write(self) -> None:
        """The writer thread: one transaction per batch, then the pruning, on a connection of
        its own. Whatever goes wrong turns the cache off; the thread keeps taking batches
        (and dropping them) until ``close``, so nothing waits on it."""
        connection = None
        try:
            connection = self._open_or_reset()
        except BaseException as error:  # the run must not wait on a dead writer
            self._fail(error)
        while (batch := self._writes.get()) is not None:
            if connection is None or self.problem is not None:
                continue
            try:
                entries = [(entry, size, _compress(data)) for entry, size, data in batch.entries]
                with connection:
                    connection.executemany(
                        "INSERT OR REPLACE INTO entries VALUES (?, ?, ?)", entries
                    )
                    connection.executemany(
                        "INSERT OR REPLACE INTO used VALUES (?, ?, ?)",
                        [(entry, batch.at, len(value)) for entry, _, value in entries],
                    )
                    connection.executemany(
                        "UPDATE used SET at = ? WHERE key = ?",
                        [(batch.at, entry) for entry in batch.used],
                    )
            except BaseException as error:
                self._fail(error)
            self._progress_at = time.monotonic()
        if connection is not None:
            try:
                if self.problem is None:
                    self._prune(connection)
            except BaseException as error:
                self._fail(error)
            finally:
                try:
                    connection.close()
                except BaseException as error:
                    self._fail(error)

    def close(self) -> None:
        """Write what is pending, prune the file if it is too large, and disconnect. Waits for
        the writer while it makes progress (see ``_wait_for_writer``): past that, what is
        unwritten is dropped."""
        import sqlite3

        self.flush()
        if self._writer is not None:
            # A writer already given up on is not waited for (it is a daemon thread).
            if self.problem is None and self._hand(None):
                self._wait_for_writer(self._writer)
            self._writer = None
        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.Error as error:
                self._fail(error)
            self._connection = None

    def _wait_for_writer(self, writer: threading.Thread) -> None:
        """Wait for the writer to finish what is queued while it keeps making progress: it
        is given up on (and what is left dropped) when no batch has finished for
        ``HAND_DEADLINE_S``, or after ``CLOSE_TIMEOUT_S`` in all."""
        start = time.monotonic()
        while writer.is_alive():
            writer.join(HAND_TIMEOUT_S)
            now = time.monotonic()
            stalled = now - max(self._progress_at, start) >= HAND_DEADLINE_S
            if writer.is_alive() and (stalled or now - start >= CLOSE_TIMEOUT_S):
                self._fail("the cache writer stopped responding")
                return

    def _prune(self, connection: sqlite3.Connection) -> None:
        """Drop the least recently used entries once the file has grown past ``max_bytes``,
        down to about ``PRUNE_TO`` of it. The bound is approximate: pages are only freed
        whole, and the journal (up to ``JOURNAL_LIMIT``) is not counted."""
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        if pages * page_size <= self.max_bytes:
            return
        target = int(self.max_bytes * PRUNE_TO)
        total = connection.execute("SELECT COALESCE(SUM(bytes), 0) FROM used").fetchone()[0]
        excess = int(total) - target
        stale: list[tuple[bytes]] = []
        freed = 0
        for entry, size in connection.execute("SELECT key, bytes FROM used ORDER BY at"):
            if freed >= excess:
                break
            stale.append((bytes(entry),))
            freed += int(size)
        with connection:
            connection.executemany("DELETE FROM entries WHERE key = ?", stale)
            connection.executemany("DELETE FROM used WHERE key = ?", stale)
        # Each step of the pragma frees a page, and fetching should run them all, but
        # Python 3.11's sqlite3 stops after the first; so repeat until nothing is free.
        free = connection.execute("PRAGMA freelist_count").fetchall()[0][0]
        for _ in range(free):
            connection.execute("PRAGMA incremental_vacuum").fetchall()
            if not connection.execute("PRAGMA freelist_count").fetchall()[0][0]:
                break

    def __enter__(self) -> MeasurementCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _reason(error: BaseException) -> str:
    """Why the cache is off, in a few words: the OS's or SQLite's own."""
    if isinstance(error, OSError) and error.strerror:
        where = f": {error.filename}" if error.filename else ""
        return f"{error.strerror.lower()}{where}"
    return str(error) or type(error).__name__


@dataclass(frozen=True)
class _Batch:
    """Entries to write (key, size, serialized value, compressed when written), keys used
    again, and the time."""

    entries: list[tuple[bytes, int, bytes]]
    used: list[bytes]
    at: int


def clear(path: Path | None = None) -> bool:
    """Delete the cache files; True if there was anything to delete."""
    base = path if path is not None else cache_dir() / FILENAME
    removed = False
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            Path(f"{base}{suffix}").unlink()
            removed = True
        except FileNotFoundError:
            pass
    return removed


def describe(path: Path | None = None) -> tuple[Path, int, int, str | None]:
    """The cache file, its size in bytes (the journal included), its entry count, and why it
    cannot be used, if it cannot: its folder is not writable, or it cannot be read."""
    import sqlite3

    base = path if path is not None else cache_dir() / FILENAME
    size = sum(
        Path(f"{base}{suffix}").stat().st_size
        for suffix in ("", "-journal")
        if Path(f"{base}{suffix}").exists()
    )
    entries = 0
    problem: str | None = None
    folder = base.parent
    existing = next((parent for parent in (folder, *folder.parents) if parent.exists()), None)
    if existing is not None and not os.access(existing, os.W_OK | os.X_OK):
        problem = f"{existing} is not writable"
    if base.exists():
        try:
            with closing(
                sqlite3.connect(base.absolute().as_uri() + "?mode=ro", uri=True, timeout=TIMEOUT_S)
            ) as db:
                entries = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        except sqlite3.Error as error:
            problem = problem or str(error)
            entries = 0
    return base, size, entries, problem


def disabled_by_environment() -> bool:
    """Whether ``STYLEPROFILE_NO_CACHE`` turns the cache off (any value but empty or 0)."""
    return os.environ.get(ENVIRONMENT, "") not in ("", "0")
