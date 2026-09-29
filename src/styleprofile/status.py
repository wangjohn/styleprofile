"""The command line's progress line: the phase, chunks done, words a second and time left,
redrawn in place on stderr. The command shows it only when stderr is a terminal, so piped and
redirected output never carries it."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from typing import TextIO

from styleprofile.core import Phase, Progress

LABELS = {
    Phase.READ: "reading",
    Phase.LOAD_PARSER: "loading spaCy",
    Phase.MEASURE: "measuring",
    Phase.CALIBRATE: "calibrating",
    Phase.MEASURE_CONTRAST: "measuring the contrast set",
    Phase.CONTRAST: "learning the contrast",
}
# Redraw at most this often, in seconds; a phase change or the end of a phase always redraws.
REDRAW_S = 0.1
# Rates and time left appear once a phase has run this long, in seconds, so they are not
# guesses from the first few chunks.
SETTLE_S = 1.0
# When measuring with the parser looks like taking longer than this, in seconds, say once how
# to make it faster.
SLOW_S = 60.0
TIP = "tip: --no-syntax is ~10x faster"
# ... once the estimate rests on this share of the chunks, or this many seconds.
TIP_SHARE = 0.05
TIP_AFTER_S = 5.0


def duration(seconds: float) -> str:
    """``8s``, ``1m 05s`` or ``1h 02m``."""
    whole = max(0, round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class StatusLine:
    """A ``progress`` callback that draws one updating line on ``stream``. Call ``clear``
    before writing anything else to the stream (a ``DONE`` event clears it too)."""

    def __init__(
        self,
        stream: TextIO,
        *,
        clock: Callable[[], float] = time.monotonic,
        columns: int | None = None,
    ) -> None:
        self.stream = stream
        self.clock = clock
        self.columns = columns
        self.shown = ""
        self.tipped = False
        self._phase: Phase | None = None
        self._started = 0.0
        # The phase's first finished chunk (time, chunks, words): rates are taken from it,
        # since before it come start-up costs (spaCy's workers loading their models) that
        # the rest of the phase does not repeat.
        self._first: tuple[float, int, int] | None = None
        self._drawn = float("-inf")

    def __call__(self, event: Progress) -> None:
        now = self.clock()
        if event.phase is Phase.DONE:
            self.clear()
            return
        label = LABELS.get(event.phase)
        if label is None:  # BUILD, SCORE and EVALUATE are the measuring that follows
            return
        changed = event.phase is not self._phase
        if changed:
            self._phase, self._started, self._first = event.phase, now, None
        if self._first is None and event.done:
            self._first = (now, event.done, event.words or 0)
        finished = event.total is not None and event.done == event.total
        if not (changed or finished or now - self._drawn >= REDRAW_S):
            return
        text = label
        if event.total is not None and event.done is not None:
            text += f" {event.done:,}/{event.total:,} chunks"
            first = self._first
            if first is not None and now - first[0] >= SETTLE_S and event.done > first[1]:
                elapsed, done = now - first[0], event.done - first[1]
                if event.words:
                    text += f", {(event.words - first[2]) / elapsed:,.0f} words/s"
                left = elapsed / done * (event.total - event.done)
                if event.done < event.total:
                    text += f", about {duration(left)} left"
                # Judged on 5% of the chunks, or 5 seconds, of steady progress.
                steady = done >= TIP_SHARE * event.total or elapsed >= TIP_AFTER_S
                total = now - self._started + left
                if event.parsing and steady and not self.tipped and total > SLOW_S:
                    self.tipped = True
                    self.clear()
                    self.stream.write(TIP + "\n")
        self._draw(text)
        self._drawn = now

    def _draw(self, text: str) -> None:
        columns = self.columns or shutil.get_terminal_size((80, 24)).columns
        text = text[: max(columns - 1, 0)]
        self.stream.write("\r" + text + "\033[K")
        self.stream.flush()
        self.shown = text

    def clear(self) -> None:
        """Erase the line, if one is shown."""
        if self.shown:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self.shown = ""
