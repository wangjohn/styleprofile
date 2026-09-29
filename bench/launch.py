"""Run the ``styleprofile`` command in this process, then save what it and its workers used.

    python bench/launch.py USAGE.json ARGS...

is ``python -m styleprofile ARGS...``, except that on exit it writes the CPU time (user +
system) of this process and of every child it waited for, spaCy's worker processes included,
to USAGE.json. ``os.wait4`` in ``run.py`` cannot see those children's CPU on every platform
(macOS leaves out grandchildren), and ``getrusage(RUSAGE_CHILDREN)`` here always can.
"""

from __future__ import annotations

import json
import resource
import runpy
import sys


def main() -> int:
    out, sys.argv = sys.argv[1], ["styleprofile", *sys.argv[2:]]
    try:
        runpy.run_module("styleprofile", run_name="__main__", alter_sys=True)
        code = 0
    except SystemExit as done:
        code = done.code if isinstance(done.code, int) else (0 if done.code is None else 1)
    usage = {
        who: {"cpu_s": rusage.ru_utime + rusage.ru_stime}
        for who, rusage in (
            ("self", resource.getrusage(resource.RUSAGE_SELF)),
            ("children", resource.getrusage(resource.RUSAGE_CHILDREN)),
        )
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(usage, handle)
    return code


if __name__ == "__main__":
    sys.exit(main())
