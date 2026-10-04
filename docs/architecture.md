# Architecture

The public library exports in `styleprofile/__init__.py` call the pipelines in `api.py`.
`api.py` coordinates domain modules; it does not implement corpus preparation or the
statistical calculations. The CLI uses the same pipelines.

| Module | Responsibility |
| --- | --- |
| `corpus/reading.py` | Read files, folders and JSONL; decode and convert input formats. |
| `corpus/types.py`, `corpus/ids.py` | Shared text and chunk values, document identity and names. |
| `corpus/windows.py`, `corpus/duplicates.py` | Window and pool text; remove repeated documents. |
| `corpus/preparation.py` | Prepare inputs, split corpora, pair edits and preserve grouping. |
| `measure.py`, `surface.py`, `syntax.py` | Measure text patterns, with optional parser support. |
| `reference.py`, `calibration.py` | Build reference distributions and calibrate comparisons. |
| `scoring.py`, `stats.py` | Calculate scores and summarize measured chunks. |
| `reports.py`, `schema.py` | Validate report versions and fields; load and save reports. |
| `settings.py`, `runtime.py` | Resolve library settings and coordinate measurement and progress. |
| `results.py` | Library profile, score and evaluation objects and their report views. |
| `display.py`, `cli.py` | Render reports and provide command-line workflows. |

Files or named `Text` values become `Chunk` values during reading. Corpus preparation
keeps document identities through splitting, windowing, pooling and duplicate removal.
Measurement produces feature counts. Reference construction summarizes those counts and
calibrates lengths, contrast comparisons and any requested paragraph checks. Scoring
compares a draft's measurements with the saved reference distributions. Result objects
expose those reports to library callers; report I/O validates and writes them, and display
renders them as text.

The documented lower-level `styleprofile.profile.build_reference`, `score` and `Chunk`
imports remain aliases to their domain owners. New internal imports use the owning
modules directly. Moving a function does not change its calculations, report layout,
cache policy or syntax fallback behavior.
