UV ?= uv

.PHONY: sync test lint format typecheck check demo bench bench-quick snapshots

sync:
	$(UV) sync --extra syntax

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run pyright

check: test lint typecheck

# Build a reference from the sample corpus in examples/, score a draft against it, and show
# the saved score. Outputs go to profiles/, which is git-ignored.
demo:
	$(UV) run styleprofile build examples/writer --contrast examples/llm-drafts \
		-o profiles/demo-writer.json
	$(UV) run styleprofile score examples/draft.md profiles/demo-writer.json \
		-o profiles/demo-draft.json
	$(UV) run styleprofile show profiles/demo-draft.json > /dev/null

# Benchmark every case in bench/targets.py on generated corpora (bench/gen.py) and compare with
# the performance targets. Results go to bench/results.json. Pass options with BENCH, e.g.
# `make bench BENCH="--case big --repeat 3"`. bench-quick runs only the case CI runs. Compare
# with a base revision as CI does with `make bench-quick BENCH="--against origin/main --repeat 5"`.
bench:
	$(UV) run python bench/run.py $(BENCH)

bench-quick:
	$(UV) run python bench/run.py --quick $(BENCH)

# Regenerate the CLI output snapshots in tests/snapshots/ after an intended output change.
# It installs the syntax extra first, so the syntax snapshots are refreshed too.
snapshots:
	UPDATE_SNAPSHOTS=1 $(UV) run --extra syntax pytest tests/test_snapshots.py
