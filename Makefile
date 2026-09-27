UV ?= uv

.PHONY: sync test lint format typecheck check demo

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
