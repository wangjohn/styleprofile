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

# Build a reference from the sample corpus in examples/ and score a draft against it.
# Outputs go to profiles/, which is git-ignored.
demo:
	$(UV) run styleprofile examples/writer --window-words 500 \
		--contrast examples/llm-drafts --output profiles/demo-writer.json
	$(UV) run styleprofile examples/draft.md --window-words 500 \
		--reference profiles/demo-writer.json --output profiles/demo-draft.json
