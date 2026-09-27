UV ?= uv

.PHONY: sync test lint format typecheck check

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
