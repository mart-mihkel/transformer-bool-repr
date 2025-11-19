.PHONY: sync
sync:
	uv sync

.PHONY: format
format: sync
	uv run ruff format

.PHONY: fomat-check
fomat-check: sync
	uv run ruff format --check

.PHONY: lint
lint: sync
	uv run ruff check

.PHONY: test
test: sync
	uv run pytest

.PHONY: check
check: fomat-check lint test
