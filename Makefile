.PHONY: sync
sync:
	uv sync

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check

.PHONY: test
test:
	uv run pytest

.PHONY: check
check: format lint test
