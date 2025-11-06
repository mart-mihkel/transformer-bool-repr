.PHONY: sync
sync:
	uv sync

.PHONY: format
format: sync
	uv run ruff format

.PHONY: lint
lint: sync
	uv run ruff check
