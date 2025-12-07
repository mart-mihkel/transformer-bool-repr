.PHONY: sync
sync:
	uv sync
	uv pip install -e .

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check --fix

.PHONY: test
test:
	uv run pytest

.PHONY: check
check: format lint test
