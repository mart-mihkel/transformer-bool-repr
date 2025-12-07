sync:
	uv sync
	uv pip install --editable .

notebook:
	uv run marimo edit notebooks

format:
	@uv run ruff format

lint:
	@uv run ruff check --fix

test:
	@uv run pytest

check: format lint test
