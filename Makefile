sync:
	uv sync
	uv pip install --editable .

marimo:
	uv run marimo edit notebooks

pre-commit:
	uv run ruff check --fix
	uv run ruff format
	uv run ty check
	uv run pytest
