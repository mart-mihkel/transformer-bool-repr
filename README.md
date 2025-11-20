# Boolean Function Representation in Transformers

Algebraic normal form/Fourier expansion of boolean functions in transformer
networks

## Development

Use [uv](https://docs.astral.sh/uv/) for project management.

### Environment Setup

#### Installing uv

##### Windows (pip)

```bash
pip install uv
```

##### macOS/Linux (direct installer)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

- `uv sync --all-groups` - install all runtime and development dependencies.

### Make Commands

- `make sync` - setup local venv.
- `make format` - run formatter.
- `make lint` - run linter.

### Running Tests

- `uv run pytest` - run all tests.
- `uv run tests/run_tests.py` - run a specific test file.

### Running Scripts

- `uv run <file_name>.py`
