.PHONY: test lint type-check check

test:
	uv run pytest

lint:
	uv run ruff check .

type-check:
	uv run mypy .

check: lint type-check test
