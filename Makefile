.PHONY: install test lint format build
install:
	python -m pip install -e ".[dev]"
test:
	python -m pytest
lint:
	ruff check .
format:
	ruff format .
build:
	python -m build
