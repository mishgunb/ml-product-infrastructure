UV := sh scripts/uv.sh
.DEFAULT_GOAL := check
.PHONY: setup check demo hooks

setup:
	$(UV) sync --locked --group dev
	$(UV) run --no-sync pre-commit install

check:
	$(UV) run --no-sync ruff check .
	$(UV) run --no-sync ruff format --check .
	$(UV) run --no-sync mypy

demo:
	$(UV) run --no-sync python scripts/debug_checks.py

hooks:
	$(UV) run --no-sync pre-commit run --all-files
