UV := sh scripts/uv.sh
CONFIG ?= configs/data.json
DATA_ARGS ?=
COMPOSE ?= docker compose
DATA := $(UV) run --no-sync python -m ml_product_infrastructure.dataset
.DEFAULT_GOAL := check
.PHONY: setup check demo hooks

setup:
	$(UV) sync --locked --group dev
	$(UV) run --no-sync pre-commit install
	@test -f .env || cp .env.example .env

check:
	$(UV) run --no-sync ruff check .
	$(UV) run --no-sync ruff format --check .
	$(UV) run --no-sync mypy

demo:
	$(UV) run --no-sync python scripts/debug_checks.py

hooks:
	$(UV) run --no-sync pre-commit run --all-files

.PHONY: test s3-up s3-down ingest data data-download data-convert data-split data-stats data-publish

test:
	$(UV) run --no-sync pytest -q

s3-up:
	$(COMPOSE) up -d --wait s3
	$(COMPOSE) run --rm --build s3-init

s3-down:
	$(COMPOSE) down

ingest:
	$(DATA).ingest --config $(CONFIG) $(DATA_ARGS)

data:
	$(DATA).pipeline --config $(CONFIG) $(DATA_ARGS)

data-download:
	$(DATA).download --config $(CONFIG) $(DATA_ARGS)

data-convert:
	$(DATA).convert --config $(CONFIG) $(DATA_ARGS)

data-split:
	$(DATA).split --config $(CONFIG) $(DATA_ARGS)

data-stats:
	$(DATA).stats --config $(CONFIG) $(DATA_ARGS)

data-publish:
	$(DATA).publish --config $(CONFIG) $(DATA_ARGS)

.PHONY: test-s3
test-s3:
	RUN_S3_TESTS=1 $(UV) run --no-sync pytest -q tests/test_s3_integration.py
