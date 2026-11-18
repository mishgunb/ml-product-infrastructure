UV := sh scripts/uv.sh
CONFIG ?= configs/data.json
DATA_ARGS ?=
COMPOSE ?= docker compose
DATA := $(UV) run --no-sync python -m ml_product_infrastructure.dataset
.DEFAULT_GOAL := check
.PHONY: setup check demo hooks

setup:
	$(UV) sync --locked --group dev --extra train --extra export --extra serve
	$(UV) run --no-sync pre-commit install
	$(UV) run --no-sync python scripts/setup_env.py

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

.PHONY: infra-up infra-down experiments train
infra-up: s3-up
	$(COMPOSE) build mlflow
	$(COMPOSE) up -d --wait mlflow
	COMPOSE="$(COMPOSE)" $(UV) run --extra train --no-sync python -m ml_product_infrastructure.experiments.grid --build-only

infra-down:
	$(COMPOSE) --profile training down

experiments: CONFIG = configs/grid.json
experiments:
	COMPOSE="$(COMPOSE)" $(UV) run --extra train --no-sync python -m ml_product_infrastructure.experiments.grid --config $(CONFIG)

train: CONFIG = configs/train.json
train:
	COMPOSE="$(COMPOSE)" $(UV) run --extra train --no-sync python -m ml_product_infrastructure.experiments.grid --single --config $(CONFIG)

SERVING := $(UV) run --no-sync python -m ml_product_infrastructure.serving
SERVING_CONFIG ?= configs/serving.json
SERVING_ARGS ?=
SERVICE_HOST ?= Linux в Docker, $(shell $(COMPOSE) exec -T triton uname -m), $(shell $(COMPOSE) exec -T triton nproc) логических CPU, клиент и сервис на одной машине

.PHONY: export serve-up verify load-test serve-down
export:
	$(SERVING).export --config $(SERVING_CONFIG) $(SERVING_ARGS)

serve-up:
	test -s model_repository/cifar_cnn/1/model.onnx && test -s artifacts/serving/stats.json
	$(COMPOSE) --profile serving up -d --wait --build --force-recreate triton prediction-api

verify:
	$(SERVING).verify --config $(SERVING_CONFIG) $(SERVING_ARGS)

load-test:
	$(SERVING).load_test --config $(SERVING_CONFIG) --service-host "$(SERVICE_HOST)" $(SERVING_ARGS)

serve-down:
	$(COMPOSE) --profile training --profile serving down
