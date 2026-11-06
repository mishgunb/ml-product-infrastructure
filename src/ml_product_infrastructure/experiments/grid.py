"""Хостовый launcher: один отдельный Docker-контейнер для каждого запуска."""

import argparse
import copy
import itertools
import json
import os
import secrets
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml_product_infrastructure.dataset.files import write_json
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import RunConfig, load_run, mapping, number, text


def override(base: dict[str, object], changes: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(base)
    for path, value in changes.items():
        parts = path.split(".")
        current = result
        for part in parts[:-1]:
            current = mapping(current[part])
        if parts[-1] not in current:
            raise ValueError(f"Неизвестный параметр сетки: {path}")
        current[parts[-1]] = value
    return result


def expand_grid(base: RunConfig, grid: dict[str, object], suffix: str) -> list[RunConfig]:
    keys = sorted(grid)
    options: list[list[object]] = []
    for key in keys:
        value = grid[key]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Пустая/неверная ось сетки {key}")
        options.append(value)
    result: list[RunConfig] = []
    for index, combination in enumerate(itertools.product(*options), start=1):
        updated = override(base.to_dict(), dict(zip(keys, combination, strict=True)))
        config = RunConfig.from_dict(updated)
        updated["run_name"] = (
            f"cnn-lr{config.train.lr:g}-bs{config.train.batch_size}-s{config.seed}-{suffix}-{index}"
        )
        result.append(RunConfig.from_dict(updated))
    return result


@dataclass
class RunResult:
    config: RunConfig
    metrics: dict[str, object]

    @property
    def val_loss(self) -> float:
        return number(mapping(self.metrics["best_val"])["loss"])

    def to_dict(self) -> dict[str, object]:
        return {"config": self.config.to_dict(), "metrics": self.metrics}


def choose_best(results: list[RunResult]) -> RunResult:
    if not results:
        raise ValueError("Нет завершённых запусков сетки")
    # Test намеренно не читается при ранжировании.
    return min(results, key=lambda result: (result.val_loss, result.config.run_name))


def repository_head() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Закоммитьте изменения перед сборкой обучающего образа")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_trainer(compose: list[str], commit: str) -> None:
    subprocess.run(
        [
            *compose,
            "--profile",
            "training",
            "build",
            "--build-arg",
            f"GIT_COMMIT={commit}",
            "trainer",
        ],
        check=True,
    )


def execute_run(
    config: RunConfig, compose: list[str], commit: str, suite: str, index: int
) -> RunResult:
    generated = Path(".tools/generated-experiments") / suite / f"{index:02d}.json"
    write_json(generated, config.to_dict())
    container_name = f"hw3-{suite.lower()}-{index}"
    subprocess.run(
        [
            *compose,
            "--profile",
            "training",
            "run",
            "--rm",
            "--no-deps",
            "--name",
            container_name,
            "-e",
            f"EXPECTED_GIT_COMMIT={commit}",
            "trainer",
            "--config",
            f"/workspace/{generated.as_posix()}",
        ],
        check=True,
    )
    store = S3Store.from_env()
    key = f"models/{config.experiment_name}/{config.run_name}/metrics.json"
    with store.client.get_object(Bucket=store.bucket, Key=key)["Body"] as stream:
        metrics = mapping(json.loads(stream.read()))
    return RunResult(config, metrics)


def results_markdown(baseline: RunResult, runs: list[RunResult], final: RunResult) -> str:
    lines = [
        "| Запуск | LR | Batch | Train N | Эпохи | Лучшая эпоха | Val loss | Val acc | Val F1 | Test loss | Test acc | Test F1 | Время, с |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(runs, key=lambda item: (item.val_loss, item.config.run_name))
    for name, result in [
        ("Baseline linear", baseline),
        *[(f"CNN #{i}", r) for i, r in enumerate(ranked, 1)],
        ("Выбранная CNN, полный train", final),
    ]:
        cfg, met = result.config, result.metrics
        val, test = mapping(met["best_val"]), mapping(met["test"])
        count = cfg.data.max_train_samples if cfg.data.max_train_samples is not None else "весь"
        lines.append(
            f"| {name} | {cfg.train.lr:g} | {cfg.train.batch_size} | {count} | {cfg.train.epochs} | {met['best_epoch']} | {number(val['loss']):.4f} | {number(val['accuracy']):.4f} | {number(val['macro_f1']):.4f} | {number(test['loss']):.4f} | {number(test['accuracy']):.4f} | {number(test['macro_f1']):.4f} | {number(met['training_seconds']):.1f} |"
        )
    return "\n".join(lines) + "\n"


def run_grid(path: Path, compose: list[str], commit: str, suite: str) -> None:
    spec = mapping(json.loads(path.read_text()))
    base = load_run(Path(text(spec["base"])))
    baseline_values = override(base.to_dict(), mapping(spec["baseline"]))
    baseline_values["run_name"] = f"baseline-linear-s{base.seed}-{suite}"
    baseline = execute_run(RunConfig.from_dict(baseline_values), compose, commit, suite, 0)
    configs = expand_grid(base, mapping(spec["grid"]), suite)
    runs = [execute_run(config, compose, commit, suite, i) for i, config in enumerate(configs, 1)]
    winner = choose_best(runs)
    selected = override(winner.config.to_dict(), mapping(spec["final"]))
    selected["run_name"] = (
        f"selected-lr{winner.config.train.lr:g}-bs{winner.config.train.batch_size}-s{base.seed}-{suite}"
    )
    final = execute_run(RunConfig.from_dict(selected), compose, commit, suite, len(configs) + 1)
    summary = {
        "suite_id": suite,
        "git_commit": commit,
        "selection_basis": "minimum grid best_val.loss; test ignored",
        "selected_grid_run": winner.config.run_name,
        "baseline": baseline.to_dict(),
        "grid": [r.to_dict() for r in runs],
        "final": final.to_dict(),
    }
    output = Path(text(spec["output_dir"])) / suite
    write_json(output / "summary.json", summary)
    (output / "results.md").write_text(results_markdown(baseline, runs, final))
    store = S3Store.from_env()
    store.put_new(
        output / "summary.json",
        f"models/{base.experiment_name}/suites/{suite}/summary.json",
        "application/json",
    )
    print((output / "results.md").read_text(), flush=True)
    print(
        f"Итог: {output / 'summary.json'}; выбран {winner.config.run_name} только по val loss",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline, сетка и итоговый запуск — отдельные контейнеры"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/grid.json"))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--suite-id")
    args = parser.parse_args()
    # Загружаем .env для host CLI; контейнеры получают внутренние адреса через Compose.
    S3Store.from_env()
    commit = repository_head()
    compose = shlex.split(os.environ.get("COMPOSE", "docker compose"))
    build_trainer(compose, commit)
    if args.build_only:
        return
    suite = args.suite_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + secrets.token_hex(2)
    if args.single:
        execute_run(load_run(args.config), compose, commit, suite, 0)
    else:
        run_grid(args.config, compose, commit, suite)


if __name__ == "__main__":
    main()
