"""Один запуск в контейнере: train → лучший val checkpoint → единственный test."""

import argparse
import csv
import json
import os
import random
import socket
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ml_product_infrastructure.dataset.files import CLASSES, write_json
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import RunConfig, flatten_config, load_run
from ml_product_infrastructure.experiments.data import prepare_data
from ml_product_infrastructure.experiments.engine import BestCheckpoint, epoch
from ml_product_infrastructure.experiments.models import Classifier


def train(config: RunConfig) -> dict[str, object]:
    commit = os.environ.get("TRAINING_GIT_COMMIT", "")
    if len(commit) != 40 or os.environ.get("EXPECTED_GIT_COMMIT", commit) != commit:
        raise ValueError(
            "Образ не соответствует текущему git-коммиту; пересоберите через make infra-up"
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.train.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    store = S3Store.from_env()
    destination = f"models/{config.experiment_name}/{config.run_name}/"
    store.put_bytes_new(
        json.dumps({"git_commit": commit}).encode(), f"_locks/{destination.rstrip('/')}.json"
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(
        run_name=config.run_name,
        tags={
            "git_commit": commit,
            "selection_basis": "val_loss",
            "model_type": config.model.type,
        },
    ) as active:
        run_id = str(active.info.run_id)
        params = flatten_config(config.to_dict())
        params["git_commit"] = commit
        mlflow.log_params(params)
        mlflow.log_dict(config.to_dict(), "config.json")
        output = Path(os.environ["TRAIN_OUTPUT_DIR"]) / config.experiment_name / config.run_name
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config.to_dict())
        data = prepare_data(config)
        mlflow.log_param("data.manifest_sha256", data.manifest_sha256)
        mlflow.log_param("data.actual_train_samples", len(data.train))
        write_json(output / "train_ids.json", data.train_ids)
        generator = torch.Generator().manual_seed(config.seed)
        train_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            data.train,
            batch_size=config.train.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        val_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            data.val, batch_size=config.train.batch_size, shuffle=False, num_workers=0
        )
        model = Classifier(config.model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
        )
        best = BestCheckpoint(output / "model.pt")
        history: list[dict[str, object]] = []
        started = time.perf_counter()
        for index in range(1, config.train.epochs + 1):
            train_scores, _ = epoch(model, train_loader, data.mean, data.std, optimizer)
            val_scores, _ = epoch(model, val_loader, data.mean, data.std)
            metrics = {f"val_{k}": v for k, v in val_scores.to_dict().items()}
            metrics["train_loss"] = train_scores.loss
            mlflow.log_metrics(metrics, step=index)
            history.append({"epoch": index, **metrics})
            best.consider(
                val_scores.loss,
                index,
                model,
                {
                    "config": config.to_dict(),
                    "classes": CLASSES,
                    "mean": data.mean.flatten().tolist(),
                    "std": data.std.flatten().tolist(),
                    "git_commit": commit,
                    "manifest_sha256": data.manifest_sha256,
                },
            )
            print(
                f"{config.run_name} epoch={index}: train_loss={train_scores.loss:.5f}, val_loss={val_scores.loss:.5f}, val_acc={val_scores.accuracy:.4f}, val_macro_f1={val_scores.macro_f1:.4f}",
                flush=True,
            )
        duration = time.perf_counter() - started
        # Восстанавливаем минимум val loss; последний epoch может быть хуже.
        best.restore(model)
        val_best, confusion = epoch(model, val_loader, data.mean, data.std)
        # Единственный вызов оценки test — после фиксации лучшего checkpoint.
        test_loader: DataLoader[tuple[Tensor, ...]] = DataLoader(
            data.test, batch_size=config.train.batch_size, shuffle=False, num_workers=0
        )
        test_best, _ = epoch(model, test_loader, data.mean, data.std)
        metrics_result: dict[str, object] = {
            "run_name": config.run_name,
            "container_id": socket.gethostname(),
            "mlflow_run_id": run_id,
            "git_commit": commit,
            "data_s3_path": config.data.s3_path,
            "manifest_sha256": data.manifest_sha256,
            "best_epoch": best.epoch,
            "training_seconds": duration,
            "best_val": val_best.to_dict(),
            "test": test_best.to_dict(),
            "history": history,
            "selection_basis": "val_loss",
            "test_evaluations": 1,
        }
        mlflow.log_metrics(
            {
                "best_epoch": float(best.epoch),
                "training_seconds": duration,
                **{f"best_val_{key}": value for key, value in val_best.to_dict().items()},
                **{f"test_{key}": value for key, value in test_best.to_dict().items()},
            }
        )
        with (output / "confusion-val.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true/predicted", *CLASSES])
            for label, row in zip(CLASSES, confusion.tolist(), strict=True):
                writer.writerow([label, *row])
        write_json(output / "metrics.json", metrics_result)
        for name in (
            "model.pt",
            "config.json",
            "metrics.json",
            "train_ids.json",
            "confusion-val.csv",
        ):
            mlflow.log_artifact(str(output / name))
        for name, mime in (
            ("model.pt", "application/octet-stream"),
            ("config.json", "application/json"),
            ("metrics.json", "application/json"),
        ):
            store.put_new(output / name, destination + name, mime)
        print(
            f"FINISHED {config.run_name}: best_epoch={best.epoch}, val={val_best.to_dict()}, test={test_best.to_dict()}",
            flush=True,
        )
        return metrics_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Один CPU-эксперимент по JSON-конфигу")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(load_run(args.config))


if __name__ == "__main__":
    main()
