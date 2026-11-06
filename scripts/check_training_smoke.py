"""Проверить модель в S3 и метрики/артефакты реального MLflow после smoke-запуска."""

import csv
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import torch
from mlflow import MlflowClient

from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import integer, load_run, mapping, number, text


def main() -> None:
    store = S3Store.from_env()
    cfg = load_run(Path(".tools/ci-smoke.json"))
    prefix = f"models/{cfg.experiment_name}/{cfg.run_name}/"
    with store.client.get_object(Bucket=store.bucket, Key=prefix + "metrics.json")[
        "Body"
    ] as stream:
        result = mapping(json.loads(stream.read()))
    history = result["history"]
    assert isinstance(history, list) and len(history) == 2
    best = min(history, key=lambda row: number(mapping(row)["val_loss"]))
    assert result["best_epoch"] == mapping(best)["epoch"]
    assert result["test_evaluations"] == 1 and result["selection_basis"] == "val_loss"
    uri = os.environ["MLFLOW_TRACKING_URI"]
    with urlopen(uri + "/api/3.0/mlflow/server-info", timeout=10) as response:
        server_info = mapping(json.loads(response.read()))
    assert server_info["multipart_downloads_enabled"] is False
    assert server_info["multipart_uploads_enabled"] is False
    client = MlflowClient(tracking_uri=uri)
    run_id = text(result["mlflow_run_id"])
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert len(client.get_metric_history(run_id, "val_loss")) == 2
    assert {f.path for f in client.list_artifacts(run_id)} >= {
        "model.pt",
        "config.json",
        "metrics.json",
        "confusion-val.csv",
    }
    assert run.data.params["data.s3_path"] == cfg.data.s3_path
    with tempfile.TemporaryDirectory() as directory:
        matrix = Path(client.download_artifacts(run_id, "confusion-val.csv", directory))
        with matrix.open() as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 11
        assert sum(int(v) for row in rows[1:] for v in row[1:]) == 10
        tracker_model = Path(client.download_artifacts(run_id, "model.pt", directory))
        path = Path(directory) / "s3-model.pt"
        store.download(prefix + "model.pt", path)
        assert tracker_model.read_bytes() == path.read_bytes()
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        assert checkpoint["best_epoch"] == integer(result["best_epoch"])
        assert checkpoint["metadata"]["mean"] == [
            float(torch.tensor(v, dtype=torch.float32)) for v in [0.3, 0.4, 0.5]
        ]
    print(
        "PASS: best checkpoint, single test evaluation, full config, epoch metrics and S3 artifacts"
    )


if __name__ == "__main__":
    main()
