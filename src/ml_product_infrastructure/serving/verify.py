"""Сравнение метрик всего test через публичный POST /predict с метриками в S3."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import numpy as np

from ml_product_infrastructure.dataset.files import CLASSES, ImageRecord
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import mapping, number
from ml_product_infrastructure.experiments.metrics import scores_from_confusion
from ml_product_infrastructure.serving.source import Source, download, fetch_test_images, parser

TOLERANCES = {"loss": 1e-5, "accuracy": 1e-4, "macro_f1": 2e-4}


def predict(client: httpx.Client, root: Path, row: ImageRecord) -> tuple[int, int, float]:
    name = row["file_name"]
    prediction = mapping(
        client.post(
            "/predict", files={"file": ("test.png", (root / name).read_bytes(), "image/png")}
        )
        .raise_for_status()
        .json()
    )
    values = prediction["probabilities"]
    if not isinstance(values, list) or len(values) != len(CLASSES):
        raise ValueError(f"Некорректные вероятности: {name}")
    probabilities = [number(value) for value in values]
    best = int(np.argmax(probabilities))
    if (
        any(value < 0 or value > 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)
        or prediction["class_id"] != best
        or prediction["class_name"] != CLASSES[best]
    ):
        raise ValueError(f"Некорректное предсказание: {name}")
    truth = row["label"]
    return truth, best, -math.log(max(probabilities[truth], float(np.finfo(np.float64).tiny)))


def main() -> None:
    cli = parser(__doc__ or "")
    cli.add_argument("--url", default="http://127.0.0.1:8080")
    cli.add_argument("--cache", type=Path, default=Path("data/serving"))
    cli.add_argument("--workers", type=int, default=10)
    for metric, tolerance in TOLERANCES.items():
        cli.add_argument(f"--{metric.replace('_', '-')}-tol", type=float, default=tolerance)
    args = cli.parse_args()
    tolerances = {name: number(getattr(args, name + "_tol")) for name in TOLERANCES}
    if args.workers < 1 or min(tolerances.values()) < 0:
        cli.error("Нужны workers ≥ 1 и неотрицательные допуски")

    store = S3Store.from_env()
    source = Source.read(args.config)
    rows, root, digest = fetch_test_images(store, source, args.cache)
    with TemporaryDirectory() as temporary:
        metrics = Path(temporary) / "metrics.json"
        download(store, source.metrics, metrics)
        reference = mapping(json.loads(metrics.read_text()))
    if (
        reference["manifest_sha256"] != digest
        or reference["data_s3_path"] != source.dataset_prefix
    ):
        raise ValueError("Метрики относятся к другой версии данных")
    expected = mapping(reference["test"])
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    losses: list[float] = []
    with (
        httpx.Client(base_url=args.url, timeout=10.0) as client,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        client.get("/health").raise_for_status()
        for truth, best, loss in pool.map(partial(predict, client, root), rows):
            matrix[truth, best] += 1
            losses.append(loss)
    actual = scores_from_confusion(math.fsum(losses) / len(losses), matrix).to_dict()
    print(f"Test: {len(losses)} изображений\n")
    print("| Метрика | Обучение | Сервис | Абс. разница | Допуск |")
    print("| --- | ---: | ---: | ---: | ---: |")
    passed = True
    for name, value in actual.items():
        before = number(expected[name])
        delta = abs(value - before)
        tolerance = tolerances[name]
        passed &= delta <= tolerance
        print(f"| {name} | {before:.10f} | {value:.10f} | {delta:.3g} | {tolerance:g} |")
    if not passed:
        raise SystemExit("Метрики выходят за пределы допусков")


if __name__ == "__main__":
    main()
