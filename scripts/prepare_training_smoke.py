"""Создать крошечную отдельную версию только для интеграционной проверки контейнеров."""

import tempfile
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from ml_product_infrastructure.dataset.files import (
    CLASSES,
    ImageRecord,
    annotation,
    sha256_file,
    write_json,
)
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import load_run
from ml_product_infrastructure.experiments.grid import override


def main() -> None:
    store = S3Store.from_env()
    token = uuid.uuid4().hex[:12]
    prefix = f"processed/cifar10/ci-smoke-{token}"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "images").mkdir()
        counts = {"train": 30, "val": 10, "test": 10}
        for split, size in counts.items():
            rows: list[ImageRecord] = []
            for index in range(size):
                label = index % 10
                # Непересекающиеся train/val ID; исходный test сохраняет test_....
                image_id = (
                    f"train_{index + (30 if split == 'val' else 0):05d}"
                    if split != "test"
                    else f"test_{index:05d}"
                )
                name = f"images/{image_id}.png"
                pixels = np.full((32, 32, 3), label * 20, dtype=np.uint8)
                Image.fromarray(pixels).save(root / name)
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": name,
                        "label": label,
                        "label_name": CLASSES[label],
                        "sha256": sha256_file(root / name),
                    }
                )
            write_json(root / f"annotations/{split}.json", annotation(rows))
        write_json(
            root / "stats.json",
            {
                "mean": [0.3, 0.4, 0.5],
                "std": [0.2, 0.3, 0.4],
                "scale": "uint8 / 255.0",
                "train_annotation_sha256": sha256_file(root / "annotations/train.json"),
            },
        )
        files = [
            "annotations/train.json",
            "annotations/val.json",
            "annotations/test.json",
            "stats.json",
        ]
        write_json(
            root / "manifest.json",
            {
                "counts": counts,
                "files": [
                    {
                        "path": name,
                        "sha256": sha256_file(root / name),
                        "size_bytes": (root / name).stat().st_size,
                    }
                    for name in files
                ],
            },
        )
        for path in sorted(root.rglob("*")):
            if path.is_file():
                store.put_new(
                    path,
                    prefix + "/" + path.relative_to(root).as_posix(),
                    "image/png" if path.suffix == ".png" else "application/json",
                )
    base = load_run(Path("configs/train.json"))
    config = override(
        base.to_dict(),
        {
            "experiment_name": "cifar10-ci",
            "run_name": f"smoke-{token}",
            "data.s3_path": prefix,
            "data.max_train_samples": 20,
            "train.epochs": 2,
            "train.batch_size": 10,
            "train.threads": 1,
            "model.channels": [4, 8],
            "model.hidden": 8,
        },
    )
    write_json(Path(".tools/ci-smoke.json"), config)
    print(f"Крошечная CI-версия готова: {prefix}; конфиг .tools/ci-smoke.json")


if __name__ == "__main__":
    main()
