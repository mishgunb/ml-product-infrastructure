"""Среднее и population std RGB по пикселям только обучающей выборки."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import (
    ImageRecord,
    read_annotation,
    sha256_file,
    write_json,
)


def compute_stats(root: Path, records: list[ImageRecord], workers: int) -> dict[str, object]:
    if not records:
        raise ValueError("Нельзя посчитать статистики пустого train")

    def moments(row: ImageRecord) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
        with Image.open(root / row["file_name"]) as image:
            if image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(f"Некорректное RGB изображение: {row['image_id']}")
            pixels = np.asarray(image, dtype=np.uint64)
        return pixels.sum(axis=(0, 1)), (pixels * pixels).sum(axis=(0, 1))

    sums = np.zeros(3, dtype=np.uint64)
    squares = np.zeros(3, dtype=np.uint64)
    with ThreadPoolExecutor(workers) as pool:
        for first, second in pool.map(moments, records):
            sums += first
            squares += second
    count = len(records) * 32 * 32
    mean = sums.astype(np.float64) / (count * 255)
    variance = squares.astype(np.float64) / (count * 255**2) - mean**2
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0)).tolist(),
        "channels": ["R", "G", "B"],
        "scale": "uint8 / 255.0",
        "split": "train",
        "num_images": len(records),
        "pixels_per_channel": count,
        "ddof": 0,
    }


def stats(config: PipelineConfig) -> None:
    root = config.output_dir
    records = read_annotation(root / "annotations/train.json")["images"]
    result = compute_stats(root, records, config.workers)
    result["train_annotation_sha256"] = sha256_file(root / "annotations/train.json")
    write_json(root / "stats.json", result)
    print(f"Статистики только по {len(records)} train-изображениям: {root / 'stats.json'}")


if __name__ == "__main__":
    run_cli(stats, "Посчитать mean/std RGB только по train, в масштабе [0, 1]")
