"""Загрузка проверенной версии S3 в общий кеш контейнеров и RGB-тензоры."""

import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import TensorDataset

from ml_product_infrastructure.dataset.files import (
    CLASSES,
    ImageRecord,
    read_annotation,
    sha256_file,
)
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import RunConfig, integer, mapping, number, text


def training_subset(rows: list[ImageRecord], limit: int | None, seed: int) -> list[ImageRecord]:
    if limit is None or limit == len(rows):
        return rows.copy()
    if not 10 <= limit < len(rows):
        raise ValueError("max_train_samples должен быть от 10 до размера train или null")
    groups = [[row for row in rows if row["label"] == label] for label in range(10)]
    quotas = [len(group) * limit // len(rows) for group in groups]
    order = sorted(range(10), key=lambda i: (-(len(groups[i]) * limit % len(rows)), i))
    for i in order[: limit - sum(quotas)]:
        quotas[i] += 1
    generator = random.Random(seed)
    result: list[ImageRecord] = []
    for group, quota in zip(groups, quotas, strict=True):
        generator.shuffle(group)
        result.extend(group[:quota])
    return sorted(result, key=lambda row: row["image_id"])


def fetch_version(store: S3Store, prefix: str, cache: Path) -> tuple[Path, str]:
    prefix = prefix.rstrip("/") + "/"
    with store.client.get_object(Bucket=store.bucket, Key=prefix + "manifest.json")[
        "Body"
    ] as stream:
        manifest_bytes = stream.read()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = mapping(json.loads(manifest_bytes))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("В manifest отсутствует files")
    expected = {f"annotations/{name}.json" for name in ("train", "val", "test")} | {"stats.json"}
    if {text(mapping(entry).get("path")) for entry in entries} != expected:
        raise ValueError("Некорректные пути в manifest")
    root = cache / store.bucket / prefix.rstrip("/")
    root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        info = mapping(entry)
        name, expected_hash = text(info["path"]), text(info["sha256"])
        path = root / name
        if not path.exists() or sha256_file(path) != expected_hash:
            store.download(prefix + name, path)
        if sha256_file(path) != expected_hash or path.stat().st_size != integer(
            info["size_bytes"]
        ):
            raise ValueError(f"Метаданные S3 не прошли проверку: {name}")
    rows: list[ImageRecord] = []
    seen: set[str] = set()
    counts = mapping(manifest.get("counts"))
    for name in ("train", "val", "test"):
        split = read_annotation(root / f"annotations/{name}.json")["images"]
        ids = {row["image_id"] for row in split}
        if seen & ids or len(split) != integer(counts.get(name)):
            raise ValueError(f"Повторяющиеся ID или неверный размер {name}")
        seen.update(ids)
        rows.extend(split)

    def download_one(row: ImageRecord) -> None:
        path = root / row["file_name"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            store.download(prefix + row["file_name"], path)
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"PNG S3 не прошёл проверку: {row['image_id']}")

    with ThreadPoolExecutor(16) as pool:
        for index, _ in enumerate(pool.map(download_one, rows), start=1):
            if index % 20_000 == 0:
                print(f"Кеш данных проверен: {index}/{len(rows)}", flush=True)
    (root / "manifest.json").write_bytes(manifest_bytes)
    return root, digest


@dataclass
class DatasetBundle:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    mean: Tensor
    std: Tensor
    train_ids: list[str]
    manifest_sha256: str


def tensor_dataset(root: Path, rows: list[ImageRecord]) -> TensorDataset:
    def load(row: ImageRecord) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
        with Image.open(root / row["file_name"]) as image:
            if image.mode != "RGB" or image.size != (32, 32):
                raise ValueError(f"Ожидался RGB PNG 32×32: {row['image_id']}")
            return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()

    with ThreadPoolExecutor(8) as pool:
        images = np.stack(list(pool.map(load, rows)))
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    return TensorDataset(torch.from_numpy(images), labels)


def prepare_data(config: RunConfig) -> DatasetBundle:
    root, digest = fetch_version(
        S3Store.from_env(), config.data.s3_path, Path(os.environ["TRAIN_CACHE_DIR"])
    )
    stats = mapping(json.loads((root / "stats.json").read_text()))
    if stats.get("train_annotation_sha256") != sha256_file(root / "annotations/train.json"):
        raise ValueError("stats.json относится к другой train-разметке")

    def channel_tensor(name: str) -> Tensor:
        values = stats.get(name)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"В stats.json требуется RGB {name}")
        return torch.tensor([number(v) for v in values], dtype=torch.float32).view(1, 3, 1, 1)

    mean, std = channel_tensor("mean"), channel_tensor("std")
    if bool(torch.any(std <= 0)) or stats.get("scale") != "uint8 / 255.0":
        raise ValueError("Неподдерживаемые статистики нормализации")
    train = training_subset(
        read_annotation(root / "annotations/train.json")["images"],
        config.data.max_train_samples,
        config.seed,
    )
    val = read_annotation(root / "annotations/val.json")["images"]
    test = read_annotation(root / "annotations/test.json")["images"]
    if read_annotation(root / "annotations/train.json")["classes"] != CLASSES:
        raise ValueError("Ожидались классы CIFAR-10")
    return DatasetBundle(
        tensor_dataset(root, train),
        tensor_dataset(root, val),
        tensor_dataset(root, test),
        mean,
        std,
        [row["image_id"] for row in train],
        digest,
    )
