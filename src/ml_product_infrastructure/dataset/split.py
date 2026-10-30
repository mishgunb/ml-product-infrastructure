"""Детерминированное стратифицированное разбиение исходного train."""

import random

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import (
    CLASSES,
    ImageRecord,
    annotation,
    read_annotation,
    write_json,
)


def stratified_split(
    records: list[ImageRecord],
    val_size: int,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    groups = [[row for row in records if row["label"] == label] for label in range(len(CLASSES))]
    total = len(records)
    if total == 0 or not len(CLASSES) <= val_size <= total - len(CLASSES):
        raise ValueError("Разбиение должно оставлять каждый класс и в train, и в val")
    quotas = [len(group) * val_size // total for group in groups]
    remainder = val_size - sum(quotas)
    priority = sorted(
        range(len(groups)), key=lambda label: (-(len(groups[label]) * val_size % total), label)
    )
    for label in priority[:remainder]:
        quotas[label] += 1
    if any(not 0 < quota < len(group) for quota, group in zip(quotas, groups, strict=True)):
        raise ValueError("Для этого размера val невозможно сохранить все классы в обеих выборках")
    generator = random.Random(seed)
    train: list[ImageRecord] = []
    val: list[ImageRecord] = []
    for group, quota in zip(groups, quotas, strict=True):
        group.sort(key=lambda row: row["image_id"])
        generator.shuffle(group)
        val.extend(group[:quota])
        train.extend(group[quota:])
    train.sort(key=lambda row: row["image_id"])
    val.sort(key=lambda row: row["image_id"])
    return train, val


def split(config: PipelineConfig) -> None:
    root = config.output_dir / "annotations"
    original = read_annotation(root / "_train_full.json")["images"]
    train, val = stratified_split(original, config.val_size, config.seed)
    write_json(root / "train.json", annotation(train))
    write_json(root / "val.json", annotation(val))
    # test.json и PNG не меняются; val сохраняет идентификаторы train_....
    print(f"Разбиение seed={config.seed}: train={len(train)}, val={len(val)}; test не изменён")


if __name__ == "__main__":
    run_cli(split, "Выделить стратифицированную валидацию, сохранив test и имена PNG")
