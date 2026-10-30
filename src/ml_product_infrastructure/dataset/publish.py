"""Проверка набора и публикация неизменяемой версии; manifest записывается последним."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import (
    ImageRecord,
    read_annotation,
    sha256_file,
    write_json,
)
from ml_product_infrastructure.dataset.s3 import S3Store


def git_commit() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(
            "Перед публикацией закоммитьте изменения: git-коммит должен точно описывать код"
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def validate_dataset(config: PipelineConfig) -> tuple[dict[str, int], list[ImageRecord]]:
    root = config.output_dir
    counts: dict[str, int] = {}
    records: list[ImageRecord] = []
    all_ids: set[str] = set()
    for split, expected in [
        ("train", 50_000 - config.val_size),
        ("val", config.val_size),
        ("test", 10_000),
    ]:
        rows = read_annotation(root / f"annotations/{split}.json")["images"]
        if len(rows) != expected:
            raise ValueError(f"В {split} ожидалось {expected} изображений, получено {len(rows)}")
        ids = {row["image_id"] for row in rows}
        if ids & all_ids:
            raise ValueError("Выборки содержат общие image_id")
        prefix = "test_" if split == "test" else "train_"
        if any(not item.startswith(prefix) for item in ids):
            raise ValueError(f"В {split} попали изображения другого исходного split")
        all_ids.update(ids)
        counts[split] = len(rows)
        records.extend(rows)
    expected_ids = {f"train_{i:05d}" for i in range(50_000)} | {
        f"test_{i:05d}" for i in range(10_000)
    }
    if all_ids != expected_ids:
        raise ValueError("Набор идентификаторов отличается от исходного CIFAR-10")
    # Проверяем, что параметры в конфиге действительно описывают записанное разбиение.
    from ml_product_infrastructure.dataset.split import stratified_split

    original = read_annotation(root / "annotations/_train_full.json")["images"]
    expected_train, expected_val = stratified_split(original, config.val_size, config.seed)
    for name, expected_rows in [("train", expected_train), ("val", expected_val)]:
        if read_annotation(root / f"annotations/{name}.json")["images"] != expected_rows:
            raise ValueError(f"{name}.json не соответствует seed/val_size; запустите шаг split")
    with ThreadPoolExecutor(config.workers) as pool:
        hashes = pool.map(lambda row: sha256_file(root / row["file_name"]), records)
        for row, digest in zip(records, hashes, strict=True):
            if row["sha256"] != digest:
                raise ValueError(f"PNG изменён: {row['file_name']}")
    stats_value: object = json.loads((root / "stats.json").read_text())
    if not isinstance(stats_value, dict) or stats_value.get("num_images") != counts["train"]:
        raise ValueError("stats.json не соответствует train; повторите шаг stats")
    if stats_value.get("train_annotation_sha256") != sha256_file(root / "annotations/train.json"):
        raise ValueError("stats.json рассчитан по другому разбиению; повторите шаг stats")
    return counts, records


def publish(config: PipelineConfig) -> None:
    store = S3Store.from_env()
    store.assert_version_absent(config.prefix)
    commit = git_commit()
    counts, records = validate_dataset(config)
    root = config.output_dir
    metadata_paths = [f"annotations/{name}.json" for name in ("train", "val", "test")] + [
        "stats.json"
    ]
    manifest = {
        "schema_version": 1,
        "dataset": "cifar10",
        "version": config.version,
        "counts": counts,
        "image_count": len(records),
        "git_commit": commit,
        "source": {
            "url": config.source_url,
            "raw_key": config.raw_key,
            "sha256": config.source_sha256,
        },
        "processing": {"val_size": config.val_size, "seed": config.seed},
        "files": [
            {
                "path": name,
                "size_bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in metadata_paths
        ],
    }
    write_json(root / "manifest.json", manifest)
    reservation = json.dumps({"git_commit": commit, "version": config.version}).encode()
    store.put_bytes_new(reservation, f"_locks/{config.prefix.rstrip('/')}.json")

    def upload_image(row: ImageRecord) -> None:
        store.put_new(root / row["file_name"], config.prefix + row["file_name"], "image/png")

    with ThreadPoolExecutor(config.workers) as pool:
        for number, _ in enumerate(pool.map(upload_image, records), start=1):
            if number % 5000 == 0:
                print(f"Загрузка PNG: {number}/{len(records)}", flush=True)
    for name in metadata_paths:
        store.put_new(root / name, config.prefix + name, "application/json")
    # Только наличие manifest.json означает, что версия полностью опубликована.
    store.put_new(root / "manifest.json", config.prefix + "manifest.json", "application/json")
    print(f"Готово: s3://{store.bucket}/{config.prefix} — {counts}")


if __name__ == "__main__":
    run_cli(publish, "Опубликовать новую версию PNG/JSON; существующая версия не перезаписывается")
