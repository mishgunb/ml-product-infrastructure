"""S3-источники для экспорта и проверки сервиса."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ml_product_infrastructure.dataset.files import ImageRecord, read_annotation, sha256_file
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.experiments.config import mapping, text


@dataclass(frozen=True)
class Source:
    checkpoint: str
    dataset: str

    @classmethod
    def read(cls, path: Path) -> "Source":
        value = mapping(json.loads(path.read_text()))
        return cls(text(value["checkpoint"]), text(value["dataset"]).rstrip("/"))

    @property
    def metrics(self) -> str:
        return self.checkpoint.rsplit("/", 1)[0] + "/metrics.json"

    @property
    def dataset_prefix(self) -> str:
        return urlparse(self.dataset).path.strip("/")


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, default=Path("configs/serving.json"))
    return result


def locate(store: S3Store, uri: str) -> tuple[S3Store, str]:
    url = urlparse(uri)
    if url.scheme != "s3" or not url.netloc or not url.path.strip("/"):
        raise ValueError(f"Требуется путь s3://bucket/key: {uri}")
    return S3Store(store.client, url.netloc), url.path.lstrip("/")


def download(store: S3Store, uri: str, path: Path) -> None:
    bucket, key = locate(store, uri)
    bucket.download(key, path)


def manifest(store: S3Store, source: Source) -> tuple[dict[str, str], str]:
    bucket, key = locate(store, source.dataset + "/manifest.json")
    with bucket.client.get_object(Bucket=bucket.bucket, Key=key)["Body"] as body:
        raw = body.read()
    items = mapping(json.loads(raw))["files"]
    if not isinstance(items, list):
        raise ValueError("В manifest отсутствует files")
    hashes = {text(mapping(item)["path"]): text(mapping(item)["sha256"]) for item in items}
    return hashes, hashlib.sha256(raw).hexdigest()


def checked_download(store: S3Store, uri: str, path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        download(store, uri, path)
    if sha256_file(path) != digest:
        raise ValueError(f"Неверная контрольная сумма: {uri}")


def fetch_test_images(
    store: S3Store, source: Source, cache: Path
) -> tuple[list[ImageRecord], Path, str]:
    hashes, digest = manifest(store, source)
    root = cache / digest
    annotation = "annotations/test.json"
    checked_download(
        store, f"{source.dataset}/{annotation}", root / annotation, hashes[annotation]
    )
    rows = read_annotation(root / annotation)["images"]

    def fetch(row: ImageRecord) -> None:
        name = row["file_name"]
        checked_download(store, f"{source.dataset}/{name}", root / name, row["sha256"])

    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(fetch, rows):
            pass
    return rows, root, digest
