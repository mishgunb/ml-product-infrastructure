"""Проверки воспроизводимости, отсутствия утечки и неизменяемых S3-записей."""

import hashlib
import io
import pickle
from collections import Counter
from pathlib import Path

import boto3
import numpy as np
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber
from PIL import Image

from ml_product_infrastructure.dataset.config import PipelineConfig
from ml_product_infrastructure.dataset.convert import CifarUnpickler, convert
from ml_product_infrastructure.dataset.files import (
    CLASSES,
    ImageRecord,
    annotation,
    read_annotation,
    sha256_file,
    write_json,
)
from ml_product_infrastructure.dataset.ingest import ingest
from ml_product_infrastructure.dataset.s3 import ObjectExistsError, S3Store
from ml_product_infrastructure.dataset.split import split, stratified_split
from ml_product_infrastructure.dataset.stats import compute_stats


def record(index: int, label: int, split: str = "train") -> ImageRecord:
    image_id = f"{split}_{index:05d}"
    return {
        "image_id": image_id,
        "file_name": f"images/{image_id}.png",
        "label": label,
        "label_name": CLASSES[label],
        "sha256": "0" * 64,
    }


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        source_url="https://example.invalid/cifar.tar.gz",
        source_sha256="0" * 64,
        raw_key="raw/cifar-10-python.tar.gz",
        raw_path=tmp_path / "raw/archive.tar.gz",
        output_root=tmp_path / "processed",
        version="v1",
        val_size=10,
        seed=42,
        workers=2,
    )


def test_stratified_split_covers_all_ids_without_overlap_and_is_repeatable() -> None:
    rows = [record(i, i % 10) for i in range(100)]
    train, val = stratified_split(rows, 23, 42)
    assert len(train) == 77 and len(val) == 23
    assert {r["image_id"] for r in train}.isdisjoint({r["image_id"] for r in val})
    assert sorted(train + val, key=lambda r: r["image_id"]) == rows
    assert sorted(Counter(r["label"] for r in val).values()) == [2] * 7 + [3] * 3
    assert (train, val) == stratified_split(list(reversed(rows)), 23, 42)
    assert val != stratified_split(rows, 23, 43)[1]


def test_split_writes_identical_json_and_never_changes_test_or_images(
    config: PipelineConfig,
) -> None:
    root = config.output_dir
    write_json(
        root / "annotations/_train_full.json", annotation([record(i, i % 10) for i in range(30)])
    )
    write_json(root / "annotations/test.json", annotation([record(0, 0, "test")]))
    test_before = (root / "annotations/test.json").read_bytes()
    split(config)
    train_before = (root / "annotations/train.json").read_bytes()
    val_before = (root / "annotations/val.json").read_bytes()
    split(config)
    assert train_before == (root / "annotations/train.json").read_bytes()
    assert val_before == (root / "annotations/val.json").read_bytes()
    assert test_before == (root / "annotations/test.json").read_bytes()
    assert all(
        row["image_id"].startswith("train_")
        for row in read_annotation(root / "annotations/val.json")["images"]
    )


def test_rgb_stats_only_use_supplied_training_images(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    rows = [record(0, 0), record(1, 1)]
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(tmp_path / rows[0]["file_name"])
    Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)).save(
        tmp_path / rows[1]["file_name"]
    )
    # Посторонний test не должен влиять ни на mean, ни на std.
    Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)).save(
        tmp_path / "images/test_00000.png"
    )
    stats = compute_stats(tmp_path, rows, workers=2)
    assert stats["mean"] == [0.5, 0.5, 0.5]
    assert stats["std"] == [0.5, 0.5, 0.5]
    assert stats["num_images"] == 2
    assert stats["pixels_per_channel"] == 2048


def test_untrusted_pickle_global_is_rejected_without_execution() -> None:
    with pytest.raises(ValueError, match="Запрещённый pickle global"):
        CifarUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_corrupted_archive_is_rejected_before_deserializing(config: PipelineConfig) -> None:
    config.raw_path.parent.mkdir(parents=True)
    config.raw_path.write_bytes(b"not a trusted archive")
    with pytest.raises(ValueError, match="SHA-256"):
        convert(config)
    assert not config.output_dir.exists()


@pytest.fixture
def store() -> S3Store:
    client = boto3.client(
        "s3",
        endpoint_url="http://unit-test.invalid",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    return S3Store(client, "test-bucket")


def test_ingest_existing_raw_does_not_download_or_upload(
    store: S3Store,
    config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(S3Store, "from_env", lambda: store)

    def forbidden_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("Нельзя скачивать существующий raw повторно")

    monkeypatch.setattr("ml_product_infrastructure.dataset.ingest.urlopen", forbidden_download)
    with Stubber(store.client) as stub:
        stub.add_response("head_object", {}, {"Bucket": store.bucket, "Key": config.raw_key})
        ingest(config)
        stub.assert_no_pending_responses()
    assert not config.raw_path.exists()


def test_conditional_upload_refuses_existing_object(store: S3Store, tmp_path: Path) -> None:
    path = tmp_path / "object"
    path.write_bytes(b"original")
    with Stubber(store.client) as stub:
        stub.add_client_error(
            "put_object",
            service_error_code="PreconditionFailed",
            http_status_code=412,
            expected_params={
                "Bucket": store.bucket,
                "Key": "raw/object",
                "Body": ANY,
                "IfNoneMatch": "*",
                "ContentType": "application/gzip",
                "Metadata": {"sha256": hashlib.sha256(b"original").hexdigest()},
            },
        )
        with pytest.raises(ObjectExistsError):
            store.put_new(path, "raw/object", "application/gzip")


def test_existing_and_partial_versions_are_both_blocked(store: S3Store) -> None:
    with Stubber(store.client) as stub:
        stub.add_response(
            "list_objects_v2",
            {"KeyCount": 1},
            {"Bucket": store.bucket, "Prefix": "processed/cifar10/v1/", "MaxKeys": 1},
        )
        with pytest.raises(ObjectExistsError, match="новую --version"):
            store.assert_version_absent("processed/cifar10/v1/")
    with Stubber(store.client) as stub:
        stub.add_response(
            "list_objects_v2",
            {"KeyCount": 0},
            {"Bucket": store.bucket, "Prefix": "processed/cifar10/v2/", "MaxKeys": 1},
        )
        stub.add_response(
            "head_object", {}, {"Bucket": store.bucket, "Key": "_locks/processed/cifar10/v2.json"}
        )
        with pytest.raises(ObjectExistsError, match="зарезервирована"):
            store.assert_version_absent("processed/cifar10/v2/")


def test_permission_errors_are_not_treated_as_missing_objects(store: S3Store) -> None:
    with Stubber(store.client) as stub:
        stub.add_client_error("head_object", service_error_code="403", http_status_code=403)
        with pytest.raises(ClientError):
            store.exists("raw/object")


def test_annotation_rejects_path_traversal(tmp_path: Path) -> None:
    row = record(0, 0)
    row["file_name"] = "../../outside.png"
    write_json(tmp_path / "annotation.json", annotation([row]))
    with pytest.raises(ValueError, match="путь PNG"):
        read_annotation(tmp_path / "annotation.json")


def test_file_sha256_matches_bytes(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"dataset")
    assert sha256_file(path) == hashlib.sha256(b"dataset").hexdigest()


def test_official_download_resumes_partial_archive_and_keeps_bytes(
    config: PipelineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from urllib.request import Request

    from ml_product_infrastructure.dataset.ingest import download_source

    content = b"trusted original archive bytes"
    config = replace(config, source_sha256=hashlib.sha256(content).hexdigest())
    config.raw_path.parent.mkdir(parents=True)
    temporary = config.raw_path.with_suffix(config.raw_path.suffix + ".download")
    temporary.write_bytes(content[:4])

    class Response(io.BytesIO):
        status = 206
        headers = {
            "Content-Length": str(len(content) - 4),
            "Content-Range": f"bytes 4-{len(content) - 1}/{len(content)}",
        }

        def geturl(self) -> str:
            return config.source_url

    def opener(request: Request, timeout: int) -> Response:
        assert request.get_header("Range") == "bytes=4-"
        assert timeout > 0
        return Response(content[4:])

    monkeypatch.setattr("ml_product_infrastructure.dataset.ingest.urlopen", opener)
    download_source(config)
    assert config.raw_path.read_bytes() == content
    assert not temporary.exists()
