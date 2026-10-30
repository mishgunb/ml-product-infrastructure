"""Контракт настоящего RustFS; запускается в CI и через make test-s3."""

import os
import uuid
from pathlib import Path

import pytest

from ml_product_infrastructure.dataset.s3 import ObjectExistsError, S3Store

pytestmark = pytest.mark.skipif(os.environ.get("RUN_S3_TESTS") != "1", reason="нужен make s3-up")


def test_real_s3_preserves_conditional_objects_and_bucket(tmp_path: Path) -> None:
    configured = S3Store.from_env()
    bucket = f"cifar-integration-{uuid.uuid4().hex[:12]}"
    store = S3Store(configured.client, bucket)
    store.client.create_bucket(Bucket=bucket)
    key = "raw/immutable.bin"
    first = tmp_path / "first"
    first.write_bytes(b"original raw bytes")
    second = tmp_path / "second"
    second.write_bytes(b"replacement must be rejected")
    try:
        store.put_new(first, key, "application/octet-stream")
        with pytest.raises(ObjectExistsError):
            store.put_new(second, key, "application/octet-stream")
        store.download(key, tmp_path / "downloaded")
        assert (tmp_path / "downloaded").read_bytes() == first.read_bytes()
        store.assert_version_absent("processed/cifar10/v1/")
        store.put_bytes_new(b"{}", "_locks/processed/cifar10/v1.json")
        with pytest.raises(ObjectExistsError):
            store.assert_version_absent("processed/cifar10/v1/")
        with pytest.raises(ObjectExistsError):
            store.put_bytes_new(b"{}", "_locks/processed/cifar10/v1.json")
    finally:
        # Удаляется только отдельный временный бакет этого теста.
        for obj in store.client.list_objects_v2(Bucket=bucket).get("Contents", []):
            store.client.delete_object(Bucket=bucket, Key=obj["Key"])
        store.client.delete_bucket(Bucket=bucket)
