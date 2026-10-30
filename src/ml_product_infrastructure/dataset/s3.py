"""Работа с S3: переменные окружения и условные записи без перезаписи."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import sha256_file


class ObjectExistsError(RuntimeError):
    """Условная запись отклонена: ключ уже занят."""


class S3Store:
    def __init__(self, client: S3Client, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> S3Store:
        load_dotenv(Path(".env"), override=False)

        def required(name: str) -> str:
            value = os.environ.get(name)
            if not value:
                raise ValueError(f"Задайте {name} в окружении или .env (см. .env.example)")
            return value

        client = boto3.client(
            "s3",
            endpoint_url=required("S3_ENDPOINT_URL"),
            aws_access_key_id=required("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=required("S3_SECRET_ACCESS_KEY"),
            region_name=required("S3_REGION"),
            config=Config(
                s3={"addressing_style": "path"},
                retries={"mode": "standard", "max_attempts": 5},
                max_pool_connections=64,
                connect_timeout=10,
                read_timeout=120,
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        return cls(client, required("S3_BUCKET"))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def assert_version_absent(self, prefix: str) -> None:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        if response.get("KeyCount", 0):
            raise ObjectExistsError(f"Версия {prefix} уже существует. Укажите новую --version")
        if self.exists(f"_locks/{prefix.rstrip('/')}.json"):
            raise ObjectExistsError(
                f"Версия {prefix} зарезервирована другой/прерванной загрузкой. "
                "Укажите новую --version"
            )

    def put_new(self, path: Path, key: str, content_type: str) -> None:
        with path.open("rb") as stream:
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=stream,
                    IfNoneMatch="*",
                    ContentType=content_type,
                    Metadata={"sha256": sha256_file(path)},
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] in {"PreconditionFailed", "412"}:
                    raise ObjectExistsError(
                        f"Объект уже существует: s3://{self.bucket}/{key}"
                    ) from exc
                raise

    def put_bytes_new(self, content: bytes, key: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                IfNoneMatch="*",
                ContentType="application/json",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"PreconditionFailed", "412"}:
                raise ObjectExistsError(f"Объект уже существует: {key}") from exc
            raise

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            self.client.download_file(self.bucket, key, str(temporary))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def initialize_bucket(config: PipelineConfig) -> None:
    del config
    store = S3Store.from_env()
    try:
        store.client.head_bucket(Bucket=store.bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        store.client.create_bucket(Bucket=store.bucket)
    print(f"Бакет готов: s3://{store.bucket}")


if __name__ == "__main__":
    run_cli(initialize_bucket, "Автоматическое создание бакета S3")
