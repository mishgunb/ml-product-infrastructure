"""Автоматически создать отдельный бакет артефактов MLflow."""

import os

from botocore.exceptions import ClientError

from ml_product_infrastructure.dataset.s3 import S3Store


def main() -> None:
    store = S3Store.from_env()
    bucket = os.environ["MLFLOW_BUCKET"]
    if bucket == store.bucket:
        raise ValueError("MLflow и датасеты должны использовать разные бакеты")
    try:
        store.client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        store.client.create_bucket(Bucket=bucket)
    print(f"Бакет артефактов MLflow готов: s3://{bucket}")


if __name__ == "__main__":
    main()
