"""Сырой объект S3 → data/raw с обязательной проверкой SHA-256."""

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import verify_archive
from ml_product_infrastructure.dataset.s3 import S3Store


def download(config: PipelineConfig) -> None:
    if config.raw_path.exists():
        verify_archive(config.raw_path, config.source_sha256)
        print(f"Локальный raw проверен, повторное скачивание не нужно: {config.raw_path}")
        return
    S3Store.from_env().download(config.raw_key, config.raw_path)
    try:
        verify_archive(config.raw_path, config.source_sha256)
    except ValueError:
        config.raw_path.unlink()
        raise
    print(f"Архив получен из S3 и проверен: {config.raw_path}")


if __name__ == "__main__":
    run_cli(download, "Скачать сырой архив из S3 в data/raw")
