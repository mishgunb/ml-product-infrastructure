"""Все независимые шаги обработки одной командой."""

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.convert import convert
from ml_product_infrastructure.dataset.download import download
from ml_product_infrastructure.dataset.publish import git_commit, publish
from ml_product_infrastructure.dataset.s3 import S3Store
from ml_product_infrastructure.dataset.split import split
from ml_product_infrastructure.dataset.stats import stats


def pipeline(config: PipelineConfig) -> None:
    # Повторная версия отклоняется до скачивания и дорогой обработки.
    S3Store.from_env().assert_version_absent(config.prefix)
    git_commit()
    for step in (download, convert, split, stats, publish):
        print(f"\n=== {step.__name__} ===", flush=True)
        step(config)


if __name__ == "__main__":
    run_cli(
        pipeline, "Конвейер CIFAR-10: скачать из S3 → PNG/JSON → split → stats → новая версия S3"
    )
