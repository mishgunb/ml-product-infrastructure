"""Официальный архив → неизменяемый raw-объект в S3."""

import shutil
import time
from http.client import IncompleteRead
from urllib.request import Request, urlopen

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import verify_archive
from ml_product_infrastructure.dataset.s3 import ObjectExistsError, S3Store


def download_source(config: PipelineConfig) -> None:
    """Повторяет прерванные HTTPS-загрузки, используя Range, если сервер его поддерживает."""
    config.raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.raw_path.with_suffix(config.raw_path.suffix + ".download")
    for attempt in range(6):
        offset = temporary.stat().st_size if temporary.exists() else 0
        request = Request(
            config.source_url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        try:
            with urlopen(request, timeout=120) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("Источник перенаправил загрузку на небезопасный HTTP")
                partial = response.status == 206 and offset > 0
                if partial and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Сервер вернул неверный диапазон архива")
                length = response.headers.get("Content-Length")
                start = offset if partial else 0
                with temporary.open("ab" if partial else "wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                if length is not None and temporary.stat().st_size - start != int(length):
                    raise OSError("Скачивание архива прервано до Content-Length")
            try:
                verify_archive(temporary, config.source_sha256)
            except ValueError:
                temporary.unlink(missing_ok=True)
                raise
            temporary.replace(config.raw_path)
            return
        except (OSError, IncompleteRead) as exc:
            if attempt == 5:
                raise OSError(
                    "Архив не скачан за 6 попыток; повторите make ingest для продолжения"
                ) from exc
            print(f"Загрузка прервана, попытка {attempt + 2}/6: {exc}", flush=True)
            time.sleep(2)


def ingest(config: PipelineConfig) -> None:
    store = S3Store.from_env()
    if store.exists(config.raw_key):
        print(f"Raw уже в S3, скачивание и загрузка пропущены: {config.raw_key}")
        return
    if config.raw_path.exists():
        verify_archive(config.raw_path, config.source_sha256)
    else:
        print(f"Скачивание официального архива: {config.source_url}", flush=True)
        download_source(config)
    try:
        store.put_new(config.raw_path, config.raw_key, "application/gzip")
    except ObjectExistsError:
        print("Другой запуск уже загрузил raw-архив; существующий объект сохранён")
        return
    print(f"Raw сохранён без изменений: s3://{store.bucket}/{config.raw_key}")


if __name__ == "__main__":
    run_cli(ingest, "Скачать официальный архив и однократно загрузить его в raw/ S3")
