"""Проверенный CIFAR pickle → PNG + JSON без распаковки tar на диск."""

import hashlib
import io
import json
import pickle
import tarfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL import __version__ as pillow_version

from ml_product_infrastructure.dataset.config import PipelineConfig, run_cli
from ml_product_infrastructure.dataset.files import (
    CLASSES,
    ImageRecord,
    annotation,
    verify_archive,
    write_json,
)


class CifarUnpickler(pickle.Unpickler):
    """Допускает только необходимые CIFAR классы numpy, без произвольных глобалов."""

    def find_class(self, module: str, name: str) -> object:
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
        }
        if (module, name) not in allowed:
            raise ValueError(f"Запрещённый pickle global: {module}.{name}")
        return cast(object, super().find_class(module, name))


def read_batch(archive: tarfile.TarFile, name: str) -> tuple[npt.NDArray[np.uint8], list[int]]:
    member = archive.getmember(f"cifar-10-batches-py/{name}")
    if not member.isfile() or member.size > 40_000_000:
        raise ValueError(f"Некорректный участник архива {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Не удалось прочитать {name}")
    with stream:
        value: object = CifarUnpickler(stream, encoding="bytes").load()
    if not isinstance(value, dict):
        raise ValueError(f"Некорректный batch {name}")
    pixels: object = value.get(b"data")
    labels: object = value.get(b"labels")
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
        raise ValueError(f"Ожидался uint8 ndarray в {name}")
    if pixels.shape != (10_000, 3072):
        raise ValueError(f"Некорректная форма данных {name}: {pixels.shape}")
    if not isinstance(labels, list) or len(labels) != 10_000:
        raise ValueError(f"Некорректный список меток {name}")
    if any(type(label) is not int or not 0 <= label < len(CLASSES) for label in labels):
        raise ValueError(f"Метка вне диапазона в {name}")
    data = cast(npt.NDArray[np.uint8], pixels)
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), cast(list[int], labels)


def save_image(
    index: int,
    *,
    root: Path,
    split: str,
    offset: int,
    pixels: npt.NDArray[np.uint8],
    labels: list[int],
) -> ImageRecord:
    image_id = f"{split}_{offset + index:05d}"
    file_name = f"images/{image_id}.png"
    buffer = io.BytesIO()
    Image.fromarray(pixels[index]).save(buffer, format="PNG", compress_level=6)
    content = buffer.getvalue()
    (root / file_name).write_bytes(content)
    label = labels[index]
    return {
        "image_id": image_id,
        "file_name": file_name,
        "label": label,
        "label_name": CLASSES[label],
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def convert(config: PipelineConfig) -> None:
    # Проверяем целиком до любого чтения pickle, даже если PNG уже конвертированы.
    verify_archive(config.raw_path, config.source_sha256)
    root = config.output_dir
    marker = root / "_conversion.json"
    signature = {"source_sha256": config.source_sha256, "pillow": pillow_version, "schema": 1}
    if (
        marker.exists()
        and json.loads(marker.read_text()) == signature
        and (root / "annotations/_train_full.json").exists()
        and (root / "annotations/test.json").exists()
    ):
        print(f"PNG уже сконвертированы: {root}")
        return
    (root / "images").mkdir(parents=True, exist_ok=True)
    train: list[ImageRecord] = []
    test: list[ImageRecord] = []
    batches = [(f"data_batch_{i}", "train", (i - 1) * 10_000) for i in range(1, 6)]
    batches.append(("test_batch", "test", 0))
    with (
        tarfile.open(config.raw_path, "r:gz") as archive,
        ThreadPoolExecutor(config.workers) as pool,
    ):
        for batch_name, split, offset in batches:
            pixels, labels = read_batch(archive, batch_name)

            save = partial(
                save_image, root=root, split=split, offset=offset, pixels=pixels, labels=labels
            )
            records = list(pool.map(save, range(len(labels))))
            (train if split == "train" else test).extend(records)
            print(f"Конвертация: {batch_name}, {len(records)} PNG", flush=True)
    write_json(root / "annotations/_train_full.json", annotation(train))
    write_json(root / "annotations/train.json", annotation(train))
    write_json(root / "annotations/test.json", annotation(test))
    write_json(marker, signature)


if __name__ == "__main__":
    run_cli(convert, "Конвертировать проверенный официальный CIFAR-10 pickle в PNG/JSON")
