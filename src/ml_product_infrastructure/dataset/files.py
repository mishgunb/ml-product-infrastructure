"""Хеши, стабильный JSON и проверка схемы разметки."""

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict, cast

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ImageRecord(TypedDict):
    image_id: str
    file_name: str
    label: int
    label_name: str
    sha256: str


class Annotation(TypedDict):
    classes: list[str]
    images: list[ImageRecord]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 архива {path} не совпадает: {actual}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_annotation(path: Path) -> Annotation:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("classes") != CLASSES:
        raise ValueError(f"Неверный список классов в {path}")
    rows: object = value.get("images")
    if not isinstance(rows, list):
        raise ValueError(f"Отсутствует список images в {path}")
    seen: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError(f"Некорректная запись в {path}")
        image_id, label = item.get("image_id"), item.get("label")
        if not isinstance(image_id, str) or not re.fullmatch(r"(train|test)_\d{5}", image_id):
            raise ValueError(f"Некорректный image_id в {path}")
        if image_id in seen:
            raise ValueError(f"Повторяющийся image_id {image_id}")
        seen.add(image_id)
        if type(label) is not int or not 0 <= label < len(CLASSES):
            raise ValueError(f"Некорректная метка {image_id}")
        if item.get("label_name") != CLASSES[label]:
            raise ValueError(f"Имя класса не соответствует метке {image_id}")
        if item.get("file_name") != f"images/{image_id}.png":
            raise ValueError(f"Неверный путь PNG для {image_id}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Некорректный SHA-256 для {image_id}")
    return cast(Annotation, value)


def annotation(rows: list[ImageRecord]) -> Annotation:
    return {"classes": CLASSES.copy(), "images": rows}
