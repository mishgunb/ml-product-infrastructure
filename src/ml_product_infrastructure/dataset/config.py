"""Конфигурация всех шагов через JSON и аргументы командной строки."""

import argparse
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from botocore.exceptions import BotoCoreError, ClientError


@dataclass(frozen=True)
class PipelineConfig:
    source_url: str
    source_sha256: str
    raw_key: str
    raw_path: Path
    output_root: Path
    version: str
    val_size: int
    seed: int
    workers: int

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version

    @property
    def prefix(self) -> str:
        return f"processed/cifar10/{self.version}/"

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version):
            raise ValueError("Версия должна быть одним именем, например v1 или seed-123")
        if not re.fullmatch(r"[a-f0-9]{64}", self.source_sha256):
            raise ValueError("source_sha256 должен содержать SHA-256 доверенного архива")
        if not self.source_url.startswith("https://"):
            raise ValueError("Источник должен использовать HTTPS")
        if not self.raw_key.startswith("raw/") or ".." in self.raw_key.split("/"):
            raise ValueError("raw_key должен находиться внутри raw/")
        if not 10 <= self.val_size <= 49_990:
            raise ValueError("val_size должен быть от 10 до 49990")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed должен быть от 0 до 2**32 - 1")
        if not 1 <= self.workers <= 64:
            raise ValueError("workers должен быть от 1 до 64")


def load_config(path: Path) -> PipelineConfig:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Конфиг должен быть JSON-объектом")
    data = cast(dict[str, object], value)

    def text(key: str) -> str:
        item = data.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"В конфиге требуется строка {key}")
        return item

    def integer(key: str) -> int:
        item = data.get(key)
        if type(item) is not int:
            raise ValueError(f"В конфиге требуется целое число {key}")
        return item

    config = PipelineConfig(
        source_url=text("source_url"),
        source_sha256=text("source_sha256"),
        raw_key=text("raw_key"),
        raw_path=Path(text("raw_path")),
        output_root=Path(text("output_root")),
        version=text("version"),
        val_size=integer("val_size"),
        seed=integer("seed"),
        workers=integer("workers"),
    )
    config.validate()
    return config


def run_cli(
    action: Callable[[PipelineConfig], None],
    description: str,
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--version")
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        config = replace(
            config,
            version=args.version if args.version is not None else config.version,
            val_size=args.val_size if args.val_size is not None else config.val_size,
            seed=args.seed if args.seed is not None else config.seed,
            workers=args.workers if args.workers is not None else config.workers,
        )
        config.validate()
        action(config)
    except (ValueError, RuntimeError, OSError, BotoCoreError, ClientError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
