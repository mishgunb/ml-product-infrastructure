"""Строго проверяемый JSON-конфиг одного запуска."""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Ожидался JSON-объект со строковыми ключами")
    return cast(dict[str, object], value)


def text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Ожидалась непустая строка")
    return value


def integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Ожидалось целое число")
    return value


def number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Ожидалось конечное число")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise ValueError("Ожидалось конечное число")
    return result


def exact_keys(data: dict[str, object], keys: set[str]) -> None:
    if set(data) != keys:
        raise ValueError(f"Ключи конфига: ожидались {sorted(keys)}, получены {sorted(data)}")


@dataclass(frozen=True)
class DataConfig:
    s3_path: str
    max_train_samples: int | None


@dataclass(frozen=True)
class ModelConfig:
    type: str
    channels: tuple[int, ...]
    hidden: int


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    threads: int


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    run_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "RunConfig":
        root = mapping(value)
        exact_keys(root, {"experiment_name", "run_name", "seed", "data", "model", "train"})
        data, model, train = (mapping(root[key]) for key in ("data", "model", "train"))
        exact_keys(data, {"s3_path", "max_train_samples"})
        exact_keys(model, {"type", "channels", "hidden"})
        exact_keys(train, {"epochs", "batch_size", "lr", "weight_decay", "threads"})
        channels = model["channels"]
        if not isinstance(channels, (list, tuple)):
            raise ValueError("model.channels должен быть списком")
        limit = data["max_train_samples"]
        config = cls(
            experiment_name=text(root["experiment_name"]),
            run_name=text(root["run_name"]),
            seed=integer(root["seed"]),
            data=DataConfig(
                text(data["s3_path"]).rstrip("/"), None if limit is None else integer(limit)
            ),
            model=ModelConfig(
                text(model["type"]), tuple(integer(c) for c in channels), integer(model["hidden"])
            ),
            train=TrainConfig(
                integer(train["epochs"]),
                integer(train["batch_size"]),
                number(train["lr"]),
                number(train["weight_decay"]),
                integer(train["threads"]),
            ),
        )
        for name in (config.experiment_name, config.run_name):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,179}", name):
                raise ValueError(
                    "Имена эксперимента и запуска: буквы, цифры, точки, дефисы и подчёркивания"
                )
        if not re.fullmatch(
            r"processed/cifar10/[A-Za-z0-9][A-Za-z0-9._-]{0,63}", config.data.s3_path
        ):
            raise ValueError("data.s3_path должен указывать на версию processed/cifar10/...")
        if not 0 <= config.seed < 2**32:
            raise ValueError("seed вне диапазона [0, 2**32)")
        if limit is not None and integer(limit) < 10:
            raise ValueError("max_train_samples должен быть не меньше 10 или null")
        if config.model.type not in {"linear", "simple_cnn"}:
            raise ValueError("Модель должна быть linear или simple_cnn")
        if not 2 <= len(config.model.channels) <= 3 or any(
            not 1 <= c <= 256 for c in config.model.channels
        ):
            raise ValueError("Нужны 2–3 положительных размера channels, не более 256")
        if not 1 <= config.model.hidden <= 4096:
            raise ValueError("model.hidden вне диапазона")
        if not 1 <= config.train.epochs <= 1000 or not 1 <= config.train.batch_size <= 4096:
            raise ValueError("Некорректные epochs/batch_size")
        if not 0 < config.train.lr <= 1 or config.train.weight_decay < 0:
            raise ValueError("Некорректные lr/weight_decay")
        if not 1 <= config.train.threads <= 16:
            raise ValueError("train.threads должен быть от 1 до 16")
        return config


def load_run(path: Path) -> RunConfig:
    return RunConfig.from_dict(json.loads(path.read_text()))


def flatten_config(value: dict[str, object], prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten_config(mapping(item), name))
        else:
            result[name] = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
    return result
