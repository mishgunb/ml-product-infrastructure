"""Одна эпоха и выбор чекпоинта строго по validation loss."""

import math
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml_product_infrastructure.experiments.metrics import Scores, scores_from_confusion


class BestCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.loss = math.inf
        self.epoch = 0

    def consider(
        self, loss: float, epoch: int, model: nn.Module, metadata: dict[str, object]
    ) -> bool:
        if not math.isfinite(loss):
            raise ValueError("Validation loss не является конечным числом")
        if loss >= self.loss:
            return False
        self.loss, self.epoch = loss, epoch
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "metadata": metadata, "best_epoch": epoch},
            self.path,
        )
        return True

    def restore(self, model: nn.Module) -> None:
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    mean: Tensor,
    std: Tensor,
    optimizer: Optimizer | None = None,
) -> tuple[Scores, npt.NDArray[np.int64]]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            inputs = (images.to(dtype=torch.float32) / 255.0 - mean) / std
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = cast(Tensor, model(inputs))
            loss = cast(Tensor, criterion(logits, labels))
            if optimizer is not None:
                torch.autograd.backward(loss)
                optimizer.step()
            size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * size
            total += size
            predicted = logits.detach().argmax(dim=1)
            confusion += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    if total == 0:
        raise ValueError("Пустая выборка")
    matrix = cast(npt.NDArray[np.int64], confusion.numpy())
    return scores_from_confusion(total_loss / total, matrix), matrix
