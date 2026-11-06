"""Метрики из матрицы ошибок, с фиксированными десятью классами."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Scores:
    loss: float
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return {"loss": self.loss, "accuracy": self.accuracy, "macro_f1": self.macro_f1}


def scores_from_confusion(loss: float, matrix: npt.NDArray[np.int64]) -> Scores:
    if matrix.shape != (10, 10) or np.any(matrix < 0) or matrix.sum() == 0:
        raise ValueError("Нужна непустая матрица ошибок 10×10")
    diagonal = matrix.diagonal().astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(10), where=denominator != 0)
    return Scores(
        loss=loss, accuracy=float(diagonal.sum() / matrix.sum()), macro_f1=float(f1.mean())
    )
