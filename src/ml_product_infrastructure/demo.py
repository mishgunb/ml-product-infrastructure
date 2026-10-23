"""Небольшой пример расчёта метрики модели."""


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Доля совпадений; списки должны быть непустыми и одинаковой длины."""
    if not y_true:
        raise ValueError("Выборка не должна быть пустой")
    if len(y_true) != len(y_pred):
        raise ValueError("Число предсказаний должно совпадать с числом ответов")
    matches = sum(actual == predicted for actual, predicted in zip(y_true, y_pred, strict=True))
    return matches / len(y_true)


def main() -> None:
    """Показать расчёт accuracy на небольшой выборке."""
    score = accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    print(f"Accuracy: {score:.2%}")
