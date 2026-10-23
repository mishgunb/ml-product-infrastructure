"""Скрипт для проверки линтера и типизации: make demo."""

from ml_product_infrastructure.demo import accuracy


def evaluate(predictions: list[int]) -> float:
    """Оценить четыре предсказания на демонстрационных ответах."""
    return accuracy([0, 1, 1, 0], predictions)


if __name__ == "__main__":
    print(f"Accuracy: {evaluate([0, 1, 0, 0]):.2%}")
