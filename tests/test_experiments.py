"""Защита выбора по val, лучшего checkpoint и воспроизводимого CPU-обучения."""

from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from ml_product_infrastructure.experiments.config import ModelConfig, load_run
from ml_product_infrastructure.experiments.data import training_subset
from ml_product_infrastructure.experiments.engine import BestCheckpoint
from ml_product_infrastructure.experiments.grid import (
    RunResult,
    choose_best,
    expand_grid,
    override,
)
from ml_product_infrastructure.experiments.metrics import scores_from_confusion
from ml_product_infrastructure.experiments.models import Classifier
from tests.test_dataset import record


def test_grid_contains_six_unique_configs_without_mutating_base() -> None:
    base = load_run(Path("configs/train.json"))
    runs = expand_grid(
        base, {"train.lr": [0.0003, 0.001, 0.003], "train.batch_size": [64, 128]}, "unit"
    )
    assert len(runs) == len({r.run_name for r in runs}) == 6
    assert {(r.train.lr, r.train.batch_size) for r in runs} == {
        (lr, bs) for lr in (0.0003, 0.001, 0.003) for bs in (64, 128)
    }
    assert base.train.lr == 0.001 and base.train.batch_size == 128
    assert all(r.data == base.data and r.seed == base.seed for r in runs)
    with pytest.raises(ValueError, match="Неизвестный параметр"):
        override(base.to_dict(), {"train.typo": 3})


def test_selection_uses_val_loss_even_when_test_prefers_other_run() -> None:
    config = load_run(Path("configs/train.json"))
    first = RunResult(config, {"best_val": {"loss": 0.4}, "test": {"loss": 9.0}})
    second = RunResult(config, {"best_val": {"loss": 0.6}, "test": {"loss": 0.01}})
    assert choose_best([second, first]) is first


def test_checkpoint_restores_best_epoch_instead_of_last(tmp_path: Path) -> None:
    model = nn.Linear(1, 1, bias=False)
    best = BestCheckpoint(tmp_path / "model.pt")
    for index, loss in enumerate([0.8, 0.2, 0.7], start=1):
        with torch.no_grad():
            model.weight.fill_(index)
        best.consider(loss, index, model, {"source": "unit"})
    assert best.epoch == 2
    best.restore(model)
    assert float(model.weight.item()) == 2.0
    checkpoint = torch.load(best.path, weights_only=True)
    assert checkpoint["best_epoch"] == 2
    with pytest.raises(ValueError, match="конечным"):
        best.consider(float("nan"), 4, model, {})


def test_macro_f1_counts_all_ten_classes_and_handles_empty_predictions() -> None:
    matrix = np.zeros((10, 10), dtype=np.int64)
    matrix[0, 0], matrix[0, 1], matrix[1, 1] = 1, 1, 2
    scores = scores_from_confusion(0.7, matrix)
    assert scores.accuracy == 0.75
    assert scores.macro_f1 == pytest.approx((2 / 3 + 4 / 5) / 10)


def test_training_subset_is_fixed_and_stratified() -> None:
    rows = [record(i, i % 10) for i in range(100)]
    selected = training_subset(rows, 30, 42)
    assert selected == training_subset(rows, 30, 42)
    assert selected != training_subset(rows, 30, 43)
    assert set(Counter(row["label"] for row in selected).values()) == {3}
    assert len(training_subset(rows, None, 42)) == 100


@pytest.mark.parametrize("kind", ["linear", "simple_cnn"])
def test_model_has_ten_outputs_for_rgb_cifar(kind: str) -> None:
    model = Classifier(ModelConfig(kind, (4, 8), 16))
    output = cast(Tensor, model(torch.zeros((2, 3, 32, 32))))
    assert output.shape == (2, 10)


def test_config_can_change_without_commit_but_runtime_code_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml_product_infrastructure.experiments.grid import repository_head

    changed = ["configs/train.json"]

    def output(command: list[str], text: bool) -> str:
        assert text
        if command[1] == "diff":
            return "\n".join(changed)
        if command[1] == "ls-files":
            return ""
        return "a" * 40

    monkeypatch.setattr(
        "ml_product_infrastructure.experiments.grid.subprocess.check_output", output
    )
    assert repository_head() == "a" * 40
    changed.append("src/ml_product_infrastructure/experiments/train.py")
    with pytest.raises(ValueError, match="Закоммитьте"):
        repository_head()
