"""Линейный baseline и небольшая CNN, без загрузки предобученных весов."""

from typing import cast

from torch import Tensor, nn

from ml_product_infrastructure.experiments.config import ModelConfig


class Classifier(nn.Module):
    def __init__(self, config: ModelConfig, classes: int = 10) -> None:
        super().__init__()
        if config.type == "linear":
            self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, classes))
            return
        layers: list[nn.Module] = []
        previous, side = 3, 32
        for channel in config.channels:
            layers.extend([nn.Conv2d(previous, channel, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            previous, side = channel, side // 2
        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(previous * side * side, config.hidden),
                nn.ReLU(),
                nn.Linear(config.hidden, classes),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.network(inputs))
