"""HTTP API для классификации изображений через Triton."""

import io
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import httpx
import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

from ml_product_infrastructure.dataset.files import CLASSES
from ml_product_infrastructure.experiments.config import mapping, number

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4096 * 4096


def normalization(path: Path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    values = mapping(json.loads(path.read_text()))
    channels: list[npt.NDArray[np.float32]] = []
    for name in ("mean", "std"):
        array = values.get(name)
        if not isinstance(array, list) or len(array) != 3:
            raise ValueError(f"В stats.json требуется RGB {name}")
        channels.append(np.asarray([number(value) for value in array], dtype=np.float32))
    mean, std = channels
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Некорректные параметры нормализации")
    if values.get("scale") != "uint8 / 255.0":
        raise ValueError("Некорректный масштаб в stats.json")
    return mean[:, None, None], std[:, None, None]


def preprocess(
    data: bytes, mean: npt.NDArray[np.float32], std: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    with Image.open(io.BytesIO(data), formats=("PNG", "JPEG")) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError("Размер изображения превышает 16 777 216 пикселей")
        image = source.convert("RGB")
        if image.size != (32, 32):
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    return (pixels / np.float32(255) - mean) / std


@dataclass(frozen=True)
class Serving:
    client: httpx.AsyncClient
    model_path: str
    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        model = os.environ.get("TRITON_MODEL_NAME", "cifar_cnn")
        version = os.environ.get("TRITON_MODEL_VERSION", "1")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model) or not version.isdecimal():
            raise ValueError("Некорректное имя или версия модели Triton")
        mean, std = normalization(
            Path(os.environ.get("STATS_PATH", "artifacts/serving/stats.json"))
        )
        async with httpx.AsyncClient(
            base_url=os.environ.get("TRITON_URL", "http://127.0.0.1:8000"), timeout=5.0
        ) as connection:
            api.state.serving = Serving(
                connection, f"/v2/models/{model}/versions/{version}", mean, std
            )
            yield

    api = FastAPI(lifespan=lifespan)

    @api.get("/health")
    async def health() -> dict[str, str]:
        serving = cast(Serving, api.state.serving)
        try:
            for route in ("/v2/health/ready", f"{serving.model_path}/ready"):
                (await serving.client.get(route)).raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Triton или версия модели недоступны") from exc
        return {"status": "ready"}

    @api.post("/predict")
    async def predict(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        data = await file.read(MAX_BYTES + 1)
        if not data:
            raise HTTPException(400, "Файл пуст")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Файл превышает 5 МиБ")
        serving = cast(Serving, api.state.serving)
        try:
            inputs = await run_in_threadpool(preprocess, data, serving.mean, serving.std)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise HTTPException(
                422, "Ожидается читаемое PNG или JPEG допустимого размера"
            ) from exc
        # Triton HTTP binary extension: FP32 tensor follows the JSON header in wire order.
        binary = inputs.astype("<f4", copy=False).tobytes()
        header = json.dumps(
            {
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1, 3, 32, 32],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": len(binary)},
                    }
                ],
                "outputs": [{"name": "logits", "parameters": {"binary_data": False}}],
            }
        ).encode()
        try:
            response = await serving.client.post(
                f"{serving.model_path}/infer",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Inference-Header-Content-Length": str(len(header)),
                },
                content=header + binary,
            )
            payload = response.raise_for_status().json()
            if not isinstance(payload["model_name"], str) or not isinstance(
                payload["model_version"], str
            ):
                raise ValueError("Некорректные метаданные модели")
            logits = np.asarray(payload["outputs"][0]["data"], dtype=np.float64)
            if logits.shape != (len(CLASSES),) or not np.isfinite(logits).all():
                raise ValueError("Некорректный ответ модели")
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise HTTPException(502, "Не удалось получить предсказание от Triton") from exc
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        index = int(probabilities.argmax())
        return {
            "class_id": index,
            "class_name": CLASSES[index],
            "probabilities": probabilities.tolist(),
            "model_name": payload["model_name"],
            "model_version": payload["model_version"],
        }

    return api


app = create_app()
