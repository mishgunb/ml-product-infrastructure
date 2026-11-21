"""Веб-демо с прокси к HTTP-обёртке ДЗ №4."""

import json
import math
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ml_product_infrastructure.experiments.config import mapping, number, text

MAX_BYTES = 5 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


@dataclass
class Demo:
    client: httpx.AsyncClient
    classes: list[str]
    examples: dict[str, str]
    directory: Path


def read_assets(directory: Path) -> tuple[list[str], dict[str, str]]:
    metadata = mapping(json.loads((directory / "assets.json").read_text()))
    labels, examples = metadata["classes"], metadata["examples"]
    if not isinstance(labels, list) or not isinstance(examples, list):
        raise ValueError("Подготовьте классы и примеры через make demo-prepare")
    classes = [text(label) for label in labels]
    selected = {text(mapping(item)["id"]): text(mapping(item)["class_name"]) for item in examples}
    if len(classes) != 10 or len(set(classes)) != 10 or not 3 <= len(selected) <= 5:
        raise ValueError("Ожидались 10 классов и 3–5 примеров CIFAR-10")
    for image_id, label in selected.items():
        if (
            not re.fullmatch(r"test_\d{5}", image_id)
            or label not in classes
            or not (directory / f"{image_id}.png").is_file()
        ):
            raise ValueError("Некорректные примеры для демо")
    return classes, selected


def validate_prediction(payload: object, classes: list[str]) -> dict[str, object]:
    prediction = mapping(payload)
    values = prediction["probabilities"]
    if not isinstance(values, list) or len(values) != len(classes):
        raise ValueError("Неверный размер вектора вероятностей")
    probabilities = [number(value) for value in values]
    if any(value < 0 or value > 1 for value in probabilities) or not math.isclose(
        sum(probabilities), 1, abs_tol=1e-6
    ):
        raise ValueError("Некорректные вероятности")
    best = max(range(len(classes)), key=probabilities.__getitem__)
    if prediction["class_id"] != best or prediction["class_name"] != classes[best]:
        raise ValueError("Список классов демо не соответствует модели")
    return {
        "class_id": best,
        "class_name": classes[best],
        "probabilities": probabilities,
        "model_name": text(prediction["model_name"]),
        "model_version": text(prediction["model_version"]),
        "classes": classes,
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        directory = Path(os.environ.get("DEMO_ASSETS_PATH", "artifacts/demo"))
        classes, examples = read_assets(directory)
        address = os.environ["PREDICTION_API_URL"]
        async with httpx.AsyncClient(base_url=address, timeout=10.0) as client:
            api.state.demo = Demo(client, classes, examples, directory)
            yield

    api = FastAPI(lifespan=lifespan)
    api.mount("/static", StaticFiles(directory=STATIC), name="static")

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @api.get("/health")
    async def health() -> dict[str, str]:
        # Интерфейс остаётся доступен, чтобы показать ошибку остановленного API.
        return {"status": "ready"}

    @api.get("/api/examples")
    async def examples() -> dict[str, object]:
        demo = cast(Demo, api.state.demo)
        return {
            "classes": demo.classes,
            "examples": [
                {
                    "id": image_id,
                    "class_name": label,
                    "image_url": f"/api/examples/{image_id}/image",
                }
                for image_id, label in demo.examples.items()
            ],
        }

    @api.get("/api/examples/{image_id}/image")
    async def example_image(image_id: str) -> FileResponse:
        demo = cast(Demo, api.state.demo)
        if image_id not in demo.examples:
            raise HTTPException(404, "Пример не найден")
        return FileResponse(demo.directory / f"{image_id}.png", media_type="image/png")

    @api.post("/api/predict")
    async def predict(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        content = await file.read(MAX_BYTES + 1)
        if not content:
            raise HTTPException(400, "Файл пуст. Выберите изображение PNG или JPEG.")
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "Изображение слишком большое. Максимальный размер — 5 МиБ.")
        demo = cast(Demo, api.state.demo)
        try:
            response = await demo.client.post(
                "/predict",
                files={"file": (file.filename or "image", content, file.content_type)},
            )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                504, "Сервис предсказания не ответил за 10 секунд. Попробуйте ещё раз."
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                502, "Сервис предсказания недоступен. Попробуйте ещё раз после его запуска."
            ) from exc
        if 400 <= response.status_code < 500:
            raise HTTPException(
                response.status_code, "Не удалось прочитать изображение. Выберите PNG или JPEG."
            )
        if response.is_error:
            raise HTTPException(502, "Сервис предсказания вернул ошибку. Попробуйте позже.")
        try:
            return validate_prediction(response.json(), demo.classes)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                502, "Сервис вернул некорректный ответ. Попробуйте позже."
            ) from exc

    return api


app = create_app()
