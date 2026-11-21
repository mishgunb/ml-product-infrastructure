"""Прокси демо сохраняет классы данных и показывает ошибки вместо зависания."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ml_product_infrastructure.dataset.files import CLASSES
from ml_product_infrastructure.demo_ui.app import create_app


@pytest.mark.parametrize("scenario", ["success", "offline", "timeout", "bad_file", "bad_response"])
def test_demo_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    examples = [{"id": f"test_{index:05d}", "class_name": CLASSES[index]} for index in range(3)]
    (tmp_path / "assets.json").write_text(json.dumps({"classes": CLASSES, "examples": examples}))
    for item in examples:
        (tmp_path / f"{item['id']}.png").write_bytes(b"example")
    monkeypatch.setenv("DEMO_ASSETS_PATH", str(tmp_path))
    monkeypatch.setenv("PREDICTION_API_URL", "http://wrapper.test")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://wrapper.test/predict"
        assert request.method == "POST"
        assert "multipart/form-data" in request.headers["content-type"]
        assert b"image-content" in request.content
        if scenario == "offline":
            raise httpx.ConnectError("internal transport details", request=request)
        if scenario == "timeout":
            raise httpx.ReadTimeout("internal timeout details", request=request)
        if scenario == "bad_file":
            return httpx.Response(422, json={"detail": "image decoder details"})
        if scenario == "bad_response":
            return httpx.Response(200, content=b"not valid JSON")
        return httpx.Response(
            200,
            json={
                "class_id": 0,
                "class_name": CLASSES[0],
                "probabilities": [1.0] + [0.0] * 9,
                "model_name": "actual_model",
                "model_version": "7",
            },
        )

    original_client = httpx.AsyncClient

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return original_client(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    with TestClient(create_app()) as browser:
        assert browser.get("/health").status_code == 200
        assert browser.get("/api/examples").json()["classes"] == CLASSES
        assert browser.get("/api/examples/unknown/image").status_code == 404
        response = browser.post(
            "/api/predict", files={"file": ("test.png", b"image-content", "image/png")}
        )
        expected_status = {
            "success": 200,
            "offline": 502,
            "timeout": 504,
            "bad_file": 422,
            "bad_response": 502,
        }[scenario]
        assert response.status_code == expected_status
        payload = response.json()
        if scenario == "success":
            assert payload["classes"] == CLASSES
            assert payload["model_name"] == "actual_model"
            assert payload["model_version"] == "7"
        else:
            assert payload["detail"]
            assert "internal" not in payload["detail"]
            assert "decoder" not in payload["detail"]
        assert browser.get("/health").status_code == 200
