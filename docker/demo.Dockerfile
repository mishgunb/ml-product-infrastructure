FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
FROM python:3.12.8-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
RUN uv sync --locked --extra serve --no-dev --no-editable
CMD ["/app/.venv/bin/uvicorn", "ml_product_infrastructure.demo_ui.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
