FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
FROM python:3.12.8-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --no-editable
ENTRYPOINT ["/app/.venv/bin/python", "-m", "ml_product_infrastructure.dataset.s3"]
