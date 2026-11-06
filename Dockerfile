FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
FROM python:3.12.8-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --extra train --no-dev --no-editable
ARG GIT_COMMIT
ENV TRAINING_GIT_COMMIT=${GIT_COMMIT}
ENTRYPOINT ["/app/.venv/bin/python", "-m", "ml_product_infrastructure.experiments.train"]
