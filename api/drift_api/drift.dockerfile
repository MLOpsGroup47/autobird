FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY api/ ./api
COPY src/ ./src
COPY models/ ./models
COPY configs/ configs/
COPY README.md README.md

RUN uv sync --locked --no-cache --no-install-project

CMD exec uv run uvicorn api.app.drift:app --port $PORT --host 0.0.0.0 --workers 1
