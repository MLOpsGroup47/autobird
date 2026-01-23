FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*



WORKDIR /app

# Copy dependency files from the context
COPY uv.lock pyproject.toml ./


RUN uv sync --locked --no-install-project --no-cache

COPY api/ ./api
COPY src/ ./src
COPY data/processed/ ./data/processed
COPY models/ ./models
COPY configs/ ./configs
COPY README.md README.md



CMD exec uv run uvicorn api.app.inference:app --port $PORT --host 0.0.0.0 --workers 1