FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

EXPOSE $PORT

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY api/ ./api
COPY src/ ./src
COPY configs/ configs/
COPY README.md README.md

RUN uv sync --locked --no-cache --no-install-project

CMD exec uv run uvicorn api.drift_api.drift:app --port $PORT --host 0.0.0.0 --workers 1
