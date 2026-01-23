# Base Python image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL maintainer="Holger Floelyng"

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY configs/ configs/
COPY src/ src/
COPY tasks.py .



RUN uv sync --locked --no-cache --no-install-project
ENV PYTHONPATH="/app/src"
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["uv", "run", "python", "-m",  "call_of_birds_autobird.train"]