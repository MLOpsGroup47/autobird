FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL maintainer="Holger Floelyng"

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock .
COPY pyproject.toml .
COPY README.md .
COPY tasks.py .
COPY src/ ./src


RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/call_of_birds_autobird/train.py"]