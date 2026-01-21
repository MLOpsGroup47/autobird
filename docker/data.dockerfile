FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src

RUN uv sync --frozen --no-dev --no-install-project

ENTRYPOINT ["uv", "run", "python", "-u", "src/call_of_birds_autobird/data.py"]

