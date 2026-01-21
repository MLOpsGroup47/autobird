FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /api/app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY uv.lock .
COPY pyproject.toml .
COPY README.md .
COPY tasks.py .
COPY src/ ./src


RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/call_of_birds_autobird/train.py"]