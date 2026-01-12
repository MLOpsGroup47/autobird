# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY tasks.py tasks.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/


WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/call_of_birds_autobird/train.py"]
