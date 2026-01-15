import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "call_of_birds_autobird"
PYTHON_VERSION = "3.12"


def _pty() -> bool:
    return not WINDOWS


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


# @task
# def train(ctx: Context) -> None:
#     """Train model."""
#     ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# docker helpers
@task
def docker_build_multi(ctx: Context, entrypoint: str) -> None:
    """Build multi-platform docker image."""
    ctx.run(f"docker build --platform linux/amd64,linux/arm64 -f {entrypoint}.dockerfile . -t {entrypoint}:latest")


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# Git helpers
@task
def git(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")


# uv helpers
@task
def uvp(ctx: Context) -> None:
    """Install lib and add to uv project."""
    ctx.run(f"uv pip install --python {PYTHON_VERSION} --no-deps {PROJECT_NAME}")
    ctx.run("uv add .")


# data and training tasks
@task
def preprocess(ctx, raw_dir="data/voice_of_birds", processed_dir="data/processed"):
    """Preprocess audio into tensors (local, using uv)."""
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.data preprocess {raw_dir} {processed_dir}",
        echo=True,
        pty=_pty(),
    )


@task
def train(ctx, data_dir="data/processed", epochs=5, batch_size=32, lr=1e-3):
    """Train model (local, using uv)."""
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.train "
        f"--data-path {data_dir} --epochs {epochs} --batch-size {batch_size} --lr {lr}",
        echo=True,
        pty=_pty(),
    )


@task
def tests(ctx):
    """Run tests (local, using uv)."""
    ctx.run("uv run coverage run -m pytest tests/")
    ctx.run("uv run coverage report -m")
