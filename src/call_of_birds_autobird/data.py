import typer
import hydra

from pathlib import Path
from typing import List
from omegaconf import DictConfig

from call_of_func.data.processing import preprocess_cfg
from call_of_func.utils.get_configs import _load_cfg

app = typer.Typer()
config_dir = (Path(__file__).resolve().parents[2] / "configs").as_posix()

@hydra.main(version_base=None, config_path=config_dir, config_name="config")
def preprocess_hydra(cfg: DictConfig) -> None:
    preprocess_cfg(cfg)

### Main preprocessing pipeline
@app.command()
def preprocess_typer(
    config: str = typer.Option("config", help="Hydra config name"),
    o: List[str] = typer.Option(None, "--o", help="Hydra overrides"),
    raw_dir: Path = typer.Option(None, help="Override raw_dir over config"),
    processed_dir: Path = typer.Option(None, help="Override processed_dir"),
    renamed_files: bool = typer.Option(False, help="Rename dir/files to remove spaces"),
) -> None:
    cfg = _load_cfg(config_name=config, overrides=o or [])
    preprocess_cfg(
        cfg=cfg,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        renamed_files=renamed_files,
    )


if __name__ == "__main__":
    app()