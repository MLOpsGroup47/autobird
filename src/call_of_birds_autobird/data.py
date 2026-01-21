from pathlib import Path
from typing import List

import hydra
import typer
from call_of_func.data.processing import preprocess_cfg
from call_of_func.utils.get_configs import _load_cfg
from call_of_func.utils.get_source_path import CONFIG_DIR
from omegaconf import DictConfig

app = typer.Typer()

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
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