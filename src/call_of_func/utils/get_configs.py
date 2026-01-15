
from pathlib import Path
from typing import List, Optional, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def _load_cfg(config_name: str = "default", overrides: Optional[List[str]] = None) -> DictConfig:
    """Loads Hydra config from ./configs/data/<config_name>.yaml
    and its defaults tree.
    """
    project_root = Path(__file__).resolve().parents[3]
    config_dir = (project_root / "configs").resolve()

    overrides = overrides or []

    # Hydra needs an absolute config dir
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg