from pathlib import Path

import hydra
from omegaconf import DictConfig

from call_of_func.train.train_engine import train_from_cfg

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = (ROOT / "configs").as_posix()

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig) -> None:
    train_from_cfg(cfg)

if __name__ == "__main__":
    train()