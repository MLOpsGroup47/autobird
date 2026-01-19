from pathlib import Path

import hydra
from call_of_func.train.train_engine import train_from_cfg
from omegaconf import DictConfig
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = (ROOT / "configs").as_posix()

# Load .env from project root to utilize wandb
load_dotenv(dotenv_path=ROOT / ".env", override=False)

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def train(cfg: DictConfig) -> None:
    train_from_cfg(cfg)

if __name__ == "__main__":
    train()