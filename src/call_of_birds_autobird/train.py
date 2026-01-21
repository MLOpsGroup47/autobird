from pathlib import Path

import hydra
from call_of_func.train.train_engine import training
from call_of_func.utils.get_source_path import ROOT, CONFIG_DIR
from dotenv import load_dotenv
from omegaconf import DictConfig

# Load .env from project root to utilize wandb
load_dotenv(dotenv_path=ROOT / ".env", override=False)

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def run_train(cfg: DictConfig) -> None:
    training(cfg)

if __name__ == "__main__":
    run_train()

