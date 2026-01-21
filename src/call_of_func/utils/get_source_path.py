
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = (ROOT / "configs").as_posix()

def get_root() -> Path:
    """Returns the project root path."""
    return ROOT

def get_config_dir() -> str:
    """Returns the absolute path to the configs directory as a string."""
    return CONFIG_DIR