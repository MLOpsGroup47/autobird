from dataclasses import dataclass
from pathlib import Path
from typing import Union

@dataclass
class PathConfig:
    root: Union[Path, str]
    raw_dir: Union[Path, str]
    processed_dir: Union[Path, str]
    reports_dir: Union[Path, str]
    eval_dir: Union[Path, str]
    ckpt_dir: Union[Path, str]
    x_train: Union[Path, str]
    y_train: Union[Path, str]
    x_val: Union[Path, str]
    y_val: Union[Path, str]

    def resolve(self) -> "PathConfig":
        # resolve project root first
        root = Path(self.root).expanduser().resolve()

        def r(p: Path) -> Path:
            p = Path(p).expanduser()
            return p if p.is_absolute() else (root / p).resolve()

        self.root = root
        self.raw_dir = r(self.raw_dir)
        self.processed_dir = r(self.processed_dir)
        self.reports_dir = r(self.reports_dir)
        self.eval_dir = r(self.eval_dir)
        self.ckpt_dir = r(self.ckpt_dir)
        self.x_train = r(self.x_train)
        self.y_train = r(self.y_train)
        self.x_val = r(self.x_val)
        self.y_val = r(self.y_val)

        # create dirs that should exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.x_train.mkdir(parents=True, exist_ok=True)
        self.y_train.mkdir(parents=True, exist_ok=True)
        self.x_val.mkdir(parents=True, exist_ok=True)
        self.y_val.mkdir(parents=True, exist_ok=True)

        return self
