from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathConfig:
    root: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    ckpt_dir: Path

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
        self.ckpt_dir = r(self.ckpt_dir)

        # create dirs that should exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        return self
