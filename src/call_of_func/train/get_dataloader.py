import json
from pathlib import Path
from typing import List, Optional, Tuple
import gcsfs

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.train.train_helper import rm_rare_classes

def _load_class_names(processed_dir: Path) -> Optional[List[str]]:
    if str(processed_dir).startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(f"{processed_dir}/labels.json", "r") as f:
            return json.load(f)
    else:
        label_path = processed_dir / "labels.json"
        if not label_path.exists():
            return None
        return json.loads(label_path.read_text(encoding="utf8"))


def _load_tensor(path: str | Path) -> torch.Tensor:
    path_str = str(path)
    if path_str.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(path_str, "rb") as f:
            return torch.load(f)
    else:
        return torch.load(Path(path_str)) 
    
def maybe_path(p):
    return p if str(p).startswith("gs://") else Path(p)


def build_dataloader(
    cfg: DictConfig,
    prune_rare: bool = True,
) -> Tuple[DataLoader, DataLoader, int, Optional[List[str]]]:
    """Build dataloaders using the already composed Hydra cfg."""
    paths = PathConfig(
        root=Path(cfg.paths.root),
        raw_dir=Path(cfg.paths.raw_dir),
        processed_dir=Path(cfg.paths.processed_dir),
        reports_dir=Path(cfg.paths.reports_dir),
        eval_dir=Path(cfg.paths.eval_dir),
        ckpt_dir=Path(cfg.paths.ckpt_dir),
        x_train=Path(cfg.paths.x_train),
        y_train=Path(cfg.paths.y_train),
        x_val=Path(cfg.paths.x_val),
        y_val=Path(cfg.paths.y_val),
    )

    # hyperparams from hydra
    hp = cfg.train.hp
    min_samples = int(hp.sample_min)

    x_train = _load_tensor(paths.x_train)
    y_train = _load_tensor(paths.y_train).long()
    x_val   = _load_tensor(paths.x_val)
    y_val   = _load_tensor(paths.y_val).long()
    print(f"Loaded data: train={len(y_train)}, val={len(y_val)}")

    class_names = _load_class_names(paths.processed_dir)
    new_names = class_names

    if prune_rare:
        x_train, y_train, x_val, y_val, new_names = rm_rare_classes(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            min_samples=min_samples,
            class_names=class_names,
        )

    n_classes = int(y_train.max().item()) + 1

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(hp.batch_size),
        shuffle=bool(hp.shuffle_train),
        num_workers=int(hp.num_workers),
        pin_memory=bool(hp.pin_memory),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=int(hp.batch_size),
        shuffle=bool(hp.shuffle_val),
        num_workers=int(hp.num_workers),
        pin_memory=bool(hp.pin_memory),
    )

    return train_loader, val_loader, n_classes, new_names
