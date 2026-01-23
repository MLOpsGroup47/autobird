import json
from pathlib import Path
from typing import List, Optional, Tuple
import fsspec
import gcsfs
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.train.train_helper import rm_rare_classes


def _load_tensor(path) -> torch.Tensor:
    # fsspec.open handles gs:// and local paths automatically
    with fsspec.open(str(path), "rb") as f:
        return torch.load(f, map_location="cpu")

def _load_class_names(processed_dir) -> Optional[list[str]]:
    # Join paths correctly whether string (GCS) or Path (Local)
    path_str = f"{str(processed_dir)}/labels.json"
    
    try:
        with fsspec.open(path_str, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    
def safe_path(p):
    if p is None: return None
    p_str = str(p)
    if "://" in p_str:  # Catch gs://, s3://, etc.
        return p_str
    return Path(p_str)


def build_dataloader(
    cfg: DictConfig,
    prune_rare: bool = True,
) -> Tuple[DataLoader, DataLoader, int, Optional[List[str]]]:
    """Build dataloaders using the already composed Hydra cfg."""
    paths = PathConfig(
        root=safe_path(cfg.paths.root),
        raw_dir=safe_path(cfg.paths.raw_dir),
        processed_dir=safe_path(cfg.paths.processed_dir),
        reports_dir=safe_path(cfg.paths.reports_dir),
        eval_dir=safe_path(cfg.paths.eval_dir),
        ckpt_dir=safe_path(cfg.paths.ckpt_dir),
        x_train=safe_path(cfg.paths.x_train),
        y_train=safe_path(cfg.paths.y_train),
        x_val=safe_path(cfg.paths.x_val),
        y_val=safe_path(cfg.paths.y_val),
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
