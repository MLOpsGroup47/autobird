import json
from pathlib import Path
from typing import List, Optional, Tuple

import gcsfs
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from call_of_func.dataclasses.pathing import PathConfig


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
) -> tuple[DataLoader, DataLoader, int, Optional[List[str]], Optional[DistributedSampler]]:
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

    rank = int(cfg.runtime.rank)
    world_size = int(cfg.runtime.world_size)
    local_rank = int(cfg.runtime.local_rank)

    # hyperparams from hydra
    hp = cfg.train.hp

    x_train = _load_tensor(paths.x_train)
    y_train = _load_tensor(paths.y_train).long()
    x_val   = _load_tensor(paths.x_val)
    y_val   = _load_tensor(paths.y_val).long()
    
    if rank == 0:
        print("-------- Run settings --------")
        print(f"Runs settings {hp}")
        print(f"--> DDP world_size {world_size}")

        print("-------- Starting train --------")
        print(f"-----> Running {hp.epochs} epochs")
        print(f"----> Learning rate {hp.lr}")
        print(f"---> Batch size {hp.batch_size}")
        print(f"Loaded data: train={len(y_train)}, val={len(y_val)}")

    class_names = _load_class_names(paths.processed_dir)
    new_names = class_names

    n_classes = int(y_train.max().item()) + 1

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    use_ddp =  world_size > 1

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None

    if use_ddp:
        # In DDP, each process sees a different shard of the dataset
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(hp.batch_size),
            sampler=train_sampler,
            shuffle=False,
            num_workers=int(hp.num_workers),
            pin_memory=bool(hp.pin_memory),
            drop_last=bool(getattr(hp, "drop_last", False)),
        )
    else:
        # single-process
        if bool(getattr(hp, "use_weighted_sampler", False)):
            counts = torch.bincount(y_train, minlength=n_classes).float()
            w_per_class = 1.0 / (counts + 1e-6)
            sample_w = w_per_class[y_train]  # [N]

            weighted_sampler = WeightedRandomSampler(
                weights=sample_w,
                num_samples=len(sample_w),
                replacement=True,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=int(hp.batch_size),
                sampler=weighted_sampler,
                shuffle=False,  # must be False when sampler
                num_workers=int(hp.num_workers),
                pin_memory=bool(hp.pin_memory),
                drop_last=bool(getattr(hp, "drop_last", False)),
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=int(hp.batch_size),
                shuffle=bool(hp.shuffle_train),
                num_workers=int(hp.num_workers),
                pin_memory=bool(hp.pin_memory),
                drop_last=bool(getattr(hp, "drop_last", False)),
            )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(hp.batch_size),
        sampler=val_sampler if use_ddp else None,
        shuffle=False if use_ddp else bool(hp.shuffle_val),
        num_workers=int(hp.num_workers),
        pin_memory=bool(hp.pin_memory),
        drop_last=False,
    )

    return train_loader, val_loader, n_classes, new_names, train_sampler