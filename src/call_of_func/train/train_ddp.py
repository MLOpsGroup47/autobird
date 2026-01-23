from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from call_of_func.train.train_engine import training


def ddp_setup() -> Tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Safe on non-DDP too."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Only initialize on real multi-GPU CUDA runs
    if world_size > 1 and torch.cuda.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_train(cfg: DictConfig) -> None:
    rank, world_size, local_rank = ddp_setup()

    # runtime node must be OmegaConf
    if not hasattr(cfg, "runtime") or cfg.runtime is None:
        cfg.runtime = OmegaConf.create()

    cfg.runtime.rank = rank
    cfg.runtime.world_size = world_size
    cfg.runtime.local_rank = local_rank

    try:
        training(cfg)
    finally:
        ddp_cleanup()
