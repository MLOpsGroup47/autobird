from typing import Optional

import torch
from omegaconf import DictConfig
import torch.distributed as dist


def accuracy(logits, y) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

## ddp helper ##
def _get_runtime(cfg: DictConfig) -> tuple[int, int, int]:
    """Read runtime injucted by torchrun"""
    rt = getattr(cfg, "runtime", None)
    if rt is None:
        return 0, 1, 0
    rank= int(getattr(rt, "rank", 0))
    world_size = int(getattr(rt, "world_size", 1))
    local_rank = int(getattr(rt, "local_rank", 0))
    return rank, world_size, local_rank

def _get_device(local_rank: int) -> torch.device:
    """Select device. Cuda if availeble. MPS if on mac: else CPU"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ddp_is_active(world_size: int, device: torch.device) -> bool:
    """DDP only active when multi-process AND CUDA AND process group initialized."""
    return (
        world_size > 1
        and device.type == "cuda"
        and dist.is_available()
        and dist.is_initialized()
    )