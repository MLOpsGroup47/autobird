import importlib
from typing import Any

import torch
from hydra.utils import instantiate


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    return instantiate(cfg.train.optim, params=model.parameters())


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    # Check for slr specifically based on your train_engine logic
    if "slr" not in cfg.train or cfg.train.slr is None:
        return None
    return instantiate(cfg.train.slr, optimizer=optimizer)