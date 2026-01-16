import torch
import importlib
from typing import Any



def _locate(path: str) -> Any:
    """
    'torch.optim.Adam' -> <class torch.optim.adam.Adam>
    """
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)

def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    opt_cls = _locate(cfg.Optimizer.type)
    kwargs = dict(cfg.Optimizer)
    kwargs.pop("type")
    return opt_cls(model.parameters(), **kwargs)

def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    if "Scheduler" not in cfg or cfg.Scheduler is None:
        return None
    sched_cls = _locate(cfg.Scheduler.type)
    kwargs = dict(cfg.Scheduler)
    kwargs.pop("type")
    return sched_cls(optimizer, **kwargs)
