from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def build_profiler(cfg, device: torch.device):
    prof_cfg = cfg.train.prof  # profiling params

    # gate
    if not bool(getattr(prof_cfg, "enabled", False)):
        return None

    # output dir from path config
    out_dir = Path(cfg.paths.profile_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)

    prof = profile(
        activities=acts,
        schedule=schedule(
            wait=int(getattr(prof_cfg, "wait", 1)),
            warmup=int(getattr(prof_cfg, "warmup", 1)),
            active=int(getattr(prof_cfg, "active", 3)),
            repeat=int(getattr(prof_cfg, "repeat", 1)),
        ),
        record_shapes=bool(getattr(prof_cfg, "record_shapes", True)),
        profile_memory=bool(getattr(prof_cfg, "profile_memory", True)),
        with_stack=bool(getattr(prof_cfg, "with_stack", False)),
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
    )
    return prof
