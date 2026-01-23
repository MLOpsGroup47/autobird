from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from call_of_birds_autobird.model import Model
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function

import wandb
from call_of_func.data.data_calc import create_fq_mask, specaugment
from call_of_func.train.get_dataloader import build_dataloader
from call_of_func.train.get_optim import build_optimizer, build_scheduler
from call_of_func.train.train_checkpoint import save_checkpoints
from call_of_func.train.train_helper import (
    _ddp_is_active,
    _get_device,
    _get_runtime,
    accuracy,
)
from call_of_func.utils.get_trackers import build_profiler


### epoch run
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    fq_mask,
    time_mask,
    amp: bool,
    grad_clip: float,
    prof,
) -> Tuple[float, float]:
    """This function train one full epoch."""
    model.train()
    run_loss = 0
    run_acc = 0.0
    total = 0
    
    params = model.module.parameters() if hasattr(model, "module") else model.parameters()

    for x,y in loader: 
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with record_function("specaugment"):
            x = specaugment(
                x, 
                fq_mask=fq_mask, 
                time_mask=time_mask,
            )
        
        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            assert scaler is not None
            with autocast():
                logits = model(x)
                loss = criterion(logits,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

        bs = x.size(0)
        run_loss += loss.item() * bs
        run_acc += accuracy(logits, y) * bs
        total += bs
        if prof is not None:
            prof.step()

    return run_loss / total, run_acc / total

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, 
    loader, 
    criterion, 
    device: torch.device,
    world_size: int,
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        val_loss += loss.item() * bs
        val_acc += accuracy(logits, y) * bs
        total += bs
    if _ddp_is_active(world_size, device):
        t = torch.tensor([val_loss, val_acc, total], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss, val_acc, total = t.tolist()
    return float(val_loss / total), float(val_acc / total)

def training(cfg) -> None:

    MODEL_DIR = Path("/gcs/birdcage-bucket/models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # device = _get_device()
    # print(f"Training on device: {device}")
    # print(f"cwd: {Path.cwd()}")
    rank, world_size, local_rank = _get_runtime(cfg)
    device = _get_device(local_rank)
    is_main = (rank == 0)

    if is_main: # Only print this in rank == 0, so we dont get duplicates 
        print(f"Training on device: {device}")
        print(f"cwd: {Path.cwd()}")

    # Hyperparameters config
    hp = cfg.train.hp

    # prof initialization 
    prof = build_profiler(cfg, device) if cfg is not None else None
    if is_main: # So we dont get duplicate wandb logs
        if prof is not None:
            print("PROFILER ENABLED")
            print("profile_dir (cfg):", cfg.paths.profile_dir)
            print("profile_dir (resolved):", Path(cfg.paths.profile_dir).resolve())
        else:
            print("PROFILER DISABLED (cfg.train.prof.enabled is False)")

    run = None
    try:
        if prof is not None:
            prof.__enter__()

        # wandb initialization
        use_wandb = bool(getattr(hp, "use_wandb", True)) 
        is_main = (rank == 0)
        if is_main and use_wandb: # only on main, no duplicates
            wandb_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", None),
                entity=os.getenv("WANDB_ENTITY", None),
                config=wandb_cfg,
                name=f"dm{hp.d_model}_L{hp.n_layers}_H{hp.n_heads}_bs{hp.batch_size}_lr{hp.lr}",
            )
            
        # dataloaders
        train_loader, val_loader, n_classes, class_names, train_sampler = build_dataloader(cfg)

        model: nn.Module = Model(
            n_classes=n_classes,
            d_model=int(hp.d_model),
            n_heads=int(hp.n_heads),
            n_layers=int(hp.n_layers),
        ).to(device)
            
        if _ddp_is_active(world_size, device):
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        criterion = nn.CrossEntropyLoss() 
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        fq_mask, time_mask = create_fq_mask(fq_mask=8, time_mask=20)  # # fq_mask for specaugment
        scaler = GradScaler() if (bool(hp.amp) and device.type == "cuda") else None

        for epoch in range(int(hp.epochs)):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            tr_loss, tr_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                fq_mask=fq_mask,
                time_mask=time_mask,
                amp=bool(hp.amp),
                grad_clip=float(hp.grad_clip),
                prof= prof,
            )
            va_loss, va_acc = validate_one_epoch(
                model=model, 
                loader=val_loader, 
                criterion=criterion, 
                device=device, 
                world_size=world_size,
            )

            if scheduler is not None:
                scheduler.step()

            # save only rank 0, so on duplicates
            if is_main:
                save_checkpoints(
                    ckpt_dir=Path(cfg.paths.ckpt_dir),
                    epoch=epoch + 1,  
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                    n_classes=n_classes,
                    hp={
                        "d_model": int(hp.d_model),
                        "n_heads": int(hp.n_heads),
                        "n_layers": int(hp.n_layers),
                        "batch_size": int(hp.batch_size),
                        "lr": float(hp.lr),
                        "min_samples": int(cfg.preprocessing.min_samples),
                    },
                    val_loss=float(va_loss),
                    val_acc=float(va_acc),
                    class_names=class_names,
                )

                if run is not None:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train/loss": tr_loss,
                            "train/acc": tr_acc,
                            "val/loss": va_loss,
                            "val/acc": va_acc,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

                    print(
                        f"Epoch {epoch+1}/{int(hp.epochs)} | "
                        f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                        f"val loss {va_loss:.4f} acc {va_acc:.4f}"
                    )
    finally:
        save_path = MODEL_DIR / "model.pth"
        if int(os.environ.get("RANK", 0)) == 0:
            torch.save(model.state_dict(), save_path)
            print("Master process saved the model.")
        if prof is not None:
            prof.__exit__(None, None, None)
        if run is not None:
            wandb.finish()