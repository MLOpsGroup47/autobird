from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn
from call_of_birds_autobird.model import Model
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, record_function

import wandb
from call_of_func.data.data_calc import create_fq_mask, specaugment
from call_of_func.train.get_dataloader import build_dataloader
from call_of_func.train.get_optim import build_optimizer, build_scheduler
from call_of_func.train.train_checkpoint import save_checkpoints
from call_of_func.train.train_helper import accuracy, get_device
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
    """This function train one full epoch.
    
    Arg
    """
    model.train()
    run_loss = 0
    run_acc = 0.0
    total = 0

    for x,y in loader: 
        x = x.to(device)
        y = y.to(device)
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    loader, criterion, 
    device: torch.device,
) -> Tuple[float, float]:
    
    model.eval()
    run_loss = 0.0
    run_acc = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        run_loss += loss.item() * bs
        run_acc += accuracy(logits, y) * bs
        total += bs

    return run_loss / total, run_acc / total

def training(cfg) -> None:
    device = get_device()
    print(f"Training on device: {device}")
    print(f"cwd: {Path.cwd()}")

    #configs 
    hp = cfg.train.hp

    # prof initialization 
    prof = build_profiler(cfg, device) if cfg is not None else None
    run = None
    try:
        if prof is not None:
            prof.__enter__()

        # wandb initialization
        use_wandb = bool(getattr(hp, "use_wandb", True)) 
        if use_wandb:
            wandb_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            run = wandb.init(
                project=os.getenv("WANDB_PROJECT", None),
                entity=os.getenv("WANDB_ENTITY", None),
                config=wandb_cfg,
                name=f"dm{hp.d_model}_L{hp.n_layers}_H{hp.n_heads}_bs{hp.batch_size}_lr{hp.lr}",
            )
            
        # dataloaders (prune rare based on hp.sample_min)
        train_loader, val_loader, n_classes, class_names = build_dataloader(
            cfg=cfg,
            prune_rare=True,
        )

        model = Model(
            n_classes=n_classes,
            d_model=int(hp.d_model),
            n_heads=int(hp.n_heads),
            n_layers=int(hp.n_layers),
        ).to(device)


        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        fq_mask, time_mask = create_fq_mask(fq_mask=8, time_mask=20)  # make configurable later if you want
        scaler = GradScaler() if (bool(hp.amp) and device.type == "cuda") else None

        for epoch in range(int(hp.epochs)):
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
            va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, device)


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
                    "sample_min": int(hp.sample_min),
                },
                val_loss=float(va_loss),
                val_acc=float(va_acc),
                class_names=class_names,
            )


            if scheduler is not None:
                scheduler.step()


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
        if prof is not None:
            prof.__exit__(None, None, None)
        if run is not None:
            wandb.finish()