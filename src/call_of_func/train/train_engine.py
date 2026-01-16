from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio  # type: ignore
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, record_function

from call_of_birds_autobird.model import Model
from call_of_func.train.optim import build_optimizer, build_scheduler
from call_of_func.train.get_dataloader import build_dataloader

def accuracy(logits, y) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def create_fq_mask(fq_mask: int, time_mask: int):
    fq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=fq_mask)  # freq mask
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)  # time mask
    return fq_mask, time_mask

def specaugment(x: torch.tensor, fq_mask, time_mask) -> torch.tensor:
    x = x.squeeze(1)  # [B, Mels, Time]
    x = fq_mask(x)
    x = time_mask(x)
    return x.unsqueeze(1)  # [B, 1, Mels, Time

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
) -> Tuple[float, float]:
    
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
                loss = criterion(x,y)
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

def train_from_cfg(cfg) -> None:
    device = get_device()
    print(f"Training on device: {device}")
    print(f"cwd: {Path.cwd()}")

    hp = cfg.train.hyperparams.hyperparameters

    # dataloaders (prune rare based on hp.sample_min)
    train_loader, val_loader, n_classes, new_names = build_dataloader(
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
    optimizer = build_optimizer(model, cfg.train.optimizer)
    scheduler = build_scheduler(optimizer, cfg.train.scheduler)

    fq_mask, time_mask = create_fq_mask(fq_mask=8, time_mask=20)  # make configurable later if you want
    scaler = GradScaler() if (bool(hp.amp) and device.type == "cuda") else None

    # profiler 
    prof = None
    if getattr(hp, "profile_run", False):
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        prof = profile(
            activities=acts,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("reports/torch_prof"),
        )
        prof.__enter__()

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
        )
        va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{int(hp.epochs)} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

    if prof is not None:
        prof.__exit__(None, None, None)