from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
import torch.nn as nn
from call_of_func.train.get_dataloader import build_dataloader
from call_of_func.train.train_helper import get_device
from omegaconf import DictConfig

from call_of_birds_autobird.model import Model


@torch.no_grad()
def _eval_loop(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct_sum = 0.0

    all_preds = []
    all_y = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        correct_sum += float((logits.argmax(dim=1) == y).float().sum().item())
        total += bs

        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_y.append(y.detach().cpu())

    preds = torch.cat(all_preds, dim=0)
    ys = torch.cat(all_y, dim=0)

    avg_loss = loss_sum / total
    avg_acc = correct_sum / total
    return avg_loss, avg_acc, preds, ys


def _macro_f1(preds: torch.Tensor, ys: torch.Tensor, n_classes: int) -> float:
    # No sklearn dependency: compute per-class precision/recall and average F1
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (ys == c)).sum().item())
        fp = int(((preds == c) & (ys != c)).sum().item())
        fn = int(((preds != c) & (ys == c)).sum().item())

        if tp == 0 and (fp == 0 or fn == 0):
            # If class never predicted and/or never present -> define F1 = 0
            f1s.append(0.0)
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return float(sum(f1s) / len(f1s)) if f1s else 0.0


@hydra.main(version_base=None, config_path=str((Path(__file__).resolve().parents[2] / "configs")), config_name="config")
def run_eval(cfg: DictConfig) -> None:
    device = get_device()
    print(f"Eval on device: {device}")
    ckpt_path = Path(cfg.paths.ckpt_dir) / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train first (and ensure checkpoint saving is enabled)."
        )

    # Build dataloader EXACTLY like training (incl. prune_rare)
    # For eval, we typically prune the same way as training did, so the label space matches.
    _, val_loader, n_classes, class_names = build_dataloader(cfg=cfg, prune_rare=True)

    # Load checkpoint (includes n_classes used during training)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_n_classes = int(ckpt.get("n_classes", n_classes))

    hp = cfg.train.hp
    model = Model(
        n_classes=ckpt_n_classes,
        d_model=int(hp.d_model),
        n_heads=int(hp.n_heads),
        n_layers=int(hp.n_layers),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, preds, ys = _eval_loop(model, val_loader, criterion, device)

    macro_f1 = _macro_f1(preds, ys, ckpt_n_classes)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Macro F1: {macro_f1:.4f}")

    # Save report
    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "eval_metrics.txt"
    out.write_text(
        f"checkpoint: {ckpt_path}\n"
        f"val_loss: {val_loss:.6f}\n"
        f"val_acc: {val_acc:.6f}\n"
        f"macro_f1: {macro_f1:.6f}\n"
        f"n_classes: {ckpt_n_classes}\n"
        f"class_names_present: {class_names is not None}\n"
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    run_eval()
