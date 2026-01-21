import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoints(
    *,
    ckpt_dir: Path,
    epoch: int,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    scheduler_state: Optional[Dict[str, Any]],
    n_classes: int,
    hp: Dict[str, Any],
    val_loss: float,
    val_acc: float,
    class_names: Optional[list[str]] = None,
) -> None:
    """Save training checkpoints.

    - 'last.pt' is overwritten every epoch
    - 'best.pt' is updated when validation accuracy improves
    """
    # Check that checkpoint directory exists
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Payload to save
    payload = {
        "epoch": int(epoch),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "n_classes": int(n_classes),
        "hp": hp,
        "metrics": {
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        },
        "class_names": class_names,
    }

    # Always save last checkpoint
    last_ckpt = ckpt_dir / "last.pt"
    torch.save(payload, last_ckpt)

    # Load best validation accuracy so far if exists
    best_metric_path = ckpt_dir / "best_metric.json"
    best_acc = -1.0

    if best_metric_path.exists():
        try:
            best_acc = float(
                json.loads(best_metric_path.read_text(encoding="utf8"))["best_val_acc"]
            )
        except Exception:
            best_acc = -1.0

    # Saves best checkpoint if val_acc improves
    if float(val_acc) > best_acc:
        best_ckpt = ckpt_dir / "best.pt"
        torch.save(payload, best_ckpt)

        best_metric_path.write_text(
            json.dumps(
                {
                    "best_val_acc": float(val_acc),
                    "epoch": int(epoch),
                }
            ),
            encoding="utf8",
        )
