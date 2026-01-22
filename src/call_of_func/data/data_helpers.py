import json
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import typer


def rn_dir(root: Path = typer.Argument("data/voice_of_birds", exists=True)) -> None:
    """Rename directories to replace spaces with underscores.

    Arg:
        root: dir to rename
    """
    # rename dirs in root
    for p in sorted(root.iterdir()):
        if p.is_dir():
            new_name = p.name.replace(" ", "_")
            if new_name != p.name:
                p.rename(p.with_name(new_name))
            print(f"{p.name} -> {new_name}")


def rn_mp3(cfg, exists=True) -> None:
    """Rename audio files to replace spaces with underscores.

    Arg:
        root: dir to rename
    """
    # rename files in root
    for f in cfg.root.rglob("*"):
        if f.is_file() and f.suffix.lower() in cfg.audio_exts:
            new_name = f.name.replace(" ", "_")
            if new_name != f.name:
                f.rename(f.with_name(new_name))
                print(f"{f.name} -> {new_name}")


def load_data(processed_dir: Path = Path("data/processed")):
    """Load processed data tensors from disk.

    Args:
        processed_dir: Path to processed data directory.

    Returns:
        x_train, y_train, x_val, y_val, classes, train_chunk_starts, val_chunk_starts
    """
    ROOT = Path(__file__).resolve().parents[2]  # /app
    processed_dir = Path(processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = (ROOT / processed_dir).resolve()

    # classes
    with open(processed_dir / "labels.json", "r", encoding="utf8") as fh:
        classes = json.load(fh)

    # tensors
    x_train = torch.load(processed_dir / "train_x.pt")
    y_train = torch.load(processed_dir / "train_y.pt")
    x_val = torch.load(processed_dir / "val_x.pt")
    y_val = torch.load(processed_dir / "val_y.pt")

    # json list
    with open(processed_dir / "train_group.json", "r", encoding="utf8") as fh:
        train_group = json.load(fh)
    with open(processed_dir / "val_group.json", "r", encoding="utf8") as fh:
        val_group = json.load(fh)

    # chunk starts are tensors (binary)
    train_chunk_starts = torch.load(processed_dir / "train_chunk_starts.pt")
    val_chunk_starts = torch.load(processed_dir / "val_chunk_starts.pt")

    return x_train, y_train, x_val, y_val, classes, train_group, val_group, train_chunk_starts, val_chunk_starts



def _compute_global_norm_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std over dataset tensor X: shape [N, 1, Mels, Time]."""
    mean = X.mean(dim=(0, 1, 3), keepdim=True)  # [1, 1, Mels, 1]
    std = X.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-8)  # [1, 1, Mels, 1]
    return mean, std

def filter_data(
    y_train: torch.Tensor,
    min_samples: int,
    class_names: Optional[list[str]] = None,
) -> Tuple[torch.Tensor, list[int], Optional[list[str]]]:
    y_train = y_train.long()
    num_classes = int(y_train.max().item()) + 1
    train_counts = torch.bincount(y_train, minlength=num_classes)

    keep = (train_counts >= min_samples).nonzero(as_tuple=True)[0]
    keep_sorted = keep.tolist()

    print(f"Keeping {len(keep_sorted)}/{num_classes} classes with >= {min_samples} train samples")

    old_to_new = torch.full((num_classes,), -1, dtype=torch.long)
    old_to_new[keep] = torch.arange(len(keep), dtype=torch.long)

    new_class_names = None
    if class_names is not None:
        new_class_names = [class_names[i] for i in keep_sorted]

    return old_to_new, keep_sorted, new_class_names


def _apply_mapping_with_meta(
    x: torch.Tensor,
    y: torch.Tensor,
    group: list[str],
    chunk_starts: list[float],
    old_to_new: torch.Tensor,
):
    y = y.long()
    mapped = old_to_new[y]
    mask = mapped >= 0

    x2 = x[mask]
    y2 = mapped[mask]

    # mask is torch bool; use it to filter python lists
    mask_list = mask.cpu().numpy().tolist()
    group2 = [g for g, m in zip(group, mask_list) if m]
    starts2 = [s for s, m in zip(chunk_starts, mask_list) if m]

    return x2, y2, group2, starts2