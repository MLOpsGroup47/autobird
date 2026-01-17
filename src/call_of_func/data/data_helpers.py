import typer
import torch
import json
from pathlib import Path

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
