import json
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig

audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _index_dataset(
    raw_dir: Path,
    cfg: DataConfig,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Return list of (filepath, label_id) and class names."""
    # find class subfolders
    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in: {raw_dir}")  # check if subdirs exist

    classes = [p.name for p in class_dirs]  # class names
    class_to_id = {c: i for i, c in enumerate(classes)}  # map class name to id

    # create list of (filepath, label_id)
    items: List[Tuple[Path, int]] = []
    for cdir in class_dirs:
        for f in cdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in audio_exts:
                items.append((f, class_to_id[cdir.name]))

    # check if any audio files found
    if not items:
        raise ValueError(f"No audio files found under: {raw_dir}")

    return items, classes


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio, return mono float32 array and sample rate."""
    try:
        x, sr = sf.read(str(path), always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return x.astype(np.float32), int(sr)
    except Exception:
        # fallback: librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


def _chunk_audio(
    x: np.ndarray,
    pre_cfg: PreConfig,
    data_cfg: DataConfig,
) -> List[Tuple[np.ndarray, int]]:
    """Split audio into (chunk, start_sample) tuples.

    Returns a list of (np.ndarray, int).
    """
    # compute lengths of clip and stride in samples
    clip_len = int(data_cfg.clip_sec * pre_cfg.sr)
    stride_len = max(1, int(pre_cfg.sr * data_cfg.stride_sec))

    chunks: List[Tuple[np.ndarray, int]] = []
    n = len(x)

    # chunking loop
    start = 0
    while start < n:
        end = start + clip_len
        chunk = x[start:end]
        if len(chunk) < clip_len:
            if not data_cfg.pad_last:
                break
            chunk = np.pad(chunk, (0, clip_len - len(chunk)), mode="constant")
        chunks.append((chunk, start))
        start += max(1, stride_len)
    return chunks


def _recording_id(path: Path) -> str:
    """A stable recording ID from filepath."""
    return f"{path.parent.name}_{path.stem}"


def _split_by_groups(
    items: List[Tuple[Path, int]],
    cfg: DataConfig,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Split items into train/val sets by recording ID groups."""
    rng = random.Random(cfg.seed)

    # group by recording ID
    groups: dict[str, List[Tuple[Path, int]]] = {}
    for path, label in items:
        rec_id = _recording_id(path)
        if rec_id not in groups:
            groups[rec_id] = []
        groups[rec_id].append((path, label))
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n_train = int(len(group_keys) * cfg.train_split)  # number of train groups
    train_id = set(group_keys[:n_train])  # id's of train groups

    train_items: List[Tuple[Path, int]] = []
    val_items: List[Tuple[Path, int]] = []

    for rec_id, group_items in groups.items():
        if rec_id in train_id:
            train_items.extend(group_items)  # add all items in group train
        else:
            val_items.extend(group_items)  # add all items in group val

    return train_items, val_items


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
