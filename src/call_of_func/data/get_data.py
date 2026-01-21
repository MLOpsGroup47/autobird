import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from call_of_func.data.data_calc import _log_mel
from call_of_func.data.data_helpers import _compute_global_norm_stats
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig

audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _index_dataset(raw_dir: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
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
        # fallback: torchaudio
        wav, sr = torchaudio.load(str(path))  # wav: [channels, time]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x = wav.squeeze(0).cpu().numpy().astype(np.float32)
        return x, int(sr)


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



def _save_split(
        split_name: str, 
        split_items: List[Tuple[Path, int]],
        paths: PathConfig,
        pre_cfg: PreConfig,
        data_cfg: DataConfig
        ) -> None:
    """Process and save a data split.

    Args:
        split_name: Train/val split
        split_items: (filepath, label_id) tuples for the split
        paths: PathConfig with processed_dir
        pre_cfg: Preprocessing config
        data_cfg: Data config

    Returns:
        None
    """
    X, y, group = [], [], []
    chunk_starts = []

    for path, label_id in split_items:
        try:
            x, sr = _load_audio(path)

            if sr != pre_cfg.sr:
                x_t = torch.from_numpy(x).float().unsqueeze(0)  # [1, time]
                x_t = torchaudio.functional.resample(x_t, orig_freq=sr, new_freq=pre_cfg.sr)
                x = x_t.squeeze(0).cpu().numpy().astype(np.float32)
                sr = pre_cfg.sr

            rid = _recording_id(path)

            chunks = _chunk_audio(
                x,
                pre_cfg=pre_cfg,
                data_cfg=data_cfg,
            )

            for chunk, start_sample in chunks:
                # RMS filter
                if float(np.sqrt(np.mean(chunk**2))) < pre_cfg.min_rms:
                    continue

                S = _log_mel(chunk, cfg=pre_cfg)  # [n_mels, time]

                # Mel variance filter
                if float(S.std()) < pre_cfg.min_mel_std:
                    continue

                X.append(torch.from_numpy(S).unsqueeze(0))  # [1, n_mels, time]
                y.append(label_id)
                group.append(rid)
                chunk_starts.append(start_sample / pre_cfg.sr)

        except Exception as e:
            print(f"Skipping bad audio: {path} -> {e}")
            continue

    if not X:
        print(f"No valid audio for split '{split_name}', skipping save.")
        return

    x_tensor = torch.stack(X, dim=0)  # [N, 1, n_mels, time]
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Normalize using train stats
    if split_name == "train":
        mean, std = _compute_global_norm_stats(x_tensor)
        torch.save(mean, paths.processed_dir / "train_mean.pt")
        torch.save(std, paths.processed_dir / "train_std.pt")
    else:
        mean = torch.load(paths.processed_dir / "train_mean.pt")
        std = torch.load(paths.processed_dir / "train_std.pt")

    x_tensor = (x_tensor - mean) / std

    # Save tensors
    torch.save(x_tensor, paths.processed_dir / f"{split_name}_x.pt")
    torch.save(y_tensor, paths.processed_dir / f"{split_name}_y.pt")

    with open(paths.processed_dir / f"{split_name}_group.json", "w", encoding="utf8") as fh:
        json.dump(group, fh, ensure_ascii=False)

    torch.save(
        torch.tensor(chunk_starts, dtype=torch.float32),
        paths.processed_dir / f"{split_name}_chunk_starts.pt",
    )

    print(f"Saved split '{split_name}': {len(y_tensor)} samples.")