import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
) -> tuple[List[Tuple[Path, int]], 
    List[Tuple[Path, int]], 
    List[Tuple[Path, int]],
]:
    """Split items into train/val sets by recording ID groups."""
    rng = random.Random(cfg.seed)

    # group by recording ID
    groups: Dict[str, List[Tuple[Path, int]]] = {}
    group_label: Dict[str, int] = {}  # assumes each rec_id has one label
    for path, label in items:
        rec_id = _recording_id(path)
        groups.setdefault(rec_id, []).append((path, label))
        if rec_id not in group_label:
            group_label[rec_id] = label
        else:
            # if a rec_id has multiple labels, that indicates a dataset issue
            if group_label[rec_id] != label:
                raise ValueError(f"Recording group {rec_id} has multiple labels!")

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    # build label -> list of rec_ids
    by_label: Dict[int, List[str]] = {}
    for rec_id in group_keys:
        lbl = group_label[rec_id]
        by_label.setdefault(lbl, []).append(rec_id)

    # seed splits to ensure class coverage
    train_id, val_id, test_id = set(), set(), set()

    for lbl, rec_ids in by_label.items():
        rng.shuffle(rec_ids)
        if len(rec_ids) >= 3:
            # guarantee 1 val + 1 test
            val_id.add(rec_ids[0])
            test_id.add(rec_ids[1])
            train_id.update(rec_ids[2:])
        elif len(rec_ids) == 2:
            # guarantee val 
            val_id.add(rec_ids[0])
            train_id.add(rec_ids[1])
        else:
            # only 1 group -> must be train
            train_id.add(rec_ids[0])

    # fill remaining groups to approximate requested ratios
    assigned = train_id | val_id | test_id
    remaining = [gid for gid in group_keys if gid not in assigned]
    rng.shuffle(remaining)

    n_groups = len(group_keys)
    desired_train = int(n_groups * cfg.train_split)
    desired_test  = int(n_groups * cfg.test_split)
    desired_val   = n_groups - desired_train - desired_test  # remainder

    # helper to add remaining until desired counts reached
    def fill(target_set: set, desired_count: int):
        while len(target_set) < desired_count and remaining:
            target_set.add(remaining.pop())

    # fill in a stable order
    fill(test_id, desired_test)
    fill(val_id, desired_val)
    fill(train_id, desired_train)

    # if anything left (due to rounding/seed overfill), put in train
    train_id.update(remaining)

    # 5) build item lists
    train_items: List[Tuple[Path, int]] = []
    val_items: List[Tuple[Path, int]] = []
    test_items: List[Tuple[Path, int]] = []

    for rec_id, group_items in groups.items():
        if rec_id in train_id:
            train_items.extend(group_items)
        elif rec_id in val_id:
            val_items.extend(group_items)
        elif rec_id in test_id:
            test_items.extend(group_items)
        else:
            # should not happen
            train_items.extend(group_items)

    return train_items, val_items, test_items

def _build_split(
    split_items: List[Tuple[Path, int]],
    pre_cfg: PreConfig,
    data_cfg: DataConfig,
):
    """Build tensors for one split (no saving, no normalization).

    Returns:
        x_tensor: [N, 1, n_mels, time]
        y_tensor: [N]
        group: list[str]
        chunk_starts: list[float]
    """
    X, y, group = [], [], []
    chunk_starts = []

    for path, label_id in split_items:
        try:
            x, sr = _load_audio(path)

            if sr != pre_cfg.sr:
                x_t = torch.from_numpy(x).float().unsqueeze(0)
                x_t = torchaudio.functional.resample(
                    x_t, orig_freq=sr, new_freq=pre_cfg.sr
                )
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

                S = _log_mel(chunk, cfg=pre_cfg)

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
        raise RuntimeError("No valid audio found for this split.")

    x_tensor = torch.stack(X, dim=0)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return x_tensor, y_tensor, group, chunk_starts
