import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import typer

import soundfile as sf
import librosa
import torch


app = typer.Typer()
audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# -------
# Renaming utilities
# -------
@app.command()
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


@app.command()
def rn_mp3(root: Path = typer.Argument("data/voice_of_birds", exists=True)) -> None:
    """Rename audio files to replace spaces with underscores.

    Arg:
        root: dir to rename
    """
    # rename files in root
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in audio_exts:
            new_name = f.name.replace(" ", "_")
            if new_name != f.name:
                f.rename(f.with_name(new_name))
                print(f"{f.name} -> {new_name}")


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class PreConfig:
    """Preprocessing configuration."""
    sr: int = 16000  # sampling rate
    clip_sec: float = 5.0  # clip length in seconds
    n_fft: int = 1024  # fft window
    hop_length: int = 512  # hop length
    n_mels: int = 64  # number of mel bands
    fq_min: int = 20  # min frequency
    fq_max: int = 8000  # max frequency
    train_split: float = 0.8  # train/val split
    seed: int = 4  # random seed


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
        # fallback: librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), int(sr)


def _chunk_audio(
    x: np.ndarray,
    sr: int,
    clip_sec: float,
    stride_sec: float,
    pad_last: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """Split audio into (chunk, start_sample) tuples.

    Returns a list of (np.ndarray, int).
    """
    # compute lengths of clip and stride in samples
    clip_len = int(clip_sec * sr)
    stride_len = max(1, int(sr * stride_sec))

    chunks: List[Tuple[np.ndarray, int]] = []
    n = len(x)

    # chunking loop
    start = 0
    while start < n:
        end = start + clip_len
        chunk = x[start:end]
        if len(chunk) < clip_len:
            if not pad_last:
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
    train_split: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Split items into train/val sets by recording ID groups."""
    rng = random.Random(seed)

    # group by recording ID
    groups: dict[str, List[Tuple[Path, int]]] = {}
    for path, label in items:
        rec_id = _recording_id(path)
        if rec_id not in groups:
            groups[rec_id] = []
        groups[rec_id].append((path, label))
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n_train = int(len(group_keys) * train_split)  # number of train groups
    train_id = set(group_keys[:n_train])  # id's of train groups

    train_items: List[Tuple[Path, int]] = []
    val_items: List[Tuple[Path, int]] = []

    for rec_id, group_items in groups.items():
        if rec_id in train_id:
            train_items.extend(group_items)  # add all items in group train
        else:
            val_items.extend(group_items)  # add all items in group val

    return train_items, val_items


def _log_mel(
    x: np.ndarray,
    sr: int,
    cfg: PreConfig,
) -> np.ndarray:
    """Compute log-mel spectrogram: shape [n_mels, time]."""
    # mel power spectrogram
    S = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fq_min,
        fmax=min(cfg.fq_max, sr // 2),
        power=2.0,
    )
    # log compression
    S = np.log(S + 1e-6).astype(np.float32)
    return S


def _compute_global_norm_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std over dataset tensor X: shape [N, 1, Mels, Time]."""
    mean = X.mean()  # [1, 1, Mels, 1]
    std = X.std().clamp_min(1e-8)  # [1, 1, Mels, 1]
    return mean, std


### Main preprocessing pipeline
@app.command()
def preprocess(
    raw_dir: Path = typer.Argument("data/voice_of_birds", exists=True),
    processed_dir: Path = typer.Argument("data/processed"),
    target_sr: int = typer.Option(16000, help="Target sampling rate"),
    clip_sec: float = typer.Option(5.0, help="Clip length in seconds"),
    n_fft: int = typer.Option(1024, help="FFT window size"),
    hop_length: int = typer.Option(512, help="Hop length"),
    n_mels: int = typer.Option(64, help="Number of mel bands"),
    fq_min: int = typer.Option(20, help="Min frequency"),
    fq_max: int = typer.Option(8000, help="Max frequency"),
    train_split: float = typer.Option(0.8, help="Train/validation split"),
    seed: int = typer.Option(4, help="Random seed"),
    renamed_files: bool = False,
) -> None:
    """Process raw audio data and save processed tensors.
    Args:
        raw_dir: Raw data directory
        processed_dir: Processed data save directory
        target_sr: Target sampling rate
        clip_sec: Clip length in seconds
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        fq_min: Min frequency
        fq_max: Max frequency
        train_split: Train/validation split
        seed: Random seed
        renamed_files: Whether to rename files and directories to replace spaces with underscores

    returns:
        None
    """
    def _coerce(val, typ, fallback):
        """Check if input type is correct, else try to cast or use fallback.
        Args:
            val: input value
            typ: desired type
            fallback: fallback value if casting fails

        Returns:
            coerced value of desired type or fallback
        """
        # If value is already correct type, return it
        if isinstance(val, typ):
            return val
        # Typer may pass ArgumentInfo/OptionInfo objects; try to read `.default`
        default = getattr(val, "default", None)
        if default is not None:
            try:
                return typ(default)
            except Exception:
                pass
        # Try to cast directly, else use fallback
        try:
            return typ(val)
        except Exception:
            return fallback

    # check that inputs are correct types with _coerce
    target_sr = _coerce(target_sr, int, 16000)
    clip_sec = _coerce(clip_sec, float, 5.0)
    n_fft = _coerce(n_fft, int, 1024)
    hop_length = _coerce(hop_length, int, 512)
    n_mels = _coerce(n_mels, int, 64)
    fq_min = _coerce(fq_min, int, 20)
    fq_max = _coerce(fq_max, int, 8000)
    seed = _coerce(seed, int, 4)
    renamed_files = _coerce(renamed_files, bool, False)

    # load config
    cfg = PreConfig(
        sr=target_sr,
        clip_sec=clip_sec,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fq_min=fq_min,
        fq_max=fq_max,
        train_split=train_split,
        seed=seed,
    )

    # check paths
    if not isinstance(raw_dir, Path):
        raw_dir = Path("data/voice_of_birds")
    else:
        raw_dir = raw_dir.expanduser().resolve()

    if not isinstance(processed_dir, Path):
        processed_dir = Path("data/processed")
    else:
        processed_dir = processed_dir.expanduser().resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    items, classes = _index_dataset(raw_dir)

    # For convenience, rename files and directories to replace spaces with underscores
    if renamed_files:
        rn_dir(raw_dir)
        rn_mp3(raw_dir)

    train_items, val_items = _split_by_groups(
        items=items,
        train_split=cfg.train_split,
        seed=cfg.seed,
    )
    splits = {"train": train_items, "val": val_items}

    # Save class label names separately
    with open(processed_dir / "labels.json", "w", encoding="utf8") as fh:  # save class names
        json.dump(classes, fh, ensure_ascii=False)

    def save_split(split_name: str, split_items: List[Tuple[Path, int]]) -> None:
        """Process and save a data split.

        Args:
            split_name: Train/val split
            split_items: (filepath, label_id) tuples for the split

        Returns:
            None
        """
        X, y, group = [], [], []
        chunk_starts = []

        for path, label_id in split_items:
            try:
                x, sr = _load_audio(path)
                if sr != cfg.sr:
                    x = librosa.resample(x, orig_sr=sr, target_sr=cfg.sr)

                rid = _recording_id(path)

                chunks = _chunk_audio(
                    x,
                    sr=cfg.sr,
                    clip_sec=cfg.clip_sec,
                    stride_sec=cfg.clip_sec / 2,
                    pad_last=True,
                )

                for chunk, start_sample in chunks:
                    S = _log_mel(chunk, sr=cfg.sr, cfg=cfg)  # [n_mels, time]
                    X.append(torch.from_numpy(S).unsqueeze(0))  # [1, n_mels, time]
                    y.append(label_id)
                    group.append(rid)
                    chunk_starts.append(start_sample / cfg.sr)

            except Exception as e:
                print(f"Skipping bad audio: {path} -> {e}")
                continue

        if not X:
            print(f"No valid audio for split '{split_name}', skipping save.")
            return

        x_tensor = torch.stack(X, dim=0)  # [N, 1, n_mels, time]
        y_tensor = torch.tensor(y, dtype=torch.long)  # [N]

        mean, std = _compute_global_norm_stats(x_tensor)  # compute mean/std
        x_tensor = (x_tensor - mean) / std  # normalize

        torch.save(x_tensor, processed_dir / f"{split_name}_x.pt")  # save tensors
        torch.save(y_tensor, processed_dir / f"{split_name}_y.pt")

        with open(processed_dir / f"{split_name}_group.json", "w", encoding="utf8") as fh:
            json.dump(group, fh, ensure_ascii=False)  # save group ids

        torch.save(torch.tensor(chunk_starts, dtype=torch.float32), processed_dir / f"{split_name}_chunk_starts.pt")
        print(f"Saved split '{split_name}': {len(y_tensor)} samples.")  # log save

    for split_name, split_items in splits.items():
        save_split(split_name, split_items)  # process and save each split


@app.command()
def load_data(processed_dir: Path = Path("data/processed")):
    """Load processed data tensors from disk.
    Args:
        processed_dir: Path to processed data directory.
    returns:
        x_train, y_train, x_val, y_val, classes, train_chunk_starts, val_chunk_starts
    """
    processed_dir = Path(processed_dir).expanduser().resolve()

    # classes
    with open(processed_dir / "labels.json", "r", encoding="utf8") as fh:
        classes = json.load(fh)

    # tensors
    x_train = torch.load(processed_dir / "train_x.pt")
    y_train = torch.load(processed_dir / "train_y.pt")
    x_val   = torch.load(processed_dir / "val_x.pt")
    y_val   = torch.load(processed_dir / "val_y.pt")

    # json lists (if you still want them)
    with open(processed_dir / "train_group.json", "r", encoding="utf8") as fh:
        train_group = json.load(fh)
    with open(processed_dir / "val_group.json", "r", encoding="utf8") as fh:
        val_group = json.load(fh)

    # chunk starts are tensors (binary)
    train_chunk_starts = torch.load(processed_dir / "train_chunk_starts.pt")
    val_chunk_starts   = torch.load(processed_dir / "val_chunk_starts.pt")

    return x_train, y_train, x_val, y_val, classes, train_group, val_group, train_chunk_starts, val_chunk_starts

if __name__ == "__main__":
    app()
