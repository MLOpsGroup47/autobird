import os
import json
import math
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


@app.command()
def rn_dir(root: Path = typer.Argument("data/voice_of_birds", exists=True)) -> None:
    """Rename directories to replace spaces with underscores.

    arg:
        root: dir to rename
    """

    root = root
    for p in sorted(root.iterdir()):
        if p.is_dir():
            new_name = p.name.replace(" ", "_")
            if new_name != p.name:
                p.rename(p.with_name(new_name))
            print(f"{p.name} -> {new_name}")

@app.command()
def rn_mp3(root_dir: Path = typer.Argument("data/voice_of_birds", exists=True)) -> None:
    """Rename audio files to replace spaces with underscores.

    arg:
        root_dir: dir to rename
    """

    root = root_dir 

    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in audio_exts:
            new_name = f.name.replace(" ", "_")
            if new_name != f.name:
                f.rename(f.with_name(new_name))
                print(f"{f.name} -> {new_name}")



@dataclass(frozen=True)
class PreConfig:
    sr: int = 16000 # sampling rate
    clip_sec: float = 5.0  # clip length in seconds
    n_fft: int = 1024  # fft window
    hop_length: int = 512 # hop length
    n_mels: int = 64  # number of mel bands
    fq_min: int = 20  # min frequency
    fq_max: int = 8000 # max frequency
    train_split: float = 0.8  # train/val split
    seed: int = 4  # random seed


def _index_dataset(raw_dir: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """Return list of (filepath, label_id) and class names."""

    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in: {raw_dir}")

    classes = [p.name for p in class_dirs]
    class_to_id = {c: i for i, c in enumerate(classes)}

    items: List[Tuple[Path, int]] = []
    for cdir in class_dirs:
        for f in cdir.rglob("*"):
            if f.is_file() and f.suffix.lower() in audio_exts:
                items.append((f, class_to_id[cdir.name]))

    if not items:
        raise ValueError(f"No audio files found under: {raw_dir}")

    return items, classes

def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio with soundfile (fast, reliable).
    Returns mono float32 array in range [-1, 1] and sample rate.
    """
    # Primary: soundfile (libsndfile)
    try:
        x, sr = sf.read(str(path), always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return x.astype(np.float32), int(sr)
    except Exception:
        # Fallback 1: librosa (uses audioread/ffmpeg backends)
        try:
            y, sr = librosa.load(str(path), sr=None, mono=False)
            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.mean(axis=0)
            return y.astype(np.float32), int(sr)
        except Exception:
            # Fallback 2: torchaudio
            try:
                import torchaudio

                waveform, sr = torchaudio.load(str(path))
                # waveform: [channels, time]
                if waveform.ndim == 2:
                    waveform = waveform.mean(dim=0)
                return waveform.numpy().astype(np.float32), int(sr)
            except Exception as e:
                raise RuntimeError(f"Failed to read audio {path}: {e}") from e

def _fix_length(x: np.ndarray, sr: int, clip_seconds: float, rng: random.Random) -> np.ndarray:
    """Random crop if long; zero-pad if short."""
    target_len = int(sr * clip_seconds)
    if len(x) > target_len:
        start = rng.randint(0, len(x) - target_len)
        x = x[start : start + target_len]
    elif len(x) < target_len:
        pad = target_len - len(x)
        x = np.pad(x, (0, pad), mode="constant")
    return x

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
    std = X.std().clamp_min(1e-8)    # [1, 1, Mels, 1]
    return mean, std


@app.command()
def preprocess(raw_dir: Path = typer.Argument("data/voice_of_birds", exists=True), 
               processed_dir: Path = typer.Argument("data/processed"),
               target_sr: int = typer.Option(16000, help="Target sampling rate"),
               clip_sec: float = typer.Option(5.0, help="Clip length in seconds"),
               n_ffts: int = typer.Option(1024, help="FFT window size"),
               hop_length: int = typer.Option(512, help="Hop length"),
               n_mels: int = typer.Option(64, help="Number of mel bands"),
               fq_min: int = typer.Option(20, help="Min frequency"),
               fq_max: int = typer.Option(8000, help="Max frequency"),
               train_split: float = typer.Option(0.8, help="Train/validation split"),   
               seed: int = typer.Option(4, help="Random seed"),
               renamed_files: bool = False
                ) -> None:
    

    def _coerce(val, typ, fallback):
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

    target_sr = _coerce(target_sr, int, 16000)
    clip_sec = _coerce(clip_sec, float, 5.0)
    n_ffts = _coerce(n_ffts, int, 1024)
    hop_length = _coerce(hop_length, int, 512)
    n_mels = _coerce(n_mels, int, 64)
    fq_min = _coerce(fq_min, int, 20)
    fq_max = _coerce(fq_max, int, 8000)
    seed = _coerce(seed, int, 4)
    renamed_files = _coerce(renamed_files, bool, False)

    cfg = PreConfig(
        sr=target_sr,
        clip_sec=clip_sec,
        n_fft=n_ffts,
        hop_length=hop_length,
        n_mels=n_mels,
        fq_min=fq_min,
        fq_max=fq_max,
        seed=seed)
    
    if not isinstance(raw_dir, Path) or raw_dir.__class__.__name__ == "ArgumentInfo":
        raw_dir = Path("data/voice_of_birds")
    else:
        raw_dir = raw_dir.expanduser().resolve()

    if not isinstance(processed_dir, Path) or processed_dir.__class__.__name__ == "ArgumentInfo":
        processed_dir = Path("data/processed")
    else:
        processed_dir = processed_dir.expanduser().resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    items, classes = _index_dataset(raw_dir)

    if renamed_files:
        rn_dir(raw_dir)
        rn_mp3(raw_dir)
    
    rng = random.Random(cfg.seed)
    rng.shuffle(items)

    n_train = int(len(items) * cfg.train_split)
    train_items = items[:n_train]
    val_items = items[n_train:]

    splits = {"train": train_items, "val": val_items}

    # Save class label names separately
    with open(processed_dir / "labels.json", "w", encoding="utf8") as fh:
        json.dump(classes, fh, ensure_ascii=False)

    def save_split(split_name: str, split_items: List[Tuple[Path, int]]) -> None:
        x_list = []
        y_list = []

        for i, (path, label_id) in enumerate(split_items):
            try:
                x, sr = _load_audio(path)
            except Exception as e:
                print(f"Skipping bad audio: {path} -> {e}")
                continue

            try:
                if sr != cfg.sr:
                    x = librosa.resample(x, orig_sr=sr, target_sr=cfg.sr)
                x = _fix_length(x, sr=cfg.sr, clip_seconds=cfg.clip_sec, rng=rng)
                S = _log_mel(x, sr=cfg.sr, cfg=cfg)  # [n_mels, time]
                x_list.append(torch.from_numpy(S).unsqueeze(0))  # [1, n_mels, time]
                y_list.append(label_id)
            except Exception as e:
                print(f"Failed processing {path}, skipping -> {e}")
                continue

        if not x_list:
            print(f"No valid audio for split '{split_name}', skipping save.")
            return

        x_tensor = torch.stack(x_list, dim=0)  # [N, 1, n_mels, time]
        y_tensor = torch.tensor(y_list, dtype=torch.long)  # [N]

        mean, std = _compute_global_norm_stats(x_tensor)
        x_tensor = (x_tensor - mean) / std

        torch.save(x_tensor, processed_dir / f"{split_name}_x.pt")
        torch.save(y_tensor, processed_dir / f"{split_name}_y.pt")

    for split_name, split_items in splits.items():
        save_split(split_name, split_items)

if __name__ == "__main__":
    app()