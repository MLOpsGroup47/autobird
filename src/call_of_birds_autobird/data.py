import json
from pathlib import Path
from typing import List, Tupleq

import librosa
import numpy as np
import torch
import typer

from call_of_func.data.data_helpers import rn_dir, rn_mp3
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import PreConfig, DataConfig
from call_of_func.data.get_data import (
    _index_dataset,
    _chunk_audio,
    _load_audio,
    _recording_id,
    _split_by_groups,
)

app = typer.Typer()
root = Path(__file__).parents[2]
PATHS = PathConfig(
    root = root,
    raw_dir=Path("data/voice_of_bird"),
    processed_dir=Path("data/processed"),
    reports_dir=Path("reports/figures"),
    ckpt_dir=Path("models/checkpoins")
)

def _log_mel(
    x: np.ndarray,
    cfg: PreConfig,
) -> np.ndarray:
    """Compute log-mel spectrogram: shape [n_mels, time]."""
    # mel power spectrogram
    S = librosa.feature.melspectrogram(
        y=x,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fq_min,
        fmax=min(cfg.fq_max, cfg.sr // 2),
        power=2.0,
    )
    # log compression
    S = np.log(S + 1e-6).astype(np.float32)
    return S


def _compute_global_norm_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std over dataset tensor X: shape [N, 1, Mels, Time]."""
    mean = X.mean(dim=(0, 1, 3), keepdim=True)  # [1, 1, Mels, 1]
    std = X.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-8)  # [1, 1, Mels, 1]
    return mean, std

### Main preprocessing pipeline
@app.command()
def preprocess(
    raw_dir: Path = typer.Option(None, help="Override raw_dir"),
    processed_dir: Path = typer.Option(None, help="Override processed_dir"),
    target_sr: int = typer.Option(16000, help="Target sampling rate"),
    clip_sec: float = typer.Option(5.0, help="Clip length in seconds"),
    stride_sec: float = typer.Option(2.5, help="Stride length in secounds"),
    n_fft: int = typer.Option(1024, help="FFT window size"),
    hop_length: int = typer.Option(512, help="Hop length"),
    n_mels: int = typer.Option(64, help="Number of mel bands"),
    fq_min: int = typer.Option(20, help="Min frequency"),
    fq_max: int = typer.Option(8000, help="Max frequency"),
    train_split: float = typer.Option(0.8, help="Train/validation split"),
    seed: int = typer.Option(4, help="Random seed"),
    pad_last: bool = True,
    renamed_files: bool = False,
) -> None:
    """Process raw audio data and save processed tensors.

    Args:
        raw_dir: Raw data directory
        processed_dir: Processed data save directory
        target_sr: Target sampling rate
        clip_sec: Clip length in seconds
        stride_sec: Stride length in secounds
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
        fq_min: Min frequency
        fq_max: Max frequency
        train_split: Train/validation split
        seed: Random seed
        renamed_files: Whether to rename files and directories to replace spaces with underscores

    Returns:
        None
    """
    # load config
    pre_cfg = PreConfig(
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

    data_cfg = DataConfig(
        train_split=train_split,
        seed=seed,
        clip_sec=clip_sec,
        stride_sec=stride_sec,
        pad_last=pad_last
    )

    paths = PATHS
    if raw_dir is not None or processed_dir is not None:
        paths = PathConfig(
            root            = PATHS.root,
            raw_dir         = raw_dir or PATHS.raw_dir,
            processed_dir   = processed_dir or PATHS.processed_dir,
            reports_dir     = PATHS.reports_dir,
            ckpt_dir        = PATHS.ckpt_dir,
        ).resolve()

    print(f"Project root: {paths.root}")
    print(f"Raw data directory: {paths.raw_dir}")
    print(f"Processed data directory: {paths.processed_dir}")

    # validate and create
    if not paths.raw_dir.exists() or not paths.raw_dir.is_dir():
        raise typer.BadParameter(f"Raw data directory does not exist: {raw_dir}")
    processed_dir.mkdir(parents=True, exist_ok=True)

    if renamed_files:
        rn_dir(paths.raw_dir)
        rn_mp3(paths.raw_dir)

    # index dataset
    items, classes = _index_dataset(paths.raw_dir)

    train_items, val_items = _split_by_groups(
        items=items,
        train_split=data_cfg.train_split,
        seed=data_cfg.seed,
    )
    splits = {"train": train_items, "val": val_items}

    # Save class label names separately
    with open(paths.processed_dir / "labels.json", "w", encoding="utf8") as fh:  # save class names
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
                if sr != pre_cfg.sr:
                    x = librosa.resample(x, orig_sr=sr, target_sr=pre_cfg.sr)

                rid = _recording_id(path)

                chunks = _chunk_audio(
                    x,
                    sr=pre_cfg.sr,
                    clip_sec=pre_cfg.clip_sec,
                    stride_sec=pre_cfg.clip_sec / 2,
                    pad_last=data_cfg.pad_last,
                )

                for chunk, start_sample in chunks:
                    if float(np.sqrt(np.mean(chunk**2))) < pre_cfg.min_rms:
                        continue  # skip low rms
                    
                    S = _log_mel(chunk, cfg=pre_cfg)  # [n_mels, time]

                    if float(S.std()) < pre_cfg.min_mel_std:
                        continue  # skip low mel std

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

        if split_name == "train":
            mean, std = _compute_global_norm_stats(x_tensor)  # compute mean/std
            torch.save(mean, paths.processed_dir / "train_mean.pt")  # save mean
            torch.save(std, paths.processed_dir / "train_std.pt")    # save std
        else:
            mean = torch.load(paths.processed_dir / "train_mean.pt")  # load train mean
            std = torch.load(paths.processed_dir / "train_std.pt")    # load train std

        x_tensor = (x_tensor - mean) / std  # normalize

        torch.save(x_tensor, paths.processed_dir / f"{split_name}_x.pt")  # save tensors
        torch.save(y_tensor, paths.processed_dir / f"{split_name}_y.pt")

        with open(paths.processed_dir / f"{split_name}_group.json", "w", encoding="utf8") as fh:
            json.dump(group, fh, ensure_ascii=False)  # save group ids

        torch.save(torch.tensor(chunk_starts, dtype=torch.float32), processed_dir / f"{split_name}_chunk_starts.pt")
        print(f"Saved split '{split_name}': {len(y_tensor)} samples.")  # log save

    for split_name, split_items in splits.items():
        save_split(split_name, split_items)  # process and save each split

if __name__ == "__main__":
    app()
