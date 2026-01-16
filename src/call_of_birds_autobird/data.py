import json
from pathlib import Path
from typing import List, Tuple

import hydra
import librosa
import numpy as np
import torch
import typer
from call_of_func.data.data_helpers import rn_dir, rn_mp3
from call_of_func.data.get_data import (
    _chunk_audio,
    _index_dataset,
    _load_audio,
    _recording_id,
    _split_by_groups,
)
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig
from call_of_func.utils.get_configs import _load_cfg
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

app = typer.Typer()
root = Path(__file__).parents[2]
PATHS = PathConfig(
    root=root,
    raw_dir=Path("data/voice_of_birds"),
    processed_dir=Path("data/processed"),
    reports_dir=Path("reports/figures"),
    ckpt_dir=Path("models/checkpoins"),
)


def _log_mel(x: np.ndarray, cfg: PreConfig) -> np.ndarray:
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
    config: str = typer.Option("default", help="Hydra config name"),
    o: List[str] = typer.Option(None, "--o", help="Hydra overrides"),
    raw_dir: Path = typer.Option(None, help="Override raw_dir over config"),
    processed_dir: Path = typer.Option(None, help="Override processed_dir"),
    renamed_files: bool = typer.Option(False, help="Rename dir/files to remove spaces"),
) -> None:
    """Process raw audio data and save processed tensors.

    Args:
        config: Hydra config name
        o: Hydra overrides
        raw_dir: Raw data directory
        processed_dir: Processed data save directory
        renamed_files: Whether to rename files and directories to replace spaces with underscores

    Returns:
        None
    """
    # load config
    cfg = _load_cfg(config_name=config, overrides=o or [])

    paths = PathConfig(
        root=Path(cfg.paths.root),
        raw_dir=Path(cfg.paths.raw_dir),
        processed_dir=Path(cfg.paths.processed_dir),
        reports_dir=Path(cfg.paths.reports_dir),
        ckpt_dir=Path(cfg.paths.ckpt_dir),
    )
    if raw_dir is not None or processed_dir is not None:
        paths = PathConfig(
            root=paths.root,
            raw_dir=raw_dir or paths.raw_dir,
            processed_dir=processed_dir or paths.processed_dir,
            reports_dir=paths.reports_dir,
            ckpt_dir=paths.ckpt_dir,
        ).resolve()

    pre_cfg = PreConfig(
        sr=cfg.preprocessing.sr,
        clip_sec=cfg.preprocessing.clip_sec,
        n_fft=cfg.preprocessing.n_fft,
        hop_length=cfg.preprocessing.hop_length,
        n_mels=cfg.preprocessing.n_mels,
        fq_min=cfg.preprocessing.fq_min,
        fq_max=cfg.preprocessing.fq_max,
        min_rms=cfg.preprocessing.min_rms,
        min_mel_std=cfg.preprocessing.min_mel_std,
    )

    data_cfg = DataConfig(
        train_split=cfg.data.train_split,
        seed=cfg.data.seed,
        clip_sec=cfg.data.clip_sec,
        stride_sec=cfg.data.stride_sec,
        pad_last=True,
    )

    print(f"Project root: {paths.root}")
    print(f"Raw data directory: {paths.raw_dir}")
    print(f"Processed data directory: {paths.processed_dir}")

    # validate and create
    if not paths.raw_dir.exists() or not paths.raw_dir.is_dir():
        raise typer.BadParameter(f"Raw data directory does not exist: {paths.raw_dir}")
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    if renamed_files:
        rn_dir(paths.raw_dir)
        rn_mp3(paths.raw_dir)

    # index dataset
    items, classes = _index_dataset(paths.raw_dir)

    train_items, val_items = _split_by_groups(items=items, cfg=data_cfg)

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
                    pre_cfg=pre_cfg,
                    data_cfg=data_cfg,
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
            torch.save(std, paths.processed_dir / "train_std.pt")  # save std
        else:
            mean = torch.load(paths.processed_dir / "train_mean.pt")  # load train mean
            std = torch.load(paths.processed_dir / "train_std.pt")  # load train std

        x_tensor = (x_tensor - mean) / std  # normalize

        torch.save(x_tensor, paths.processed_dir / f"{split_name}_x.pt")  # save tensors
        torch.save(y_tensor, paths.processed_dir / f"{split_name}_y.pt")

        with open(paths.processed_dir / f"{split_name}_group.json", "w", encoding="utf8") as fh:
            json.dump(group, fh, ensure_ascii=False)  # save group ids

        torch.save(
            torch.tensor(chunk_starts, dtype=torch.float32), paths.processed_dir / f"{split_name}_chunk_starts.pt"
        )
        print(f"Saved split '{split_name}': {len(y_tensor)} samples.")  # log save

    for split_name, split_items in splits.items():
        save_split(split_name, split_items)  # process and save each split


if __name__ == "__main__":
    app()
