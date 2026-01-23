import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import typer
from omegaconf import DictConfig

from call_of_func.data.data_helpers import (
    _apply_mapping_with_meta,
    filter_data,
    rn_dir,
    rn_mp3,
)
from call_of_func.data.get_data import _build_split, _compute_global_norm_stats, _index_dataset, _split_by_groups
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig
from call_of_func.utils.get_configs import _load_cfg


def preprocess_cfg(
    cfg: DictConfig,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    renamed_files: bool = False,
) -> None:
    # Paths 
    paths = PathConfig(
        root=Path(cfg.paths.root),
        raw_dir=Path(cfg.paths.raw_dir),
        processed_dir=Path(cfg.paths.processed_dir),
        reports_dir=Path(cfg.paths.reports_dir),
        eval_dir=Path(cfg.paths.eval_dir),
        ckpt_dir=Path(cfg.paths.ckpt_dir),
        x_train=Path(cfg.paths.x_train),
        y_train=Path(cfg.paths.y_train),
        x_val=Path(cfg.paths.x_val),
        y_val=Path(cfg.paths.y_val),
    )

    if raw_dir is not None or processed_dir is not None:
        paths = PathConfig(
            root=paths.root,
            raw_dir=raw_dir or paths.raw_dir,
            processed_dir=processed_dir or paths.processed_dir,
            reports_dir=paths.reports_dir,
            eval_dir=paths.eval_dir,
            ckpt_dir=paths.ckpt_dir,
            x_train=paths.x_train,
            y_train=paths.y_train,
            x_val=paths.x_val,
            y_val=paths.y_val,
        ).resolve()

    # Preprocessing config 
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
        min_samples=cfg.preprocessing.min_samples,
    )

    # Data split config
    data_cfg = DataConfig(
        train_split=cfg.data.train_split,
        test_split=cfg.data.test_split,
        seed=cfg.data.seed,
        clip_sec=cfg.data.clip_sec,
        stride_sec=cfg.data.stride_sec,
        pad_last=True,
    )

    print(f"Project root: {paths.root}")
    print(f"Raw data directory: {paths.raw_dir}")
    print(f"Processed data directory: {paths.processed_dir}")
    print(
        f"Train split: {data_cfg.train_split} | "
        f"Val split: {round(1 - (data_cfg.train_split + data_cfg.test_split), 2)} | "
        f"Test split: {data_cfg.test_split}"
    )
    print(f"Pruning classes with < {pre_cfg.min_samples} train samples")

    if not paths.raw_dir.exists() or not paths.raw_dir.is_dir():
        raise typer.BadParameter(f"Raw data directory does not exist: {paths.raw_dir}")
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    if renamed_files:
        rn_dir(paths.raw_dir)
        rn_mp3(paths.raw_dir)

    items, classes = _index_dataset(paths.raw_dir)
    train_items, val_items, test_items = _split_by_groups(items=items, cfg=data_cfg)

    # build raw tensor splits
    train_x, train_y, train_group, train_chunk_starts = _build_split(
        split_items=train_items, pre_cfg=pre_cfg, data_cfg=data_cfg
    )
    val_x, val_y, val_group, val_chunk_starts = _build_split(
        split_items=val_items, pre_cfg=pre_cfg, data_cfg=data_cfg
    )
    test_x, test_y, test_group, test_chunk_starts = _build_split(
        split_items=test_items, pre_cfg=pre_cfg, data_cfg=data_cfg
    )

    # map the excludes train groups to val and test set
    old_to_new, keep_idx, new_class_names = filter_data(
        y_train=train_y,
        min_samples=pre_cfg.min_samples,
        class_names=classes,
    )
    # apply prune to each split
    train_x, train_y, train_group, train_chunk_starts = _apply_mapping_with_meta(
        train_x, train_y, train_group, train_chunk_starts, old_to_new
    )
    val_x, val_y, val_group, val_chunk_starts = _apply_mapping_with_meta(
        val_x, val_y, val_group, val_chunk_starts, old_to_new
    )
    test_x, test_y, test_group, test_chunk_starts = _apply_mapping_with_meta(
        test_x, test_y, test_group, test_chunk_starts, old_to_new
    )

    print(f"After pruning/remap: train={len(train_y)} val={len(val_y)} test={len(test_y)} classes={len(new_class_names or [])}")

    # save labels.json 
    with open(paths.processed_dir / "labels.json", "w", encoding="utf8") as fh:
        json.dump(new_class_names, fh, ensure_ascii=False)

    # compute normalization from train only, save stats, normalize all splits
    mean, std = _compute_global_norm_stats(train_x)
    torch.save(mean, paths.processed_dir / "train_mean.pt")
    torch.save(std, paths.processed_dir / "train_std.pt")

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    # save tensors
    torch.save(train_x, paths.processed_dir / "train_x.pt")
    torch.save(train_y, paths.processed_dir / "train_y.pt")
    torch.save(val_x, paths.processed_dir / "val_x.pt")
    torch.save(val_y, paths.processed_dir / "val_y.pt")
    torch.save(test_x, paths.processed_dir / "test_x.pt")
    torch.save(test_y, paths.processed_dir / "test_y.pt")

    # 8) save meta files
    with open(paths.processed_dir / "train_group.json", "w", encoding="utf8") as fh:
        json.dump(train_group, fh, ensure_ascii=False)
    with open(paths.processed_dir / "val_group.json", "w", encoding="utf8") as fh:
        json.dump(val_group, fh, ensure_ascii=False)
    with open(paths.processed_dir / "test_group.json", "w", encoding="utf8") as fh:
        json.dump(test_group, fh, ensure_ascii=False)

    torch.save(torch.tensor(train_chunk_starts, dtype=torch.float32), paths.processed_dir / "train_chunk_starts.pt")
    torch.save(torch.tensor(val_chunk_starts, dtype=torch.float32), paths.processed_dir / "val_chunk_starts.pt")
    torch.save(torch.tensor(test_chunk_starts, dtype=torch.float32), paths.processed_dir / "test_chunk_starts.pt")

    print("Saved pruned+remapped+normalized splits + labels + meta.")
