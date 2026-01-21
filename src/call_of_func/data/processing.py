import json
from pathlib import Path
from typing import List

import numpy as np
import typer
from omegaconf import DictConfig

from call_of_func.data.data_helpers import rn_dir, rn_mp3
from call_of_func.data.get_data import (
    _index_dataset,
    _save_split,
    _split_by_groups,
)
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
    print(f"Train split: {data_cfg.train_split} | Val split: {round(1-(data_cfg.train_split+data_cfg.test_split), 1)} | test split: {data_cfg.test_split}")

    if not paths.raw_dir.exists() or not paths.raw_dir.is_dir():
        raise typer.BadParameter(f"Raw data directory does not exist: {paths.raw_dir}")
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    if renamed_files:
        rn_dir(paths.raw_dir)
        rn_mp3(paths.raw_dir)

    items, classes = _index_dataset(paths.raw_dir)
    train_items, val_items, test_items = _split_by_groups(items=items, cfg=data_cfg)
    splits = {"train": train_items, "val": val_items, "test": test_items}

    with open(paths.processed_dir / "labels.json", "w", encoding="utf8") as fh:
        json.dump(classes, fh, ensure_ascii=False)

    for split_name, split_items in splits.items():
        _save_split(
            split_name=split_name,
            split_items=split_items,
            paths=paths,
            pre_cfg=pre_cfg,
            data_cfg=data_cfg,
        )
