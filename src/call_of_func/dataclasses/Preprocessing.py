"""Dataclasses for preprocessing configuration."""

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class PreConfig:
    """Preprocessing configuration."""
    sr: int = 16000  # sampling rate
    clip_sec: float = 5.0 # clip length in seconds
    n_fft: int = 1024 # fft window
    hop_length: int = 512 # hop length
    n_mels: int = 64  # number of mel bands
    fq_min: int = 20 # min frequency
    fq_max: int = 8000# max frequency
    min_rms: float = 0.005 # min rms for valid audio
    min_mel_std: float = 0.10  # min mel std for valid audio
    min_samples: int = 50

@dataclass(frozen=True)
class DataConfig:
    train_split: float = 0.8
    test_split: float = 0.1
    seed: int = 4
    clip_sec: float = 5.0
    stride_sec: float = 2.5
    pad_last: bool = True
 