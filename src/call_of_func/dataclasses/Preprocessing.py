"""Dataclasses for preprocessing configuration."""

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class PreConfig:
    """Preprocessing configuration."""
    sr: int  # sampling rate
    clip_sec: float # clip length in seconds
    n_fft: int  # fft window
    hop_length: int # hop length
    n_mels: int  # number of mel bands
    fq_min: int # min frequency
    fq_max: int  # max frequency
    min_rms: float  # min rms for valid audio
    min_mel_std: float  # min mel std for valid audio


@dataclass(frozen=True)
class DataConfig:
    train_split: float 
    seed: int
    clip_sec: float
    stride_sec: float 
    pad_last: bool
