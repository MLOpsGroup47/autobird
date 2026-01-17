
from typing import Tuple

import librosa
import numpy as np
import torch

from call_of_func.dataclasses.Preprocessing import PreConfig


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