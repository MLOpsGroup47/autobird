
from typing import Tuple

import torchaudio
import numpy as np
import torch

from call_of_func.dataclasses.Preprocessing import PreConfig


def _log_mel(x: np.ndarray, cfg: PreConfig) -> np.ndarray:
    """Compute log-mel spectrogram: shape [n_mels, time]."""
    x_t = torch.from_numpy(x).float().unsqueeze(0)  # [1, time]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.fq_min,
        f_max=min(cfg.fq_max, cfg.sr // 2),
        power=2.0,
    )(x_t)  # [1, n_mels, time]

    S = torch.log(mel + 1e-6).squeeze(0).cpu().numpy().astype(np.float32)  # [n_mels, time]
    return S



def _compute_global_norm_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std over dataset tensor X: shape [N, 1, Mels, Time]."""
    mean = X.mean(dim=(0, 1, 3), keepdim=True)  # [1, 1, Mels, 1]
    std = X.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-8)  # [1, 1, Mels, 1]
    return mean, std