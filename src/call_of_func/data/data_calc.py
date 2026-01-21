
from typing import Tuple

import numpy as np
import torch
import torchaudio

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



def create_fq_mask(fq_mask: int, time_mask: int):
    try: # if torchaudio fail import for none gpu
        import torchaudio  # type: ignore
    except Exception:
        return None, None
    fq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=fq_mask)  # freq mask
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask)  # time mask
    return fq_mask, time_mask

def specaugment(x: torch.Tensor, fq_mask, time_mask) -> torch.Tensor:
    if fq_mask is None or time_mask is None:  # if torchaudio fail import
        return x  # no-op
    x = x.squeeze(1)  # [B, Mels, Time]
    x = fq_mask(x)
    x = time_mask(x)
    return x.unsqueeze(1)  # [B, 1, Mels, Time]
