from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F

from call_of_birds_autobird.model import Model
from call_of_func.data.data_calc import _log_mel
from call_of_func.data.get_data import _chunk_audio
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig


def _idx_to_name(idx_to_label, i: int) -> str:
    """
    labels.json may be either:
      - list: [name0, name1, ...]
      - dict: {"0": name0, "1": name1, ...}
    """
    if isinstance(idx_to_label, list):
        return idx_to_label[i]
    return idx_to_label[str(i)]


def _load_norm_stats(processed_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_path = processed_dir / "train_mean.pt"
    std_path = processed_dir / "train_std.pt"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(f"Missing norm stats: {mean_path} or {std_path}")

    mean = torch.load(mean_path, map_location="cpu")
    std = torch.load(std_path, map_location="cpu")

    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-8)

    return mean, std



def _resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return x, sr

    wav = torch.from_numpy(x).float()
    if wav.ndim != 1:
        wav = wav.view(-1)
    wav = wav.unsqueeze(0)  # [1, T]

    wav = F.resample(wav, orig_freq=sr, new_freq=target_sr)

    x_rs = wav.squeeze(0).cpu().numpy().astype(np.float32)
    return x_rs, target_sr


def inference_load(
    x: np.ndarray,
    sr: int,
    pre_cfg: PreConfig,
    data_cfg: DataConfig,
    norm_stats: Tuple[float, float],  # (mean, std) from training
    device: torch.device,
) -> torch.Tensor:
    """
    Build model-ready tensor: [B, 1, n_mels, frames] where B is number of chunks.
    Ensures preprocessing matches training:
      - resample to pre_cfg.sr
      - same chunking settings (clip/stride/pad)
      - same log-mel params
      - same normalization stats
    """
    # 1) resample to training sr
    x, sr = _resample_if_needed(x, sr, int(pre_cfg.sr))

    # 2) chunk audio
    chunks = _chunk_audio(x, pre_cfg=pre_cfg, data_cfg=data_cfg)
    if len(chunks) == 0:
        raise ValueError("No chunks produced. Check clip_sec/stride/pad settings.")

    # 3) log-mel per chunk
    mels: List[np.ndarray] = []
    for chunk in chunks:
        chunk_audio = chunk[0] if isinstance(chunk, (tuple, list)) else chunk
        mel = _log_mel(chunk_audio, cfg=pre_cfg)  # [n_mels, frames]
        mels.append(mel.astype(np.float32))

    # 4) stack -> [B, n_mels, frames]
    mel_batch = np.stack(mels, axis=0)

    # 5) add channel dim -> [B, 1, n_mels, frames]
    mel_batch = mel_batch[:, None, :, :]

    # 6) to torch
    x_out = torch.from_numpy(mel_batch)  # float32, CPU

    # 7) normalize with TRAIN stats (supports scalar OR per-mel vector)
    mean, std = norm_stats
    mean_t = mean.to(dtype=torch.float32)
    std_t = std.to(dtype=torch.float32).clamp_min(1e-8)

    # x_out: [B, 1, n_mels, frames]
    if mean_t.numel() == 1:
        x_out = (x_out - mean_t) / std_t
    else:
        # assume per-mel stats: [n_mels]
        if mean_t.ndim != 1:
            mean_t = mean_t.view(-1)
        if std_t.ndim != 1:
            std_t = std_t.view(-1)

        # reshape to broadcast over [B, 1, n_mels, frames]
        mean_t = mean_t.view(1, 1, -1, 1)
        std_t = std_t.view(1, 1, -1, 1)

        x_out = (x_out - mean_t) / std_t

    # 8) move to device
    x_out = x_out.to(device)
    return x_out


@torch.no_grad()
def predict_file(
    x: torch.Tensor,  # [B, 1, n_mels, frames]
    model: Model,
    processed_dir: Path,
    agg: str = "vote",  # "vote" or "mean_prob"
) -> Dict[str, object]:
    """
    Predict on all chunks; aggregate to file-level label.
    """
    model.eval()

    logits = model(x)                 # [B, C]
    probs = torch.softmax(logits, -1) # [B, C]
    chunk_preds = probs.argmax(-1)    # [B]

    # labels.json
    label_path = processed_dir / "labels.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}")
    with open(label_path, "r", encoding="utf8") as f:
        idx_to_label = json.load(f)

    if agg == "vote":
        vals, counts = torch.unique(chunk_preds, return_counts=True)
        winner = int(vals[counts.argmax()].item())
        file_label = _idx_to_name(idx_to_label, winner)

    elif agg == "mean_prob":
        mean_prob = probs.mean(dim=0)  # [C]
        winner = int(mean_prob.argmax().item())
        file_label = _idx_to_name(idx_to_label, winner)

    else:
        raise ValueError(f"Unknown agg='{agg}'. Use 'vote' or 'mean_prob'.")

    return {
        "label": file_label,
        "winner_idx": winner,
        "chunk_preds": chunk_preds.detach().cpu().tolist(),
        "chunk_probs": probs.detach().cpu().tolist(),
    }


def _read_audio_mono(file: str) -> Tuple[np.ndarray, int]:
    """
    Read audio as mono float32 numpy + sr.
    Tries soundfile first, falls back to torchaudio.
    """
    try:
        x, sr = sf.read(file, always_2d=False)
        if isinstance(x, np.ndarray) and x.ndim == 2:
            # soundfile returns [T, C]
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        return x, int(sr)
    except Exception:
        wav, sr = torchaudio.load(file)  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x = wav.squeeze(0).cpu().numpy().astype(np.float32)
        return x, int(sr)


if __name__ == "__main__":
    # -----------------------------
    # USER SETTINGS
    # -----------------------------
    file = "/Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/project/mlops_project/data/voice_of_birds/West_Mexican_Chachalaca_sound/West_Mexican_Chachalaca28.mp3"
    ckpt_name = "best.pt"

    paths = PathConfig(
        root=Path("."),
        raw_dir=Path("data/voice_of_birds"),
        processed_dir=Path("data/processed"),
        reports_dir=Path("reports/figures"),
        eval_dir=Path("reports/eval"),
        ckpt_dir=Path("models/checkpoints"),
        x_train=Path("data/processed/train_x.pt"),
        y_train=Path("data/processed/train_y.pt"),
        x_val=Path("data/processed/val_x.pt"),
        y_val=Path("data/processed/val_y.pt"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # LOAD CHECKPOINT + MODEL
    # -----------------------------
    ckpt_path = paths.ckpt_dir / ckpt_name
    state = torch.load(ckpt_path, map_location="cpu")

    n_classes = int(state["n_classes"])
    hp = state.get("hp", {})

    model = Model(
        n_classes=n_classes,
        d_model=int(hp.get("d_model", 64)),
        n_heads=int(hp.get("n_heads", 2)),
        n_layers=int(hp.get("n_layers", 1)),
    ).to(device)

    model.load_state_dict(state["model_state"], strict=True)

    # -----------------------------
    # BUILD PRE/DATA CFG (MUST MATCH TRAINING)
    # -----------------------------
    # IMPORTANT: set these to the SAME values used in preprocessing/training.
    # If your project stores them in Hydra cfg, you should load them from there.
    pre_cfg = PreConfig(
        sr=16000,
        clip_sec=5.0,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        fq_min=20,
        fq_max=8000,
        min_rms=0.0,
        min_mel_std=0.0,
        min_samples=0,
    )

    data_cfg = DataConfig(
        train_split=0.8,
        test_split=0.1,
        seed=4,
        clip_sec=5.0,
        stride_sec=2.5,
        pad_last=True,
    )

    # -----------------------------
    # LOAD AUDIO + INFERENCE FEATURES
    # -----------------------------
    x_np, sr = _read_audio_mono(file)


    mean_std = _load_norm_stats(paths.processed_dir)

    x_in = inference_load(
    x=x_np,
    sr=sr,
    pre_cfg=pre_cfg,
    data_cfg=data_cfg,
    norm_stats=mean_std,
    device=device,
    )
    print("Input shape:", x_in.shape)  # [B, 1, n_mels, frames]

    # -----------------------------
    # PREDICT
    # -----------------------------
    out = predict_file(
        x=x_in,
        model=model,
        processed_dir=paths.processed_dir,
        agg="vote",
    )
    print("Predicted:", out["label"], "| idx:", out["winner_idx"])
