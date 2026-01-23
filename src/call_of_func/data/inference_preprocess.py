from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from call_of_birds_autobird.model import Model
from fastapi import FastAPI, File, UploadFile

from call_of_func.data.data_calc import _log_mel
from call_of_func.data.get_data import _chunk_audio
from call_of_func.data.new_inference import _load_norm_stats
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig


def inference_load(
    x: np.ndarray,
    sr: int,
    path_cfg: PathConfig,
    pre_cfg: Optional[PreConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load a single audio file and return a model-ready tensor."""
    pre_cfg = pre_cfg or PreConfig()
    data_cfg = data_cfg or DataConfig()

    # 1) chunk audio
    chunks = _chunk_audio(x, pre_cfg=pre_cfg, data_cfg=data_cfg)
    if len(chunks) == 0:
        # Fixed the 'file' name error here by using a generic message
        raise ValueError("No chunks produced. Check audio length vs clip_sec settings.")

    # 2) log-mel per chunk
    mels: List[np.ndarray] = []
    for chunk in chunks:
        chunk_audio = chunk[0] if isinstance(chunk, (tuple, list)) else chunk
        mel = _log_mel(chunk_audio, cfg=pre_cfg)
        mels.append(mel.astype(np.float32))

    # 3) stack and add channel dim -> [B, 1, n_mels, frames]
    mel_batch = np.stack(mels, axis=0)[:, None, :, :]
    x_out = torch.from_numpy(mel_batch)

    # 5) add channel dim -> [B, 1, n_mels, frames]
    mel_batch = mel_batch[:, None, :, :]

    # 6) to torch
    x_out = torch.from_numpy(mel_batch)  # float32

    # 7) optional normalization (use SAME stats as training)
    norm_stats = _load_norm_stats(processed_dir=Path(path_cfg.processed_dir))

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

    if device is not None:
        x_out = x_out.to(device)

    return x_out.squeeze(1) if x_out.shape[1]==1 and x_out.shape[2] == 1 else x_out  # remove channel dim if model expects [B, n_mels, frames]

@torch.no_grad()
def predict_file(
    x: torch.Tensor,
    model: Model,
    paths: PathConfig,
    device: torch.device,
    agg: str = "vote",
) -> Dict[str, object]:
    model.eval()
    x = x.to(device)
    
    logits = model(x)  
    probs = torch.softmax(logits, dim=-1)  
    chunk_preds = probs.argmax(dim=-1)  

    # Type narrowing for Mypy safety
    p_proc = Path(paths.processed_dir)
    label_path = p_proc / "labels.json"
    
    with open(label_path, "r", encoding="utf8") as f:
        idx_to_label = json.load(f)

    if agg == "vote":
        vals, counts = torch.unique(chunk_preds, return_counts=True)
        winner = int(vals[counts.argmax()].item())
        # JSON keys are strings, so cast winner to str
        file_label = idx_to_label[str(winner)] 
    elif agg == "mean_prob":
        mean_prob = probs.mean(dim=0)
        winner = int(mean_prob.argmax().item())
        file_label = idx_to_label[str(winner)]
    else:
        raise ValueError(f"Unknown agg='{agg}'.")

    return {
        "label": file_label,
        "winner_idx": winner,
        "chunk_preds": chunk_preds.cpu().tolist(),
        "chunk_probs": probs.cpu().tolist(),
    }


def paths_from_hydra_cfg(cfg) -> PathConfig:
    return PathConfig(
        root=Path(cfg.pathing.paths.root),
        raw_dir=Path(cfg.pathing.paths.raw_dir),
        processed_dir=Path(cfg.pathing.paths.processed_dir),
        reports_dir=Path(cfg.pathing.paths.reports_dir),
        eval_dir=Path(cfg.pathing.paths.eval_dir),
        ckpt_dir=Path(cfg.pathing.paths.ckpt_dir),
        x_train=Path(cfg.pathing.paths.x_train),
        y_train=Path(cfg.pathing.paths.y_train),
        x_val=Path(cfg.pathing.paths.x_val),
        y_val=Path(cfg.pathing.paths.y_val),
    )


if __name__ == "__main__":
    file = "data/voice_of_birds/Brazilian_Tinamou_sound/Brazilian_Tinamou16.mp3"
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

    # --- FIX START ---
    # Narrow the type to Path to allow the / operator
    p_ckpt = Path(paths.ckpt_dir)
    ckpt_path = p_ckpt / ckpt_name
    # --- FIX END ---

    state = torch.load(ckpt_path, map_location=device)
    n_classes = int(state["n_classes"])
    hp = state.get("hp", {}) 

    # Load model
    model = Model(
        n_classes=n_classes,
        d_model=int(hp.get("d_model", 64)),
        n_heads=int(hp.get("n_heads", 2)),
        n_layers=int(hp.get("n_layers", 1)),
    )

    model.load_state_dict(state["model_state"])

    try:
        x_raw, sr = sf.read(file, always_2d=False)
        if x_raw.ndim == 2:
            x_raw = x_raw.mean(axis=1)
    except Exception:
        wav, sr = torchaudio.load(file) 
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x_raw = wav.squeeze(0).cpu().numpy().astype(np.float32)
    
    # Process audio into mel-spectrogram chunks
    x_tensor = inference_load(x_raw, sr, path_cfg=paths, device=device)
    print("Input shape:", x_tensor.shape)  # [B, 1, n_mels, frames]

    # Run prediction
    out = predict_file(x_tensor, model=model, paths=paths, device=device, agg="vote")
    print("Predicted:", out["label"])