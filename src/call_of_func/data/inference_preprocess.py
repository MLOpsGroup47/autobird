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
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig


def inference_load(
    x: np.ndarray,
    sr: int,
    pre_cfg: Optional[PreConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    norm_stats: Optional[Tuple[float, float]] = None,  # (mean, std) from training
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load a single audio file and return a model-ready tensor: [B, 1, n_mels, frames] where B = number of chunks."""
    pre_cfg = pre_cfg or PreConfig()
    data_cfg = data_cfg or DataConfig()

    # 1) load audio

    # 2) chunk audio (should return a list/iterable of chunks)
    chunks = _chunk_audio(x, pre_cfg=pre_cfg, data_cfg=data_cfg)
    if len(chunks) == 0:
        raise ValueError(f"No chunks produced for {file}. Check clip_sec/stride/pad settings.")

    # 3) log-mel per chunk -> list of [n_mels, frames]
    mels: List[np.ndarray] = []
    for chunk in chunks:
        # some implementations return (chunk, start_time) or (chunk, meta...)
        # handle both: if it's a tuple/list, take first element as audio
        if isinstance(chunk, (tuple, list)):
            chunk_audio = chunk[0]
        else:
            chunk_audio = chunk

        mel = _log_mel(chunk_audio, cfg=pre_cfg)  # [n_mels, frames]
        mels.append(mel.astype(np.float32))

    # 4) stack -> [B, n_mels, frames]
    mel_batch = np.stack(mels, axis=0)

    # 5) add channel dim -> [B, 1, n_mels, frames]
    mel_batch = mel_batch[:, None, :, :]

    # 6) to torch
    x_out = torch.from_numpy(mel_batch)  # float32

    # 7) optional normalization (use SAME stats as training)
    if norm_stats is not None:
        mean, std = norm_stats
        std = max(float(std), 1e-8)
        x_out = (x_out - float(mean)) / std

    # 8) optional device
    if device is not None:
        x_out = x_out.to(device)

    return x_out


@torch.no_grad()
def predict_file(
    x: torch.Tensor,
    model: Model,
    paths: PathConfig,
    device: torch.device,
    agg: str = "vote",  # "vote" or "mean_prob"
) -> Dict[str, object]:
    """Predict on ALL chunks in x: [B,1,n_mels,frames].

    Returns a dict with:
      - label (file-level)
      - chunk_preds
      - chunk_probs (optional)
    """
    model.eval()

    # Forward on all chunks
    x = x.to(device)
    logits = model(x)  # [B, C]
    probs = torch.softmax(logits, dim=-1)  # [B, C]
    chunk_preds = probs.argmax(dim=-1)  # [B]

    # Load label map
    label_path = paths.processed_dir / "labels.json"
    with open(label_path) as f:
        idx_to_label = json.load(f)  # keys often strings

    # Aggregate
    if agg == "vote":
        # majority vote on predicted class indices
        vals, counts = torch.unique(chunk_preds, return_counts=True)
        winner = vals[counts.argmax()].item()
        file_label = idx_to_label[winner]
    elif agg == "mean_prob":
        mean_prob = probs.mean(dim=0)  # [C]
        winner = mean_prob.argmax().item()
        file_label = idx_to_label[str(winner)]
    else:
        raise ValueError(f"Unknown agg='{agg}'. Use 'vote' or 'mean_prob'.")

    return {
        "label": file_label,
        "chunk_preds": chunk_preds.detach().cpu().tolist(),
        "chunk_probs": probs.detach().cpu().tolist(),  # remove if too heavy
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
    file = "/Users/holgermaxfloelyng/Desktop/BioMed/MSc_Biomed/SEM_3/project/mlops_project/data/voice_of_birds/West_Mexican_Chachalaca_sound/West_Mexican_Chachalaca4.mp3"
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

    ckpt_path = paths.ckpt_dir / ckpt_name
    state = torch.load(ckpt_path, map_location=device)
    n_classes = int(state["n_classes"])
    hp = state.get("hp", {})  # might be dict or OmegaConf-like



    # Load model
    model = Model(
        n_classes=n_classes,
        d_model=int(hp.get("d_model", 64)),
        n_heads=int(hp.get("n_heads", 2)),
        n_layers=int(hp.get("n_layers", 1)),
    )

    model.load_state_dict(state["model_state"])
    # local defaults (no hydra)

    try:
        x, sr = sf.read(file, always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
    except Exception:
        # fallback: torchaudio
        wav, sr = torchaudio.load(file)  # wav: [channels, time]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x = wav.squeeze(0).cpu().numpy().astype(np.float32)
    
    x = inference_load(x, sr)
    print("Input shape:", x.shape)  # [B, 1, n_mels, frames]



    out = predict_file(x, model=model, paths=paths, device=device, agg="vote")
    print("Predicted:", out["label"])