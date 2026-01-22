from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List
import json
import numpy as np
import torch
from typing import Optional, Tuple, Union, List, Dict
from call_of_birds_autobird.model import Model
from call_of_func.data.get_data import _load_audio, _chunk_audio
from call_of_func.data.data_calc import _log_mel
from call_of_func.dataclasses.Preprocessing import PreConfig, DataConfig
from call_of_func.dataclasses.pathing import PathConfig


def inference_load(
    file: Union[str, Path],
    pre_cfg: Optional[PreConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    norm_stats: Optional[Tuple[float, float]] = None,  # (mean, std) from training
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load a single audio file and return a model-ready tensor:
        [B, 1, n_mels, frames]
    where B = number of chunks.
    """
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Audio file not found: {file}")

    pre_cfg = pre_cfg or PreConfig()
    data_cfg = data_cfg or DataConfig()

    # 1) load audio
    x, sr = _load_audio(str(file))  # x typically: (samples,) or (channels, samples)

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
    paths: PathConfig,
    device: Optional[torch.device] = None,
    ckpt_name: str = "best.pt",
    agg: str = "vote",  # "vote" or "mean_prob"
) -> Dict[str, object]:
    """
    Predict on ALL chunks in x: [B,1,n_mels,frames]
    Returns a dict with:
      - label (file-level)
      - chunk_preds
      - chunk_probs (optional)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Model()
    ckpt_path = paths.ckpt_dir / ckpt_name
    state = torch.load(ckpt_path, map_location=device)
    n_classes = int(state["n_classes"])
    hp = state.get("hp", {})  # might be dict or OmegaConf-like

    model = Model(
        n_classes=n_classes,
        d_model=int(hp.get("d_model", 64)),
        n_heads=int(hp.get("n_heads", 2)),
        n_layers=int(hp.get("n_layers", 1)),
    )
    model.load_state_dict(state["model_state"])
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
    file = "data/voice_of_birds/Highland_Tinamou_sound/Highland_Tinamou5.mp3"

    # local defaults (no hydra)
    x = inference_load(file)
    print("Input shape:", x.shape)  # [B, 1, n_mels, frames]

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

    out = predict_file(x, paths=paths, agg="vote")
    print("Predicted:", out["label"])