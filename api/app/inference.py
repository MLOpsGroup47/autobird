from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from call_of_birds_autobird.model import Model
from call_of_func.data.inference_preprocess import inference_load, predict_file
from call_of_func.dataclasses.pathing import PathConfig
from fastapi import FastAPI, File, UploadFile

BUCKET_FOLDER = "/gcs/birdcage-bucket/"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("Loading model for inference...")
    global model, device, paths

    ckpt_name = "last.pt"

    paths = PathConfig(
        root=Path("../../"),
        raw_dir=Path("../../data/voice_of_birds"),
        processed_dir=Path("../../data/processed"),
        reports_dir=Path("../../reports/figures"),
        eval_dir=Path("../../reports/eval"),
        ckpt_dir= Path("../../models/checkpoints"),
        x_train=Path("../../data/processed/train_x.pt"),
        y_train=Path("../../data/processed/train_y.pt"),
        x_val=Path("../../data/processed/val_x.pt"),
        y_val=Path("../../data/processed/val_y.pt"),
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

    model.to(device)

    yield

    print("Cleaning up")
    
    del model, device, paths



app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    
    return {"Hello": "Welcome to AutoBird Inference API",
            "Help":"Use /classify/ endpoint to POST an audio file for classification."}

@app.post("/classify/")
async def caption(audio: UploadFile = File(...)):


    try:
        x, sr = sf.read(audio.file, always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        sr = int(sr)
    except Exception:
        # fallback: torchaudio
        wav, sr = torchaudio.load(audio.file)  # wav: [channels, time]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x = wav.squeeze(0).cpu().numpy().astype(np.float32)
        sr = int(sr)

    x = inference_load(x, sr)
    print("Input shape:", x.shape)  # [B, 1, n_mels, frames]



    out = predict_file(x, model=model, paths=paths, device=device, agg="vote")
    return out['label']
    


def predict_step(file: str, model: Model, paths: PathConfig, device: torch.device):
    
    # Simulate FastAPI UploadFile by reading file and wrapping in BytesIO

    
    try:
        x, sr = sf.read(file, always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        sr = int(sr)
    except Exception:
        # fallback: torchaudio
        wav, sr = torchaudio.load(file)  # wav: [channels, time]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        x = wav.squeeze(0).cpu().numpy().astype(np.float32)
        sr = int(sr)

    x = inference_load(x, sr)
    print("Input shape:", x.shape)  # [B, 1, n_mels, frames]



    out = predict_file(x, model=model, paths=paths, device=device, agg="vote")
    return out



if __name__ == "__main__":
    ckpt_name = "last.pt"

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

    model.to(device)
      
    print(predict_step('data/voice_of_birds/Andean Guan_sound/Andean Guan2.mp3', model=model, paths=paths, device=device)['label'])