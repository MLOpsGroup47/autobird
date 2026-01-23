import os
import pickle
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from call_of_birds_autobird.model import Model
from call_of_func.data.inference_preprocess import inference_load, predict_file
from call_of_func.dataclasses.pathing import PathConfig
from fastapi import BackgroundTasks, FastAPI, File, UploadFile

# Global state for model, device and paths
model: Model
device: torch.device
paths: PathConfig

BUCKET_FOLDER = "/gcs/birdcage-bucket/"
MODEL_NAME = "last.pt"

def resolve_checkpoint_path(ckpt_name: str, paths: PathConfig) -> Path: # Return Path specifically
    """Resolve checkpoint path based on environment."""
    if os.getenv("K_SERVICE") or os.getenv("RUNNING_IN_GCP") == "1":
        # Ensure the GCS path is also a Path object for consistency
        return Path(BUCKET_FOLDER) / "models" / ckpt_name
    else:
        # Cast to Path to support the / operator
        return Path(paths.ckpt_dir) / ckpt_name

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("Loading model for inference...")
    global model, device, paths


    paths = PathConfig( 
        root=Path("."),
        raw_dir=Path("data/voice_of_birds"),
        processed_dir=Path("data/processed"),
        reports_dir=Path("reports/figures"),
        eval_dir=Path("reports/eval"),
        ckpt_dir= Path("models/checkpoints"),
        x_train=Path("data/processed/train_x.pt"),
        y_train=Path("data/processed/train_y.pt"),
        x_val=Path("data/processed/val_x.pt"),
        y_val=Path("data/processed/val_y.pt"),
    )
    
    if not (Path.cwd() / "pyproject.toml").exists():
        paths = PathConfig( 
            root=Path("."),
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

    paths = paths.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint_path(MODEL_NAME, paths)
    state = torch.load(ckpt_path, map_location=device)
    n_classes = int(state["n_classes"])
    if isinstance(state.get("hp"), dict):
        class HP:
            pass
        hp = HP()
        for k, v in state["hp"].items():
            setattr(hp, k, v)
        state["hp"] = hp
    
    hp = state.get("hp", {})  # might be dict or OmegaConf-like

    # Load model
    model = Model(
        n_classes=n_classes,
        d_model=int(hp.d_model),
        n_heads=int(hp.n_heads),
        n_layers=int(hp.n_layers),
    ).to(device)

    model.load_state_dict(state["model_state"])


    with open("prediction_database.csv", "w") as file:
        file.write("time, audio_file, prediction\n")


    yield

    print("Cleaning up")
    
    del model, device, paths



app = FastAPI(lifespan=lifespan)


def add_to_database(
    now: str,
    audio_file: str,
    prediction: str,
) -> None:
    """Simple function to add prediction to database."""
    with open("prediction_database.csv", "a") as file:       
        file.write(f"{now}, {audio_file}, {prediction}\n")


@app.get("/")
def read_root():
    
    return {"Hello": "Welcome to AutoBird Inference API",
            "Help":"Use /classify/ endpoint to POST an audio file for classification."}

@app.get("/files/")
def list_files():
    """List model files in the upload folder."""
    # Use os.fspath or Path conversion to satisfy Mypy
    if os.getenv("K_SERVICE") or os.getenv("RUNNING_IN_GCP") == "1":
        base = Path(BUCKET_FOLDER) / "models"
    else:
        base = Path(paths.ckpt_dir)
    
    try:
        # os.listdir works best with str; .iterdir() is the Path alternative
        files = sorted([f.name for f in base.iterdir() if f.is_file()])
    except Exception:
        files = []
    return {"files": files}


@app.post("/classify/")
async def caption(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
):
    try:
        try:
            x_in, sr = sf.read(audio.file, always_2d=False)
            if x_in.ndim == 2:
                x_in = x_in.mean(axis=1)
            x_in = x_in.astype(np.float32)
            sr = int(sr)
        except Exception as e:
            print(f"SF Read Error: {e}")
            # fallback: torchaudio
            audio.file.seek(0)
            wav, sr = torchaudio.load(audio.file)  # wav: [channels, time]
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            x_in = wav.squeeze(0).cpu().numpy().astype(np.float32)
            sr = int(sr)

        x = inference_load(
            x=x_in,
            sr=sr,
            device=device,
            path_cfg=paths,
        )

        out = predict_file(x, model=model, paths=paths, device=device, agg="vote")
        now = str(datetime.now(tz=timezone.utc))
        background_tasks.add_task(add_to_database, now, str(audio.filename), str(out['label']))

        return out['label']
    except Exception as e:
        print(f"Server Error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    


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


