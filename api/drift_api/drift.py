import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, Response
from google.cloud import storage

from evidently.legacy.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.legacy.report import Report


BUCKET = os.getenv("GCP_BUCKET", "birdcage-bucket")
PREFIX = os.getenv("PROCESSED_PREFIX", "data/processed").strip("/")

TRAIN_X = "train_x.pt"
VAL_X   = "val_x.pt"
TRAIN_Y = "train_y.pt"
VAL_Y   = "val_y.pt"

app = FastAPI(title="AutoBird Drift API")


def extract_features(x: torch.Tensor) -> pd.DataFrame:
    if x.ndim != 4:
        raise ValueError(f"Expected [N,1,n_mels,time], got {tuple(x.shape)}")

    x = x.float()
    S = x[:, 0, :, :]  # [N, n_mels, time]

    mel_mean = S.mean(dim=(1, 2)).cpu().numpy()
    mel_std  = S.std(dim=(1, 2)).cpu().numpy()

    frame_energy = (S**2).mean(dim=1)  # [N, time]
    energy_mean = frame_energy.mean(dim=1).cpu().numpy()
    energy_std  = frame_energy.std(dim=1).cpu().numpy()

    return pd.DataFrame(
        {
            "mel_mean": mel_mean,
            "mel_std": mel_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
        }
    )


def gcs_load(name: str) -> torch.Tensor:
    tmp = Path(tempfile.gettempdir()) / "autobird_drift"
    tmp.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    blob = client.bucket(BUCKET).blob(f"{PREFIX}/{name}")
    local = tmp / name

    if not local.exists():
        blob.download_to_filename(str(local))

    return torch.load(local, map_location="cpu")


@app.get("/")
def root():
    return {"service": "autobird-drift"}


@app.get("/drift/run")
def drift_run(sample_n: int = 0):
    train_x = gcs_load(TRAIN_X)
    val_x   = gcs_load(VAL_X)
    train_y = gcs_load(TRAIN_Y)
    val_y   = gcs_load(VAL_Y)

    ref = extract_features(train_x)
    cur = extract_features(val_x)

    ref["target"] = train_y.cpu().numpy()
    cur["target"] = val_y.cpu().numpy()

    if sample_n and 0 < sample_n < len(cur):
        cur = cur.sample(n=sample_n, random_state=42).reset_index(drop=True)

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )
    report.run(reference_data=ref, current_data=cur)
    d = report.as_dict()

    dataset_drift = None
    for m in d.get("metrics", []):
        r = m.get("result", {})
        if "dataset_drift" in r:
            dataset_drift = r["dataset_drift"]

    return {
        "ok": True,
        "dataset_drift": dataset_drift,
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
    }

@app.get("/drift/report")
def drift_report(sample_n: int = 0):
    train_x = gcs_load(TRAIN_X)
    val_x   = gcs_load(VAL_X)
    train_y = gcs_load(TRAIN_Y)
    val_y   = gcs_load(VAL_Y)

    ref = extract_features(train_x)
    cur = extract_features(val_x)

    ref["target"] = train_y.cpu().numpy()
    cur["target"] = val_y.cpu().numpy()

    if sample_n and 0 < sample_n < len(cur):
        cur = cur.sample(n=sample_n, random_state=42).reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    html = report.get_html()
    return Response(content=html, media_type="text/html")

