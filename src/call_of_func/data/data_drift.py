from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)


PROCESSED_DIR = Path("data/processed")  # <-- tilpas til jeres PathConfig.processed_dir
REPORT_PATH = Path("reports/audio_drift_report.html")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_audio_features(x: torch.Tensor) -> pd.DataFrame:
    """
    x: [N, 1, n_mels, time] (som jeres get_data.py gemmer)
    Return: DataFrame med N rækker og numeriske feature-kolonner.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [N,1,n_mels,time], got {tuple(x.shape)}")

    x = x.float()
    # fjern kanal-dim
    S = x[:, 0, :, :]  # [N, n_mels, time]

    # globale stats
    mel_mean = S.mean(dim=(1, 2)).cpu().numpy()
    mel_std = S.std(dim=(1, 2)).cpu().numpy()

    # energi pr. frame (over mels) -> stilhed proxy
    frame_energy = (S**2).mean(dim=1)  # [N, time]
    energy_mean = frame_energy.mean(dim=1).cpu().numpy()
    energy_std = frame_energy.std(dim=1).cpu().numpy()

    # silence ratio: andel af frames under en relativ threshold
    # (threshold = 10% af median-energi pr sample)
    med = frame_energy.median(dim=1).values  # [N]
    thr = 0.1 * med.unsqueeze(1)            # [N,1] broadcast
    silence_ratio = (frame_energy < thr).float().mean(dim=1).cpu().numpy()

    # band energy ratios (low/mid/high mel bins)
    n_mels = S.shape[1]
    low_end = max(1, int(0.33 * n_mels))
    mid_end = max(low_end + 1, int(0.66 * n_mels))

    band_energy = (S**2).mean(dim=2)  # [N, n_mels]
    total = band_energy.sum(dim=1).clamp_min(1e-12)  # [N]

    low_ratio = (band_energy[:, :low_end].sum(dim=1) / total).cpu().numpy()
    mid_ratio = (band_energy[:, low_end:mid_end].sum(dim=1) / total).cpu().numpy()
    high_ratio = (band_energy[:, mid_end:].sum(dim=1) / total).cpu().numpy()

    df = pd.DataFrame(
        {
            "mel_mean": mel_mean,
            "mel_std": mel_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "silence_ratio": silence_ratio,
            "low_band_ratio": low_ratio,
            "mid_band_ratio": mid_ratio,
            "high_band_ratio": high_ratio,
        }
    )
    return df


def main():
    # ---- 3) load tensors ----
    train_x = torch.load(PROCESSED_DIR / "train_x.pt", map_location="cpu")
    test_x = torch.load(PROCESSED_DIR / "test_x.pt", map_location="cpu")

    # optional: target drift kræver target kolonne i begge (hvis I vil)
    train_y = torch.load(PROCESSED_DIR / "train_y.pt", map_location="cpu")
    test_y = torch.load(PROCESSED_DIR / "test_y.pt", map_location="cpu")

    reference_data = extract_audio_features(train_x)
    current_data = extract_audio_features(test_x)

    # Hvis du vil have TargetDriftPreset med:
    reference_data["target"] = train_y.cpu().numpy()
    current_data["target"] = test_y.cpu().numpy()

    # ---- 4) evidently report ----
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(str(REPORT_PATH))

    print(f"Saved drift report to: {REPORT_PATH.resolve()}")
    print(reference_data.head())
    print(current_data.head())


if __name__ == "__main__":
    main()
