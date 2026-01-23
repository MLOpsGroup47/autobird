from pathlib import Path

import pandas as pd
import torch
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.legacy.report import Report
from hydra import compose, initialize

from call_of_func.dataclasses.pathing import PathConfig


def extract_audio_features(x: torch.Tensor) -> pd.DataFrame:
    """Extracts structured (tabular) features from log-mel tensors.

    Args:
        x: Tensor of shape [N, 1, n_mels, time] (like in the preprocessing pipeline).

    Returns:
        A pandas DataFrame with one row per sample and numeric feature columns.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [N, 1, n_mels, time], got {tuple(x.shape)}")

    x = x.float()
    S = x[:, 0, :, :]  # [N, n_mels, time]

    # Global statistics over the entire log-mel spectrogram
    mel_mean = S.mean(dim=(1, 2)).cpu().numpy()
    mel_std = S.std(dim=(1, 2)).cpu().numpy()

    # Frame-wise energy (mean over mel bins) -> indicator for loudness / silence
    frame_energy = (S**2).mean(dim=1)  # [N, time]
    energy_mean = frame_energy.mean(dim=1).cpu().numpy()
    energy_std = frame_energy.std(dim=1).cpu().numpy()

    return pd.DataFrame(
        {
            "mel_mean": mel_mean,
            "mel_std": mel_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
        }
    )


def main() -> None:
    """Create an Evidently drift report by comparing reference (train) vs current (val) feature distributions.
   
    HTML report saved under reports/monitoring/
   
    """
    # Path loading from hydraconfiguration
    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(config_name="pathing/path_config")

    paths = cfg.paths

    monitoring_dir = Path(paths.reports_dir) / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    report_path = monitoring_dir / "audio_drift_report.html"
    
    train_x = torch.load(paths.x_train, map_location="cpu")
    val_x   = torch.load(paths.x_val, map_location="cpu")
    train_y = torch.load(paths.y_train, map_location="cpu")
    val_y   = torch.load(paths.y_val, map_location="cpu")

    # Extract tabular features for drift detection
    reference_data = extract_audio_features(train_x)
    current_data = extract_audio_features(val_x)

    # Add target column to enable target drift checks
    reference_data["target"] = train_y.cpu().numpy()
    current_data["target"] = val_y.cpu().numpy()

    # Create and run Evidently report
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(str(report_path))

    print(f"Saved drift report to: {report_path.resolve()}")
    print("Reference data preview:")
    print(reference_data.head())
    print("Current data preview:")
    print(current_data.head())


if __name__ == "__main__":
    main()
