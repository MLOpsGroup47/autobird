import os
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio  # type: ignore
import typer
from call_of_func.data.get_data import load_data
from call_of_func.train.train_helper import accuracy, rm_rare_classes

# from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile, record_function

from call_of_birds_autobird.model import Model

## setup app, root dir, config file, device and profiler
app = typer.Typer()
root = Path(__file__).resolve().parents[2]  # project root
configs = root / "configs" / "train"
os.chdir(root)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@app.command()
@hydra.main(config_path=str(configs), config_name="hyperparam", version_base=None)
def train(cfg, data_path: str = "data/processed", profile_run: bool = False):
    """Train the model.

    Args:
        cfg (DictConfig): Configuration object from Hydra.
            containing hyperparameters.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            batch_size (int): Batch size.
            sample_min (int): Minimum samples per class.
        data_path (str): Path to the processed data.
        profile_run (bool): Whether to enable profiling.

    Returns:
        None
    """
    print(f"Training on device: {device}")
    print(f"Using data from: {data_path}")
    print(f"Current working directory: {Path.cwd()}")

    # check if run on cpu or cuda
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    prof = None
    if profile_run:  # if true profiler is active
        prof = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("reports/torch_prof"),
        )
        prof.__enter__()
    global_step = 0

    # load hyperparameters 
    epochs = int(cfg.hyperparameters.epochs)
    lr = float(cfg.hyperparameters.lr)
    batch_size = int(cfg.hyperparameters.batch_size)
    sample_min = int(cfg.hyperparameters.sample_min)
    print(f"Training with epochs={epochs}, lr={lr}, batch_size={batch_size}, sample_min={sample_min}")

    fq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)  # freq mask
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)  # time mask

    x_train, y_train, x_val, y_val, classes, *_ = load_data(Path(data_path))
    print(f"Loaded data: train={len(y_train)}, val={len(y_val)}, classes={len(classes)}")
    x_train, y_train, x_val, y_val, classes = rm_rare_classes(  # remove rare classes
        x_train, y_train, x_val, y_val, classes, min_samples=sample_min
    )
    # balanced sampler
    class_counts = torch.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Init model
    model = Model(n_classes=len(classes), d_model=64, n_heads=2, n_layers=1).to(device)

    # criterion, optimizer, cude on? and scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_cuda = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)  # scaler to optimize backpro
    # mby add scheduler later

    dataset_train = data.TensorDataset(x_train, y_train)
    dataset_val = data.TensorDataset(x_val, y_val)
    is_cuda = device.type == "cuda"
    dataloader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )
    dataloader_val = data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=4 if is_cuda else 0, pin_memory=is_cuda
    )

    for epoch in range(epochs):
        # Training
        model.train()
        run_loss = 0.0
        run_acc = 0.0
        total = 0

        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)

            with record_function("specaugment"):
                # specaugment
                x = x.squeeze(1)  # [B, Mels, Time]
                x = fq_mask(x)
                x = time_mask(x)
                x = x.unsqueeze(1)  # [B, 1, Mels, Time]

            if use_cuda:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                bs = x.size(0)
                run_acc += accuracy(logits, y) * bs
                run_loss += loss.item() * bs
                total += bs
        epoch_loss = run_loss / total
        epoch_acc = run_acc / total
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader_val:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                val_loss += loss.item() * bs
                val_acc += accuracy(logits, y) * bs
                total += bs
        val_epoch_loss = val_loss / total
        val_accuracy = val_acc / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    train()
