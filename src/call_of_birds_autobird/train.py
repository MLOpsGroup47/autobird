import typer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchaudio
from dataclasses import dataclass
from pathlib import Path
from call_of_birds_autobird.model import Model
from call_of_birds_autobird.data import load_data
from call_of_func.train_helper import rm_rare_classes

app = typer.Typer()

def accuracy(logits, y) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclass(frozen=True)
class hyperparams:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001

@app.command()
def train(
    data_path: str = "data/processed", 
    epochs: int = hyperparams.epochs, 
    lr: float = hyperparams.learning_rate, 
    batch_size: int = hyperparams.batch_size,
    sample_min: int = 100
    ):
    """Train the model.
    Args:
        data_path (str): Path to the processed data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
    returns:
        None
    """
    fq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8) # freq mask
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20) # time mask

    x_train, y_train, x_val, y_val, classes, *_ = load_data(Path(data_path))
    print(f"Loaded data: train={len(y_train)}, val={len(y_val)}, classes={len(classes)}")
    x_train, y_train, x_val, y_val, classes = rm_rare_classes( # remove rare classes
        x_train, y_train, x_val, y_val, classes, 
        min_samples=sample_min
    )
    model = Model(n_classes=len(classes)).to(device)
    
    # balanced sampler
    class_counts = torch.bincount(y_train)
    sample_weights = 1.0 / class_counts[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset_train = data.TensorDataset(x_train, y_train)
    dataset_val = data.TensorDataset(x_val, y_val)
    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=2, pin_memory=True)
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for epoch in range(epochs):
        # Training
        model.train()
        run_loss = 0.0
        run_acc = 0.0
        total = 0

        for x, y in dataloader_train:
            x = x.to(device)
            y = y.to(device)

            # specaugment
            x = x.squeeze(1)  # [B, Mels, Time]
            x = fq_mask(x)
            x = time_mask(x)
            x = x.unsqueeze(1)  # [B, 1, Mels, Time]

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            run_acc += accuracy(logits, y) 
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
                val_acc += accuracy(logits, y)
                total += bs
        val_epoch_loss = val_loss / total
        val_accuracy = val_acc / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    train()
