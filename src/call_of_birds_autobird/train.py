import typer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from dataclasses import dataclass
from pathlib import Path
from call_of_birds_autobird.model import Model
from call_of_birds_autobird.data import load_data

app = typer.Typer()

@dataclass(frozen=True)
class hyperparams:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001

@app.command()
def train(
    data_path: str = "data/processed", 
    epochs: int = hyperparams.epochs, 
    lr: float = hyperparams.learning_rate, 
    batch_size: int = hyperparams.batch_size):
    """Train the model.
    Args:
        data_path (str): Path to the processed data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
    returns:
        None
    """

    x_train, y_train, x_val, y_val, classes, train_group, val_group, train_chunk_starts, val_chunk_starts = load_data(Path(data_path))
    model = Model(n_classes=len(classes))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_train = data.TensorDataset(x_train, y_train)
    dataset_val = data.TensorDataset(x_val, y_val)
    dataloader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for x, y in dataloader_train:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_loss = running_loss / len(dataloader_train.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader_val:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        val_epoch_loss = val_loss / len(dataloader_val.dataset)
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    train()
