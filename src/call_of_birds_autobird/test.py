import typer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from pathlib import Path

from call_of_birds_autobird.data import load_data
from call_of_birds_autobird.model import Model

app = typer.Typer()

@app.command()
def overfit(
    data_path: str = "data/processed",
    batch_size: int = 32,
    steps: int = 300,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_val, y_val, classes, *_ = load_data(Path(data_path))

    n_classes = len(classes)
    print("classes:", n_classes)
    print("x_train:", x_train.shape, "y_train:", y_train.shape)
    print("device:", device)

    n_small = min(len(y_train), batch_size * 2)
    x_small = x_train[:n_small]
    y_small = y_train[:n_small]

    loader = data.DataLoader(
        data.TensorDataset(x_small, y_small),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = Model(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(1) == y).float().mean().item()
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d}/{steps}  loss={loss.item():.4f}  acc={acc:.4f}")

if __name__ == "__main__":
    app()
