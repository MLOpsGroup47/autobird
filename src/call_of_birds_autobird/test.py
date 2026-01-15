import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import typer
from omegaconf import DictConfig, OmegaConf

from call_of_birds_autobird.data import load_data
from call_of_birds_autobird.model import Model

app = typer.Typer()

ROOT = Path(__file__).resolve().parents[2] 
os.chdir(ROOT)
print(f"Real current working directory: {Path.cwd()}")
config_dir = ROOT / "configs"
print(f"Config directory: {config_dir}")

@app.command()
@hydra.main(config_path= str(config_dir), config_name="config", version_base=None)
def overfit(cfg: DictConfig, data_path: str = "data/processed"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_val, y_val, classes, *_ = load_data(data_path)
    n_classes = len(classes)

    bs = int(cfg.hyperparameters.batch_size)
    steps = int(cfg.hyperparameters.step)  # or rename to "steps" in yaml
    lr = float(cfg.hyperparameters.lr)

    print("classes:", n_classes)
    print("device:", device)
    print("hp:", {"lr": lr, "batch_size": bs, "steps": steps})

    n_small = min(len(y_train), bs * 2)
    x_small = x_train[:n_small]
    y_small = y_train[:n_small]

    loader = data.DataLoader(
        data.TensorDataset(x_small, y_small),
        batch_size=bs,
        shuffle=True,
        drop_last=True,
    )

    model = Model(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    it = iter(loader)
    for i in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(1) == y).float().mean().item()
        if i % 25 == 0 or i == 1:
            print(f"step {i:4d}/{steps}  loss={loss.item():.4f}  acc={acc:.4f}")

if __name__ == "__main__":
    overfit()   