from pathlib import Path

import torch

p = Path("data/processed")
xtr = torch.load(p/"train_x.pt")
ytr = torch.load(p/"train_y.pt")
xva = torch.load(p/"val_x.pt")
yva = torch.load(p/"val_y.pt")

print("train_x:", tuple(xtr.shape), xtr.dtype, "mean/std:", float(xtr.mean()), float(xtr.std()))
print("train_y:", tuple(ytr.shape), ytr.dtype, "min/max:", int(ytr.min()), int(ytr.max()))
print("val_x:", tuple(xva.shape), xva.dtype, "mean/std:", float(xva.mean()), float(xva.std()))
print("val_y:", tuple(yva.shape), yva.dtype, "min/max:", int(yva.min()), int(yva.max()))
