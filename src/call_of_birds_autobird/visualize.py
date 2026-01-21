from pathlib import Path

import matplotlib.pyplot as plt
import torch
from call_of_func.utils.get_source_path import CONFIG_DIR, ROOT
from matplotlib import colormaps

data_root = ROOT / "data" / "processed" / "train_x.pt"
print(f"Data path:{data_root}")
train_x = torch.load(data_root)

def plot_spectogram():
    plt.figure(figsize=(12, 6))
    plt.imshow(train_x[12, 0].numpy(), cmap='magma')
    plt.title('Sample from train_x')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_label_distribution(y, title="Label distribution"):
    import matplotlib.pyplot as plt
    import torch

    counts = torch.bincount(y)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(counts)), counts.numpy())
    plt.xlabel("Class index")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

