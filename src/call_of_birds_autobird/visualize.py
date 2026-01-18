import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colormaps
data_root = Path.cwd().parent / "data" / "processed" / "train_x.pt"
print(f"Data path:{data_root}")
train_x = torch.load(data_root)

def plot_spectogram():
    
    plt.figure(figsize=(12, 6))
    plt.imshow(train_x[12, 0].numpy(), cmap='magma')
    plt.title('Sample from train_x')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
