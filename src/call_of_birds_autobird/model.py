from torch import nn
import torch

import torch.nn as nn
import torchaudio


class CNNTransformer(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.proj = nn.Linear(64, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.head = nn.Linear(d_model, n_classes) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, Mels, Frames]
        x = self.cnn(x)                    # [B, C, M', T']
        x = x.mean(dim=2)                  # [B, C, T']
        x = x.transpose(1, 2)              # [B, T', C]
        x = self.proj(x)                   # [B, T', D]
        x = self.transformer(x)            # [B, T', D]
        x = x.mean(dim=1)                  # temporal pooling
        return self.head(x)
    
class Model(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.model = CNNTransformer(n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

if __name__ == "__main__":
    model = Model(n_classes=10)
    x = torch.randn(4, 1, 64, 128)  # [B, 1, Mels, Frames]
    y = model(x)
    print(y.shape)  # should be [4, 10]
