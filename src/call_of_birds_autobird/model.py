import torch
import torch.nn as nn
from torch import nn


class CNNTransformer(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, max_len: int = 512):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),
        )

        self.proj = nn.Linear(64, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.drop = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, Mels, Frames]
        x = self.cnn(x)  # [B, C, M', T']
        x = x.mean(dim=2)  # [B, C, T']
        x = x.transpose(1, 2)  # [B, T', C]
        x = self.proj(x)  # [B, T', D]

        T = x.size(1)
        if T >self.pos.size(1):
            raise ValueError(
                f"Sequence length T'={T} exceeds max_len={self.pos.size(1)}. "
                f"Increase max_len in the model."
            )
        x = x +self.pos[:,:T,:]
        x = self.drop(x)
        x = self.layernorm(x)
        x = self.transformer(x)  # [B, T', D]
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled)


class Model(nn.Module):
    def __init__(self, n_classes: int = 10, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, max_len= 512):
        super().__init__()
        self.model = CNNTransformer(n_classes=n_classes, d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_len=max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    model = Model(n_classes=10)
    x = torch.randn(4, 1, 64, 128)  # [B, 1, Mels, Frames]
    y = model(x)
    print(y.shape)  # should be [4, 10]
