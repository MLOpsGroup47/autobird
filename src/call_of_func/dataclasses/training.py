from dataclasses import dataclass
from typing import FrozenSet

@dataclass
class hyperparameter():
    epochs: int = 15
    lr: float = 0.0003
    batch_size: int = 32
    sample_min: int = 50
    d_model: int = 64
    n_head: int = 2
    n_layers: int = 1
    pin_memory: bool = False
    shuffle_train: bool = True
    shuffle_val: bool = False
    num_workers: int = 2

