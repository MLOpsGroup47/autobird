from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass
class hyperparameters:
    epochs: int = 15
    lr: float = 3e-4
    batch_size: int = 32
    sample_min: int = 50
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 1
    pin_memory: bool = False
    shuffle_train: bool = True
    shuffle_val: bool = False
    num_workers: int = 2
    amp: bool = True
    grad_clip: float = 1.0

@dataclass
class OptimizerConfig:
    type: str = "torch.optim.Adam"
    lr: float = 3e-4
    weight_decay: float = 1e-4

@dataclass
class SchedulerConfig:
    type: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 5
    gamma: float = 0.1

@dataclass
class TrainConfig:
    hyperparameters: hyperparameters = field(default_factory=hyperparameters)
    Optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    Scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)