from pathlib import Path

import hydra
import pytest
import torch
import torch.nn as nn
from call_of_birds_autobird.model import Model
from call_of_func.data.data_calc import create_fq_mask
from call_of_func.train.get_dataloader import build_dataloader
from call_of_func.train.get_optim import build_optimizer
from call_of_func.train.train_engine import train_one_epoch
from call_of_func.train.train_helper import _get_device, accuracy
from call_of_func.utils.get_source_path import CONFIG_DIR
from call_of_func.utils.get_trackers import build_profiler
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler


def test_accuracy():
    """Test the accuracy calculation."""
    # Batch size 4, 3 classes
    logits = torch.tensor([
        [10.0, 1.0, 0.0],  # Pred: 0
        [1.0, 10.0, 0.0],  # Pred: 1
        [0.0, 1.0, 10.0],  # Pred: 2
        [10.0, 0.0, 0.0],  # Pred: 0
    ])
    
    # All correct
    y_true = torch.tensor([0, 1, 2, 0])
    acc = accuracy(logits, y_true)
    assert acc == 1.0, "Accuracy should be 1.0 for all correct predictions"
    
    # None correct
    y_wrong = torch.tensor([1, 2, 0, 1])
    acc_wrong = accuracy(logits, y_wrong)
    assert acc_wrong == 0.0, "Accuracy should be 0.0 for all incorrect predictions"
    
    # Half correct
    y_half = torch.tensor([0, 0, 2, 1]) # Correct, Wrong, Correct, Wrong
    acc_half = accuracy(logits, y_half)
    assert acc_half == 0.5, "Accuracy should be 0.5 for half correct predictions"




def test_single_train_step():
    """Smoke test for a single training step."""   
    device = torch.device("cpu") # Test on CPU for CI compatibility
    n_classes = 5
    batch_size = 4
    
    # Initialize components
    model = Model(n_classes=n_classes, d_model=32, n_heads=2, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy batch [B, 1, Mels, Time]
    # Model expects [B, 1, Mels, Frames]
    # Mels=64 (default in some configs, but flexible), Time=128
    input_shape = (batch_size, 1, 64, 128)
    x = torch.randn(*input_shape).to(device)
    y = torch.randint(0, n_classes, (batch_size,)).to(device)
    
    # Initial mode
    model.train()
    
    # Forward
    logits = model(x)
    assert logits.shape == (batch_size, n_classes), "Logits shape is incorrect, expected (batch_size, n_classes)"
    
    # Loss
    loss = criterion(logits, y)
    assert not torch.isnan(loss), "Loss is NaN"
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Just checking it runs without error and gradients are populated
    # Not all parameters may receive gradients if they are unused, 
    # but at least some should.
    grads_found = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
             grads_found = True
             break
    assert grads_found, "No gradients were computed!"

# @hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
# def test_train_one_epoch(cfg: DictConfig):

#     device = get_device()

#     #configs 
#     hp = cfg.train.hp

#     prof = build_profiler(cfg, device) if cfg is not None else None
#     # dataloaders (prune rare based on hp.sample_min)
#     train_loader, val_loader, n_classes, class_names = build_dataloader(
#         cfg=cfg,
#         prune_rare=True,
#     )

#     model = Model(
#         n_classes=n_classes,
#         d_model=int(hp.d_model),
#         n_heads=int(hp.n_heads),
#         n_layers=int(hp.n_layers),
#     ).to(device)


#     criterion = nn.CrossEntropyLoss()
#     optimizer = build_optimizer(model, cfg)
#     fq_mask, time_mask = create_fq_mask(fq_mask=8, time_mask=20)  # make configurable later if you want
#     scaler = GradScaler() if (bool(hp.amp) and device.type == "cuda") else None
    
#     train_loss, val_loss = train_one_epoch(
#         model=model,
#         loader=train_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         device=device,
#         scaler=scaler,
#         fq_mask=fq_mask,
#         time_mask=time_mask,
#         amp=bool(hp.amp),
#         grad_clip=float(hp.grad_clip),
#         prof= prof,
#     )
#     assert isinstance(train_loss, float), "Train loss should be a float"
#     assert isinstance(val_loss, float), "Validation loss should be a float"
#     assert train_loss > 0, "Train loss should be positive"
#     assert val_loss > 0, "Validation loss should be positive"