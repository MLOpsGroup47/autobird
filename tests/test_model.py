import pytest
import torch
from call_of_birds_autobird.model import Model


def test_model():
    """Test the model."""
    batch_size = 4
    n_classes = 10
    n_mels = 64
    n_frames = 128
    
    model = Model(n_classes=n_classes)
    
    # Create dummy input [B, 1, Mels, Frames]
    x = torch.randn(batch_size, 1, n_mels, n_frames)
    
    # Forward pass
    y = model(x)
    
    # Check output shape
    assert y.shape == (batch_size, n_classes)
    
    # Check for NaNs
    assert not torch.isnan(y).any()
