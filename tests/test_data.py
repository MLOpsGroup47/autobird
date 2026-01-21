"""Tests for data-related components."""

from pathlib import Path

import numpy as np
import pytest
import torch
from call_of_func.data.data_helpers import _compute_global_norm_stats
from call_of_func.data.get_data import _index_dataset, _split_by_groups
from call_of_func.dataclasses.Preprocessing import DataConfig, PreConfig


def test_data_config():
    """Test the DataConfig class."""
    dataset = DataConfig(train_split=0.8, seed=0, clip_sec=10.0, stride_sec=2.0, pad_last=True)
    assert isinstance(dataset, DataConfig), "DataConfig instance creation failed"
    assert dataset.train_split == 0.8, "train_split attribute assignment incorrect"
    assert dataset.seed == 0, "seed attribute assignment incorrect"
    assert dataset.clip_sec == 10.0, "clip_sec attribute assignment incorrect"
    assert dataset.stride_sec == 2.0, "stride_sec attribute assignment incorrect"
    assert dataset.pad_last is True, "pad_last attribute assignment incorrect"


def test_compute_global_norm_stats():
    """Test the _compute_global_norm_stats function."""
    batch_size = 4
    n_mels = 64
    time_steps = 100
    
    # Create dummy tensor [N, 1, Mels, Time]
    X = torch.randn(batch_size, 1, n_mels, time_steps)
    
    # Manually add some bias to check mean calculation
    X = X + 5.0
    
    mean, std = _compute_global_norm_stats(X)
    
    # Output shape should be [1, 1, Mels, 1]
    assert mean.shape == (1, 1, n_mels, 1), "Mean shape is incorrect, expected (1, 1, n_mels, 1)"
    assert std.shape == (1, 1, n_mels, 1), "Std shape is incorrect, expected (1, 1, n_mels, 1)"
    
    # values should be close to expected distribution stats
    # mean roughly 5.0, std roughly 1.0
    assert torch.allclose(mean.mean(), torch.tensor(5.0), atol=0.5), "Mean value is incorrect"
    assert torch.allclose(std.mean(), torch.tensor(1.0), atol=0.2), "Std value is incorrect"


def test_index_dataset():
    """Test _index_dataset with actual data directory."""
    data_dir = Path(__file__).parents[1] / "data" / "voice_of_birds"
    
    if not data_dir.exists():
        pytest.skip(f"Data directory not found: {data_dir}")

    # Check if folder is not empty
    if not any(data_dir.iterdir()):
         pytest.skip(f"Data directory is empty: {data_dir}")

    try:
        items, classes = _index_dataset(data_dir)
    except ValueError as e:
         pytest.skip(f"Skipping test due to data issue: {e}")

    # Check that we found some classes
    assert len(classes) > 0, "No classes found"
    
    # Check that items were found
    assert len(items) > 0, "No items found"
    
    # Verify that class names are strings and items contains proper tuples
    assert isinstance(classes[0], str), "Class name is not a string"
    assert isinstance(items[0], tuple), "Item is not a tuple"
    assert len(items[0]) == 2, "Item tuple does not have length 2"
    assert isinstance(items[0][0], Path), "First element of item tuple is not a Path"
    assert isinstance(items[0][1], int), "Second element of item tuple is not an int"
    # Verify items paths exist
    assert items[0][0].exists(), "Item path does not exist"

    # Verify that the label_id corresponds to a valid class index
    max_label = len(classes) - 1
    for _, label_id in items[:100]: # Check first 100 items to save time
        assert 0 <= label_id <= max_label, f"Label id {label_id} out of range"

