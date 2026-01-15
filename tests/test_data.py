"""Tests for data-related components."""

from call_of_func.dataclasses.Preprocessing import DataConfig


def test_data_config():
    """Test the DataConfig class."""
    dataset = DataConfig(train_split=0.8, seed=0, clip_sec=10.0, stride_sec=2.0, pad_last=True)
    assert isinstance(dataset, DataConfig)
