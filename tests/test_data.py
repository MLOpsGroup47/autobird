"""Tests for data-related components."""

from torch.utils.data import Dataset

from tests import _PATH_PREPROCESSING


def test_data_config():
    """Test the DataConfig class."""
    dataset = _PATH_PREPROCESSING.DataConfig()
    assert isinstance(dataset, DataConfig)
