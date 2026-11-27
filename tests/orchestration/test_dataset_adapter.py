"""
Unit tests for InMemoryDatasetLoader helper in Module 8 (Orchestration).
"""

import torch
import pytest

from app.orchestration.dataset_adapter import InMemoryDatasetLoader, TorchDataLoaderAdapter
from app.evaluation.dataset_loader import BaseDatasetLoader


def test_in_memory_dataset_loader_interface():
    inputs = torch.randn(10, 3, 32, 32)
    targets = torch.randint(0, 5, (10,))
    loader = InMemoryDatasetLoader(inputs=inputs, targets=targets, dataset_name="test_data", batch_size=4)

    assert isinstance(loader, BaseDatasetLoader)
    assert loader.dataset_name == "test_data"
    assert len(loader) == 10

    batches = list(loader.iterate_batches())
    assert len(batches) == 3  # 4 + 4 + 2

    b1_pixels, b1_targets, b1_list = batches[0]
    assert b1_pixels.shape == (4, 3, 32, 32)
    assert b1_targets.shape == (4,)
    assert len(b1_list) == 4
    assert b1_list == targets[:4].tolist()


def test_torch_data_loader_adapter():
    inputs = torch.randn(12, 3, 16, 16)
    targets = torch.randint(0, 4, (12,))
    ds = torch.utils.data.TensorDataset(inputs, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=4)

    adapter = TorchDataLoaderAdapter(dl, dataset_name="tensor_dataloader")
    assert isinstance(adapter, BaseDatasetLoader)
    assert adapter.dataset_name == "tensor_dataloader"
    assert len(adapter) == 12

    batches = list(adapter.iterate_batches())
    assert len(batches) == 3
    bx, by, blist = batches[0]
    assert bx.shape == (4, 3, 16, 16)
    assert by.shape == (4,)
    assert len(blist) == 4
    assert blist == targets[:4].tolist()
