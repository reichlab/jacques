import pandas as pd
import pytest
import torch

from jacques.data_processing import assign_blocks



def test_assign_blocks_basic():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "feature1": [10, 20, 30, 40, 50, 60],
        "feature2": [1, 2, 3, 4, 5, 6],
        "target": [100, 200, 300, 400, 500, 600],
    })
    
    features = ["feature1", "feature2"]
    target = "target"
    block_size = 2

    block_list = assign_blocks(df, "date", features, target, block_size)

    # Check number of blocks
    assert len(block_list) == 3  # 6 time points / block_size 2 = 3 blocks

    # Check first block's features and target
    expected_features = torch.tensor([[10, 1], [20, 2]], dtype=torch.float32)
    expected_target = torch.tensor([[100], [200]], dtype=torch.float32)

    assert torch.all(torch.eq(block_list[0]["features"], expected_features))
    assert torch.all(torch.eq(block_list[0]["target"], expected_target))

def test_assign_blocks_empty_dataframe():
    df = pd.DataFrame(columns=["date", "feature1", "feature2", "target"])
    features = ["feature1", "feature2"]
    target = "target"

    with pytest.raises(ValueError, match="Input dataframe is empty."):
        assign_blocks(df, "date", features, target, block_size=2)

def test_assign_blocks_invalid_block_size():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "feature1": [10, 20, 30, 40, 50],
        "feature2": [1, 2, 3, 4, 5],
        "target": [100, 200, 300, 400, 500],
    })
    features = ["feature1", "feature2"]
    target = "target"

    with pytest.raises(ValueError, match="Block size is too large for the dataset."):
        assign_blocks(df, "date", features, target, 10)

def test_assign_blocks_with_leftover():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=7, freq="D"),
        "feature1": [10, 20, 30, 40, 50, 60, 70],
        "feature2": [1, 2, 3, 4, 5, 6, 7],
        "target": [100, 200, 300, 400, 500, 600, 700],
    })
    
    features = ["feature1", "feature2"]
    target = "target"
    block_size = 3

    block_list = assign_blocks(df, "date", features, target, block_size)

    # Expecting 3 blocks: First with 3 values, Second with 3, Third with 1
    assert len(block_list) == 2

    # Check last block has only 1 observation
    assert block_list[-1]["features"].shape == (3, 2)
    assert block_list[-1]["target"].shape == (3, 1)

def test_assign_blocks_large_block_size():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "feature1": [10, 20, 30, 40, 50],
        "feature2": [1, 2, 3, 4, 5],
        "target": [100, 200, 300, 400, 500],
    })
    features = ["feature1", "feature2"]
    target = "target"

    with pytest.raises(ValueError, match="Block size is too large for the dataset."):
        assign_blocks(df, "date", features, target, 10)
