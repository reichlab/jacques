import pytest
import pandas as pd
from jacques.data_processing import date_block_map, assign_blocks


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing."""
    return pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', periods=10, freq='D')
    })

def test_equal_block_size(sample_df):
    """Test when num_blocks divides the time points evenly."""
    block_map = date_block_map(sample_df, 'time', 2)  # 10 time points, 2 blocks
    assert len(block_map) == 10  # Should have 10 time points

    block_values = list(block_map.values())
    # Check that the blocks are divided evenly
    assert block_values.count(0) == 5
    assert block_values.count(1) == 5

def test_with_leftovers(sample_df):
    """Test when there are leftover time points."""
    block_map = date_block_map(sample_df, 'time', 3)  # 10 time points, 3 blocks
    assert len(block_map) == 10

    block_values = list(block_map.values())
    # With 10 points and 3 blocks, each block should have 3 or 4 points
    assert block_values.count(0) == 4  # First block takes the leftovers
    assert block_values.count(1) == 3
    assert block_values.count(2) == 3


def test_single_block(sample_df):
    """Test with a single block."""
    block_map = date_block_map(sample_df, 'time', 1)  # All points should go to block 0
    assert len(block_map) == 10

    block_values = list(block_map.values())
    assert block_values == [0] * 10


def test_no_blocks(sample_df):
    """Test with num_blocks = 0, which should raise an error."""
    with pytest.raises(ValueError, match="Number of blocks must be greater than zero."):
        date_block_map(sample_df, 'time', 0)


def test_empty_df():
    """Test with an empty dataframe."""
    empty_df = pd.DataFrame({'time': []})
    with pytest.raises(ValueError, match="Input dataframe is empty."):
        date_block_map(empty_df, 'time', 3)


def test_df_with_duplicate_times():
    """Test with a dataframe containing duplicate time entries."""
    df_with_duplicates = pd.DataFrame({
        'time': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03'])
    })
    block_map = date_block_map(df_with_duplicates, 'time', 2)  # 3 time points, 2 blocks
    assert len(block_map) == 3
    block_values = list(block_map.values())
    assert block_values.count(0) == 2
    assert block_values.count(1) == 1


# def test_assign_blocks():
#     # Create test data
#     df = pd.DataFrame({'date': ['2022-01-06', '2022-01-13', '2022-01-20', '2022-01-27', '2022-02-03'],
#                        'location': [1, 2, 3, 4, 1],
#                        'x0': [0, 1, 0, 1, 1],
#                        'x1': [0, 1, 0, 1, 1],
#                        'target': [10.0, 10.5, 10.0, 10.5, 12]})
    
#     features = ['x0', 'x1']
#     target = 'target'
#     num_blocks = 2
#     time_var = 'date'

#     # Assign blocks
#     block_list = assign_blocks(df, time_var, features, target, num_blocks)

#     # Check the output
#     assert len(block_list) == 2
#     assert block_list[0]['features'].shape == (3, 2)
#     assert block_list[0]['target'].shape == (3,1)
#     assert block_list[1]['features'].shape == (2, 2)
#     assert block_list[1]['target'].shape == (2,1)


def test_assign_blocks():
    # Create test data
    df = pd.DataFrame({'date': ['2022-01-06', '2022-01-13', '2022-01-20', '2022-01-27', '2022-02-03'],
                       'location': [1, 2, 3, 4, 1],
                       'x0': [0, 1, 0, 1, 1],
                       'x1': [0, 1, 0, 1, 1],
                       'target': [10.0, 10.5, 10.0, 10.5, 12]})
    
    features = ['x0', 'x1']
    target = 'target'
    num_blocks = 2
    time_var = 'date'

    # Assign blocks
    block_list = assign_blocks(df, time_var, features, target, num_blocks)

    # Check the output
    assert len(block_list) == 2
    assert block_list[0]['features'].shape == (3, 2)
    assert block_list[0]['target'].shape == (3,1)
    assert block_list[1]['features'].shape == (2, 2)
    assert block_list[1]['target'].shape == (2,1)