import random

import numpy as np
import tensorflow as tf
import pytest

from jacques.data_processing import calc_diffs_all_train_blocks

@pytest.fixture
def sample_block_list():
    """Fixture to create a sample block list for testing."""
    
    feature_tensors = [
        tf.constant([[1, 2], [3, 4]]),
        tf.constant([[5, 6]]),
        tf.constant([[7, 8], [9, 10], [11, 12]]),
        tf.constant([[3, 4], [5, 6]]),
        tf.constant([[1, 8], [9, 2], [3, 5], [1,2]]),
    ]

    target_tensors = [
        tf.constant([0, 5]),
        tf.constant([2]),
        tf.constant([9, 7, 3]),
        tf.constant([6, 8]),
        tf.constant([1, 4, 7, 9]),
    ]

    data_list = [{'features': f, 'target': t} for f, t in zip(feature_tensors, target_tensors)]

    return data_list

def test_calc_diffs_all_train_blocks(sample_block_list):
    """Test the calc_diffs_all_train_blocks function."""
    random.seed(9731)
    features_list = [block['features'] for block in sample_block_list]

    diffs = calc_diffs_all_train_blocks(features_list, "difference")

    diffs2 = [
        tf.constant([
            [[6, 6], [4, 4]],
            [[8, 8], [6, 6]],
            [[10, 10], [8, 8]]
            ], dtype=tf.int32),
        tf.constant([
        [[6, 0], [-2, 6], [4, 3], [6, 6]],
        [[8, 2], [0, 8], [6, 5], [8, 8]],
        [[10, 4], [2, 10], [8, 7], [10, 10]]
        ], dtype=tf.int32),
     ]

    tf.debugging.assert_equal(diffs[2][0], diffs2[0])
    tf.debugging.assert_equal(diffs[2][1], diffs2[1])