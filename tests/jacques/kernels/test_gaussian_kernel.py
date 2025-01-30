import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pytest

from jacques import kernels

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

def test_diff_x_pairs():
    x1 = np.arange(2*3*10*5).reshape(2, 3, 10, 5)
    x2 = np.arange(2*3*4*5).reshape(2, 3, 4, 5) + 1000
    x1_tf = tf.constant(x1)
    x2_tf = tf.constant(x2)

    expected_result = np.zeros((2, 3, 10, 4, 5))
    for i in range(10):
        for j in range(4):
            expected_result[:, :, i, j, :] = x1[:, :, i, :] - x2[:, :, j, :]

    actual_result = tf.expand_dims(x1_tf, -2) - tf.expand_dims(x2_tf, -3)

    actual_result = kernels.diff_x_pairs(x1_tf, x2_tf)

    assert np.all(actual_result.numpy() == expected_result) 

def test_gaussian_full_kernel():
    x1 = tf.constant(np.arange(2*3*10*5).reshape(2, 3, 10, 5), dtype='float64')
    x2 = tf.constant(np.arange(2*3*4*5).reshape(2, 3, 4, 5) + 1000, dtype='float64')

    B_chol = tfp.math.fill_triangular(np.arange(15)/25000)
    B = tf.matmul(B_chol, B_chol, transpose_b=True)

    diffs = kernels.diff_x_pairs(x1, x2)

    expected_result = np.zeros((2, 3, 10, 4))
    for b1 in range(2):
        for b2 in range(3):
            for i in range(10):
                for j in range(4):
                    expected_result[b1, b2, i, j] = tf.matmul(
                        tf.matmul(tf.expand_dims(diffs[b1, b2, i, j], 0), B),
                        tf.expand_dims(diffs[b1, b2, i, j], -1)
                    )

    expected_result = np.exp(-1.0 * expected_result)

    actual_result = kernels.gaussian_kernel(x1, x2, B_chol)

    assert(np.all(np.abs(actual_result.numpy() - expected_result) < 1e-12))
    
    
def test_gaussian_full_weights():
        x1 = tf.constant(np.arange(2*3*10*5).reshape(2, 3, 10, 5), dtype='float64')
        x2 = tf.constant(np.arange(2*3*4*5).reshape(2, 3, 4, 5) + 1000, dtype='float64')
        theta_raw = np.arange(15) - 7.5
        B_sd = tf.linalg.diag(tfp.bijectors.Softplus().forward(theta_raw[:5]))
        B_corr_chol = tfp.bijectors.CorrelationCholesky().forward(theta_raw[5:])
        B_chol = B_corr_chol @ B_sd

        kernel_val = kernels.gaussian_kernel(x1, x2, B_chol)
        expected_weights = np.zeros((2, 3, 10, 4))
        for b1 in range(2):
            for b2 in range(3):
                for i in range(10):
                    temp = kernel_val.numpy()[b1, b2, i, :]
                    expected_weights[b1, b2, i, :] = temp / np.sum(temp)

        actual_weights = kernels.kernel_weights(
            x1, x2,
            theta_raw,
            kernel = 'gaussian_full')

        # actual matches expected
        assert(np.all(np.abs(actual_weights.numpy() - expected_weights) < 1e-12))

        # within batches and x2 observations, weights sum to 1 across x1 observations
        assert(np.all(np.abs(tf.reduce_sum(actual_weights, axis=-1).numpy() - np.ones((2, 3, 10))) < 1e-12))

def test_gaussian_kernel_diffs(sample_block_list):
    diffs = [
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
    
    

    B_sd = tf.linalg.diag(tfp.bijectors.Softplus().forward([1.,2.]))
    B_corr_chol = tfp.bijectors.CorrelationCholesky().forward([1.])
    B_chol = B_corr_chol @ B_sd

    pass