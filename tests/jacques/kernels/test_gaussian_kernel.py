import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from jacques import kernels


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
