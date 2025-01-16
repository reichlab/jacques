import tensorflow as tf

from jacques.kernels import diff_x_pairs


# Test for diff_x_pairs
def test_diff_x_pairs():
    x1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    x2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    expected_result = tf.constant([
        [[-4.0, -4.0], [-6.0, -6.0]],
        [[-2.0, -2.0], [-4.0, -4.0]]
    ])
    
    result = diff_x_pairs(x1, x2)
    tf.debugging.assert_equal(result, expected_result)