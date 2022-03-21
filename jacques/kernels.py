import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def diff_x_pairs(x1, x2):
    """
    Calculate the differences between all pairs of feature vectors within
    batches. 
    
    Inputs
    ------
    x1: tensor of shape `(batch_shape) + (n1, p)`
    x2: tensor of shape `(batch_shape) + (n2, p)`
    
    Returns
    -------
    tensor of shape `(batch_shape) + (n1, n2, p)`
    
    Raises
    ------
    ValueError if batch size and feature size don't match in `x1` and `x2`
    
    Notes
    -----
    The `batch_shape` and number of features `p` must match for `x1` and
    `x2`. Within each batch, the difference in feature values is
    calculated for `n1 * n2` pairs of feature vectors in `x1` and `x2`.
    """
    # validate input shapes
    x1_shape = x1.shape.as_list()
    x2_shape = x2.shape.as_list()
    if x1_shape[-1] != x2_shape[-1]:
        raise ValueError("x1 and x2 must have the same number of features in their last dimension")

    if x1_shape[:-2] != x2_shape[:-2]:
        raise ValueError("x1 and x2 must have the same batch size")
    
    # calculate differences and return
    return tf.expand_dims(x1, -2) - tf.expand_dims(x2, -3)


def gaussian_kernel(x1, x2, B_chol):
    """
    Calculate the differences between all pairs of feature vectors within
    batches. 
    
    Inputs
    ------
    x1: tensor of shape `(batch_shape) + (n1, p)`
    x2: tensor of shape `(batch_shape) + (n2, p)`
    B_chol: cholesky factor of the Gaussian kernel bandwidth, of shape
        `(p, p)`
    
    Returns
    -------
    tensor of shape `(batch_shape) + (n1, n2)`
    
    Raises
    ------
    ValueError if the batch size doesn't match in `x1` and `x2`, or if the
    feature size doesn't match in all of `x1`, `x2`, and `B_chol`.
    
    Notes
    -----
    The `batch_shape` and number of features `p` must match for `x1` and
    `x2`. The bandwidth Cholesky factor `B_chol` must be a `p` by `p`
    matrix. Suppose the batch shape is dimension `k`. The return value at
    index `(b1, ..., bk, i, j)`, with `0 <= i <= n1` and `0 <= j <= n2` is
    exp[-1 * (x1_bij - x2_bij)^T * B_chol * B_chol^T * (x1_bij - x2_bij)],
    where x1_bij = x1[b1, ..., bk, i, j] and similar for x2_bij
    """
    # validate shapes
    # comparison of shapes of x1, x2 is delegated to diff_x_pairs
    x1_shape = x1.shape.as_list()
    B_chol_shape = B_chol.shape.as_list()
    if x1_shape[-1] != B_chol_shape[-1]:
        raise ValueError("x1, x2, and B_chol must correspond to the same number of features")
    
    if B_chol_shape != [B_chol_shape[-1], B_chol_shape[-1]]:
        raise ValueError("B_chol must be a square matrix")

    # calculate differences in features for all pairs in batches of x1, x2
    diffs = diff_x_pairs(x1, x2)
    
    # product (x1 - x2)^T * B_chol
    diff_chol_prod = tf.expand_dims(tf.matmul(diffs, B_chol), -1)
    
    # kernel value, exp[-1 * (x1 - x2)^T * B_chol * B_chol^T * (x1 - x2)]
    result = tf.exp(-1.0 * tf.matmul(diff_chol_prod, diff_chol_prod, transpose_a=True))
    
    # drop extra 1x1 dimensions from matrix product calculation
    result = tf.squeeze(result, [-1, -2])
    
    return result


def kernel_weights(x1, x2, theta_b, kernel = 'gaussian_diag'):
    """
    Calculate weights for observations in `x2` based on their similarity to
    observations in `x1` according to a specified kernel.
    
    Inputs
    ------
    x1: tensor of shape `(batch_shape) + (n1, p)`
    x2: tensor of shape `(batch_shape) + (n2, p)`
    theta_b: one-dimensional tensor of parameters used to construct a
        bandwidth matrix
    kernel: string specifying the kernel form; supported options are
        'gaussian_diag' and `gaussian_full'
    
    Returns
    -------
    tensor of shape `(batch_shape) + (n1, n2)` with observation weights.
    
    Notes
    -----
    Suppose the batch shape is dimension `k`. The return value entries at
    indices `[b1, ..., bk, i, :]` are non-negative weights that sum to 1
    and measure the similarity of each feature vector
    x2[b1, ..., bk, j, :] to the corresponding values x1[b1, ..., bk, i, :]
    """
    if kernel == 'gaussian_diag':
        # TODO: Direct implementation not requiring multiplication by a diagonal
        # Cholesky factor
        B_chol = tf.linalg.diag(theta_b)
        kernel_vals = gaussian_kernel(x1, x2, B_chol)
    elif kernel == 'gaussian_full':
        B_chol = tfp.math.fill_triangular(theta_b)
        kernel_vals = gaussian_kernel(x1, x2, B_chol)
    else:
        raise ValueError("kernel must be 'gaussian_diag' or 'gaussian_full'")
    
    # TODO: may be faster to use tf.linalg.normalize?
    weights = kernel_vals / \
        tf.math.reduce_sum(kernel_vals, axis = -1, keepdims=True)
    
    return weights


def quantile_smooth_bw(tau, theta_b_raw):
    """
    Calculate per-quantile-level bandwidth for quantile smoothing
    
    Inputs
    ------
    tau: tensor of length `k` with probability levels at which to extract
        quantile estimates
    theta_b_raw: scalar with bandwidth parameter on "raw" scale, i.e.,
        a real number
    
    Returns
    -------
    tensor of length `k` with bandwidths suitable for kernel smoothing the
        empirical quantile function
    
    Notes
    -----
    """
    # convert theta_b_raw to lie between 0 and 1;
    # theta_b \in (0, 0.25), so this represents 4 * theta_b
    theta_b_times_4 = tf.math.sigmoid(theta_b_raw)
    theta_b_times_2 = theta_b_times_4 / tf.constant(np.array(2.0))
    theta_b = theta_b_times_4 / tf.constant(np.array(4.0))
    
    # binary masks:
    # lower_mask is 1.0 if tau < 2*theta_b and 0 otherwise
    # upper_mask is 1.0 if tau > 1 - 2*theta_b and 0 otherwise
    # central_mask is 1.0 if 2*theta_b <= tau <= 1 - 2*theta_b and 0 otherwise
    lower_mask = tf.cast(
        tf.math.less(tau, theta_b_times_2),
        dtype = tau.dtype)
    upper_mask = tf.cast(
        tf.math.greater(tau, tf.constant(np.array(1.)) - theta_b_times_2),
        dtype = tau.dtype)
    central_mask = tf.ones_like(tau) - lower_mask - upper_mask
    
    # calculate bandwidth as piecewise function of tau
    bw = tf.math.multiply(
            tau - tf.math.square(tau) / theta_b_times_4,
            lower_mask) + \
        tf.math.multiply(theta_b, central_mask) + \
        tf.math.multiply(
            1 - tau - tf.math.square(1 - tau) / theta_b_times_4,
            upper_mask)
    
    return bw


def integrated_epanechnikov(z):
    """
    Integral of Epanechnikov kernel from -1 to z
    """
    # indicator of whether z \in (-1, 1)
    z_in_bounds = tf.cast(
        tf.logical_and(
            tf.math.greater(z, tf.constant(np.array(-1.0))),
            tf.math.less(z, tf.constant(np.array(1.0)))
        ),
        dtype = z.dtype)
    
    # indicator of whether z > 1
    z_greater_1 = tf.cast(
        tf.math.greater(z, tf.constant(np.array(1.0))),
        dtype = z.dtype)
    
    return tf.multiply(
        tf.constant(np.array(0.5)) + tf.constant(np.array(0.75)) * z + \
            tf.constant(np.array(-0.25)) * tf.math.pow(z, 3),
        z_in_bounds
    ) + z_greater_1


def kernel_quantile_fn(y, w, tau, theta_b_raw):
    """
    Calculate quantile estimates via a kernel smoothing of the empirical
    quantile function based on a weighted sample.
    
    Inputs
    ------
    y: tensor of shape `(batch_shape) + (n_train,)` with observed values of the
        target variable.
    w: tensor of shape `(batch_shape) + (n_test, n_train)` with observation weights
    tau: tensor of length `k` with probability levels at which to extract
        quantile estimates
    theta_b_raw: scalar with bandwidth parameter on "raw" scale, i.e.,
        a real number
    
    Returns
    -------
    tensor of shape `(batch_shape, k)` with estimated quantiles
    
    Notes
    -----
    """
    # number of batch dimensions
    batch_dims = len(y.shape) - 1
    
    # calculate bandwidth for each quantile level
    bw = quantile_smooth_bw(tau, theta_b_raw)
    
    # reshape tau and bw to (1, ..., 1, k, 1), where the number of leading
    # ones matches the length of the batch size, for later broadcasting
    # with w
    target_tau_shape = tuple([1 for i in range(batch_dims)]) + (len(tau), 1)
    tau = tf.reshape(tau, target_tau_shape)
    bw = tf.reshape(bw, target_tau_shape)
    
    # within each batch, sort y and w according to values of y
    # TODO: would probably be faster to do an initial sort within the data
    # generator rather than repeated sorts within this function
    sorted_indx = tf.argsort(y, axis=-1)
    y_sorted = tf.gather(y, sorted_indx, batch_dims=batch_dims)
    w_sorted = tf.gather(w, tf.broadcast_to(sorted_indx, w.shape), batch_dims=batch_dims)
    
    # expand w's shape to (batch_shape) + (1, n) for later
    # broadcasting with tau and bw,
    # and expand y's shape to (batch_shape) + (n, 1) for later
    # (batched) matrix multiplication with U_diff
    y_sorted = tf.expand_dims(y_sorted, -1)
    w_sorted = tf.expand_dims(w_sorted, -2)
    
    # calculate cumulative weights and prepend zero; shape is (batch_shape) + (1, n + 1)
    cum_w = tf.concat(
        [tf.zeros(w_sorted.shape[:-1] + (1,), dtype=w.dtype), tf.cumsum(w_sorted, axis=-1)],
        axis = -1)
    
    # calculate U, with shape (batch_shape) + (k, n + 1)
    U = integrated_epanechnikov((tau - cum_w) / bw)
    
    # calculate U_diff, with shape (batch_shape) + (k, n)
    U_diff = U[..., :-1] - U[..., 1:]
    
    # calculate U_diff * y_sorted, with shape (batch_shape) + (k, 1);
    # then, drop trailing dimension
    result = tf.matmul(U_diff, y_sorted)
    result = tf.squeeze(result, axis=-1)
    
    return result

