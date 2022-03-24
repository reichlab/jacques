import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import unittest

from jacques import kernels


class Test_Kernel_Smooth_Quantile_Fn(unittest.TestCase):
    def test_quantile_smooth_bw(self):
        tau = np.concatenate(
            [np.array([0.01, 0.025])] +
                [np.linspace(0.05, 0.95, 19)] +
                [np.array([0.975, 0.99])],
            axis = 0)
        theta_b_raw = np.ones(1)
        theta_b = 0.25 * np.exp(theta_b_raw) / (1.0 + np.exp(theta_b_raw))
        
        lower_tau = tau[:9]
        central_tau = tau[9:(-9)]
        upper_tau = tau[-9:]
        expected = np.concatenate(
            [lower_tau - lower_tau**2 / (4 * theta_b)] +
                [np.full_like(central_tau, theta_b)] +
                [(1 - upper_tau) - (1 - upper_tau)**2 / (4 * theta_b)],
            axis = 0
        )
        
        actual = kernels.quantile_smooth_bw(tf.constant(tau), tf.constant(theta_b_raw))
        
        # actual matches expected
        self.assertTrue(np.all(np.abs(actual.numpy() - expected) < 1e-12))
    
    
    def test_integrated_epanechnikov(self):
        # TODO
        raise NotImplementedError
    
    
    def test_kernel_quantile_fn(self):
        tau = np.concatenate(
            [np.array([0.01, 0.025])] +
                [np.linspace(0.05, 0.95, 19)] +
                [np.array([0.975, 0.99])],
            axis = 0)
        y = np.array(
            [[[1.0, 5.0, 8.0],
                [5.0, 2.0, 3.0]],
                [[11.0, 15.0, 20.0],
                [5.0, 2.0, 1.0]],
                [[1.0, 5.0, 5.0],
                [5.0, 20.0, 3.0]]]
        )
        w = np.array(
            [[[0.1, 0.5, 0.4],
                [0.333, 0.333, 0.334]],
                [[0.3, 0.2, 0.5],
                [0.0, 0.0, 1.0]],
                [[0.4, 0.4, 0.2],
                [0.2, 0.2, 0.6]]]
        )
        y_sorted = np.array(
            [[[ 1.,  5.,  8.],
                [ 2.,  3.,  5.]],
                [[11., 15., 20.],
                [ 1.,  2.,  5.]],
                [[ 1.,  5.,  5.],
                [ 3.,  5., 20.]]]
        )
        w_sorted = np.array(
            [[[0.1  , 0.5  , 0.4  ],
                [0.333, 0.334, 0.333]],
                [[0.3  , 0.2  , 0.5  ],
                [1.   , 0.   , 0.   ]],
                [[0.4  , 0.4  , 0.2  ],
                [0.6  , 0.2  , 0.2  ]]]
        )
        theta_b_raw = np.ones(1)
        bw = kernels.quantile_smooth_bw(tf.constant(tau), tf.constant(theta_b_raw))

        expected = np.zeros((3, 2, 23))
        for b1 in range(3):
            for b2 in range(2):
                for k in range(23):
                    tau_k = tau[k]
                    bw_k = bw[k]
                    cw = np.concatenate([np.array([0.0]), np.cumsum(w_sorted[b1, b2, :])], axis=0)
                    for i in range(3):
                        U_im1 = kernels.integrated_epanechnikov(
                            (tau_k - cw[i+1-1]) / bw_k
                        )
                        U_i = kernels.integrated_epanechnikov(
                            (tau_k - cw[i+1]) / bw_k
                        )
                        expected[b1, b2, k] = expected[b1, b2, k] + \
                            (U_im1 - U_i) * y_sorted[b1, b2, i]
                    

        actual = kernels.kernel_quantile_fn(
            tf.constant(y),
            tf.constant(w),
            tf.constant(tau),
            tf.constant(theta_b_raw))
        
        # actual matches expected
        self.assertTrue(np.all(np.abs(actual.numpy() - expected) < 1e-12))
    
    
    def test_kernel_quantile_fn_w_batched(self):
        tau = np.concatenate(
            [np.array([0.01, 0.025])] +
                [np.linspace(0.05, 0.95, 19)] +
                [np.array([0.975, 0.99])],
            axis = 0)
        y = np.array([[5.0, 2.0, 3.0]])
        w = np.array(
            [[0.1, 0.5, 0.4],
                [0.333, 0.333, 0.334],
                [0.3, 0.2, 0.5],
                [0.0, 0.0, 1.0],
                [0.4, 0.4, 0.2],
                [0.2, 0.2, 0.6]]
        )
        y_sorted = np.array([[ 2.,  3.,  5.]])
        w_sorted = np.array(
            [[0.5, 0.4, 0.1],
                [0.333, 0.334, 0.333],
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0],
                [0.4, 0.2, 0.4],
                [0.2, 0.6, 0.2]]
        )
        theta_b_raw = np.ones(1)
        bw = kernels.quantile_smooth_bw(tf.constant(tau), tf.constant(theta_b_raw))

        expected = np.zeros((6, 23))
        for b1 in range(6):
            for k in range(23):
                tau_k = tau[k]
                bw_k = bw[k]
                cw = np.concatenate([np.array([0.0]), np.cumsum(w_sorted[b1, :])], axis=0)
                for i in range(3):
                    U_im1 = kernels.integrated_epanechnikov(
                        (tau_k - cw[i+1-1]) / bw_k
                    )
                    U_i = kernels.integrated_epanechnikov(
                        (tau_k - cw[i+1]) / bw_k
                    )
                    expected[b1, k] = expected[b1, k] + \
                        (U_im1 - U_i) * y_sorted[0, i]
                    

        actual = kernels.kernel_quantile_fn(
            tf.constant(y),
            tf.constant(w),
            tf.constant(tau),
            tf.constant(theta_b_raw))

        # actual matches expected
        self.assertTrue(np.all(np.abs(actual.numpy() - expected) < 1e-12))

if __name__ == '__main__':
    unittest.main()
