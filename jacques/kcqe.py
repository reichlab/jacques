import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import kernels


class KCQE():
    def __init__(self, x_kernel = 'gaussian_diag', p=1) -> None:
        self.x_kernel = x_kernel
        self.p = p
        if x_kernel == 'gaussian_diag':
            self.n_param = p + 1
        elif x_kernel == 'gaussian_full':
            self.n_param = p * (p - 1) / 2 + 1
        else:
            raise ValueError("x_kernel must be 'gaussian_diag' or 'gaussian_full'")
        
        # TODO: uncomment once KCQE extends jacques base class
        # super(KCQE, self).__init__()
    
    
    def unpack_param_vec(self, param_vec):
        """
        Unpack vector of parameters into a dictionary
        
        Inputs
        ------
        param_vec: tensor of length `(self.n_param,)`
        
        Returns
        -------
        dictionary with entries `x_bw_raw` and `y_bw_raw`
        
        Raises
        ------
        ValueError if len(param_vec) != self.n_param
        """
        if len(param_vec) != self.n_param:
            raise ValueError("Require len(param_vec) == self.n_param")
        
        return {
            'x_bw_raw': param_vec[:-1],
            'y_bw_raw': param_vec[-1]
        }
    
    
    def predict(self, param_vec, x_train, y_train, x_test, tau):
        """
        Generate quantile predictions from a KCQE model
        
        Inputs
        ------
        param_vec: tensor of shape `(self.n_param,)` with vector of parameters
        x_train: tensor of shape `(batch_shape) + (n_train, p)` with training
            set features
        y_train: tensor of shape `(batch_shape) + (n_train,)` with training
            set response values
        x_test: tensor of shape `(batch_shape) + (n_test, p)` with test set
            features
        tau: tensor of length `k` with probability levels at which to extract
            quantile estimates
        
        Returns
        -------
        tensor of shape `(batch_shape) + (n_test, k)` with test set quantile
            estimates at each quantile level
        
        Raises
        ------
        ValueError if the last dimension of `x_train` and `x_test` don't match
        `self.p`
        """
        if x_train.shape.as_list()[-1] != self.p or x_test.shape.as_list()[-1] != self.p:
            raise ValueError("x_train and x_test must contain self.p features in their last dimension")
        
        param_dict = self.unpack_param_vec(param_vec)
        
        # w_train is a tensor of shape `(batch_shape) + (n_test, n_train)`
        w_train = kernels.kernel_weights(x1=x_test,
                                         x2=x_train,
                                         theta_b=param_dict['x_bw_raw'],
                                         kernel=self.x_kernel)
        
        return kernels.kernel_quantile_fn(y=tf.expand_dims(y_train, -2), # will be broadcast to match shape of w_train
                                          w=w_train,
                                          tau=tau,
                                          theta_b_raw=param_dict['y_bw_raw'])
