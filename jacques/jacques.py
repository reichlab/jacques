import pandas as pd
import numpy as np
import tensorflow as tf
import abc
import math
import random
from featurize import featurize_data


class jacques(abc.ABC):
    def generator(self, x_train_val, y_train_val, batch_size, block_size):
        """
        Traning/validation set data generator

        Parameters
        ----------
         x_train_val: 3D tensor with shape (L, T, P = 1) 
            L is the number of location l and T is the number of time point t for which
            the full feature vector x_{l,t}, possibly including lagged covariate values,
            and the response y_{l,t}, corresponding to the target variable at time t+h,
            could be calculated. P is number of features, equal to 1 for now.
            Each row is a vector x_{l,t} = [x_{l,t,1},...,x_{l,t,P}] of features for some pair 
            (l, t) in the training set.
         y_train_val: 3D tensor with length (L, T, 1)
            Each value is a forecast target variable value in the training set.
            y_{l, t} = z_{l, 1, t+h}
        batch_size: integer
            Number of blocks in each batch. Each block has size of block_size
        block_size: integer
            Number of consecutive time points in a block.
        
        Returns
        -------
        x_val: 3D tensor with shape (batch_size = 1, N, P = 1) 
            N is the number of combinations of location l and time point t in 
            a block of feature vector x_train_val. P is the number of features.
        y_val: 2D tensor with shape (batch_size = 1, N)
            Corresponding obseration data of x_val
        x_train: 3D tensor with shape (batch_size = 1, N', P = 1)
            N' is the number of combinations of location l and time point t in 
            the remaining blocks of feature vector x_train_val.
            P is the number of features.
            If x_val is the leading/ending block, 
            then x_train has the remaining blocks = all blocks - x_val - one neighbor block of x_val
            if x_val is a middle block, 
            then x_train has the remaining blocks = all blocks - x_val - two adjacent blocks of x_val
        y_train: 2D tensor with shape (batch_size = 1, N')
            Corresponding obseration data of x_train
        """

        assert batch_size == 1

        leftover = y_train_val.shape[1] % block_size
            
        block_start_index = np.arange(start = leftover, stop = y_train_val.shape[1], step = block_size)
            
        num_blocks = len(block_start_index)

        i = 0
        while True:
            if i % num_blocks==0:
                i = 0
                np.random.shuffle(block_start_index)
            
            if block_start_index[i] == leftover:
                train_idx = list(range(0, block_start_index[i])) + list(range(block_start_index[i] + 2 * block_size,y_train_val.shape[1]))
                
            elif block_start_index[i] == y_train_val.shape[1] - block_size:
                train_idx = list(range(0, block_start_index[i] - block_size))
                
            else:
                train_idx = list(range(0, block_start_index[i] - block_size)) + list(range(block_start_index[i] + 2 * block_size, y_train_val.shape[1]))

            # gather results
            x_val = x_train_val[:,block_start_index[i]: block_start_index[i] + block_size,:]
            x_val = tf.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
            x_val = tf.expand_dims(x_val, axis = 0)

            x_train = tf.gather(x_train_val, train_idx, axis = 1)
            x_train = tf.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))
            x_train = tf.expand_dims(x_train, axis = 0)

            y_val = y_train_val[:,block_start_index[i]: block_start_index[i] + block_size,:]
            y_val = tf.reshape(y_val, [-1])
            y_val = tf.expand_dims(y_val, axis = 0)

            y_train = tf.gather(y_train_val, train_idx, axis = 1)
            y_train = tf.reshape(y_train, [-1])
            y_train = tf.expand_dims(y_train, axis = 0)

            i += 1

            yield x_val, x_train, y_val, y_train
       
    
    def init_xval_split(self, data, target_var, h=1, block_size=21, batch_size=1):
        """
        Create training/validation set generator and test data
        
        Parameters
        ----------
        data: data frame 
            It has columns location, date, and a column with the response variable to forecast.
            This data frame needs to be sorted by location and date columns in ascending order.
        target_var: string
            Name of the column in the data frame with the forecast target variable.
            Default to "inc_hosp"
        h: integer
            Forecast horizon. Default to 1
        block_size: integer
            Number of consecutive time points in a block. Default to 21. 
        batch_size: integer
            Number of blocks in each batch. Each block has size of block_size. Default to 1.
        
        Returns
        -------
        num_blocks: integer
            Total number of block could be created with given dataset
        x_test: 2D tensor with shape (L, P = 1)
            Each value is test set feature for each location at forecast date.
        xval_batch_gen: generator
            A generator initialized with x_train_val and y_train_val, block_size and batch_size
        """
        x_train_val, y_train_val, x_test = featurize_data(data, target_var, h)

        # calculate number of blocks
        num_blocks = math.floor(y_train_val.shape[1]/block_size)

        xval_batch_gen = self.generator(x_train_val, y_train_val, block_size=21, batch_size=1)

        return num_blocks, x_test, xval_batch_gen

    def pinball_loss(self, y, q, tau):
        """
        Calculate pinball loss of predictions from a single model.
        Parameters
        ----------
        y: 1D tensor of length N
            observed values
        q: 2D tensor with shape (N, K)
            forecast values
        tau: 1D tensor of length K: Each slice `q[:, k]` corresponds 
        to predictions at quantile level `tau[k]`
        Returns
        -------
        Mean pinball loss over all predictions as scalar tensor 
        (mean over all i = 1, …, N and k = 1, …, K)
        """
    
        # add an extra dimension to y --> (N,1)
        y_broadcast = tf.expand_dims(y, -1)
        # broadcast y to shape (N, K)
        y_broadcast = tf.broadcast_to(y_broadcast, q.shape)
        loss = tf.reduce_mean(tf.maximum(tau*(y_broadcast - q), (tau-1)*(y_broadcast-q)))

        return loss
    
    @abc.abstractmethod
    def predict(self, param_vec, x_train, y_train, x_test, tau):
        """
        Generate quantile prediction
        
        Parameters
        ----------
        param_vec: tensor of shape `(self.n_param,)` with vector of parameters
        x_train: tensor of shape `(batch_shape) + (n_train, P)` with training
            set features
        y_train: tensor of shape `(batch_shape) + (n_train,)` with training
            set response values
        x_test: tensor of shape `(batch_shape) + (n_test, P)` with test set
            features
        tau: tensor of length `k` with probability levels at which to extract
            quantile estimates
        
        Returns
        -------
        tensor of shape `(batch_shape) + (n_test, k)` with test set quantile
            estimates at each quantile level
        """


# class some_child_class(jacques):
#     def predict(self, q, train_q = False, w = None):
#         # blah blah
