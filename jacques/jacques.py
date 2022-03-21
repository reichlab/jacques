import pandas as pd
import numpy as np
import tensorflow as tf
import abc
import math
import random

class jacques(abc.ABC):
    def featurize_data(self, data, target_var="inc_hosp", h=1):
        """
        Convert data to tensors containing features x and responses y for each location
        and time.

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

        Returns
        -------
        x_train_val: 2D tensor with shape (N, P = 1) 
            N is the number of combinations of location l and time point t for which
            the full feature vector x_{l,t}, possibly including lagged covariate values,
            and the response y_{l,t}, corresponding to the target variable at time t+h,
            could be calculated. P is number of features, equal to 1 for now.
            Each row is a vector x_{l,t} = [x_{l,t,1},...,x_{l,t,P}] of features for some pair 
            (l, t) in the training set.
        y_train_val: 1D tensor with length N
            Each value is a forecast target variable value in the training set.
            y_{l, t} = z_{l, 1, t+h}
        x_T: 2D tensor with shape (L, P = 1)
            Each value is test set feature for each location at forecast date.
        """
        assert target_var in data.columns

        # data has columns date, location, inc hosp, rate and population
        T = max(data['date'])

        # create a column for 7-day moving avg of rates
        # for each location, the first 6 days have nans
        data['moving_avg_rate'] = data.groupby('location').rolling(7)[target_var].mean().reset_index(drop=True)
        
        # create a column for h horizon ahead target for observed values. 
        # for each location, this column has h nans in the end.
        # the last nan is for forecast date.
        data['h_days_ahead_target'] = data.groupby('location')[target_var].shift(-h)

        # create x_T using data with date = forecast_date (T)
        data_T = data.loc[data["date"]== T,:]

        # x_T is (L, P = 1)
        x_T = data_T['moving_avg_rate'].values.reshape(-1, 1)

        # list of indices of rows that have as least one nan
        na_idx, _ = np.where(data.isna())
        
        # take out nans in data
        train_val = data.drop(na_idx)

        # shape is (L, (T - 6 - 1 - (h -1))), P = 1)
        features = ['moving_avg_rate']
        x_train_val = train_val.pivot(index = "location", columns = "date", values = features).to_numpy()
        x_train_val = x_train_val.reshape((x_train_val.shape[0], x_train_val.shape[1], len(features)))
        #x_train_val = train_val['moving_avg_rate'].values.reshape(-1, 1)
        
        # shape is (L, (T - 6 - 1 - (h -1))), P = 1)
        y_train_val = train_val.pivot(index = "location", columns = "date", values = 'h_days_ahead_target').to_numpy()
        y_train_val = y_train_val.reshape((y_train_val.shape[0], y_train_val.shape[1], 1))
        #y_train_val = train_val['h_days_ahead_target'].values.reshape(-1, 1)
        
        # convert everything to tensor
        x_train_val = tf.constant(x_train_val)
        y_train_val = tf.constant(y_train_val)
        x_T = tf.constant(x_T)
        
        return x_train_val, y_train_val, x_T
    
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
        x_val: 2D tensor with shape (N, P = 1) 
            N is the number of combinations of location l and time point t in 
            a block of feature vector x_train_val. P is the number of features.
        y_val: 1D tensor with shape (N,)
            Corresponding obseration data of x_val
        x_train: 2D tensor with shape (N', P = 1)
            N' is the number of combinations of location l and time point t in 
            the remaining blocks of feature vector x_train_val.
            P is the number of features.
            If x_val is the leading/ending block, 
            then x_train has the remaining blocks = all blocks - x_val - one neighbor block of x_val
            if x_val is a middle block, 
            then x_train has the remaining blocks = all blocks - x_val - two adjacent blocks of x_val
        y_train: 1D tensor with shape (N',)
            Corresponding obseration data of x_train
        """
        leftover = y_train_val.shape[1] % block_size
            
        block_start_index = np.arange(start = leftover, stop = y_train_val.shape[1]-1, step = block_size)
            
        num_blocks = len(block_start_index)

        i = 0
        while True:
            if i % num_blocks==0:
                i = 0
                np.random.shuffle(block_start_index)
            
            if block_start_index[i] == leftover:
                train_idx = list(range(0, block_start_index[i])) + list(range(block_start_index[i] + 2 * block_size,y_train_val.shape[1]))
                
            elif block_start_index[i] == y_train_val.shape[1]- 1 - block_size:
                train_idx = list(range(0, block_start_index[i] - block_size))
                
            else:
                train_idx = list(range(0, block_start_index[i] - block_size)) + list(range(block_start_index[i] + 2 * block_size, y_train_val.shape[1]))
                
            
            i += 1

            # gather results
            x_val = x_train_val[:,block_start_index[i]: block_start_index[i] + block_size,:]
            x_val = tf.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
            
            x_train = tf.gather(x_train_val, train_idx, axis = 1)
            x_train = tf.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))
            
            y_val = y_train_val[:,block_start_index[i]: block_start_index[i] + block_size,:]
            y_val = tf.reshape(y_val, [-1])
            
            y_train = tf.gather(y_train_val, train_idx, axis = 1)
            y_train = tf.reshape(y_train, [-1])

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
        x_test: 3D tensor with shape (L, T, P = 1)
            Each value is test set feature for each location at forecast date.
        xval_batch_gen: generator
            A generator initialized with x_train_val and y_train_val, block_size and batch_size
        """
        x_train_val, y_train_val, x_test = self.featurize_data(data, target_var, h)

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
    # from qenspy
    def predict(self, q, train_q, w):
        """
        Generate prediction from a quantile forecast ensemble.
        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Component prediction quantiles for observation cases i = 1, ..., N,
            quantile levels k = 1, ..., K, and models m = 1, ..., M
        train_q: boolean
            Indicator for calculating bandwidth during training or not
            It is only used in MedianQens.
        w: 2D tensor with shape (K, M)
            Component model weights, where `w[m, k]` is the weight given to
            model m for quantile level k
        Returns
        -------
        ensemble_q: 2D tensor with shape (N, K)
            Ensemble forecasts for each observation case i = 1, ..., N and
            quantile level k = 1, ..., K
        """


class some_child_class(jacques):
    def predict(self, q, train_q = False, w = None):
        # blah blah