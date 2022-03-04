import pandas as pd
import numpy as np
import tensorflow as tf
import abc

class jacques(abc.ABC):
    def featurize_data(data, target_var="inc_hosp", h=1):
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
        # for each location, the first 6 days have nan
        data['moving_avg_rate'] = data.groupby('location').rolling(7)[target_var].mean().reset_index(drop=True)
        
        # list of indices where rates are nan
        na_idx = data.loc[pd.isna(data["moving_avg_rate"]), :].index
        
        # take out nans in data
        tmp = data.drop(na_idx)
        
        # list of indices where date is the forecast date in tmp
        T_idx = tmp.loc[tmp["date"]== T,:].index

        # create x_T using data with date = forecast_date (T)
        data_T = data.loc[data["date"]== T,:]
    
        # x_T is (L, P = 1)
        x_T = data_T['moving_avg_rate'].values.reshape(-1, 1)
        
        # x_train_val is (L*(T-6-1), P = 1) 
        train_val = tmp.drop(T_idx)
        x_train_val = train_val['moving_avg_rate'].values.reshape(-1, 1)

        # tmp has value on forecast_date for each location
        grp = tmp[target_var].groupby(tmp['location'])
        # y_train_val is (L*(T-6-1), 1) 
        # pad h-1 numbers of nan at the end of each v means extending the array for every date before forecast_date (T)
        # -1 is for forecast_date (T)
        y_train_val = np.vstack([np.pad(v, (0, h-1), constant_values=np.nan)[h:].reshape(-1, 1) for _, v in grp])

        # convert everything to tensor
        x_train_val = tf.constant(x_train_val)
        y_train_val = tf.constant(y_train_val)
        x_T = tf.constant(x_T)
        
        return x_train_val, y_train_val, x_T
    
    def pinball_loss(self, y, q, tau):
        """
        Calculate pinball loss of predictions from a single model.
        Parameters
        ----------
        y: 1D tensor of length N
            observed values
        q: 2D tensor with shape (N, K)
            forecast values
        tau: 1D tensor of length K: Each slice `q[:, k, :]` corresponds 
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