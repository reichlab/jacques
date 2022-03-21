import pandas as pd
import numpy as np
import tensorflow as tf

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

