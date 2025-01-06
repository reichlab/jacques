import pandas as pd
import numpy as np
import tensorflow as tf
import abc
import math
import random
import pickle


class jacques(abc.ABC):
    def single_batch_generator(self, x_train_val, y_train_val, block_size):
        """
        Training/validation set data generator for one batch at a time

        Parameters
        ----------
        x_train_val: 3D tensor with shape (L * D, T, P)
            L is the number of location l and T is the number of time point t for which
            the full feature vector x_{l,t}, possibly including lagged covariate values,
            and the response y_{l,t}, corresponding to the target variable at time t+h,
            could be calculated. P is number of features. D is the number of data sources.
            Each row is a vector x_{l,t} = [x_{l,t,1},...,x_{l,t,P}] of features for some pair
            (l, t) in the training set.
        y_train_val: 2D tensor with length (L*D, T)
            Each value is a forecast target variable value in the training set.
            y_{l, t} = z_{l, 1, t+h}
        block_size: integer
            Number of consecutive time points in a block.

        Returns
        -------
        x_val: 3D tensor with shape (batch_size = 1, N, P)
            N is the number of combinations of location l and time point t in
            a block of feature vector x_train_val. P is the number of features.
        y_val: 2D tensor with shape (batch_size = 1, N)
            Corresponding obseration data of x_val
        x_train: 3D tensor with shape (batch_size = 1, N', P)
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

        # Leftover observations from splitting into equal blocks of size blocksize
        leftover = y_train_val.shape[1] % block_size

        # removes leftover blocks from beginning, gives the index of each block split
        block_start_index = np.arange(start=leftover, stop=y_train_val.shape[1], step=block_size)

        # Total number of blocks
        num_blocks = len(block_start_index)

        i = 0
        while True:
            # if i== number of blocks, set i to 0 and shuffle the block_start_index
            if i % num_blocks == 0:
                i = 0
                np.random.shuffle(block_start_index)

            # if the first full block is selected as validation set
            if block_start_index[i] == leftover:
                # need to drop a random block from training set
                # take out block_start_index and the block after it out of choices

                # chooses random block index to drop from block_start_index not including block i
                drop_block_start_idx = random.choice(list(set(block_start_index)
                        - set([block_start_index[i], block_start_index[i] + block_size])))

                if drop_block_start_idx > block_start_index[i] + 2 * block_size:
                    # train_idx includes 3 groups of indices:
                    # (1) everything up to the start of the first block, which was
                    # selected as the validation block
                    # (2) indices between the selected block plus a one-block buffer
                    # and the start of the additional dropped block.
                    # (3) any trailing indices after the end of the additional dropped block```
                    train_idx = list(range(0, block_start_index[i])) \
                        + list(range(block_start_index[i] + 2 * block_size, drop_block_start_idx)) \
                        + list(range(drop_block_start_idx + block_size, y_train_val.shape[1]))
                else:
                    # if they are equal
                    train_idx = list(range(0, block_start_index[i])) \
                        + list(range(block_start_index[i] + 3 * block_size, y_train_val.shape[1]))
            # if the last full block is selected as validation set
            elif block_start_index[i] == y_train_val.shape[1] - block_size:
                # need to drop a random block from training set
                # take out block_start_index and the block before it out of choices

                drop_block_start_idx = random.choice(
                    list(
                        set(block_start_index)
                        - set([block_start_index[i], block_start_index[i] - block_size])
                    )
                )

                if drop_block_start_idx < block_start_index[i] - 2 * block_size:
                    train_idx = list(range(0, drop_block_start_idx)) \
                        + list(range(drop_block_start_idx + block_size,
                                     block_start_index[i] - block_size))
                else:
                    # if they are equal
                    train_idx = list(range(0, block_start_index[i] - 2 * block_size))
            # if the middle block is selected as validation set
            else:
                train_idx = list(range(0, block_start_index[i] - block_size)) \
                    + list(range(block_start_index[i] + 2 * block_size,
                                 y_train_val.shape[1]))

            # gather results
            x_val = x_train_val[:, block_start_index[i] : block_start_index[i] + block_size, :]
            x_val = tf.reshape(x_val, (x_val.shape[0] * x_val.shape[1], x_val.shape[2]))
            x_val = tf.expand_dims(x_val, axis=0)

            x_train = tf.gather(x_train_val, train_idx, axis=1)
            x_train = tf.reshape(
                x_train,
                (x_train.shape[0] * x_train.shape[1], x_train.shape[2])
            )
            x_train = tf.expand_dims(x_train, axis=0)

            y_val = y_train_val[:, block_start_index[i] : block_start_index[i] + block_size]
            y_val = tf.reshape(y_val, [-1])
            y_val = tf.expand_dims(y_val, axis=0)

            y_train = tf.gather(y_train_val, train_idx, axis=1)
            y_train = tf.reshape(y_train, [-1])
            y_train = tf.expand_dims(y_train, axis=0)

            # Drop any entries with missing data in either target or features
            mask = tf.math.isfinite(x_val).all(axis=1) & tf.math.isfinite(y_val)

            x_val = tf.boolean_mask(x_val, mask)
            y_val = tf.boolean_mask(y_val, mask)

            mask = tf.math.isfinite(x_train).all(axis=1) & tf.math.isfinite(y_train)
            x_train = tf.boolean_mask(x_train, mask)
            y_train = tf.boolean_mask(y_train, mask)


            i += 1

            yield x_val, x_train, y_val, y_train

    def generator(self, x_train_val, y_train_val, batch_size, block_size):
        """
        Training/validation set data generator

        Parameters
        ----------
        x_train_val: 3D tensor with shape (L, T, P)
            L is the number of location l and T is the number of time point t for which
            the full feature vector x_{l,t}, possibly including lagged covariate values,
            and the response y_{l,t}, corresponding to the target variable at time t+h,
            could be calculated. P is number of features.
            Each row is a vector x_{l,t} = [x_{l,t,1},...,x_{l,t,P}] of features for some pair
            (l, t) in the training set.
        y_train_val: 2D tensor with length (L, T)
            Each value is a forecast target variable value in the training set.
            y_{l, t} = z_{l, 1, t+h}
        batch_size: integer
            Number of blocks in each batch. Each block has size of block_size
        block_size: integer
            Number of consecutive time points in a block.

        Returns
        -------
        x_val: 3D tensor with shape (batch_size = 1, N, P)
            N is the number of combinations of location l and time point t in
            a block of feature vector x_train_val. P is the number of features.
        y_val: 2D tensor with shape (batch_size = 1, N)
            Corresponding obseration data of x_val
        x_train: 3D tensor with shape (batch_size = 1, N', P)
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

        gen = self.single_batch_generator(x_train_val, y_train_val, block_size)

        while True:
            xs_and_ys = [next(gen) for i in range(batch_size)]
            x_val = tf.concat([xy[0] for xy in xs_and_ys], axis=0)
            x_train = tf.concat([xy[1] for xy in xs_and_ys], axis=0)
            y_val = tf.concat([xy[2] for xy in xs_and_ys], axis=0)
            y_train = tf.concat([xy[3] for xy in xs_and_ys], axis=0)
            yield x_val, x_train, y_val, y_train

    def pinball_loss(self, y, q, tau):
        """
        Calculate pinball loss of predictions from a single model.
        Parameters
        ----------
        y: 2D tensor with shape (batch_size, N)
            observed response values
        q: 3D tensor with shape (batch_shape, N, K)
            forecast values at each quantile k in 1, …, K
        tau: 1D tensor of length K
            Each slice `q[:, :, k]` corresponds to predictions at quantile level `tau[k]`
        Returns
        -------
        Mean pinball loss over all predictions as scalar tensor
        (mean over all i = 1, …, N and k = 1, …, K)
        """

        # add an extra dimension to y --> (batch_size, N,1)
        y_broadcast = tf.expand_dims(y, -1)
        # broadcast y to shape (batch_size, N, K) - reminder N=T*L number of time points T and location L
        y_broadcast = tf.broadcast_to(y_broadcast, q.shape)
        loss = tf.reduce_mean(
            tf.maximum(tau * (y_broadcast - q), (tau - 1) * (y_broadcast - q))
        )

        return loss

    def pinball_loss_objective(self, param_vec, x_train, y_train, x_test, y_test, tau):
        """
        Pinball loss objective function for use during parameter estimation:
        a function of component weights

        Parameters
        ----------
        param_vec: 1D tensor
            parameter values in an unconstrained space (i.e., real numbers)
        x_train: 3D tensor with shape (batch_size, N_train, P)
            feature values of the training set
        y_train: 2D tensor with shape (batch_size, N_train)
            observed response values of the training set
        x_test: 3D tensor with shape (batch_size, N_test = L, P)
            feature values for each location at forecast date in test set
        y_test: 2D tensor with shape (batch_size, N_test = L)
            observed response values of the test set
        tau: 1D of length K
            quantile levels (probabilities)

        Returns
        -------
        Scalar pinball loss for predictions of y_test at
        quantile levels tau based on the training data
        """

        # q_hat has shape (batch_shape, N_test = L, K)
        q_hat = self.predict(
            param_vec=param_vec,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            tau=tau,
        )

        loss = self.pinball_loss(y_test, q_hat, tau)

        return loss

    # not using this for now
    def set_param_estimates_vec(self, param_estimates_vec):
        """
        Set parameter estimates in vector form
        Parameters
        ----------
        param_estimates_vec: 1D tensor of length K*(M-1)
        """
        self.param_estimates_vec = param_estimates_vec

    @property
    @abc.abstractmethod
    def n_param(self):
        """
        Abstract property for number of parameters of a model
        """
        pass

    @abc.abstractmethod
    def predict(self, param_vec, x_train, y_train, x_test, tau):
        """
        Generate quantile prediction

        Parameters
        ----------
        param_vec: tensor of shape `(self.n_param,)` with vector of parameters
        x_train: 3D tensor of shape `(batch_shape) + (n_train, P)` with training
            set features
        y_train: 2D  tensor of shape `(batch_shape) + (n_train,)` with training
            set response values
        x_test: 3D tensor of shape `(batch_shape) + (n_test, P)` with test set
            features
        tau: tensor of length `k` with probability levels at which to extract
            quantile estimates

        Returns
        -------
        tensor of shape `(batch_shape) + (n_test, k)` with test set quantile
            estimates at each quantile level
        """

    def fit(
        self,
        xval_batch_gen,
        num_blocks,
        tau,
        optim_method,
        num_epochs,
        learning_rate,
        batch_size=1,
        init_param_vec=None,
        verbose=False,
        save_frequency=None,
        save_path=None,):
        """
        Estimate model parameters

        Parameters
        ----------
        xval_batch_gen: generator
            Training/validation set data generator from the `generator()`.
        num_blocks: integer
            Total number of blocks to be created with given dataset
        batch_size: integer
            Number of blocks in each batch. Each block has size of block_size. Default to 1.
            This means each gradient descent iteration sees forecasts for only one time block.
        tau: 1D tensor of length K
            Quantile levels (probabilities) at which to forecast
        init_param_vec: optional 1D tensor of length K*(M-1)
            Optional initial values for the weights during estimation
        optim_method: string
            Method for optimization. For now, only support "adam" or "sgd".
        num_epochs: integer
            Number of iterations for optimization.
            One epoch consists of a run through all training set time blocks.
        learning_rate: scalar tensor or a float value
            The learning rate
        verbose: boolean
            If True, intermediate messages are printed during estimation. Defaults to False.
        save_frequency: integer
            Defaults to None.
            Intermediate state of parameter estimation is saved every `save_frequency` epochs.
        save_path: string
            Defaults to None. Path to save parameter estimation snapshots.
        """
        # initialize init_param_vec
        if init_param_vec == None:
            # all zeros
            init_param_vec = tf.constant(np.zeros(self.n_param), dtype=np.float32)

            # does not work well :(
            # He weight initialization
            # weight ~ N(0.0, sqrt(2/n))
            # std = math.sqrt(2.0 / self.n_param)
            # init_param_vec = np.random.rand(self.n_param) * std
            # init_param_vec = tf.constant(init_param_vec)

        # declare variable representing parameters to estimate
        param_vec_var = tf.Variable(initial_value=init_param_vec, name="param_vec", dtype=np.float32)

        # create optimizer
        if optim_method == "adam":
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        elif optim_method == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

        # number of batches
        num_batches = math.ceil(num_blocks / batch_size)

        # initiate loss trace
        lls_ = np.zeros(num_epochs * num_batches, np.float32)
        i = 0

        # create a list of trainable variables
        trainable_variables = [param_vec_var]

        for epoch in range(num_epochs):
            for batch_ind in range(num_batches):

                x_val, x_train, y_val, y_train = next(xval_batch_gen)

                x_val = tf.cast(x_val, tf.float32)
                x_train = tf.cast(x_train, tf.float32)
                y_val = tf.cast(y_val, tf.float32)

                with tf.GradientTape() as tape:
                    loss = self.pinball_loss_objective(
                        param_vec=param_vec_var,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_val,
                        y_test=y_val,
                        tau=tau,)

                grads = tape.gradient(loss, trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 10.0)
                optimizer.apply_gradients(zip(grads, trainable_variables))
                lls_[i] = loss
                i += 1

                if verbose:
                    print("epoch idx = %d" % epoch)
                    print("batch idx = %d" % batch_ind)
                    print("loss idx = %d" % (epoch + 1) * (batch_ind + 1))
                    print("param estimates vec = ")
                    print(param_vec_var.numpy())
                    print("loss = ")
                    print(loss.numpy())
                    print("grads = ")
                    print(grads)
                if save_frequency is not None and save_path is not None:

                    if ((epoch + 1) * (batch_ind + 1)) % save_frequency == 0:
                        # save parameter estimates and loss trace
                        params_to_save = {
                            "param_estimates_vec": param_vec_var.numpy(),
                            "loss_trace": lls_,
                        }

                        pickle.dump(params_to_save, open(str(save_path), "wb"))

        # set parameter estimates
        # self.set_param_estimates_vec(params_vec_var.numpy())
        self.loss_trace = lls_

        return param_vec_var


# class some_child_class(jacques):
#     def predict(self, q, train_q = False, w = None):
#         # blah blah
