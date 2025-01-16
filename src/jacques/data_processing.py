import random

import numpy as np
import polars as pl
import tensorflow as tf

from jacques.kernels import diff_x_pairs


def date_block_map(df, time_var, num_blocks):
    """
    Parameters
    __________

    df: pandas dataframe
    time_var: str
        Name of the time variable
    num_blocks: integer
        Total number of blocks to be created with given dataset
    batch_size: integer
        Number of blocks in each batch. Each block has size of block_size. Default to 1.
        This means each gradient descent iteration sees forecasts for only one time block.

    Returns
    _______________
    block_map: dict
        Dictionary with time points as keys and block assignments as values
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    unique_times = df[time_var].unique()
    time_points = len(unique_times)

    if num_blocks <= 0:
        raise ValueError("Number of blocks must be greater than zero.")

    block_size = time_points // num_blocks

    block_assignments = np.repeat(np.arange(num_blocks), block_size)

    # if there are remaining observations, assign them to the first block
    leftover = time_points % num_blocks

    #All of the leftover time points get assigned to the first block
    if leftover > 0:
        block_assignments = np.concatenate((np.repeat(0, leftover), block_assignments))

    
    #Create a dictionary of time points and their corresponding block assignments
    block_map = dict(zip(unique_times, block_assignments))
    
    return block_map


def assign_blocks(df, time_var, features, target, num_blocks):
    """
    Assigns each time point in the dataset to a block
    
    Parameters
    __________
    df: pandas dataframe
    time_var: str
        Name of the time variable
    block_map: dict
        Dictionary with time points as keys and block assignments as values

    Returns
    _______
    block_list: list of dictionaries
        Each dictionary contains the features and target values for a given block
    """

    block_map = date_block_map(df, time_var, num_blocks)

    df['block'] = df[time_var].map(block_map).astype(int)
    
    block_list = []
    for i in range(num_blocks):
        data_dict = {
            'features' : tf.constant(pl.from_pandas(df).filter(pl.col("block") == i).select(features), dtype = tf.float32),
            'target': tf.constant(pl.from_pandas(df).filter(pl.col("block") == i).select(target), dtype = tf.float32)
        }
        block_list.append(data_dict)

    return block_list


def validation_training_pairings(num_blocks):
    """
    Parameters
    __________
    num_blocks: integer
        Total number of blocks to be created with given dataset

    Returns
    __________
    matrix: 2D numpy array
        A matrix identifying which training blocks to use for each validation block.
        Entry i,j is 1 if the jth block is used for training when the ith block is used for validation, and 0 otherwise.
    """
    if(num_blocks < 4):
        raise ValueError("Number of blocks must be greater than 3")
    # Create an n x n matrix filled with 1s
    matrix = np.ones((num_blocks, num_blocks), dtype=int)
    
    # Set the main diagonal and the two adjacent diagonals to 0
    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(matrix[1:], 0)
    np.fill_diagonal(matrix[:, 1:], 0)
    
    #For the first block, we remove the following block, and one random block from the rest
    matrix[0, random.randint(2, num_blocks - 1)] = 0 

    # For the last block, we remove the previous block, and one random block from the rest
    matrix[num_blocks - 1, random.randint(0, num_blocks - 3)] = 0

    return matrix

# Note that when implemented, this will need to be passed only the features from the block_list
def calc_diffs_one_train_block(features_list, test_block, train_blocks_mask, distance_metric = "difference"):
    """
    Parameters
    __________
    list_of_tensors: list of tensors
        List of tensors containing the features and target values for each block
    test_block: integer
        The block to be used for training
    validation_training_pairings: 2D numpy array
        A matrix identifying which training blocks to use for each validation block.
        Entry i,j is 1 if the jth block is used for training when the ith block is used for validation, and 0 otherwise.


    Returns
    _______
    diffs: list of tensors
        List of tensors containing the differences between the test blocks and a training block
    """

    
    train_blocks = [block for block, keep in zip(features_list, train_blocks_mask) if keep]
    test_block = features_list[test_block]

    if distance_metric is "difference":
        diffs = [diff_x_pairs(test_block, train_block) for train_block in train_blocks]
    
    # Here we can add functionality for distance or for dot product similarity (Or should this go in diff_x_pairs?)
    else:
        raise ValueError("Invalid distance metric")

    return diffs


def calc_diffs_all_train_blocks(block_list, distance_metric):

    all_diffs = [calc_diffs_one_train_block(block_list, test_block, validation_training_pairings[test_block], distance_metric) for test_block in range(len(block_list))]

    return all_diffs