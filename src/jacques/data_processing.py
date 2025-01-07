import numpy as np
import random
import pandas as pd
import polars as pl
import tensorflow as tf


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


def generate_train_blocks(num_blocks):
    """
    Generator, which iterates through validation blocks and identifies which training blocks to use.
    For each validation block, we will remove 3 blocks (generally the selected block and the 2 adjacent blocks)

    Parameters
    __________
    num_blocks: integer
        Total number of blocks to be created with given dataset

    Yields:
    ________

    train_blocks: list
    
    """
    # For each validation block, we will remove 3 blocks (generally the selected block and the 2 adjacent blocks)
    for i in range(num_blocks):
        # Initialize list of ones to signify which training blocks will correspond to each validation block
        train_blocks = [1] * num_blocks
        if i == 0:
            # For the first block, we drop the first two blocks and then a radom one selected from the remaining
            random_block = random.randint(2, num_blocks - 1)
            train_blocks[0], train_blocks[1], train_blocks[random_block] = 0, 0, 0
            yield train_blocks
        elif i == num_blocks - 1:
            # For the last block, we drop the last two blocks and then a radom one selected from the remaining
            random_block = random.randint(0, num_blocks - 3)
            train_blocks[-1], train_blocks[-2], train_blocks[random_block] = 0, 0, 0
            yield train_blocks
        else:
            # For all other blocks, we drop the selected block and the two adjacent blocks
            train_blocks[i - 1], train_blocks[i], train_blocks[i + 1] = 0, 0, 0
            yield train_blocks
    
# Now a function that returns the training and test set block numbers for each batch
def train_blocks_design(num_blocks):
    """
    Iterates through the block assignment generator and returns a dataframe with the training and test block assignments for each batch.

    Parameters
    __________
    num_blocks: integer
        Total number of blocks to be created with given dataset
    batch_size: integer
        Number of blocks in each batch. Each block has size of block_size. Default to 1.
        This means each gradient descent iteration sees forecasts for only one time block.

    Returns
    ________
    block_assignments: pandas dataframe
        (num_blocks x num_blocks) Dataframe with training and test block assignments for each batch
        Row i corresponds to the block assignments when block i is the validation block.
        Entry i, j  is a 1 if block j is used as a training block when block i si the validation block, and 0 otherwise.

    """
    #initialize array, num_blocks by num_blocks
    block_assignments = []
    block_generator = generate_train_blocks(num_blocks)

    for row in block_generator:
        block_assignments.append(row)

    return pd.DataFrame(block_assignments)
