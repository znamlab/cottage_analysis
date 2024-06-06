import numpy as np
import pandas as pd


def closest_smaller_value(arr, value):
    smaller_values = arr[arr < value]
    if len(smaller_values) > 0:
        return np.max(smaller_values)
    else:
        return None


def closest_bigger_value(arr, value):
    bigger_values = arr[arr > value]
    if len(bigger_values) > 0:
        return np.min(bigger_values)
    else:
        return None


def find_zeros_before_ones(arr):
    """Function to find a sequence of zeros before ones (to find open loop recordings before closed loop)

    Args:
        arr (list or np.array): a list or array containing 0s and 1s.

    Returns:
        (list, list): list of arrays of indices of zeros and ones where zeros are before ones
    """
    arr = np.array(arr)
    changes_01 = np.where(np.diff(arr) == 1)[0] + 1  # idx of changes from 0 to 1
    changes_10 = np.where(np.diff(arr) == -1)[0] + 1  # idx of changes from 1 to 0
    zeros = []
    ones = []

    if len(changes_01) > 0:
        for idx in changes_01:
            assert arr[idx] == 1
            zero_start = closest_smaller_value(
                changes_10, idx
            )  # find the start of the zero sequence
            if zero_start is None:  # if the sequence starts with 0
                zero_start = 0
            zero_idx = np.arange(zero_start, idx)
            zeros.append(zero_idx)
            one_stop = closest_bigger_value(
                changes_10, idx
            )  # find the end of the one sequence
            if one_stop is None:  # if the sequence ends with 1
                one_stop = len(arr)
            one_idx = np.arange(idx, one_stop)
            ones.append(one_idx)
    return zeros, ones


def open_before_closed_trials_df(trials_df):
    arr = trials_df["closed_loop"].values
    zeros, ones = find_zeros_before_ones(arr)
    all_trials_df = []
    if zeros is not None:
        for openloop_trials, closedloop_trials in zip(zeros, ones):
            trials_df_subset = pd.concat(
                [trials_df.iloc[openloop_trials], trials_df.iloc[closedloop_trials]],
                ignore_index=True,
            )
            all_trials_df.append(trials_df_subset)
    return all_trials_df
