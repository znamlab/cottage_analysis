import numpy as np


def calculate_r_squared(y, y_hat):
    """Calculate R squared as the fraction of variance explained.

    Args:
        y: true values
        y_hat: predicted values

    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    residual_var = np.sum((y_hat - y) ** 2)
    total_var = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - residual_var / total_var
    return r_squared


def threshold(arr, lower_thr=None, upper_thr=None):
    arr_cp = arr.copy()
    if lower_thr != None:
        arr_cp[arr_cp < lower_thr] = lower_thr
    if upper_thr != None:
        arr_cp[arr_cp > upper_thr] = upper_thr

    return arr_cp
