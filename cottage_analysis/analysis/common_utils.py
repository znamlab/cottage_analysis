import numpy as np
import scipy
import pandas as pd
from scipy.optimize import curve_fit
import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import find_depth_neurons
from pathlib import Path


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


def iterate_fit(
    func, X, y, lower_bounds, upper_bounds, niter=5, p0_func=None, verbose=False
):
    """Iterate fitting to avoid local minima.

    Args:
        func: function to fit
        X: independent variables
        y: dependent variable
        lower_bounds: lower bounds for the parameters
        upper_bounds: upper bounds for the parameters
        niter: number of iterations
        p0_func: function to generate initial parameters

    Returns:
        popt_best: best parameters
        rsq_best: best R squared

    """
    popt_arr = []
    rsq_arr = []
    valid = ~np.isnan(X) & ~np.isnan(y)
    if np.any(~valid):
        print(f"Warning: {np.sum(~valid)} NaN values in X or y")
        X = X[valid]
        y = y[valid]
    np.random.seed(42)
    for i_iter in range(niter):
        if p0_func is not None:
            p0 = p0_func()
        else:
            # generate random initial parameters from a standard normal distribution for unbounded parameters
            # otherwise, draw from a uniform distribution between lower and upper bounds
            p0 = np.random.normal(0, 1, len(lower_bounds))
            for i in range(len(lower_bounds)):
                if np.isinf(lower_bounds[i]) or np.isinf(upper_bounds[i]):
                    continue
                else:
                    p0[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])
        popt, _ = curve_fit(
            func,
            X,
            y,
            maxfev=1000000, #100000
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            p0=p0,
        )

        pred = func(np.array(X), *popt)
        r_sq = calculate_r_squared(y, pred)
        popt_arr.append(popt)
        rsq_arr.append(r_sq)
        if verbose:
            print(f"Iteration {i_iter}, R^2 = {r_sq}")
    idx_best = np.argmax(np.array(rsq_arr))
    popt_best = popt_arr[idx_best]
    rsq_best = rsq_arr[idx_best]
    return popt_best, rsq_best


def get_confidence_interval(
    arr=(),
    mean_arr=(),
    sem_arr=(),
    axis=1,
    sig_level=0.05,
):
    """Get confidence interval of an input array.

    Args:
        arr (np.ndarray, optional): 2d array, for example ndepths x ntrials to calculate confidence interval across trials.
        mean_arr (np.ndarray, optional): mean array, if the original array is not provided.
        sem_arr (np.ndarray, optional): SEM array, if the original array is not provided.
        sig_level (float, optional): Significant level. Default 0.05.
        method (str): Method to calculate confidence interval, "normal" or "bootstrap". Default "normal".
    Returns:
        CI_low (np.1darray): lower bound of confidence interval.
        CI_high (np.1darray): upper bound of confidence interval.
    """

    z = scipy.stats.norm.ppf((1 - sig_level / 2))
    if len(arr) > 0:
        sem = scipy.stats.sem(arr, axis=axis, nan_policy="omit")
        CI_low = np.nanmean(arr, axis=axis) - z * sem
        CI_high = np.nanmean(arr, axis=axis) + z * sem
    elif len(mean_arr) > 0 and len(sem_arr) > 0:
        CI_low = mean_arr - z * sem
        CI_high = mean_arr + z * sem
    else:
        print("Error: you need to input either [arr] or [mean_arr and sem_arr]")
        CI_low = []
        CI_high = []
    return CI_low, CI_high


def get_bootstrap_ci(arr, sig_level=0.05, n_bootstraps=1000):
    """Calculate confidence interval using bootstrap method.

    Args:
        arr (np.ndarray): 2d array, for example ndepths x ntrials to calculate confidence interval across trials.
        sig_level (float): Significant level.
        n_bootstraps (int): Number of bootstraps.

    Returns:
        CI_low (np.1darray): lower bound of confidence interval.
        CI_high (np.1darray): upper bound of confidence interval.
    """
    ndepths, ntrials = arr.shape
    bootstrapped_means = np.zeros((n_bootstraps, ndepths))
    for i in range(n_bootstraps):
        for idepth in range(ndepths):
            resampled = arr[idepth, np.random.randint(0, ntrials, ntrials)]
            bootstrapped_means[i, idepth] = np.nanmean(resampled)
    CI_low = np.percentile(bootstrapped_means, 100 * sig_level / 2, axis=0)
    CI_high = np.percentile(bootstrapped_means, 100 * (1 - sig_level / 2), axis=0)
    return CI_low, CI_high


def choose_trials_subset(trials_df, choose_trials):
    depth_list = find_depth_neurons.find_depth_list(trials_df)
    trial_number = len(trials_df) // len(depth_list)

    if choose_trials == None:  # fit all trials
        trials_df_chosen = trials_df
        sfx = ""
        choose_trial_nums = np.arange(trial_number)
    else:
        if choose_trials == "odd":  # fit odd trials
            trials_df_chosen = pd.DataFrame(columns=trials_df.columns)
            # choose odd trials from trials_df
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[::2, :]
                trials_df_chosen = pd.concat([trials_df_chosen, trials_df_depth])
            choose_trial_nums = np.arange(trial_number)[::2]
        if choose_trials == "even":  # fit even trials
            trials_df_chosen = pd.DataFrame(columns=trials_df.columns)
            # choose even trials from trials_df
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[1::2, :]
                trials_df_chosen = pd.concat([trials_df_chosen, trials_df_depth])
            choose_trial_nums = np.arange(trial_number)[1::2]
        if choose_trials is not None and isinstance(
            choose_trials, list
        ):  # if choose_trials is a given list
            trials_df_chosen = pd.DataFrame(columns=trials_df.columns)
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[choose_trials, :]
                trials_df_chosen = pd.concat(
                    [trials_df_chosen, trials_df_depth], ignore_index=True
                )
            choose_trial_nums = choose_trials
        sfx = "_crossval"

    return trials_df_chosen, choose_trial_nums, sfx


def find_thresh_sequence(
    array,
    length,
    shift,
    threshold_min=None,
    threshold_max=None,
):
    """Find a sequance within an array where the values are below a threshold for a certain length.

    Args:
        array (np.1darray): array to be searched
        length (int): length of sequence
        shift (int): length of shift to the right (positive) or left (negative); positive shift = the sequence is in the past
        threshold_min (float, optional): min threshold. Defaults to None.
        threshold_max (float, optional): max threshold. Defaults to None.

    Returns:
        _type_: _description_
    """
    # shift an array by the shift amount
    if (threshold_min is None) and (threshold_max is None):
        print("WARNING: No threshold is given. Full array is returned.")
        indices = np.arange(len(array))
    else:
        if threshold_min is None:
            mask = array < threshold_max
        elif threshold_max is None:
            mask = array > threshold_min
        else:
            mask = (array > threshold_min) & (array < threshold_max)
        conv = np.convolve(mask, np.ones(length, dtype=int), "valid")
        indices = np.where(conv >= length)[0]
        indices = indices + int(shift) - 1

        # Get rid of the indices that's larger than the length of the array
        indices = indices[indices < len(array)]

        return indices


def fill_missing_elements(arr, fill_n):
    # if an element is larger than the previous element by more than 1, insert the consecutive x more integers after the previous element in the array.
    diffs = np.diff(arr)
    gap_indices = np.where(diffs > 1)[0]

    # Generate the missing numbers for each gap
    if len(gap_indices) > 0:
        new_elems = [np.arange(arr[i] + 1, arr[i] + fill_n) for i in gap_indices]

        # Concatenate the original array with the new elements and flatten
        filled_array = np.sort(np.concatenate((arr, np.concatenate(new_elems))))

    else:
        filled_array = arr

    return filled_array
