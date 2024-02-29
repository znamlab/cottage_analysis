import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
from scipy.stats import spearmanr
import flexiznam as flz

from sklearn.model_selection import StratifiedKFold

from cottage_analysis.analysis import common_utils, size_control
from functools import partial

print = partial(print, flush=True)


def gaussian_func(x, a, x0, log_sigma, b, min_sigma):
    sigma = np.exp(log_sigma) + min_sigma
    return (a * np.exp(-((x - x0) ** 2)) / (2 * sigma**2)) + b


def find_depth_list(df):
    """Return the depth list from a dataframe that contains all the depth information from
    a session

    Args:
        df (DataFrame): A dataframe (such as vs_df or trials_df) that contains all the depth
            information from a session

    Returns:
        depth_list (list): list of depth values occurred in a session

    """
    depth_list = df["depth"].unique()
    depth_list = depth_list[~np.isnan(depth_list)].tolist()
    depth_list.sort()

    return depth_list


def average_dff_for_all_trials(
    trials_df,
    rs_thr=0.2,
    rs_thr_max=None,
    still_only=False,
    still_time=0,
    frame_rate=15,
    closed_loop=1,
    param="depth",
):
    """Generate an array (ndepths x ntrials x ncells) for average dffs across each trial.

    Args:
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        rs_thr (float, optional): min threshold of running speed to be counted into depth tuning analysis. Defaults to 0.2 m/s.
        rs_thr_max (float, optional): max threshold of running speed to be counted into depth tuning analysis. Defaults to None.
        still_only (bool, optional): whether to only use the frames when the mouse is not running. Defaults to False.
        still_time (int, optional): Number of seconds to use when the mouse stay still. Defaults to 0.
        frame_rate (int, optional): frame rate of the recording. Defaults to 15.
    """
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    depth_list = find_depth_list(trials_df)
    if still_only:
        if rs_thr_max is None:
            print(
                "ERROR: calculating under not_running condition without rs_thr to determine max speed"
            )
        else:  # use not running data, speed < rs_thr_max
            trials_df["trial_mean_dff"] = trials_df.apply(
                lambda x: np.nanmean(
                    x.dff_stim[
                        common_utils.find_thresh_sequence(
                            array=x.RS_stim,
                            threshold_max=rs_thr_max,
                            length=still_time * frame_rate,
                            shift=still_time * frame_rate,
                        ),
                        :,
                    ],
                    axis=0,
                ),
                axis=1,
            )
    else:
        if (rs_thr is None) and (rs_thr_max is None):  # no rs_thr, use all data
            trials_df["trial_mean_dff"] = trials_df.apply(
                lambda x: np.nanmean(x.dff_stim, axis=0), axis=1
            )
        elif rs_thr_max is None:  # no rs_thr_max, use all data above rs_thr
            trials_df["trial_mean_dff"] = trials_df.apply(
                lambda x: np.nanmean(x.dff_stim[x.RS_stim > rs_thr, :], axis=0), axis=1
            )
        elif rs_thr is None:  # no rs_thr, use all data below rs_thr_max
            trials_df["trial_mean_dff"] = trials_df.apply(
                lambda x: np.nanmean(x.dff_stim[x.RS_stim < rs_thr_max, :], axis=0),
                axis=1,
            )
        else:  # use data between rs_thr and rs_thr_max
            trials_df["trial_mean_dff"] = trials_df.apply(
                lambda x: np.nanmean(
                    x.dff_stim[(x.RS_stim > rs_thr) & (x.RS_stim < rs_thr_max), :],
                    axis=0,
                ),
                axis=1,
            )

    if param == "depth":
        grouped_trials = trials_df.groupby(by="depth")
        mean_dff_arr = []
        trial_nos = []
        for depth in depth_list:
            trial_nos.append(
                np.stack(
                    np.array(grouped_trials.get_group(depth)["trial_mean_dff"].values)
                ).shape[0]
            )
        trial_no_min = np.min(trial_nos)
        for depth in depth_list:
            trial_mean_dff = np.stack(
                np.array(grouped_trials.get_group(depth)["trial_mean_dff"].values)
            )[:trial_no_min, :]
            mean_dff_arr.append(trial_mean_dff)
        mean_dff_arr = np.stack(mean_dff_arr)

    elif param == "size":
        trials_df = size_control.get_physical_size(
            trials_df, use_cols=["size", "depth"], k=1
        )
        grouped_trials = trials_df.groupby(by="physical_size")
        size_list = np.sort(trials_df["physical_size"].unique())
        trial_nos = []
        for size in size_list:
            trial_nos.append(
                np.stack(
                    np.array(grouped_trials.get_group(size)["trial_mean_dff"].values)
                ).shape[0]
            )
        trial_no_max = np.max(trial_nos)
        mean_dff_arr = np.full(
            (
                len(size_list),
                trial_no_max,
                trials_df["trial_mean_dff"].iloc[0].shape[0],
            ),
            np.nan,
        )
        for isize, size in enumerate(size_list):
            trial_mean_dff = np.stack(
                np.array(grouped_trials.get_group(size)["trial_mean_dff"].values)
            )
            mean_dff_arr[isize, : len(trial_mean_dff), :] = trial_mean_dff

    return mean_dff_arr


def find_depth_neurons(
    trials_df,
    neurons_ds,
    rs_thr=0.2,
    alpha=0.05,
):
    """Find depth neurons from all ROIs segmented.

    Args:
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        neurons_ds (Series): flexilims dataset for neurons_df.
        rs_thr (float, optional): threshold of running speed to be counted into depth tuning analysis. Defaults to 0.2 m/s.
        alpha (float, optional): significance level for anova test. Defaults to 0.05.

    Returns:
        (DataFrame, Series): (neurons_df, neurons_ds) A dataframe that contains the analysed properties for each ROI; flexilims dataset for neurons_df.


    """
    # Create an empty datafrom for neurons_df
    neurons_df = pd.DataFrame(
        columns=[
            "roi",  # ROI number
            "is_depth_neuron",  # bool, is it a depth-selective neuron or not
            "depth_neuron_anova_p",  # float, p value for depth neuron anova test
            "best_depth",  # #, depth with the maximum average response
        ]
    )
    nrois = trials_df.dff_stim.iloc[0].shape[1]
    neurons_df["roi"] = np.arange(nrois)

    # Find the averaged dFF for each trial in only closed loop recordings
    trials_df = trials_df[trials_df.closed_loop == 1]

    # Anova test to determine which neurons are depth neurons
    depth_list = find_depth_list(trials_df)
    mean_dff_arr = average_dff_for_all_trials(trials_df, rs_thr=rs_thr)

    for roi in tqdm(np.arange(nrois)):
        _, p = scipy.stats.f_oneway(*mean_dff_arr[:, :, roi])

        neurons_df.loc[roi, "depth_neuron_anova_p"] = p
        neurons_df.loc[roi, "is_depth_neuron"] = p < alpha
        neurons_df.loc[roi, "best_depth"] = depth_list[
            np.argmax(np.mean(mean_dff_arr[:, :, roi], axis=1))
        ]

    return neurons_df, neurons_ds


def fit_preferred_depth(
    trials_df,
    neurons_df,
    neurons_ds,
    closed_loop=1,
    choose_trials=None,
    depth_min=0.02,
    depth_max=20,
    rs_thr=0.2,
    niter=10,
    min_sigma=0.5,
    k_folds=1,
    param="depth",
):
    """Function to fit depth tuning with gaussian function

    Args:
        trials_df (pd.DataFrame): trials_df for this session that describes the parameters for each trial.
        neurons_df (pd.DataFrame): neurons_df for this session that describes the properties for each ROI.
        neurons_ds (Series): flexilims dataset for neurons_df.
        closed_loop (int, optional): Fit based on closed loop data or not. Defaults to 1.
        choose_trials (str or list, optional): Which trials to choose for the fit. Defaults to None. None: all trials. "odd": odd trials. "even": even trials. list: a list of trial numbers.
        depth_min (float, optional): min boundary of preferred depth in m. Defaults to 0.02.
        depth_max (float, optional): min boundary of preferred depth in m. Defaults to 20.
        rs_thr (float, optional): Running speed threshold for fiting preferred depth in m. Defaults to 0.2.
        niter (int, optional): Number of rounds of fitting iterations. Defaults to 10.
        min_sigma (float, optional): min sigma for gaussian fitting. Defaults to 0.5.
        k_folds (int, optional): Number of folds for k-fold cross-validation. Defaults to 1.
        param (str, optional): "depth" or "size". Defaults to "depth".

    Returns:
        (pd.DataFrame, Series): neurons_df, neurons_df
    """

    # Function to initialize depth tuning parameters
    if param == "depth":

        def p0_func():
            return np.concatenate(
                (
                    np.exp(np.random.normal(size=1)),
                    np.atleast_1d(np.log(neurons_df.loc[roi, "best_depth"])),
                    np.exp(np.random.normal(size=1)),
                    np.random.normal(size=1),
                )
            ).flatten()

    elif param == "size":
        neurons_df, neurons_ds = size_control.find_best_size(
            trials_df=trials_df,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            rs_thr=rs_thr,
            rs_thr_max=None,
            is_closedloop=closed_loop,
            still_only=False,
            still_time=0,
            frame_rate=15,
        )
        trials_df = size_control.get_physical_size(
            trials_df, use_cols=["size", "depth"], k=1
        )

        def p0_func():
            return np.concatenate(
                (
                    np.exp(np.random.normal(size=1)),
                    np.atleast_1d(np.log(neurons_df.loc[roi, "best_size"])),
                    np.exp(np.random.normal(size=1)),
                    np.random.normal(size=1),
                )
            ).flatten()

    # Choose trials
    depth_list = find_depth_list(trials_df)
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    trials_df_fit, choose_trial_nums, sfx = common_utils.choose_trials_subset(
        trials_df, choose_trials
    )

    if closed_loop == 1:
        protocol_sfx = "_closedloop"
    else:
        protocol_sfx = "_openloop"

    # Stratified K-fold cross validation on trials_df_fit
    # if k_folds = 1: fit all data, save the best popt results as depth_tuning_popt;
    # if k_folds > 1: fit data in nfolds, save rsq for test data as depth_tuning_test_rsq;

    # give class labels to each depth / size
    if param == "depth":
        log_depth_list = np.log(depth_list)
        log_depth_list = np.append(
            log_depth_list, (log_depth_list[-1] + log_depth_list[1] - log_depth_list[0])
        )
        log_depth_list = np.insert(
            log_depth_list,
            0,
            (log_depth_list[0] - log_depth_list[1] + log_depth_list[0]),
        )
        depth_list_expand = np.exp(log_depth_list)
        bins = (depth_list_expand[1:] + depth_list_expand[:-1]) / 2
        trials_df_fit["depth_label"] = pd.cut(
            trials_df_fit["depth"], bins=bins, labels=np.arange(len(depth_list))
        )
        Y = trials_df_fit.drop(columns=["depth_label"])
        X_label = trials_df_fit["depth_label"]
        X = trials_df_fit["depth"]
    elif param == "size":
        size_list = np.sort(trials_df_fit["physical_size"].unique())
        log_size_list = np.log(size_list)
        log_size_list = np.append(
            log_size_list, (log_size_list[-1] + log_size_list[1] - log_size_list[0])
        )
        log_size_list = np.insert(
            log_size_list, 0, (log_size_list[0] - log_size_list[1] + log_size_list[0])
        )
        size_list_expand = np.exp(log_size_list)
        bins = (size_list_expand[1:] + size_list_expand[:-1]) / 2
        trials_df_fit["size_label"] = pd.cut(
            trials_df_fit["physical_size"], bins=bins, labels=np.arange(len(size_list))
        )
        Y = trials_df_fit.drop(columns=["size_label"])
        X_label = trials_df_fit["size_label"]
        X = trials_df_fit["physical_size"]

    # thresholding running speed
    if rs_thr is None:
        Y["trial_mean_dff"] = Y.apply(lambda x: np.nanmean(x.dff_stim, axis=0), axis=1)
    else:
        Y["trial_mean_dff"] = Y.apply(
            lambda x: np.nanmean(x.dff_stim[x.RS_stim >= rs_thr, :], axis=0), axis=1
        )

    if param == "depth":
        lower_bounds = [0, np.log(depth_min), 0, -np.inf]
        upper_bounds = [np.inf, np.log(depth_max), np.inf, np.inf]
    elif param == "size":
        lower_bounds = [0, np.log(size_list_expand[0]), 0, -np.inf]
        upper_bounds = [np.inf, np.log(size_list_expand[-1]), np.inf, np.inf]

    # if k_folds = 1: fit all data, save the best popt results as depth_tuning_popt;
    gaussian_func_ = partial(gaussian_func, min_sigma=min_sigma)
    if k_folds == 1:
        # Create empty columns for fitting results
        neurons_df[f"preferred_{param}{protocol_sfx}{sfx}"] = np.nan
        neurons_df[f"{param}_tuning_popt{protocol_sfx}{sfx}"] = [[np.nan]] * len(
            neurons_df
        )
        neurons_df[f"{param}_tuning_trials{protocol_sfx}{sfx}"] = [[np.nan]] * len(
            neurons_df
        )

        for roi in tqdm(range(Y.dff_stim.iloc[0].shape[1])):
            popt, rsq = common_utils.iterate_fit(
                func=gaussian_func_,
                X=np.log(np.array(X)),
                y=np.array(np.stack(Y["trial_mean_dff"])[:, roi]).flatten(),
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                niter=niter,
                p0_func=p0_func,
            )
            neurons_df.at[roi, f"preferred_{param}{protocol_sfx}{sfx}"] = np.exp(
                popt[1]
            )
            neurons_df.at[roi, f"{param}_tuning_popt{protocol_sfx}{sfx}"] = popt
            neurons_df.at[
                roi, f"{param}_tuning_trials{protocol_sfx}{sfx}"
            ] = choose_trial_nums

    # if k_folds > 1: fit data in nfolds, save rsq for test data as depth_tuning_test_rsq;
    elif k_folds > 1:
        neurons_df[f"{param}_tuning_test_rsq{protocol_sfx}{sfx}"] = np.nan
        # initialize StratifiedKFold with the number of folds you want
        stratified_kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=42
        )

        # Loop through each roi
        for roi in tqdm(range(Y.dff_stim.iloc[0].shape[1])):
            # loop through the folds
            y_pred_all = []
            y_test_all = []
            for fold, (train_index, test_index) in enumerate(
                stratified_kfold.split(Y, X_label)
            ):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                y_test_all.append(
                    (np.stack(y_test["trial_mean_dff"])[:, roi]).flatten()
                )

                # Fit depth tuning with gaussian function
                # fit gaussian function to the average dffs for each trial (depth tuning)
                popt, _ = common_utils.iterate_fit(
                    gaussian_func_,
                    np.log(np.array(X_train)).flatten(),
                    np.array(np.stack(y_train["trial_mean_dff"])[:, roi]).flatten(),
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    niter=niter,
                    p0_func=p0_func,
                )
                y_pred = gaussian_func_(np.log(X_test), *popt)
                y_pred_all.append(y_pred)
            rsq = common_utils.calculate_r_squared(
                np.concatenate(y_test_all), np.concatenate(y_pred_all)
            )
            rval, pval = spearmanr(
                np.concatenate(y_test_all), np.concatenate(y_pred_all)
            )
            neurons_df.at[roi, f"{param}_tuning_test_rsq{protocol_sfx}{sfx}"] = rsq
            neurons_df.at[
                roi, f"{param}_tuning_test_spearmanr_rval{protocol_sfx}{sfx}"
            ] = rval
            neurons_df.at[
                roi, f"{param}_tuning_test_spearmanr_pval{protocol_sfx}{sfx}"
            ] = pval

    return neurons_df, neurons_ds
