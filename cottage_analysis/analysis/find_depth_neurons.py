import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import flexiznam as flz

from sklearn.model_selection import StratifiedKFold

from cottage_analysis.analysis import common_utils
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


def average_dff_for_all_trials(trials_df, rs_thr=0.2, closed_loop=1):
    """Generate an array (ndepths x ntrials x ncells) for average dffs across each trial.

    Args:
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        rs_thr (float, optional): threshold of running speed to be counted into depth tuning analysis. Defaults to 0.2 m/s.
    """
    trials_df = trials_df[trials_df.closed_loop == 1]
    depth_list = find_depth_list(trials_df)
    if rs_thr is None:
        trials_df["trial_mean_dff"] = trials_df.apply(
            lambda x: np.nanmean(x.dff_stim, axis=0), axis=1
        )
    else:
        trials_df["trial_mean_dff"] = trials_df.apply(
            lambda x: np.nanmean(x.dff_stim[x.RS_stim >= rs_thr, :], axis=0), axis=1
        )
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

    return mean_dff_arr


def find_depth_neurons(
    session_name,
    trials_df,
    neurons_ds,
    flexilims_session=None,
    project=None,
    rs_thr=0.2,
    alpha=0.05,
    ops=None,
):
    """Find depth neurons from all ROIs segmented.

    Args:
        session_name (str): session name. {Mouse}_{Session}.
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        flexilims_session (Series, optional): flexilims session object. Defaults to None.
        project (str, optional): project name. Defaults to None. Must be provided if flexilims_session is None.
        rs_thr (float, optional): threshold of running speed to be counted into depth tuning analysis. Defaults to 0.2 m/s.
        alpha (float, optional): significance level for anova test. Defaults to 0.05.
        ops (dict, optional): dictionary of parameters. Defaults to None.

    Returns:
        neurons_df (DataFrame): A dataframe that contains the analysed properties for each ROI

    """
    # session paths
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    root = Path(flz.PARAMETERS["data_root"]["processed"])
    session_folder = root / exp_session.path

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

    ops = {
        "depth_neuron_criteria": "anova",
        "depth_neuron_RS_threshold": rs_thr,
    }

    return neurons_df, neurons_ds, ops


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
    ops=None,
):
    '''Function to fit depth tuning with gaussian function

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
        ops (dict, optional): Options for analysis. Defaults to None.

    Returns:
        (pd.DataFrame, Series, dict): neurons_df, neurons_df, ops
    '''
    # Function to initialize depth tuning parameters
    def p0_func():
        return np.concatenate(
            (
                np.exp(np.random.normal(size=1)),
                np.atleast_1d(np.log(neurons_df.loc[roi, "best_depth"])),
                np.exp(np.random.normal(size=1)),
                np.random.normal(size=1),
            )
        ).flatten()

    # Choose trials
    depth_list = find_depth_list(trials_df)
    trial_number = len(trials_df) // len(depth_list)
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    if closed_loop:
        protocol_sfx = "_closedloop"
    else:
        protocol_sfx = "_openloop"

    if choose_trials == None:  # fit all trials
        trials_df_fit = trials_df
        sfx = ""
        choose_trial_nums = np.arange(trial_number)
    else:
        if choose_trials == "odd":  # fit odd trials
            trials_df_fit = pd.DataFrame(columns=trials_df.columns)
            # choose odd trials from trials_df
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[::2, :]
                trials_df_fit = pd.concat([trials_df_fit, trials_df_depth])
            choose_trial_nums = np.arange(trial_number)[::2]
        if choose_trials == "even":  # fit even trials
            trials_df_fit = pd.DataFrame(columns=trials_df.columns)
            # choose even trials from trials_df
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df_depth.iloc[1::2, :]
                trials_df_fit = pd.concat([trials_df_fit, trials_df_depth])
            choose_trial_nums = np.arange(trial_number)[1::2]
        if choose_trials is not None and isinstance(
            choose_trials, list
        ):  # if choose_trials is a given list
            trials_df_fit = pd.DataFrame(columns=trials_df.columns)
            for depth in depth_list:
                trials_df_depth = trials_df[trials_df.depth == depth]
                trials_df_depth = trials_df.iloc[choose_trials, :]
                trials_df_fit = pd.concat([trials_df_fit, trials_df_depth])
            choose_trial_nums = choose_trials
        sfx = "_crossval"


    # Stratified K-fold cross validation on trials_df_fit
    # if k_folds = 1: fit all data, save the best popt results as depth_tuning_popt;
    # if k_folds > 1: fit data in nfolds, save rsq for test data as depth_tuning_test_rsq;

    # give class labels to each depth
    log_depth_list = np.log(depth_list)
    log_depth_list = np.append(
        log_depth_list, (log_depth_list[-1] + log_depth_list[1] - log_depth_list[0])
    )
    log_depth_list = np.insert(
        log_depth_list, 0, (log_depth_list[0] - log_depth_list[1] + log_depth_list[0])
    )
    depth_list_expand = np.exp(log_depth_list)
    bins = (depth_list_expand[1:] + depth_list_expand[:-1]) / 2
    trials_df_fit["depth_label"] = pd.cut(
        trials_df_fit["depth"], bins=bins, labels=np.arange(len(depth_list))
    )
    X = trials_df_fit.drop(columns=["depth_label"])
    y_label = trials_df_fit["depth_label"]
    y = trials_df_fit["depth"]

    # thresholding running speed
    if rs_thr is None:
        X["trial_mean_dff"] = X.apply(lambda x: np.nanmean(x.dff_stim, axis=0), axis=1)
    else:
        X["trial_mean_dff"] = X.apply(
            lambda x: np.nanmean(x.dff_stim[x.RS_stim >= rs_thr, :], axis=0), axis=1
        )

    # if k_folds = 1: fit all data, save the best popt results as depth_tuning_popt;
    gaussian_func_ = partial(gaussian_func, min_sigma=min_sigma)
    if k_folds == 1:
        # Create empty columns for fitting results
        neurons_df[f"preferred_depth{protocol_sfx}{sfx}"] = np.nan
        neurons_df[f"depth_tuning_popt{protocol_sfx}{sfx}"] = [[np.nan]] * len(neurons_df)
        neurons_df[f"depth_tuning_trials{protocol_sfx}{sfx}"] = [[np.nan]] * len(neurons_df)
        
        for roi in tqdm(range(X.dff_stim.iloc[0].shape[1])):
            popt, rsq = common_utils.iterate_fit(
                gaussian_func_,
                np.log(np.array(y)),
                np.array(np.stack(X["trial_mean_dff"])[:, roi]).flatten(),
                lower_bounds=[0, np.log(depth_min), 0, -np.inf],
                upper_bounds=[np.inf, np.log(depth_max), np.inf, np.inf],
                niter=niter,
                p0_func=p0_func,
            )
            neurons_df.at[roi, f"preferred_depth{protocol_sfx}{sfx}"] = np.exp(popt[1])
            neurons_df.at[roi, f"depth_tuning_popt{protocol_sfx}{sfx}"] = popt
            neurons_df.at[
                roi, f"depth_tuning_trials{protocol_sfx}{sfx}"
            ] = choose_trial_nums

    # if k_folds > 1: fit data in nfolds, save rsq for test data as depth_tuning_test_rsq;
    elif k_folds > 1:
        neurons_df[f"depth_tuning_test_rsq{protocol_sfx}{sfx}"] = np.nan
        # initialize StratifiedKFold with the number of folds you want
        stratified_kfold = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=42
        )

        # Loop through each roi
        for roi in tqdm(range(X.dff_stim.iloc[0].shape[1])):
            # loop through the folds
            X_pred_all = []
            X_test_all = []
            for fold, (train_index, test_index) in enumerate(
                stratified_kfold.split(X, y_label)
            ):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                X_test_all.append((np.stack(X_test["trial_mean_dff"])[:, roi]).flatten())

                # Fit depth tuning with gaussian function
                # fit gaussian function to the average dffs for each trial (depth tuning)
                popt, _ = common_utils.iterate_fit(
                    gaussian_func_,
                    np.log(np.array(y_train)).flatten(),
                    np.array(np.stack(X_train["trial_mean_dff"])[:, roi]).flatten(),
                    lower_bounds=[0, np.log(depth_min), 0, -np.inf],
                    upper_bounds=[np.inf, np.log(depth_max), np.inf, np.inf],
                    niter=niter,
                    p0_func=p0_func,
                )
                X_pred = gaussian_func_(np.log(y_test), *popt)
                X_pred_all.append(X_pred)
            rsq = common_utils.calculate_r_squared(
                np.concatenate(X_test_all), np.concatenate(X_pred_all)
            )
            neurons_df.at[roi, f"depth_tuning_test_rsq{protocol_sfx}{sfx}"] = rsq

    ops = {
        "depth_fit_min_sigma": min_sigma,
        "depth_fit_min": depth_min,
        "depth_fit_max": depth_max,
    }
    if k_folds > 1:
        ops["depth_fit_k_folds"] = k_folds

    return neurons_df, neurons_ds, ops
