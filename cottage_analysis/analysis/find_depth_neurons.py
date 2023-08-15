import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import flexiznam as flz

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
    flexilims_session=None,
    project=None,
    rs_thr=0.2,
    alpha=0.05,
    conflicts="skip",
):
    """Find depth neurons from all ROIs segmented.

    Args:
        session_name (str): session name. {Mouse}_{Session}.
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        flexilims_session (Series, optional): flexilims session object. Defaults to None.
        project (str, optional): project name. Defaults to None. Must be provided if flexilims_session is None.
        rs_thr (float, optional): threshold of running speed to be counted into depth tuning analysis. Defaults to 0.2 m/s.
        alpha (float, optional): significance level for anova test. Defaults to 0.05.
        conflicts (str, optional): how to handle conflicts when running neurons_df. Defaults to "skip".

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

    # Create a neurons_df dataset from flexilism
    neurons_ds = flz.Dataset.from_origin(
        origin_id=exp_session.id,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        conflicts=conflicts,
    )
    neurons_ds.path = neurons_ds.path.parent / f"neurons_df.pickle"

    # if neurons_ds exists on flexilims and conflicts is set to skip, load the existing neurons_df
    if (neurons_ds.flexilims_status() != "not online") and (conflicts == "skip"):
        print("Loading existing neurons_df file...")
        return np.load(neurons_ds.path_full), neurons_df

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

    for iroi in tqdm(np.arange(nrois)):
        _, p = scipy.stats.f_oneway(*mean_dff_arr[:, :, iroi])

        neurons_df.loc[iroi, "depth_neuron_anova_p"] = p
        neurons_df.loc[iroi, "is_depth_neuron"] = p < alpha
        neurons_df.loc[iroi, "best_depth"] = depth_list[
            np.argmax(np.mean(mean_dff_arr[:, :, iroi], axis=1))
        ]

    # save neurons_df
    neurons_ds.path_full.parent.mkdir(parents=True, exist_ok=True)
    neurons_df.to_pickle(neurons_ds.path_full)

    # update flexilims
    neurons_ds.extra_attributes["depth_neuron_criteria"] = "anova"
    neurons_ds.extra_attributes["depth_neuron_RS_threshold"] = rs_thr
    neurons_ds.update_flexilims(mode="overwrite")

    return neurons_df, neurons_ds


def fit_preferred_depth(
    trials_df,
    neurons_df,
    neurons_ds,
    depth_min=0.02,
    depth_max=20,
    niter=10,
    min_sigma=0.5,
    conflicts="skip",
):
    """Fit depth tuning with 1d gaussian function to find the preferred depth of neurons in closed loop.

    Args:
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        neurons_df (DataFrame): A dataframe that contains the analysed properties for each ROI.
        neurons_ds (Dataset): A dataset that contains the analysed properties for each ROI.
        depth_min (float, optional): Lower boundary of fitted preferred depth (m). Defaults to 0.02.
        depth_max (int, optional): Upper boundary of fitted preferred depth (m). Defaults to 20.
        batch_num (int, optional): Number of batches for fitting the gaussian function. Defaults to 10.

    Returns:
        neurons_df (DataFrame): A dataframe that contains the analysed properties for each ROI.

    """
    # session paths
    session_folder = neurons_ds.path_full.parent

    if conflicts == "skip":
        return neurons_df, neurons_ds
    
    # Initialize neurons_df
    depth_list = find_depth_list(trials_df)
    neurons_df["preferred_depth_closed_loop"] = np.nan
    neurons_df["gaussian_depth_tuning_popt"] = [[np.nan]] * len(neurons_df)
    neurons_df["gaussian_depth_tuning_r_squared"] = np.nan

    # Find the averaged dFF for each trial in only closed loop recordings
    trials_df = trials_df[trials_df.closed_loop == 1]
    mean_dff_arr = average_dff_for_all_trials(trials_df)

    # Fit gaussian function to the average dffs for each trial (depth tuning)
    x = np.log(np.repeat(np.array(depth_list), mean_dff_arr.shape[1]))

    def p0_func():
        return np.concatenate(
            (
                np.exp(np.random.normal(size=1)),
                np.atleast_1d(np.log(neurons_df.loc[iroi, "best_depth"])),
                np.exp(np.random.normal(size=1)),
                np.random.normal(size=1),
            )
        ).flatten()

    for iroi in tqdm(range(mean_dff_arr.shape[2])):
        gaussian_func_ = partial(gaussian_func, min_sigma=min_sigma)
        popt, rsq = common_utils.iterate_fit(
            gaussian_func_,
            x,
            mean_dff_arr[:, :, iroi].flatten(),
            lower_bounds=[0, np.log(depth_min), 0, -np.inf],
            upper_bounds=[np.inf, np.log(depth_max), np.inf, np.inf],
            niter=niter,
            p0_func=p0_func,
        )

        neurons_df.at[iroi, "preferred_depth_closed_loop"] = np.exp(popt[1])
        neurons_df.at[iroi, "gaussian_depth_tuning_popt"] = popt
        neurons_df.at[iroi, "gaussian_depth_tuning_r_squared"] = rsq

    # save neurons_df
    neurons_df.to_pickle(session_folder / "neurons_df.pickle")
    
    # update flexilims
    neurons_ds.extra_attributes["fit_depth_min"] = depth_min
    neurons_ds.extra_attributes["fit_depth_max"] = depth_max
    neurons_ds.extra_attributes["fit_depth_min_sigma"] = min_sigma
    neurons_ds.update_flexilims(mode="overwrite")
    
    return neurons_df, neurons_ds
