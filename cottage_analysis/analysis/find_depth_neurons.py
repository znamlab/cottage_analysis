import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import flexiznam as flz

from cottage_analysis.filepath import generate_filepaths
from cottage_analysis.preprocessing import synchronisation
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


def average_dff_for_all_trials(trials_df, rs_thr=0.2):
    """Generate an array (ndepths x ntrials x ncells) for average dffs across each trial.

    Args:
        trials_df (DataFrame): trials_df dataframe for this session that describes the parameters for each trial.
        rs_thr (float, optional): threshold of running speed to be counted into depth tuning analysis. Defaults to 0.2 m/s.
    """
    depth_list = find_depth_list(trials_df)
    if rs_thr is None:
        trials_df["trial_mean_dff"] = trials_df.apply(
            lambda x: np.nanmean(x.dff_stim, axis=1), axis=1
        )
    else:
        trials_df["trial_mean_dff"] = trials_df.apply(
            lambda x: np.nanmean(x.dff_stim[:, x.RS_stim >= rs_thr], axis=1), axis=1
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
    project, mouse, session, protocol="SpheresPermTubeReward", rs_thr=0.2, alpha=0.05
):
    """Find depth neurons from all ROIs segmented.

    Args:
        project (str): project name.
        mouse (str): mouse name.
        session (str): session name.
        protocol (str): protocol name. Defaults to "SpheresPermTubeReward"
        rs_thr (float, optional): threshold of running speed to be counted into
            depth tuning analysis. Defaults to 0.2 m/s.
        alpha (float, optional): significance level for depth tuning analysis. Defaults to 0.05.

    Returns:
        neurons_df (DataFrame): A dataframe that contains the analysed properties for each ROI

    """
    root = Path(flz.PARAMETERS["data_root"]["processed"])
    session_folder = root / project / mouse / session

    # Load files
    trials_df = pd.read_pickle(session_folder / "plane0/trials_df.pickle")
    iscell = common_utils.load_is_cell_file(project, mouse, session, protocol)
    # Make an empty dataframe for saving depth neuron properties
    neurons_df = pd.DataFrame(
        columns=[
            "roi",  # ROI number
            "is_cell",  # bool, is it a cell or not
            "is_depth_neuron",  # bool, is it a depth-selective neuron or not
            "depth_neuron_anova_p",  # float, p value for depth neuron anova test
            "best_depth",  # #, depth with the maximum average response
        ]
    )
    neurons_df["roi"] = np.arange(len(iscell))
    neurons_df["is_cell"] = iscell

    # Find the averaged dFF for each trial in only closed loop recordings
    trials_df = trials_df[trials_df.closed_loop == 1]

    # Anova test to determine which neurons are depth neurons
    depth_list = find_depth_list(trials_df)
    mean_dff_arr = average_dff_for_all_trials(trials_df, rs_thr=rs_thr)

    for iroi in tqdm(np.arange(len(iscell))):
        _, p = scipy.stats.f_oneway(*mean_dff_arr[:, :, iroi])

        neurons_df.loc[iroi, "depth_neuron_anova_p"] = p
        neurons_df.loc[iroi, "is_depth_neuron"] = p < alpha
        neurons_df.loc[iroi, "best_depth"] = depth_list[
            np.argmax(np.mean(mean_dff_arr[:, :, iroi], axis=1))
        ]
    neurons_df.to_pickle(session_folder / "plane0/neurons_df.pickle")

    return neurons_df


def fit_preferred_depth(
    project,
    mouse,
    session,
    protocol,
    depth_min=0.02,
    depth_max=20,
    niter=10,
    min_sigma=0.5,
):
    """Fit depth tuning with 1d gaussian function to find the preferred depth of neurons in closed loop.

    Args:
        project (str): project name.
        mouse (str): mouse name.
        session (str): session name.
        protocol (str, optional): protocol name. Defaults to "SpheresPermTubeReward".
        depth_min (float, optional): Lower boundary of fitted preferred depth (m). Defaults to 0.02.
        depth_max (int, optional): Upper boundary of fitted preferred depth (m). Defaults to 20.
        batch_num (int, optional): Number of batches for fitting the gaussian function. Defaults to 10.

    Returns:
        neurons_df (DataFrame): A dataframe that contains the analysed properties for each ROI.

    """
    root = Path(flz.PARAMETERS["data_root"]["processed"])
    session_folder = root / project / mouse / session

    # Load files
    trials_df = pd.read_pickle(session_folder / "plane0/trials_df.pickle")
    neurons_df = pd.read_pickle(session_folder / "plane0/neurons_df.pickle")
    iscell = common_utils.load_is_cell_file(project, mouse, session, protocol)
    depth_list = find_depth_list(trials_df)

    neurons_df["is_cell"] = iscell
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
        # !! USE LOG(DEPTH LIST IN CM) WHEN CALCULATING, x = np.log(depth_list*100)
        neurons_df.at[iroi, "gaussian_depth_tuning_popt"] = popt
        neurons_df.at[iroi, "gaussian_depth_tuning_r_squared"] = rsq

    neurons_df.to_pickle(session_folder / "plane0/neurons_df.pickle")
    return neurons_df