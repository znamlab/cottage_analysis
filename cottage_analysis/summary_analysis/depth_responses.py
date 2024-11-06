import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

import flexiznam as flz
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.analysis import spheres, common_utils
from cottage_analysis.plotting import depth_selectivity_plots


def get_visually_responsive_neurons(
    trials_df, neurons_df, is_closed_loop=1, before_onset=0.5, frame_rate=15
):
    """Find visually responsive neurons.

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        is_closed_loop (int, optional): whether it's closedloop or openloop. Defaults to 1.
        before_onset (float, optional): time before onset to calculate mean response. Defaults to 0.5.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
    """
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    # Find the mean response of each trial for all ROIs
    trials_df["trial_mean_response"] = trials_df.apply(
        lambda x: np.mean(x.dff_stim, axis=0), axis=1
    )

    # Find the mean response of the blank period before the next trial for all ROIs
    trials_df["trial_mean_onset"] = trials_df.apply(
        lambda x: np.mean(x.dff_blank[-int(frame_rate * before_onset) :], axis=0),
        axis=1,
    )
    # Shift blank response down 1 to put it to the correct trial
    trials_df["trial_mean_onset"] = trials_df["trial_mean_onset"].shift(1)

    all_response = np.stack(trials_df.trial_mean_response[1:].values)
    all_onset = np.stack(trials_df.trial_mean_onset[1:].values)

    # Check whether the response is significantly higher than the blank period
    for iroi, roi in enumerate(neurons_df.roi):
        response = all_response[:, iroi]
        onset = all_onset[:, iroi]
        pval = scipy.stats.wilcoxon(response, onset).pvalue
        neurons_df.at[roi, "visually_responsive"] = (pval < 0.05) & (
            np.mean(response - onset) > 0
        )
        neurons_df.at[roi, "visually_responsive_pval"] = pval
        neurons_df.at[roi, "mean_resp"] = np.mean(response - onset)

    return neurons_df


def get_visually_responsive_all_sessions(
    flexilims_session,
    session_list,
    use_cols,
    is_closed_loop=1,
    protocol_base="SpheresPermTubeReward",
    protocol_base_list=[],
    before_onset=0.5,
    frame_rate=15,
):
    isess = 0
    for i, session_name in enumerate(session_list):
        print(f"Calculating visually responsive neurons for {session_name}")
        if len(protocol_base_list) > 0:
            protocol_base = protocol_base_list[i]

        # Load all data
        if ("PZAH6.4b" in session_name) | ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5

        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        _, trials_df = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            filter_datasets={"anatomical_only": 3},
            recording_type="two_photon",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )
        trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
        neurons_df = get_visually_responsive_neurons(
            trials_df,
            neurons_df,
            is_closed_loop=is_closed_loop,
            before_onset=before_onset,
            frame_rate=frame_rate,
        )
        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df
            else:
                neurons_df = neurons_df[use_cols]

            neurons_df["session"] = session_name
            exp_session = flz.get_entity(
                datatype="session",
                name=session_name,
                flexilims_session=flexilims_session,
            )
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=exp_session.name,
                dataset_type="suite2p_rois",
                filter_datasets={"anatomical_only": 3},
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = np.load(
                suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True
            )[:, 0]
            neurons_df["iscell"] = iscell

            if isess == 0:
                results_all = neurons_df.copy()
            else:
                results_all = pd.concat(
                    [results_all, neurons_df], axis=0, ignore_index=True
                )
            isess += 1
        else:
            print(
                f"ERROR: SESSION {session_name}: specified cols not all in neurons_df"
            )

    return results_all


def get_psth_crossval_all_sessions(
    flexilims_session,
    session_list,
    nbins=10,
    closed_loop=1,
    use_cols=[
        "preferred_depth_closedloop_crossval",
        "depth_tuning_test_rsq_closedloop",
    ],
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    verbose=1,
    corridor_length=6,
    blank_length=0,
    overwrite=False,
):
    """Calculate the PSTH for all sessions in session_list.
    Also calculate running speed PSTH; the correlation between actual and virtual running speeds for openloop sessions.

    Args:
        flexilims_session (Series): flexilims session.
        session_list (list): list of session names.
        nbins (int, optional): number of bins for raster. Defaults to 10.
        closed_loop (int, optional): whether it's closedloop or openloop. Defaults to 1.
        use_cols (list, optional): list of useful columns. Defaults to [ "preferred_depth_closedloop_crossval", "depth_tuning_test_rsq_closedloop", ].
        rs_thr_min (float, optional): running speed min threshold. Defaults to None.
        rs_thr_max (float, optional): running speed max threshold. Defaults to None.
        still_only (bool, optional): whether to only take stationary frames. Defaults to False.
        still_time (float, optional): duration of stationary time. Defaults to 1.
        verbose (bool, optional): verbose. Defaults to 1.
        corridor_length (float, optional): corridor length for one trial. Defaults to 6.
        blank_length (float, optional): length of blank period at each end of the corridor. Defaults to 0.
        overwrite (bool, optional): whether to overwrite the existing results or not. Defaults to False.

    Returns:
        pd.DataFrame: concatenated neurons_df dataframe
    """
    results_all = []
    for isess, session_name in enumerate(session_list):
        print(f"{isess}/{len(session_list)}: calculating PSTH for {session_name}")
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            conflicts="skip",
        )
        psth_path = neurons_ds.path_full.parent / "psth_crossval.pkl"
        if psth_path.exists() and not overwrite:
            results_all.append(pd.read_pickle(psth_path))
            continue
        # Load all data
        if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5
        suite2p_ds = flz.get_datasets_recursively(
            flexilims_session=flexilims_session,
            origin_name=session_name,
            dataset_type="suite2p_traces",
        )
        fs = list(suite2p_ds.values())[0][-1].extra_attributes["fs"]
        try:
            neurons_df = pd.read_pickle(neurons_ds.path_full)
        except FileNotFoundError:
            print(f"ERROR: SESSION {session_name}: neurons_ds not found")
            continue
        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df
            else:
                neurons_df = neurons_df[use_cols]

            _, trials_df = spheres.sync_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets={"anatomical_only": 3},
                recording_type="two_photon",
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
            )
            trials_df = trials_df[trials_df.closed_loop == closed_loop]
            neurons_df["session"] = session_name
            # Add roi, preferred depth, iscell to results
            exp_session = flz.get_entity(
                datatype="session",
                name=session_name,
                flexilims_session=flexilims_session,
            )
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=exp_session.name,
                dataset_type="suite2p_rois",
                filter_datasets={"anatomical_only": 3},
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = np.load(
                suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True
            )[:, 0]
            neurons_df["iscell"] = iscell
            neurons_df["psth_crossval"] = [[np.nan]] * len(neurons_df)

            # Calculate dff psth crossval
            # Get the responses for this session that are not included for calculating the cross-validated preferred depth
            choose_trials_resp = list(
                set(neurons_df.depth_tuning_trials_closedloop.iloc[0])
                - set(neurons_df.depth_tuning_trials_closedloop_crossval.iloc[0])
            )
            trials_df_resp, _, _ = common_utils.choose_trials_subset(
                trials_df, choose_trials_resp, by_depth=True
            )

            print("Calculating dff PSTH")
            for roi in tqdm(range(len(neurons_df))):
                psth, _, _ = depth_selectivity_plots.get_PSTH(
                    trials_df=trials_df_resp,
                    roi=roi,
                    is_closed_loop=closed_loop,
                    max_distance=corridor_length + blank_length,
                    min_distance=-blank_length,
                    nbins=nbins,
                    rs_thr_min=rs_thr_min,
                    rs_thr_max=rs_thr_max,
                    still_only=still_only,
                    still_time=still_time,
                    frame_rate=fs,
                    compute_ci=False,
                )
                neurons_df.at[roi, "psth_crossval"] = psth

            neurons_df.to_pickle(psth_path)
            results_all.append(neurons_df)
            if verbose:
                print(f"Finished concat neurons_df from session {session_name}")
        else:
            print(
                f"ERROR: SESSION {session_name}: specified cols not all in neurons_df"
            )
    results_all = pd.concat(results_all, axis=0, ignore_index=True)
    return results_all
