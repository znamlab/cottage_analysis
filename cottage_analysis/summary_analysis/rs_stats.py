import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import flexiznam as flz
from cottage_analysis.analysis import find_depth_neurons, spheres
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.summary_analysis import depth_responses


def calculate_openloop_rs_correlation(
    imaging_df_openloop, trials_df, separate_depths=False
):
    """Calculate the correlation between actual and virtual running speeds for openloop sessions.

    Args:
        imaging_df_openloop (pd.DataFrame): imaging dataframe for openloop sessions.
        trials_df (pd.DataFrame): trials dataframe.
        separate_depths (bool, optional): whether to calculate the correlation for each depth separately. Defaults to False.
    """
    if not separate_depths:
        rs_actual = imaging_df_openloop["RS"][
            (imaging_df_openloop["RS"].notnull())
            & (imaging_df_openloop["RS_eye"].notnull())
        ]
        rs_eye = imaging_df_openloop["RS_eye"][
            (imaging_df_openloop["RS"].notnull())
            & (imaging_df_openloop["RS_eye"].notnull())
        ]
        r_all, p_all = pearsonr(rs_actual, rs_eye)
    else:
        trials_df_openloop = trials_df[trials_df.closed_loop == 0]
        depth_list = find_depth_neurons.find_depth_list(trials_df)
        r_all = []
        p_all = []
        for depth in depth_list:
            rs_actual = np.hstack(
                trials_df_openloop[trials_df_openloop.depth == depth]["RS_stim"]
            )
            rs_eye = np.hstack(
                trials_df_openloop[trials_df_openloop.depth == depth]["RS_eye_stim"]
            )
            nan_vals = np.isnan(rs_actual) | np.isnan(rs_eye)
            rs_actual = rs_actual[~nan_vals]
            rs_eye = rs_eye[~nan_vals]
            r, p = pearsonr(rs_actual, rs_eye)
            r_all.append(r)
            p_all.append(p)
    return r_all, p_all


def get_rs_stats_all_sessions(
    flexilims_session,
    session_list,
    nbins=60,
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    corridor_length=6,
    blank_length=3,
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
    results_all = pd.DataFrame(
        columns=[
            [
                "session",
                "rs_psth_stim_closedloop",
                "rs_psth_closedloop",
                "rs_mean_trials_closedloop",
                "rs_mean_closedloop",
                "rs_psth_stim_openloop",
                "rs_psth_openloop",
                "rs_mean_trials_openloop",
                "rs_mean_openloop",
                "rs_correlation_rval_openloop",
                "rs_correlation_pval_openloop",
                "rs_correlation_rval_openloop_alldepths",
                "rs_correlation_pval_openloop_alldepths",
            ]
        ],
        index=np.arange(len(session_list)),
    )
    (
        results_all["rs_psth_stim_closedloop"],
        results_all["rs_psth_closedloop"],
        results_all["rs_mean_trials_closedloop"],
        results_all["rs_mean_closedloop"],
        results_all["rs_psth_stim_openloop"],
        results_all["rs_psth_openloop"],
        results_all["rs_mean_trials_openloop"],
        results_all["rs_mean_openloop"],
        results_all["rs_correlation_rval_openloop_alldepths"],
        results_all["rs_correlation_pval_openloop_alldepths"],
    ) = (
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
    )
    for isess, session_name in enumerate(session_list):
        print(f"{isess}/{len(session_list)}: calculating RS stats for {session_name}")
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            conflicts="skip",
        )
        save_path = neurons_ds.path_full.parent / "rs_stats.pkl"
        if save_path.exists() and not overwrite:
            results_all.iloc[isess] = pd.read_pickle(save_path)
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
        trials_df_original = trials_df.copy()
        for closed_loop in trials_df_original.closed_loop.unique():
            trials_df = trials_df_original[
                trials_df_original.closed_loop == closed_loop
            ]
            if closed_loop:
                sfx = "closedloop"
            else:
                sfx = "openloop"
            results_all.at[isess, "session"] = session_name

            # Calculate the running speed psth
            print("Calculating running speed PSTH")
            # just for stim period
            rs_psth_stim, _, _ = depth_responses.get_PSTH(
                trials_df=trials_df,
                roi=0,
                use_col="RS",
                is_closed_loop=closed_loop,
                max_distance=corridor_length,
                min_distance=0,
                nbins=nbins,
                rs_thr_min=rs_thr_min,
                rs_thr_max=rs_thr_max,
                still_only=still_only,
                still_time=still_time,
                frame_rate=fs,
                compute_ci=False,
            )
            results_all.at[isess, f"rs_psth_stim_{sfx}"] = np.expand_dims(
                rs_psth_stim, 0
            )

            # stim + some blank period
            rs_psth, _, _ = depth_responses.get_PSTH(
                trials_df=trials_df,
                roi=0,
                use_col="RS",
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
            results_all.at[isess, f"rs_psth_{sfx}"] = np.expand_dims(rs_psth, 0)

            mean_rs = find_depth_neurons.average_dff_for_all_trials(
                trials_df=trials_df,
                use_col="RS_stim",
                rs_col="RS_stim",
                rs_thr=rs_thr_min,
                rs_thr_max=rs_thr_max,
                still_only=still_only,
                still_time=still_time,
                frame_rate=fs,
                closed_loop=closed_loop,
                param="depth",
            )
            results_all.at[isess, f"rs_mean_trials_{sfx}"] = np.expand_dims(mean_rs, 0)
            results_all.at[isess, f"rs_mean_{sfx}"] = np.expand_dims(
                np.expand_dims(np.mean(mean_rs, axis=1), 0), 0
            )

        # Calculate openloop rs and rs_eye correlation
        if len(trials_df_original.closed_loop.unique()) == 2:
            print("Calculating openloop RS correlation")
            _, imaging_df_openloop = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets={"anatomical_only": 3},
                recording_type="two_photon",
                is_closedloop=0,
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                resolution=5,
                regenerate_frames=False,
            )
            r, p = calculate_openloop_rs_correlation(
                imaging_df_openloop, trials_df_original, separate_depths=False
            )
            results_all.at[isess, "rs_correlation_rval_openloop"] = r
            results_all.at[isess, "rs_correlation_pval_openloop"] = p
            results_all.loc[isess, "rs_correlation_rval_openloop"] = results_all.loc[
                isess, "rs_correlation_rval_openloop"
            ].apply(lambda x: f"{x:.15e}")
            results_all.loc[isess, "rs_correlation_pval_openloop"] = results_all.loc[
                isess, "rs_correlation_pval_openloop"
            ].apply(lambda x: f"{x:.15e}")
            r_all, p_all = calculate_openloop_rs_correlation(
                imaging_df_openloop, trials_df_original, separate_depths=True
            )
            results_all.at[isess, "rs_correlation_rval_openloop_alldepths"] = (
                np.expand_dims(np.expand_dims(r_all, 0), 0)
            )
            results_all.at[isess, "rs_correlation_pval_openloop_alldepths"] = (
                np.expand_dims(np.expand_dims(p_all, 0), 0)
            )

        # append results_df
        results_all.iloc[isess] = results_all.iloc[isess].apply(np.squeeze)
        results_all.iloc[isess].to_pickle(save_path)

    return results_all


def get_rs_of_all_sessions(
    flexilims_session,
    session_list,
):
    """
    Concatenate all running speed and optic flow speed values for all sessions in session_list.
    """
    results_all = pd.DataFrame(
        columns=[
            [
                "session",
                "is_closedloop",
            ]
        ],
        index=np.arange(len(session_list)),
    )
    results_all["RS_stim"] = [[np.nan]] * len(results_all)

    for isess, session_name in enumerate(session_list):
        print(f"{isess}/{len(session_list)}: concatenating RS & OF values for {session_name}")
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            conflicts="skip",
        )
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
        trials_df_original = trials_df.copy()
        for closed_loop in trials_df_original.closed_loop.unique():
            trials_df = trials_df_original[
                trials_df_original.closed_loop == closed_loop
            ]
            if closed_loop:
                sfx = "closedloop"
            else:
                sfx = "openloop"
            results_all.at[isess, "session"] = session_name
            results_all.at[isess, "is_closedloop"] = closed_loop
            results_all.at[isess, "RS_stim"] = np.concatenate(trials_df["RS_stim"].values).reshape(1,1,-1)
            for depth in np.sort(trials_df.depth.unique()):
                if f"OF_stim_depth_{depth}" not in results_all.columns:
                    results_all[f"OF_stim_depth_{np.round(depth,2)}"] = [[np.nan]] * len(results_all)
                results_all.at[isess, f"OF_stim_depth_{np.round(depth,2)}"] = np.concatenate(
                    trials_df[trials_df.depth == depth]["OF_stim"].values
                ).reshape(1,1,-1)
                
    return results_all
