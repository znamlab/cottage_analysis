from functools import partial
import flexiznam as flz
import numpy as np
import pandas as pd
from znamutils import slurm_it

from cottage_analysis.analysis.spheres import multidepth
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis.spheres.stimulus_reconstruction import regenerate_frames
from cottage_analysis.utilities.misc import get_str_or_recording
from cottage_analysis.io_module.visstim import get_param_log

# Import the module, not the function: fit_gaussian_blob imports this package (via
# find_depth_neurons -> size_control -> spheres), so `from ... import fit_rs_of_tuning`
# fails when fit_gaussian_blob is imported first. A module import resolves the attribute
# lazily, at call time.
from cottage_analysis.analysis import fit_gaussian_blob

print = partial(print, flush=True)


def calculate_optic_flow_angle(r, r_new, distance):
    """Calculates the optic flow angle.

    Args:
        r (float): The initial distance.
        r_new (float): The new distance.
        distance (float): The distance between the two
            points.

    Returns:
        float: The optic flow angle.
    """
    angle = np.arccos((r**2 + r_new**2 - distance**2) / (2 * r * r_new))
    return angle


def format_imaging_df(recording, imaging_df):
    """Format sphere params in imaging_df.

    Args:
        recording (Series): recording entry returned by
            flexiznam.get_entity(name=recording_name,
            project_id=project).
        imaging_df (pd.DataFrame): dataframe that contains
            info for each monitor frame.

    Returns:
        DataFrame: contains information for each monitor
            frame and vis-stim.
    """
    if "Radius" in imaging_df.columns:
        imaging_df = imaging_df.rename(columns={"Radius": "depth"})
    elif "Depth" in imaging_df.columns:
        imaging_df = imaging_df.rename(columns={"Depth": "depth"})
    # Indicate whether it's a closed loop or open loop session
    if "Playback" in recording.name:
        imaging_df["closed_loop"] = 0
    else:
        imaging_df["closed_loop"] = 1
    imaging_df["RS"] = (
        imaging_df.mouse_z_harp.diff() / imaging_df.mouse_z_harptime.diff()
    )
    # average RS eye for each imaging volume
    imaging_df["RS_eye"] = imaging_df.eye_z.diff() / imaging_df.monitor_harptime.diff()
    imaging_df.depth = imaging_df.depth / 100  # convert cm to m
    # OF for each imaging volume
    imaging_df["OF"] = imaging_df.RS_eye / imaging_df.depth
    return imaging_df


def find_stim_time(
    imaging_df, is_multidepth=False, param_log=None, diagnostics_folder=None
):
    if "stim" in imaging_df.columns and imaging_df.stim.isin([0, 1]).any():
        return imaging_df
    imaging_df["stim"] = np.nan
    if not is_multidepth:
        # easy, just find when depth is changing
        imaging_df.loc[imaging_df.depth.notnull(), "stim"] = 1
        imaging_df.loc[imaging_df.depth < 0, "stim"] = 0
    else:
        # we have 2 pb. Depth is not unique and the loggers might drift
        # first put all stim during protocol to 0
        imaging_df.loc[imaging_df.depth.notnull(), "stim"] = 0
        # then get trial start and stop, in frame logger time
        if diagnostics_folder is not None:
            diagnostics_folder.mkdir(parents=True, exist_ok=True)
            (
                harpstim_time,
                param_log_index,
                all_onsets,
                all_offsets,
                closest_offset,
            ) = multidepth.find_trial_times(param_log, debug=True)
            fig = multidepth.trial_diagnostic(
                param_log, harpstim_time, all_offsets, all_onsets
            )
            fig.savefig(diagnostics_folder / "trial_diagnostic.png")
        else:
            harpstim_time, param_log_index = multidepth.find_trial_times(param_log)
        # We know when are trial in the frame logger, find where it fits in the
        # imaging_df
        imaging_df_frame_log = np.array(imaging_df.stimulus_harptime)
        half = len(imaging_df_frame_log) // 2
        imaging_df_frame_log[:half] = np.nan_to_num(imaging_df_frame_log[:half])
        imaging_df_frame_log[half:] = np.nan_to_num(
            imaging_df_frame_log[half:], nan=imaging_df_frame_log.max() + 1
        )
        # replace the initial NaN to be put first onset at stim time
        onset_index = imaging_df_frame_log.searchsorted(harpstim_time[0])
        if onset_index[0] == 0:
            print("WARNING: stim started before imaging")
            onset_index[0] += 1  # shift by 1 frame otherwise crash at find blank-pre
        offset_index = imaging_df_frame_log.searchsorted(harpstim_time[1])
        for on, off in zip(onset_index, offset_index):
            imaging_df.loc[on:off, "stim"] = 1
    return imaging_df


def generate_trials_df(
    recording,
    imaging_df,
    is_multidepth=False,
    param_log=None,
    acceleration_time=0.5,
    add_spikes=False,
):
    """Generate a DataFrame that contains information for each trial.

    Args:
        recording (Series): recording entry returned by
            flexiznam.get_entity(name=recording_name,
            project_id=project).
        imaging_df(pd.DataFrame): dataframe that contains
            info for each imaging volume.

    Returns:
        DataFrame: contains information for each trial.

    """
    trials_df = pd.DataFrame(
        columns=[
            "trial_no",
            "depth",
            "recording_name",
            "closed_loop",
            "imaging_harptime_stim_start",
            "imaging_harptime_stim_stop",
            "imaging_harptime_blank_start",
            "imaging_harptime_blank_stop",
            "imaging_stim_start",
            "imaging_stim_stop",
            "imaging_blank_start",
            "imaging_blank_stop",
            "imaging_blank_pre_start",
            "imaging_blank_pre_stop",
            "RS_stim",  # actual running speed, m/s
            "RS_blank",
            "RS_blank_pre",
            "RS_eye_stim",  # virtual running speed, m/s
            "OF_stim",  # optic flow speed = RS/depth, rad/s
            "dff_stim",
            "dff_blank",
            "dff_blank_pre",
            "mouse_z_harp_stim",
            "mouse_z_harp_blank",
            "mouse_z_harp_blank_pre",
        ]
    )
    # Find the change of depth
    # Diagnostics folder used only for multidepth experiements
    diagnostics_folder = flz.get_processed_path(recording.path) / "diagnostics"
    imaging_df = find_stim_time(
        imaging_df, is_multidepth, param_log, diagnostics_folder=diagnostics_folder
    )
    frame_rate = 1 / np.median(np.diff(imaging_df.imaging_harptime))
    n_acc_frames = int(acceleration_time * frame_rate)
    # acceleration is the difference in RS between current frame and n_acc_frames ago
    imaging_df["acceleration_abs"] = imaging_df.RS.diff(n_acc_frames)
    # acceleration ratio is the ratio between current RS and RS n_acc_frames ago
    imaging_df["acceleration_ratio"] = imaging_df.RS / imaging_df.RS.shift(n_acc_frames)
    # max acceleration in the past n_acc_frames
    imaging_df["acceleration_abs_max"] = imaging_df.RS.rolling(n_acc_frames).apply(
        lambda x: np.abs(np.max(x) - np.min(x)), raw=True
    )
    imaging_df["acceleration_ratio_max"] = imaging_df.RS.rolling(n_acc_frames).apply(
        lambda x: (
            np.abs(np.log2(np.max(x) / np.min(x)))
            if np.min(x) > 0 and np.max(x) > 0
            else np.nan
        ),
        raw=True,
    )
    imaging_df_simple = imaging_df[
        (imaging_df["stim"].diff() != 0) & (imaging_df["stim"]).notnull()
    ].copy()
    imaging_df_simple.depth = np.round(imaging_df_simple.depth, 2)
    # Find frame or volume of imaging_df for trial start and stop
    # (depending on whether return_volume=True in generate_imaging_df)
    blank_time = 10
    start_volume_stim = imaging_df_simple[
        (imaging_df_simple["stim"] == 1)
    ].imaging_volume.values
    start_volume_blank = imaging_df_simple[
        (imaging_df_simple["stim"] == 0)
    ].imaging_volume.values
    if start_volume_blank[0] < start_volume_stim[0]:
        print("Warning: blank starts before stimulus starts! Double check!")
        start_volume_blank = start_volume_blank[1:]
        assert (
            start_volume_blank[0] > start_volume_stim[0]
        ), "Warning: 2 blank starts before stimulus starts! Double check!"

    if len(start_volume_stim) != len(
        start_volume_blank
    ):  # if trial start and blank numbers are different
        if (
            len(start_volume_stim) - len(start_volume_blank)
        ) == 1:  # last trial is not complete when stopping the recording
            stop_volume_blank = start_volume_stim[1:] - 1
            start_volume_stim = start_volume_stim[: len(start_volume_blank)]
        else:  # something is wrong
            print("Warning: incorrect stimulus trial structure! Double check!")
    else:  # if trial start and blank numbers are the same
        stop_volume_blank = start_volume_stim[1:] - 1
        last_blank_stop_time = (
            imaging_df.loc[start_volume_blank[-1]].imaging_harptime + blank_time
        )
        stop_volume_blank = np.append(
            stop_volume_blank,
            (np.abs(imaging_df.imaging_harptime - last_blank_stop_time)).idxmin(),
        )
    stop_volume_stim = start_volume_blank - 1
    start_volume_blank_pre = np.append(0, start_volume_blank[:-1])
    stop_volume_blank_pre = start_volume_stim - 1
    # Assign trial no, depth, start/stop time, start/stop imaging volume to trials_df
    # harptime are imaging trigger harp time
    trials_df.trial_no = np.arange(len(start_volume_stim))
    trials_df.depth = pd.Series(imaging_df.loc[start_volume_stim].depth.values)
    trials_df.imaging_harptime_stim_start = imaging_df.loc[
        start_volume_stim
    ].imaging_harptime.values
    trials_df.imaging_harptime_stim_stop = imaging_df.loc[
        stop_volume_stim
    ].imaging_harptime.values
    trials_df.imaging_harptime_blank_start = imaging_df.loc[
        start_volume_blank
    ].imaging_harptime.values
    trials_df.imaging_harptime_blank_stop = imaging_df.loc[
        stop_volume_blank
    ].imaging_harptime.values

    trials_df.imaging_stim_start = pd.Series(start_volume_stim)
    trials_df.imaging_stim_stop = pd.Series(stop_volume_stim)
    trials_df.imaging_blank_start = pd.Series(start_volume_blank)
    trials_df.imaging_blank_stop = pd.Series(stop_volume_blank)
    trials_df.imaging_blank_pre_start = pd.Series(start_volume_blank_pre)
    trials_df.imaging_blank_pre_stop = pd.Series(stop_volume_blank_pre)
    # If the blank stop of last trial is beyond the number of imaging frames
    if np.isnan(trials_df.imaging_blank_stop.iloc[-1]):
        trials_df.imaging_blank_stop.iloc[-1] = len(imaging_df) - 1
    # Get rid of the overlap of imaging frame no. between different trials
    mask = trials_df.imaging_stim_start == trials_df.imaging_blank_stop.shift(1)
    trials_df.loc[mask, "imaging_stim_start"] += 1

    # Assign protocol to trials_df
    if "Playback" in recording.name:
        trials_df.closed_loop = 0
    else:
        trials_df.closed_loop = 1

    def assign_values_to_df(trials_df, imaging_df, column_name, epoch):
        trials_df[f"{column_name}_{epoch}"] = trials_df.apply(
            lambda x: imaging_df[column_name]
            .loc[int(x[f"imaging_{epoch}_start"]) : int(x[f"imaging_{epoch}_stop"])]
            .values,
            axis=1,
        )
        return trials_df

    columns_to_assign = [
        "mouse_z_harp",
        "mouse_z_harp",
        "RS",
        "RS_eye",
        "OF",
        "acceleration_abs",
        "acceleration_ratio",
        "acceleration_abs_max",
        "acceleration_ratio_max",
    ]
    optional_columns = [
        "expected_optic_flow",
        "MotorSps",
        "MotorSpeed",
        "max_abs_rs2motor_diff",
        "max_abs_rs2motor_diff_ratio",
        "mean_rs2motor_diff",
    ]
    for column in optional_columns:
        if column in imaging_df.columns:
            columns_to_assign.append(column)
    for epoch in ["stim", "blank", "blank_pre"]:
        for column in columns_to_assign:
            if (column not in ["OF", "RS_eye"]) or (epoch == "stim"):
                trials_df = assign_values_to_df(trials_df, imaging_df, column, epoch)
        trials_df[f"dff_{epoch}"] = trials_df.apply(
            lambda x: np.stack(
                imaging_df.dffs.loc[
                    int(x[f"imaging_{epoch}_start"]) : int(x[f"imaging_{epoch}_stop"])
                ]
            ).squeeze(),
            axis=1,
        )
        if add_spikes and "spks" in imaging_df.columns:
            trials_df[f"spks_{epoch}"] = trials_df.apply(
                lambda x: np.stack(
                    imaging_df.spks.loc[
                        int(x[f"imaging_{epoch}_start"]) : int(
                            x[f"imaging_{epoch}_stop"]
                        )
                    ]
                ).squeeze(),
                axis=1,
            )

    # Add recording name
    trials_df.recording_name = recording.genealogy[-1]
    # Rename
    trials_df = trials_df.drop(columns=["imaging_blank_start"])

    return trials_df


def search_param_log_trials(
    harp_recording,
    trials_df,
    flexilims_session,
    vis_stim_recording=None,
    is_multidepth=False,
):
    """Add the start param logger row and stop param logger row to each
    trial. This is required for regenerate_spheres.

    Args:
        harp_recording (Series or str): Harp recording
        trials_df (pd.DataFrane): Dataframe that contails
            information for each trial.
        flexilims_session (flexilims_session): flexilims session.
        vis_stim_recording (Series or str, optional): Visual
            stimulation recording. required if `recording` does not
            contain vis_stim info. Defaults to None.
        multidepth (bool): if True, the depth is in the param log.
            Defaults to False.

    Returns:
        Dataframe: Dataframe that contails information for each trial.
    """
    harp_recording = get_str_or_recording(harp_recording, flexilims_session)
    param_log = get_param_log(
        flexilims_session,
        vis_stim_recording=vis_stim_recording,
        harp_recording=harp_recording,
        multidepth=is_multidepth,
    )

    if is_multidepth:
        harpstim_time, param_log_index = multidepth.find_trial_times(
            param_log, verbose=False
        )
        param_log_start = param_log_index[0]
        param_log_stop = param_log_index[1]
    else:
        # find trial index from param_log
        param_log["stim"] = np.nan
        if "Radius" in param_log.columns:
            param_log.loc[param_log.Radius.notnull(), "stim"] = 1
            param_log.loc[param_log.Radius < 0, "stim"] = 0
        elif "Depth" in param_log.columns:
            param_log.loc[param_log.Depth.notnull(), "stim"] = 1
            param_log.loc[param_log.Depth < 0, "stim"] = 0
        p_log_simple = param_log[
            (param_log["stim"].diff() != 0) & (param_log["stim"]).notnull()
        ]
        # find the line of param_log at which trials start and stop
        param_log_start = p_log_simple[(p_log_simple["stim"] == 1)].index
        param_log_stop = p_log_simple[(p_log_simple["stim"] == 0)].index

    # trial index for each row of param log
    trials_df["param_log_start"] = param_log_start[: len(trials_df)]
    trials_df["param_log_stop"] = param_log_stop[: len(trials_df)]

    return trials_df


def sync_all_recordings(
    session_name,
    flexilims_session=None,
    project=None,
    filter_datasets=None,
    exclude_datasets=None,
    recording_type="two_photon",
    protocol_base="SpheresPermTubeReward",
    photodiode_protocol=5,
    return_volumes=True,
    harp_is_in_recording=True,
    use_onix=False,
    conflicts="skip",
    add_spikes=False,
    sync_kwargs=None,
    ephys_kwargs=None,
):
    """Concatenate synchronisation results for all recordings in a
    session.

    Args:
        session_name (str): {mouse}_{session}
        flexilims_session (flexilims_session, optional): flexilims
            session. Defaults to None.
        project (str): project name. Defaults to None. Must be
            provided if flexilims_session is None.
        filter_datasets (dict): dictionary of filter keys and values
            to filter for the desired suite2p dataset (e.g.
            {'anatomical':3}) Default to None.
        exclude_datasets (dict): dictionary of filter keys and values
            to exclude for the undesired suite2p dataset (e.g.
            {'annotated':'yes'}) Default to None.
        recording_type (str, optional): Type of the recording.
            Defaults to "two_photon".
        protocol_base (str, optional): Base of the protocol. Defaults
            to "SpheresPermTubeReward".
        photodiode_protocol (int): number of photodiode quad colors
            used for monitoring frame refresh. Either 2 or 5 for now.
            Defaults to 5.
        return_volumes (bool): if True, return only the first frame
            of each imaging volume. Defaults to True.
        harp_is_in_recording (bool): if True, harp is in the same
            recording as the imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for
            synchronisation. Defaults to False.
        conflicts (str): how to handle conflicts. Defaults to "skip".
        sync_kwargs (dict): kwargs for synchronisation.generate_vs_df.
            Defaults to None.
        return_multiunit (bool): if True, process multiunit activity.
            Defaults to False.
        ephys_kwargs (dict): Keyword arguments for generate_spike_rate_df.
            `return_multiunit` or `exp_sd` for instance. Defaults to
            None.

    Returns:
        (pd.DataFrame, pd.DataFrame): tuple of two dataframes, one
            concatenated vs_df for all recordings, one concatenated
            trials_df for all recordings.
    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    if exp_session is None:
        raise IOError(f"No session called {session_name} found in flexilims.")
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value=recording_type,
        flexilims_session=flexilims_session,
    )
    recordings = recordings[recordings.name.str.contains(protocol_base)]
    # Special case for multidepth, the protocol_base contains the normal sphere
    if protocol_base == "SpheresPermTubeReward":
        recordings = recordings[~recordings.name.str.contains("multidepth")]

    if "exclude_reason" in recordings.columns:
        recordings = recordings[recordings["exclude_reason"].isna()]

    for i, recording_name in enumerate(recordings.name):
        print(f"Processing recording {i+1}/{len(recordings)}")

        (
            vs_df,
            imaging_df,
            trials_df,
            param_log,
            recording,
            unit_ids,
        ) = _process_single_recording_for_session(
            recording_name,
            flexilims_session,
            harp_is_in_recording,
            use_onix,
            photodiode_protocol,
            sync_kwargs,
            protocol_base,
            conflicts,
            recording_type,
            filter_datasets,
            exclude_datasets,
            return_volumes,
            ephys_kwargs,
            verbose=True,
            add_spikes=add_spikes,
        )

        if i == 0:
            vs_df_all = vs_df
            trials_df_all = trials_df
        else:
            vs_df_all = pd.concat([vs_df_all, vs_df], ignore_index=True)
            trials_df_all = pd.concat([trials_df_all, trials_df], ignore_index=True)
    print(f"Finished concatenating vs_df and trials_df")

    return vs_df_all, trials_df_all


def regenerate_frames_all_recordings(
    session_name,
    flexilims_session=None,
    project=None,
    filter_datasets=None,
    exclude_datasets=None,
    recording_type="two_photon",
    protocol_base="SpheresPermTubeReward",
    is_closedloop=1,
    is_multidepth=False,
    photodiode_protocol=5,
    return_volumes=True,
    resolution=5,
    sync_kwargs=None,
    harp_is_in_recording=True,
    use_onix=False,
    ephys_kwargs=None,
    do_regenerate_frames=True,
    verbose=True,
):
    """Concatenate regenerated frames for all recordings in a session.

    Args:
        session_name (str): {mouse}_{session}
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to
            None.
        project (str): project name. Defaults to None. Must be provided if
            flexilims_session is None.
        filter_datasets (dict): dictionary of filter keys and values to filter for the
            desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        exclude_datasets (dict): dictionary of filter keys and values to exclude for the
            undesired suite2p dataset (e.g. {'annotated':'yes'}) Default to None.
        recording_type (str, optional): Type of the recording. Defaults to "two_photon".
        protocol_base (str, optional): Base of the protocol. Defaults to
            "SpheresPermTubeReward".
        is_closedloop (bool): if True, closed loop session. Defaults to True.
        is_multidepth (bool): if True, multidepth session. Defaults to False.
        photodiode_protocol (int): number of photodiode quad colors used for monitoring
            frame refresh. Either 2 or 5 for now. Defaults to 5.
        return_volumes (bool): if True, return only the first frame of each imaging
            volume. Defaults to True.
        resolution (float): size of a pixel in degrees
        sync_kwargs (dict): kwargs for synchronisation.generate_vs_df. Defaults to None.
        harp_is_in_recording (bool): if True, harp is in the same recording as the
            imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for synchronisation. Defaults to
             False.
        ephys_kwargs (dict): Keyword arguments for generate_spike_rate_df.
            `return_multiunit` or `exp_sd` for instance. Defaults to None.
        do_regenerate_frames (bool): if True, regenerate frames. Defaults to True.
        verbose (bool): if True, print progress. Defaults to True.

    Returns:
        (np.array, pd.DataFrame): tuple, one concatenated regenerated frames for all
            recordings (nframes * y * x), one concatenated imaging_df for all recordings
    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value=recording_type,
        flexilims_session=flexilims_session,
    )
    recordings = recordings[recordings.name.str.contains(protocol_base)]
    playback_rec = recordings.name.str.contains("Playback")
    if is_closedloop:
        recordings = recordings[~playback_rec]
    else:
        recordings = recordings[playback_rec]
    multi_depth_rec = recordings.name.str.contains("multidepth")
    if is_multidepth:
        recordings = recordings[multi_depth_rec]
    else:
        recordings = recordings[~multi_depth_rec]

    conflicts = "skip"
    for i, recording_name in enumerate(recordings.name):
        print(f"Regenerating frames for recording {i+1}/{len(recordings)}")
        (
            vs_df,
            imaging_df,
            trials_df,
            param_log,
            recording,
            unit_ids,
        ) = _process_single_recording_for_session(
            recording_name,
            flexilims_session,
            harp_is_in_recording,
            use_onix,
            photodiode_protocol,
            sync_kwargs,
            protocol_base,
            conflicts,
            recording_type,
            filter_datasets,
            exclude_datasets,
            return_volumes,
            ephys_kwargs,
            verbose=True,
        )

        # Regenerate frames for this trial
        sphere_size = (
            10
            * (vs_df.OriginalSize.unique()[~np.isnan(vs_df.OriginalSize.unique())][0])
            / 0.087
        )
        assert not isinstance(sphere_size, list)
        if do_regenerate_frames:
            if "multidepth" in recording.protocol:
                separate_depth = param_log.Radius.unique()
                # remove the -9999 and nan
                separate_depth = separate_depth[~np.isnan(separate_depth)]
                separate_depth = separate_depth[separate_depth > 0]
                separate_depth = list(np.sort(separate_depth))
            else:
                separate_depth = None
            frames = regenerate_frames(
                frame_times=imaging_df.imaging_harptime,
                trials_df=trials_df,
                vs_df=vs_df,
                param_logger=param_log,
                time_column="HarpTime",
                resolution=resolution,
                sphere_size=sphere_size,
                azimuth_limits=(-120, 120),
                elevation_limits=(-40, 40),
                output_datatype="int16",
                output=None,
                separate_depths=separate_depth,
                verbose=verbose,
                # flip_x=True,
            )
            if i == 0:
                frames_all = frames
                imaging_df_all = imaging_df
            else:
                frames_all = np.concatenate((frames_all, frames), axis=0)
                imaging_df_all = pd.concat(
                    [imaging_df_all, imaging_df], ignore_index=True
                )
        else:
            frames_all = None
            if i == 0:
                imaging_df_all = imaging_df
            else:
                imaging_df_all = pd.concat(
                    [imaging_df_all, imaging_df], ignore_index=True
                )
    print(f"Finished concatenating regenerated frames and imaging_df")

    return frames_all, imaging_df_all


def get_relevant_recordings(
    recording_name, flexilims_session, harp_is_in_recording, use_onix
):
    """Get the recording, harp recording and onix recording for a given recording name.

    Args:
        recording_name (str): name of the recording.
        flexilims_session (flexilims_session): flexilims session.
        harp_is_in_recording (bool): if True, harp is in the same recording as the
            imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for synchronisation. Defaults to
            False.

    Returns:
        (recording, harp_recording, onix_rec): tuple of recording, harp recording and
            onix recording.
    """
    recording = flz.get_entity(
        datatype="recording",
        name=recording_name,
        flexilims_session=flexilims_session,
    )

    if harp_is_in_recording:
        harp_recording = recording
    else:
        harp_recording = flz.get_children(
            parent_id=recording.origin_id,
            children_datatype="recording",
            flexilims_session=flexilims_session,
            filter=dict(protocol="harpdata"),
        )
        assert (
            len(harp_recording) == 1
        ), f"{len(harp_recording)} harp recording(s) found for {recording_name}"
        harp_recording = harp_recording.iloc[0]

    if use_onix:
        onix_rec = flz.get_children(
            parent_id=recording.origin_id,
            children_datatype="recording",
            flexilims_session=flexilims_session,
            filter=dict(protocol="onix"),
        )
        assert (
            len(onix_rec) == 1
        ), f"{len(onix_rec)} onix recording(s) found for {recording_name}"
        onix_rec = onix_rec.iloc[0]
    else:
        onix_rec = None

    return recording, harp_recording, onix_rec


def _process_single_recording_for_session(
    recording_name,
    flexilims_session,
    harp_is_in_recording,
    use_onix,
    photodiode_protocol,
    sync_kwargs,
    protocol_base,
    conflicts,
    recording_type,
    filter_datasets,
    exclude_datasets,
    return_volumes,
    ephys_kwargs,
    verbose=True,
    add_spikes=False,
):
    """
    Processes a single recording to generate vs_df, imaging_df, trials_df,
    param_log, and the recording object.
    """
    if verbose:
        print(f"Processing recording: {recording_name}")
    load_onix = False if recording_type == "two_photon" else True

    recording, harp_recording, onix_rec = get_relevant_recordings(
        recording_name, flexilims_session, harp_is_in_recording, load_onix
    )

    vs_df = synchronisation.generate_vs_df(
        recording=recording,
        photodiode_protocol=photodiode_protocol,
        flexilims_session=flexilims_session,
        harp_recording=harp_recording,
        onix_recording=onix_rec if use_onix else None,
        conflicts=conflicts,
        sync_kwargs=sync_kwargs,
        protocol_base=protocol_base,
    )

    unit_ids = None
    if recording_type == "two_photon":
        imaging_df = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=recording,
            flexilims_session=flexilims_session,
            filter_datasets=filter_datasets,
            exclude_datasets=exclude_datasets,
            return_volumes=return_volumes,
            add_spikes=add_spikes,
        )
    else:  # ephys
        imaging_df, unit_ids = synchronisation.generate_spike_rate_df(
            vs_df=vs_df,
            onix_recording=onix_rec,  # Assumes onix_rec is loaded if ephys
            harp_recording=harp_recording,
            flexilims_session=flexilims_session,
            filter_datasets=filter_datasets,
            exclude_datasets=exclude_datasets,
            **(ephys_kwargs if ephys_kwargs else {}),
        )

    imaging_df = format_imaging_df(recording=recording, imaging_df=imaging_df)

    is_multidepth_protocol = "multidepth" in recording.protocol

    param_log = get_param_log(
        flexilims_session,
        vis_stim_recording=recording,
        harp_recording=harp_recording,
        multidepth=is_multidepth_protocol,
    )

    trials_df = generate_trials_df(
        recording=recording,
        imaging_df=imaging_df,
        is_multidepth=is_multidepth_protocol,
        param_log=param_log,
        add_spikes=add_spikes,
    )
    trials_df["recording"] = recording.name

    # Add param log lines to reload only sphere on screen
    trials_df = search_param_log_trials(
        harp_recording=harp_recording,
        trials_df=trials_df,
        flexilims_session=flexilims_session,
        vis_stim_recording=recording,
        is_multidepth=is_multidepth_protocol,
    )

    return vs_df, imaging_df, trials_df, param_log, recording, unit_ids


@slurm_it(
    conda_env="v1_depth_map",
    slurm_options={
        "mem": "64G",
        "time": "48:00:00",
        "partition": "ncpu",
    },
    print_job_id=True,
)
def simulate_and_fit_session(
    session_name,
    decay_tau=0.8,
    rise_tau=0.15,
    make_circular=True,
    protocol_base="SpheresPermTubeReward",
    filter_datasets=None,
    flexilims_session=None,
    project=None,
    kernel_normalization="max",
):
    """Run a full continuous simulation and fit for an entire spheres session.

    Loads ground-truth 2D Gaussian fit parameters from a pre-existing neurons_df,
    simulates continuous calcium responses across every recording in the session,
    slices the simulated trace into trials, fits 2D Gaussians to the simulated
    responses, and saves the results to a parquet file next to the neurons_df.

    Args:
        session_name (str): Session string (format: {Mouse}_{Session}).
        decay_tau (float, optional): Exponential decay time constant (in seconds) used
            for the calcium simulation. Defaults to 0.8.
        rise_tau (float, optional): Exponential rise time constant (in seconds) used
            for the calcium simulation. Defaults to 0.15.
        make_circular (bool, optional): If True, make the Gaussian circular by setting
            the major axis to the minor axis length. Defaults to True.
        protocol_base (str, optional): Base string used to filter recordings.
            Defaults to "SpheresPermTubeReward".
        filter_datasets (dict, optional): Key/value pairs used to filter the suite2p
            dataset (e.g. ``{'anatomical_only': 3}``). Defaults to None.
        flexilims_session (flexilims.session, optional): Flexilims session object.
            Required if *project* is None. Defaults to None.
        project (str, optional): Project name. Required if *flexilims_session* is None.
            Defaults to None.
        kernel_normalization (str, optional): "max" to normalize the calcium kernel's
            peak to 1, or "area" to normalize its sum (unit gain) to 1. Defaults to
            "max".

    Returns:
        pd.DataFrame: A dataframe containing:
            - ``roi``: ROI index.
            - ``popt_groundtruth``: Ground-truth circular Gaussian parameters.
            - ``popt_simulated``: Parameters recovered by fitting the simulated data.
            - ``rsq_simulated``: R-squared of the recovered fit.
            - ``fake_dff``: Concatenated simulated ΔF/F trace, one array per ROI.

        The dataframe is also saved as a parquet file next to the neurons_df dataset.
    """
    # Print parameters to have them in slurm logs
    print(f"Session: {session_name}")
    print(f"Decay tau: {decay_tau}")
    print(f"Rise tau: {rise_tau}")
    print(f"Make circular: {make_circular}")
    print(f"Filter datasets: {filter_datasets}")
    print(f"Project: {project}")
    print(f"Kernel normalization: {kernel_normalization}")
    print("\n")
    from cottage_analysis.analysis.spheres.simulation import simulate_calcium_responses

    if flexilims_session is None:
        assert project is not None, "Must provide either flexilims_session or project"
        flexilims_session = flz.get_flexilims_session(project_id=project)

    neurons_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        allow_multiple=False,
    )
    if neurons_ds is None:
        raise ValueError(f"Neurons dataset not found for session {session_name}")

    neurons_df = pd.read_pickle(neurons_ds.path_full)
    popt_list = [
        None if (isinstance(popt, float) or np.isnan(popt).any()) else popt.copy()
        for popt in neurons_df.rsof_popt_closedloop_g2d.values
    ]
    is_circ = "_circular" if make_circular else "_elliptical"
    target = neurons_ds.path_full.with_name(
        f"simulated_responses_fit_spheres_{decay_tau}_{rise_tau}{is_circ}.parquet"
    )

    # --- 1. Loop over recordings, simulate continuously, collect trials_df ---
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    if exp_session is None:
        raise IOError(f"No session called {session_name} found in flexilims.")
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value="two_photon",
        flexilims_session=flexilims_session,
    )
    recordings = recordings[recordings.name.str.contains(protocol_base)]
    # Exclude Playback and multidepth recordings (closed-loop spheres only)
    recordings = recordings[~recordings.name.str.contains("Playback")]
    recordings = recordings[~recordings.name.str.contains("multidepth")]
    if "exclude_reason" in recordings.columns:
        recordings = recordings[recordings["exclude_reason"].isna()]

    trials_df_all = None
    for i, recording_name in enumerate(recordings.name):
        print(f"Processing recording {i + 1}/{len(recordings)}")

        (
            vs_df,
            imaging_df,
            trials_df,
            _param_log,
            _recording,
            _unit_ids,
        ) = _process_single_recording_for_session(
            recording_name=recording_name,
            flexilims_session=flexilims_session,
            harp_is_in_recording=True,
            use_onix=False,
            photodiode_protocol=5,
            sync_kwargs=None,
            protocol_base=protocol_base,
            conflicts="skip",
            recording_type="two_photon",
            filter_datasets=filter_datasets,
            exclude_datasets=None,
            return_volumes=True,
            ephys_kwargs=None,
            verbose=True,
        )

        # Simulate continuous calcium trace over the entire recording
        frame_rate = 1 / np.nanmedian(imaging_df.imaging_harptime.diff())
        fake_dff_continuous = simulate_calcium_responses(
            imaging_df=imaging_df,
            popt_list=popt_list,
            tau_decay=decay_tau,
            tau_rise=rise_tau,
            frame_rate=frame_rate,
            make_circular=make_circular,
            kernel_normalization=kernel_normalization,
        )

        # Slice the continuous simulated trace into individual trials
        fake_dff_crop = []
        for _tid, trial_crop in trials_df.iterrows():
            start_offset = int(trial_crop.imaging_stim_start)
            end_offset = int(trial_crop.imaging_stim_stop) + 1
            fake_dff_crop.append(fake_dff_continuous[start_offset:end_offset, :])
        trials_df["fake_dff_stim"] = fake_dff_crop

        if trials_df_all is None:
            trials_df_all = trials_df
        else:
            trials_df_all = pd.concat([trials_df_all, trials_df], ignore_index=True)

    print("Finished simulating all recordings")

    # --- 2. Fit 2D Gaussians to simulated responses ---
    mock_trials_df = trials_df_all.copy()
    mock_trials_df["dff_stim"] = mock_trials_df["fake_dff_stim"]

    sim_neurons_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=mock_trials_df,
        model="gaussian_2d",
        trial_sfx="",
        rs_thr=0.01,
        niter=5,
        min_sigma=0.25,
        k_folds=1,
    )

    # --- 3. Construct output dataframe ---
    results = pd.DataFrame(
        {
            "roi": np.arange(len(popt_list)),
            "popt_groundtruth": popt_list,
            "popt_simulated": sim_neurons_df["rsof_popt_closedloop_g2d"].values,
            "rsq_simulated": sim_neurons_df["rsof_rsq_closedloop_g2d"].values,
        }
    )

    # Concatenate simulated trace across all trials and store one array per ROI
    fake_dff_all_trials = np.concatenate(trials_df_all["fake_dff_stim"].values, axis=0)
    fake_dff_per_roi = [
        fake_dff_all_trials[:, i] for i in range(fake_dff_all_trials.shape[1])
    ]
    results["fake_dff"] = fake_dff_per_roi

    results.to_parquet(target, index=False)
    return results
