from functools import partial
import flexiznam as flz
import numpy as np
import pandas as pd

from cottage_analysis.analysis.spheres import multidepth
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis.spheres.stimulus_reconstruction import regenerate_frames
from cottage_analysis.utilities.misc import get_str_or_recording
from cottage_analysis.io_module.visstim import get_param_log

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


def find_stim_time(imaging_df, is_multidepth=False, param_log=None):
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
        harpstim_time, param_log_index = multidepth.find_trial_times(param_log)
        onset_index = imaging_df.stimulus_harptime.values.searchsorted(harpstim_time[0])
        offset_index = imaging_df.stimulus_harptime.values.searchsorted(
            harpstim_time[1]
        )
        for on, off in zip(onset_index, offset_index):
            imaging_df.loc[on:off, "stim"] = 1
    return imaging_df


def generate_trials_df(
    recording, imaging_df, is_multidepth=False, param_log=None, acceleration_time=0.5
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
    imaging_df = find_stim_time(imaging_df, is_multidepth, param_log)
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
    optional_columns = ["expected_optic_flow", "MotorSps"]
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
        harp_is_in_recording (bool): if True, harp is in the same recording as the imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for synchronisation. Defaults to False.

    Returns:
        (recording, harp_recording, onix_rec): tuple of recording, harp recording and onix recording.
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
