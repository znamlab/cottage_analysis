"""Analysis of treadmill data

Analysis function for the spheres + motor protocols

This is a variant of the spheres protocol and depends a lot on the functions there.
"""

import warnings
import numpy as np
import pandas as pd
import flexiznam as flz

from . import spheres
from ..preprocessing import synchronisation


STEPS_PER_REV = 200
MICROSTEPPING = 1 / 4
WHEEL_RADIUS = 10.5
CIRCUMFERENCE = 2 * np.pi * WHEEL_RADIUS
# Small error in the motor speed
ACTUAL_MOTOR_SPEED = {64: 61, 32: 61.0 / 2, 16: 61.0 / 4, 8: 61.0 / 8, 4: 61.0 / 16}


def compute_response_matrix(neurons_df, trials_df_tread):
    motor_speeds = 2 ** np.arange(2, 7)
    optic_flows = 4 ** np.arange(6)

    mot_values = np.zeros((len(motor_speeds), len(optic_flows)))
    of_values = np.zeros((len(motor_speeds), len(optic_flows)))
    frame2extract = 152
    tread_responses = np.zeros(
        (len(neurons_df), len(motor_speeds), len(optic_flows), frame2extract)
    )
    for (motor, optic_flow), df in trials_df_tread.groupby(
        ["MotorSpeed", "expected_optic_flow"]
    ):
        dffs = df.dff_stim.values
        shapes = np.vstack([dff.shape for dff in dffs])
        dffs_list = []
        for idff, dff in enumerate(dffs):
            f = np.zeros(frame2extract)
            if len(dff) < frame2extract:
                f[-len(dff) :] = dff
            else:
                f = dff[len(dff) - frame2extract :]
            dffs_list.append(f)
        dffs = np.stack(dffs_list)
        avg_motor = np.nanmean(dffs, axis=0)
        std_motor = np.nanstd(dffs, axis=0)
        m_index = list(motor_speeds).index(motor)
        of_index = list(optic_flows).index(optic_flow)
        tread_responses[:, m_index, of_index, :] = avg_motor.T
        mot_values[m_index, of_index] = motor
        of_values[m_index, of_index] = optic_flow
    return motor_speeds, optic_flows, tread_responses


def sync_treadmill_sess(session_name, project, flexilims_session, filter_datasets=None):
    if filter_datasets is None:
        filter_datasets = {"anatomical_only": 3}
    if project is None:
        project = flexilims_session.project
    vs_df_tread, trials_df_tread = sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets=filter_datasets,
        recording_type="two_photon",
        photodiode_protocol=5,
        return_volumes=True,
    )
    trials_df_tread["MotorSpeed"] = np.round(
        sps2speed(trials_df_tread["MotorSps_stim"].apply(np.nanmedian).values)
    )
    trials_df_tread["expected_optic_flow"] = np.round(
        trials_df_tread["expected_optic_flow_stim"].apply(np.nanmedian).values
    )
    return vs_df_tread, trials_df_tread


def sync_all_recordings(
    session_name,
    flexilims_session=None,
    project=None,
    filter_datasets=None,
    exclude_datasets=None,
    recording_type="two_photon",
    photodiode_protocol=5,
    return_volumes=True,
    harp_is_in_recording=True,
    use_onix=False,
    conflicts="skip",
    sync_kwargs=None,
    ephys_kwargs=None,
    cut_trial_end=None,
    trial_duration=None,
    acceleration_time=0.13,
):
    """Concatenate synchronisation results for all recordings in a session.

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
        photodiode_protocol (int): number of photodiode quad colors used for monitoring
            frame refresh. Either 2 or 5 for now. Defaults to 5.
        return_volumes (bool): if True, return only the first frame of each imaging
            volume. Defaults to True.
        harp_is_in_recording (bool): if True, harp is in the same recording as the
            imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for synchronisation. Defaults to
            False.
        conflicts (str): how to handle conflicts. Defaults to "skip".
        sync_kwargs (dict): kwargs for synchronisation.generate_vs_df. Defaults to None.
        return_multiunit (bool): if True, process multiunit activity. Defaults to False.
        ephys_kwargs (dict): Keyword arguments for generate_spike_rate_df.
            `return_multiunit` or `exp_sd` for instance. Defaults to None.
        cut_trial_end (float or None): Seconds to remove at the end of each trial
        trial_duration (float): Seconds to the end of the trial to keep (new "start"
            of trial)
        acceleration_time (float or None): Acceleration time in s per cm/s. If not None,
            overrides trial_duration. Default to 0.13 (aka 61cm/s reached in 8s)

    Returns:
        (pd.DataFrame, pd.DataFrame): tuple of two dataframes, one concatenated vs_df
            for all recordings, one concatenated trials_df for all recordings.
    """
    protocol_base = "SpheresTubeMotor"
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
    if "exclude_reason" in recordings.columns:
        recordings = recordings[recordings["exclude_reason"].isna()]

    load_onix = False if recording_type == "two_photon" else True
    for i, recording_name in enumerate(recordings.name):
        print(f"Processing recording {i+1}/{len(recordings)}")
        recording, harp_recording, onix_rec = spheres.get_relevant_recordings(
            recording_name, flexilims_session, harp_is_in_recording, load_onix
        )
        vs_df = synchronisation.generate_vs_df(
            recording=recording,
            photodiode_protocol=photodiode_protocol,
            flexilims_session=flexilims_session,
            harp_recording=harp_recording,
            onix_recording=onix_rec if use_onix else None,
            project=project,
            conflicts="skip",
            sync_kwargs=sync_kwargs,
            protocol_base=protocol_base,
        )

        if recording_type == "two_photon":
            imaging_df = synchronisation.generate_imaging_df(
                vs_df=vs_df,
                recording=recording,
                flexilims_session=flexilims_session,
                filter_datasets=filter_datasets,
                exclude_datasets=exclude_datasets,
                return_volumes=return_volumes,
            )
        else:
            imaging_df, unit_ids = synchronisation.generate_spike_rate_df(
                vs_df=vs_df,
                onix_recording=onix_rec,
                harp_recording=harp_recording,
                flexilims_session=flexilims_session,
                filter_datasets=filter_datasets,
                exclude_datasets=exclude_datasets,
                **ephys_kwargs,
            )

        imaging_df = spheres.format_imaging_df(
            imaging_df=imaging_df, recording=recording
        )
        # Add the treadmill specific part
        imaging_df = process_imaging_df(
            imaging_df,
            trial_duration=trial_duration,
            cut_trial_end=cut_trial_end,
            acceleration_time=acceleration_time,
        )

        trials_df = spheres.generate_trials_df(
            recording=recording, imaging_df=imaging_df
        )

        trials_df = spheres.search_param_log_trials(
            harp_recording=harp_recording,
            trials_df=trials_df,
            flexilims_session=flexilims_session,
            vis_stim_recording=recording,
            is_multidepth="multidepth" in recording.protocol,
        )
        trials_df["recording"] = recording_name

        if i == 0:
            vs_df_all = vs_df
            trials_df_all = trials_df
        else:
            vs_df_all = pd.concat([vs_df_all, vs_df], ignore_index=True)
            trials_df_all = pd.concat([trials_df_all, trials_df], ignore_index=True)
    print(f"Finished concatenating vs_df and trials_df")

    return vs_df_all, trials_df_all


def sps2speed(
    sps,
    circumference=CIRCUMFERENCE,
    steps_per_rev=STEPS_PER_REV,
    microstepping=MICROSTEPPING,
):
    """Convert steps per second to speed in cm/s.

    Args:
        sps (float or list or np.ndarray): Steps per second.
        circumference (float, optional): Circumference of the wheel in cm. Defaults to
            local constant.
        steps_per_rev (int, optional): Number of steps per revolution. Defaults to
            local constant.
        microstepping (int, optional): Microstepping factor. Defaults to local
            constant.


    Returns:
        float or np.ndarray or None: Speed in cm/s. Returns None if input is None.
    """
    if sps is None:
        return None
    if isinstance(sps, list):
        sps = np.array(sps)

    return sps / steps_per_rev * microstepping * circumference


def process_imaging_df(
    imaging_df,
    trial_duration=None,
    cut_trial_end=None,
    motor_stability_window=0.5,
    acceleration_time=0.13,
):
    """Process the imaging dataframe to add treadmill information.

    This will take the last `trial_duration` second of each motor step

    The following columns are added:
        - MotorSpeed: Speed of the motor in cm/s.
        - is_trial_end: True if te frame is the end of a trial.
        - is_trial_start: True if the frame is the start of a trial.
        - is_stim: True if the frame is part of a trial.
        - trial_index: Index of the trial.
        - optic_flow: Optic flow in deg/s.

    Args:
        imaging_df (pd.DataFrame): Imaging dataframe.
        trial_duration (float, optional): Duration of a trial in seconds. Defaults to 2.
        cut_trial_end (float, optional): Duration to cut at the end of the trial (will
            shorten trial_duration)
        motor_stability_window (float, optional): Duration of the motor stability window
            in seconds. Defaults to 0.5.
        acceleration_time (float, optional): Acceleration time in s per cm/s. If not
            None, overrides trial_duration. Defaults to 0.13.

    Returns:
        pd.DataFrame: Imaging dataframe with treadmill information.
    """
    frame_rate = 1 / np.nanmedian(imaging_df.imaging_harptime.diff())
    assert "MotorSps" in imaging_df.columns, "Imaging df must contain MotorSps"

    imaging_df["MotorSpeed"] = np.round(sps2speed(imaging_df.MotorSps))
    imaging_df["MotorSpeed"] = imaging_df.MotorSpeed.map(ACTUAL_MOTOR_SPEED)

    # 1. Find physical trial starts and ends
    # Find trial starts, defined as first frame of motor running
    trial_starts_bool = (imaging_df.MotorSps > 0).astype(int).diff() == 1
    trial_starts = imaging_df.loc[trial_starts_bool, "imaging_harptime"].values

    # Find trial ends, defined as last frame of motor running
    trial_ends_bool = (imaging_df.MotorSps > 0).astype(int).diff() == -1
    # Shifting adds a NaN and astype(int) would make it True, fill it with False
    shifted = trial_ends_bool.shift(-1, fill_value=False)
    trial_ends = imaging_df.loc[shifted, "imaging_harptime"].values

    # 2. Apply acceleration_time (modifies start)
    if acceleration_time is not None:
        if trial_duration is not None:
            raise ValueError("Cannot provide both trial_duration and acceleration_time")
        # Find the motor speed for each trial
        # We need to find the trial index for each trial start
        # for each start, find the corresponding end
        # the end should be the first end after the start
        trial_end_indices = trial_ends.searchsorted(trial_starts)

        # get the motor speed for each trial
        motor_speeds = []
        for start, end_idx in zip(trial_starts, trial_end_indices):
            if end_idx >= len(trial_ends):
                end = imaging_df.imaging_harptime.iloc[-1]
            else:
                end = trial_ends[end_idx]

            # get the motor speed in this interval
            # use searchsorted to find indices in imaging_df
            start_idx = imaging_df.imaging_harptime.searchsorted(start)
            end_idx = imaging_df.imaging_harptime.searchsorted(end)
            motor_speeds.append(
                np.nanmedian(imaging_df.MotorSpeed.iloc[start_idx:end_idx])
            )
        motor_speeds = np.array(motor_speeds)

        # Calculate acceleration frames
        acc_frames = ((acceleration_time * motor_speeds + 0.5) * frame_rate).astype(int)

        # Shift trial starts
        # We need to add the time corresponding to acc_frames to trial_starts
        # Shift by index
        trial_start_indices = imaging_df.imaging_harptime.searchsorted(trial_starts)
        new_start_indices = trial_start_indices + acc_frames
        # Clip to be within dataframe
        new_start_indices = np.clip(new_start_indices, 0, len(imaging_df) - 1)
        trial_starts = imaging_df.imaging_harptime.iloc[new_start_indices].values

    # 3. Apply trial_duration (modifies start relative to end)
    elif trial_duration is not None:
        # If trial_duration is set, start is end - duration
        # We need to match starts and ends first to make sure we have pairs
        # But actually, the original logic just took ends and subtracted duration
        # Let's stick to the original logic for trial_duration which was simpler:
        # trial_starts = trial_ends - trial_duration
        # But we need to be careful if we have cut_trial_end as well
        # The original logic applied cut_trial_end AFTER finding starts with trial_duration
        # Wait, original logic:
        # 1. Find ends
        # 2. If trial_duration: starts = ends - duration
        # 3. If cut_trial_end: ends = ends - cut_trial_end
        # So trial_duration defines start relative to PHYSICAL end (before cut)
        trial_starts = trial_ends - trial_duration

    # 4. Apply cut_trial_end (modifies end)
    if cut_trial_end is not None:
        trial_ends = trial_ends - cut_trial_end

    # Update dataframe
    imaging_df["is_trial_start"] = False
    trial_start_index = imaging_df.imaging_harptime.searchsorted(trial_starts)
    imaging_df.loc[trial_start_index, "is_trial_start"] = True

    imaging_df["is_trial_end"] = False
    trial_end_index = imaging_df.imaging_harptime.searchsorted(trial_ends)
    imaging_df.loc[trial_end_index, "is_trial_end"] = True

    starts = imaging_df.query("is_trial_start")
    ends = imaging_df.query("is_trial_end")

    imaging_df["is_stim"] = False
    imaging_df["trial_index"] = -1

    for itrial, (start, end) in enumerate(zip(starts.index, ends.index)):
        imaging_df.loc[start:end, "is_stim"] = True
        imaging_df.loc[start:end, "trial_index"] = itrial

    imaging_df["stim"] = imaging_df["is_stim"].astype(int)

    # Calculate the optic flow
    actual_of = np.rad2deg(imaging_df.MotorSpeed / (imaging_df.depth.values * 100))
    # To get the expected_of we round in the log space

    warnings.filterwarnings("ignore")
    imaging_df["expected_optic_flow"] = 2 ** (np.round(np.log2(actual_of)))
    warnings.filterwarnings("default")

    # Add a column for the max difference between recording running speed vs motor speed
    # in a 0.5s rolling window
    nframe_window = int(motor_stability_window * frame_rate)
    # first get max and min running speed in the window
    max_running_speed = imaging_df.RS.rolling(nframe_window).max()
    min_running_speed = imaging_df.RS.rolling(nframe_window).min()
    mean_running_speed = imaging_df.RS.rolling(nframe_window).mean()
    # MotorSpeed is in cm/s, RS is in m/s
    max_running_speed_diff = (max_running_speed - imaging_df.MotorSpeed / 100).abs()
    min_running_speed_diff = (min_running_speed - imaging_df.MotorSpeed / 100).abs()
    # then get max(abs(max), abs(min))
    imaging_df["max_abs_rs2motor_diff"] = np.maximum(
        max_running_speed_diff, min_running_speed_diff
    )
    imaging_df["max_abs_rs2motor_diff_ratio"] = imaging_df.max_abs_rs2motor_diff / (
        imaging_df.MotorSpeed / 100
    )
    imaging_df["mean_rs2motor_diff"] = mean_running_speed - imaging_df.MotorSpeed / 100
    return imaging_df
