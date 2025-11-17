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
            conflicts=conflicts,
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
        imaging_df = process_imaging_df(imaging_df, trial_duration=2)

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


def process_imaging_df(imaging_df, trial_duration=2):
    """Process the imaging dataframe to add treadmill information.

    This will take the last `trial_duration` second of each motor step

    The following columns are added:
        - MotorSpeed: Speed of the motor in cm/s.
        - is_trial_end: True if the frame is the end of a trial.
        - is_trial_start: True if the frame is the start of a trial.
        - is_stim: True if the frame is part of a trial.
        - trial_index: Index of the trial.
        - optic_flow: Optic flow in deg/s.

    Args:
        imaging_df (pd.DataFrame): Imaging dataframe.
        trial_duration (float, optional): Duration of a trial in seconds. Defaults to 2.

    Returns:
        pd.DataFrame: Imaging dataframe with treadmill information.
    """

    assert "MotorSps" in imaging_df.columns, "Imaging df must contain MotorSps"

    imaging_df["MotorSpeed"] = np.round(sps2speed(imaging_df.MotorSps))
    # Find trials, defined as last 2 second of motor running
    trial_ends = (imaging_df.MotorSps > 0).astype(int).diff() == -1
    shifted = trial_ends.shift(-1)
    imaging_df["is_trial_end"] = shifted.values.astype(bool)
    trial_starts = (
        imaging_df.loc[imaging_df["is_trial_end"], "imaging_harptime"] - trial_duration
    )
    imaging_df["is_trial_start"] = False
    trial_start_index = imaging_df.imaging_harptime.searchsorted(trial_starts)
    imaging_df.loc[trial_start_index, "is_trial_start"] = True

    starts = imaging_df.query("is_trial_start")
    ends = imaging_df.query("is_trial_end")

    imaging_df["is_stim"] = False
    imaging_df["trial_index"] = -1

    for itrial, (start, end) in enumerate(zip(starts.index, ends.index)):
        imaging_df.loc[start:end, "is_stim"] = True
        imaging_df.loc[start:end, "trial_index"] = itrial

    # Calculate the optic flow
    actual_of = np.rad2deg(imaging_df.MotorSpeed / (imaging_df.depth.values * 100))
    # To get the expected_of we round in the log space

    warnings.filterwarnings("ignore")
    imaging_df["expected_optic_flow"] = 2 ** (np.round(np.log2(actual_of)))
    warnings.filterwarnings("default")

    return imaging_df
