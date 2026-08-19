"""Analysis of treadmill data

Analysis function for the spheres + motor protocols

This is a variant of the spheres protocol and depends a lot on the functions there.
"""

import warnings
import numpy as np
import pandas as pd
import flexiznam as flz
from znamutils import slurm_it

from . import spheres
from ..preprocessing import synchronisation
from .fit_gaussian_blob import fit_rs_of_tuning

STEPS_PER_REV = 200
MICROSTEPPING = 1 / 4
WHEEL_RADIUS = 10.5
CIRCUMFERENCE = 2 * np.pi * WHEEL_RADIUS
# Small error in the motor speed
ACTUAL_MOTOR_SPEED = {64: 61, 32: 61.0 / 2, 16: 61.0 / 4, 8: 61.0 / 8, 4: 61.0 / 16}

# --- Plateau-onset detector (see v1_depth_map/revisions/motor_ramps.ipynb, section D) ---
# Validated against an independent trapezoid fit and a 1 kHz reference across
# 4 sessions / 791 trials / 5 speeds (3.8-61 cm/s).
_PLATEAU_CAP_REL = 0.10
_PLATEAU_CAP_ABS = 1.5  # cm/s
_PLATEAU_SHIFT_STEP = 0.01  # s
_PLATEAU_MIN_PLATEAU_S = 1.0  # s
_PLATEAU_PRE_S = 0.5  # s of pre-command context included in the fit window


def ramp_template(t, shift, plateau_speed, acceleration):
    """Ideal trapezoidal ramp: flat zero, then a linear rise, then a flat plateau.

    Args:
        t (np.ndarray): Time in seconds, relative to the motor command onset
        shift (float): Onset of the ramp, in seconds.
        plateau_speed (float): Commanded speed this ramp rises to, in cm/s.
        acceleration (float): Rise rate of the ramp, in cm/s^2.

    Returns:
        np.ndarray: The template speed at each `t`, in cm/s: 0 before `shift`,
            `acceleration * (t - shift)` while rising, clipped at `plateau_speed`
            once the plateau is reached.
    """
    return np.clip(acceleration * (t - shift), 0.0, plateau_speed)


def _match_ramp_template(
    t,
    rs,
    plateau_speed,
    duration,
    acceleration,
    cap_rel=_PLATEAU_CAP_REL,
    cap_abs=_PLATEAU_CAP_ABS,
    shift_step=_PLATEAU_SHIFT_STEP,
    min_plateau=_PLATEAU_MIN_PLATEAU_S,
    pre=_PLATEAU_PRE_S,
):
    """Find the best-fitting onset of a trapezoidal ramp in one trial's RS trace.

    Slides `ramp_template` over the trial at candidate onsets `shift >= 0` and
    scores each with the mean per-sample absolute deviation, capped at
    `max(cap_rel * plateau_speed, cap_abs)` before averaging so that a blocked-wheel
    episode cannot dominate the fit. Returns the shift with the lowest cost.

    Args:
        t (np.ndarray): Time in seconds, relative to the motor command onset.
        rs (np.ndarray): Running speed in cm/s, same length/alignment as `t`.
        plateau_speed (float): Commanded plateau speed for this trial, in cm/s.
        duration (float): Physical motor-on duration for this trial, in seconds.
        acceleration (float): Assumed physical belt acceleration in cm/s^2 (in
            `process_imaging_df` this is `1 / acceleration_time`).
        cap_rel (float, optional): Relative component of the per-sample cost cap,
            as a fraction of `plateau_speed`. Defaults to `_PLATEAU_CAP_REL`.
        cap_abs (float, optional): Absolute floor of the per-sample cost cap, in
            cm/s -- matters at low speeds where `cap_rel * plateau_speed` would be
            smaller than the plateau's own measurement noise. Defaults to
            `_PLATEAU_CAP_ABS`.
        shift_step (float, optional): Grid resolution, in seconds, for candidate
            onset shifts. Defaults to `_PLATEAU_SHIFT_STEP`.
        min_plateau (float, optional): Seconds of plateau that must remain within
            `duration` after a candidate onset for that shift to be tried.
            Defaults to `_PLATEAU_MIN_PLATEAU_S`.
        pre (float, optional): Seconds of data before t=0 included in the fit
            window. Defaults to `_PLATEAU_PRE_S`.

    Returns:
        tuple[float, float, float]: `(t_plateau, shift, cost)` where `t_plateau`
            is the time (same origin as `t`) at which the ramp reaches
            `plateau_speed` (`shift + plateau_speed / acceleration`), `shift` is
            the fitted ramp onset, and `cost` is the normalised capped-deviation
            score at that shift, in [0, 1]. Returns `(nan, nan, nan)` if fewer
            than 10 samples fall in `[-pre, duration]`, or if no candidate shift
            leaves room for both the ramp and `min_plateau` within `duration`
            (e.g. a low-speed/short trial).
    """
    t, rs = np.asarray(t), np.asarray(rs)
    sel = (t >= -pre) & (t <= duration)
    t, rs = t[sel], rs[sel]
    if len(t) < 10:
        return np.nan, np.nan, np.nan
    ramp = plateau_speed / acceleration
    max_shift = duration - ramp - min_plateau
    if max_shift <= 0:
        return np.nan, np.nan, np.nan
    cap = max(cap_rel * plateau_speed, cap_abs)
    shifts = np.arange(0.0, max_shift, shift_step)
    template = np.clip(
        acceleration * (t[None, :] - shifts[:, None]), 0.0, plateau_speed
    )
    cost = np.minimum(np.abs(rs[None, :] - template), cap).mean(axis=1) / cap
    best = int(np.argmin(cost))
    return float(shifts[best] + ramp), float(shifts[best]), float(cost[best])


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
    method="plateau",
    margin=0.5,
    sim_popt_list=None,
    sim_tau_decay=0.8,
    sim_tau_rise=0.15,
    sim_make_circular=True,
    sim_kernel_normalization="max",
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
        method (str, optional): Onset-detection method passed to `process_imaging_df`;
            "plateau" (fits a trapezoidal ramp template to the RS trace, falling back
            to the model estimate per-trial if the fit is not possible) or "model"
            (the legacy fixed-formula estimate). Defaults to "plateau".
        margin (float, optional): Extra seconds added on top of the onset estimate
            before converting to a frame offset (formerly a hardcoded 0.5s). Applied
            identically for either method. Defaults to 0.5.
        sim_popt_list (list or None): List of 2D Gaussian fit parameters (arrays), one
            per ROI. If provided, will simulate calcium responses for the entire
            recording and add them to the trials_df as fake_dff_stim.
        sim_tau_decay (float, optional): Exponential decay time constant (in seconds)
            used for the calcium simulation. Defaults to 0.8.
        sim_tau_rise (float, optional): Exponential rise time constant (in seconds) used
            for the calcium simulation. Defaults to 0.15.
        sim_make_circular (bool, optional): If True, make the Gaussian circular by setting
            the major axis to the minor axis length. Defaults to True.
        sim_kernel_normalization (str, optional): "max" to normalize the calcium
            kernel's peak to 1, or "area" to normalize its sum (unit gain) to 1.
            Defaults to "max".

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

        # If simulating, generate continuous calcium trace over the entire recording
        if sim_popt_list is not None:
            from cottage_analysis.analysis.spheres.simulation import (
                simulate_calcium_responses,
            )

            frame_rate = 1 / np.nanmedian(imaging_df.imaging_harptime.diff())
            # Generate the continuous fake dff array (shape: n_frames x n_rois)
            fake_dff_continuous = simulate_calcium_responses(
                imaging_df=imaging_df,
                popt_list=sim_popt_list,
                tau_decay=sim_tau_decay,
                tau_rise=sim_tau_rise,
                frame_rate=frame_rate,
                make_circular=sim_make_circular,
                kernel_normalization=sim_kernel_normalization,
            )

        # Add the treadmill specific part
        imaging_df = process_imaging_df(
            imaging_df,
            trial_duration=trial_duration,
            cut_trial_end=cut_trial_end,
            acceleration_time=acceleration_time,
            method=method,
            margin=margin,
        )

        trials_df = spheres.generate_trials_df(
            recording=recording, imaging_df=imaging_df
        )

        # Slice the continuous simulated trace correctly into the cropped trials
        if sim_popt_list is not None:
            fake_dff_crop = []
            for tid, trial_crop in trials_df.iterrows():
                # Use absolute dataframe frame indices to find the exact offset
                # within the fully simulated continuous array
                # The continuous array aligns perfectly with imaging_df indices
                start_offset = int(trial_crop.imaging_stim_start)
                end_offset = int(trial_crop.imaging_stim_stop) + 1

                # Slicing from continuous array yields (n_trial_frames, n_rois)
                fake_dff_crop.append(fake_dff_continuous[start_offset:end_offset, :])
            trials_df["fake_dff_stim"] = fake_dff_crop

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
    method="plateau",
    margin=0.5,
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
            None, overrides trial_duration. Used both to compute the model-based onset
            estimate (`method="model"`) and, when `method="plateau"`, as the assumed
            physical ramp acceleration (`1 / acceleration_time`, cm/s^2) for the
            plateau-fit detector. Defaults to 0.13.
        method (str, optional): How to estimate, per trial, when the analysis window
            should open relative to the motor turning on. Only used when
            `acceleration_time` is not None. One of:
                - "plateau": onset = the time at which a trapezoidal-ramp template
                  (see `_match_ramp_template`) best matches the trial's actual `RS`
                  trace, using `acceleration_time` to fix the ramp's slope. Falls
                  back to the "model" estimate for any individual trial where the
                  detector cannot return a value (e.g. too few samples, or no room
                  for a valid ramp + plateau given that trial's speed/duration).
                - "model": onset = acceleration_time * motor_speed (the original
                  fixed-formula estimate).
            Defaults to "plateau".
        margin (float, optional): Extra seconds added on top of the onset estimate
            (from either method) before converting to a frame offset, replacing the
            previously hardcoded `+ 0.5`. Applied identically regardless of method.
            Defaults to 0.5.

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
        if method not in ("model", "plateau"):
            raise ValueError(f"Unknown method {method!r}, must be 'model' or 'plateau'")
        # Find the motor speed for each trial
        # We need to find the trial index for each trial start
        # for each start, find the corresponding end
        # the end should be the first end after the start
        trial_end_indices = trial_ends.searchsorted(trial_starts)

        # get the motor speed and physical end for each trial
        motor_speeds = []
        trial_end_times = []
        for start, end_idx in zip(trial_starts, trial_end_indices):
            if end_idx >= len(trial_ends):
                end = imaging_df.imaging_harptime.iloc[-1]
            else:
                end = trial_ends[end_idx]
            trial_end_times.append(end)

            # get the motor speed in this interval
            # use searchsorted to find indices in imaging_df
            start_idx = imaging_df.imaging_harptime.searchsorted(start)
            end_idx_frame = imaging_df.imaging_harptime.searchsorted(end)
            motor_speeds.append(
                np.nanmedian(imaging_df.MotorSpeed.iloc[start_idx:end_idx_frame])
            )
        motor_speeds = np.array(motor_speeds)
        trial_end_times = np.array(trial_end_times)

        # Onset offset (seconds after motor-on), before the margin is added
        if method == "model":
            ramp_end = acceleration_time * motor_speeds
        else:  # method == "plateau"
            acceleration_cms2 = 1.0 / acceleration_time
            harptime = imaging_df.imaging_harptime.values
            rs_cms = imaging_df.RS.values * 100.0  # RS is in m/s, detector uses cm/s
            ramp_end = np.empty(len(trial_starts), dtype=float)
            for i, (start, end, v) in enumerate(
                zip(trial_starts, trial_end_times, motor_speeds)
            ):
                duration = end - start
                sl = slice(
                    harptime.searchsorted(start - _PLATEAU_PRE_S),
                    harptime.searchsorted(end),
                )
                t_plateau, _, _ = _match_ramp_template(
                    harptime[sl] - start, rs_cms[sl], v, duration, acceleration_cms2
                )
                ramp_end[i] = (
                    t_plateau if np.isfinite(t_plateau) else acceleration_time * v
                )

        # Calculate acceleration frames
        acc_frames = ((ramp_end + margin) * frame_rate).astype(int)

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
    filter_datasets=None,
    flexilims_session=None,
    project=None,
    kernel_normalization="max",
):
    """Run a full continuous simulation and fit for an entire session.

    Args:
        session_name (str): Session string (format: {Mouse}_{Session})
        decay_tau (float, optional): Exponential decay time constant (in seconds) used
            for the calcium simulation. Defaults to 0.8.
        rise_tau (float, optional): Exponential rise time constant (in seconds) used
            for the calcium simulation. Defaults to 0.15.
        make_circular (bool, optional): If True, make the Gaussian circular by setting
            the major axis to the minor axis length. Defaults to True.
        filter_datasets (dict, optional): Key/value pairs used to filter the suite2p
            dataset (e.g. ``{'anatomical_only': 3}``). Defaults to ``None``.
        flexilims_session (flexilims.session, optional): Flexilims session object.
            Required if project is None. Defaults to None.
        project (str, optional): Project name. Required if flexilims_session is None.
            Defaults to None.
        kernel_normalization (str, optional): "max" to normalize the calcium kernel's
            peak to 1, or "area" to normalize its sum (unit gain) to 1. Defaults to
            "max".

    Returns:
        pd.DataFrame: A dataframe containing the ground-truth
            circular parameters, the actual arrays of simulated data,
            and the parameters recovered by the 2D Gaussian fitting algorithm.
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
        for popt in neurons_df.rsof_popt_closedloop_g2d_treadmill.values
    ]
    is_circ = "_circular" if make_circular else "_elliptical"
    target = neurons_ds.path_full.with_name(
        f"simulated_responses_fit_treadmill_{decay_tau}_{rise_tau}{is_circ}.parquet"
    )
    # 1. Simulate Responses Continously Over The Session
    vs_df_test, trials_df_test = sync_all_recordings(
        session_name=session_name,
        project=project,
        filter_datasets=filter_datasets,
        recording_type="two_photon",
        photodiode_protocol=5,
        sim_popt_list=popt_list,
        sim_tau_decay=decay_tau,
        sim_tau_rise=rise_tau,
        sim_make_circular=make_circular,
        sim_kernel_normalization=kernel_normalization,
    )

    # 3. Fit 2D Gaussians to Simulated Responses
    mock_trials_df = trials_df_test.copy()
    mock_trials_df["dff_stim"] = mock_trials_df["fake_dff_stim"]

    # Run standard pipeline fit procedure
    sim_neurons_df = fit_rs_of_tuning(
        trials_df=mock_trials_df,
        model="gaussian_2d",
        trial_sfx="_treadmill",
        rs_thr=0.01,
        max_acc=5,
        max_rs2motor_diff=0.5,
        niter=5,
        min_sigma=0.25,
        k_folds=1,
    )

    # 3. Construct Outputs
    results = pd.DataFrame(
        {
            "roi": np.arange(len(popt_list)),
            "popt_groundtruth": popt_list,
            "popt_simulated": sim_neurons_df["rsof_popt_closedloop_g2d"].values,
            "rsq_simulated": sim_neurons_df["rsof_rsq_closedloop_g2d"].values,
        }
    )

    # Append the raw simulated matrix (as a list of arrays from the trials)
    # We transpose this logic so that each ROI gets one list containing its whole response array
    fake_dff_all_trials = np.concatenate(trials_df_test["fake_dff_stim"].values, axis=0)
    fake_dff_per_roi = [
        fake_dff_all_trials[:, i] for i in range(fake_dff_all_trials.shape[1])
    ]
    results["fake_dff"] = fake_dff_per_roi

    results.to_parquet(target, index=False)
    return results
