from functools import partial

import flexiznam as flz
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mode, zscore
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from cottage_analysis.analysis.fit_gaussian_blob import (
    Gabor3DRFParams,
    Gaussian3DRFParams,
    gabor_3d_rf,
    gaussian_3d_rf,
)
from functools import partial
print = partial(print, flush=True)
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import mode
from cottage_analysis.preprocessing import synchronisation


def find_valid_frames(frame_times, trials_df, verbose=True):
    """Find frame numbers that are valid (not gray period, or not before or after the imaging frames) and used for regenerating sphere stimuli.

    Args:
        frame_times (np.array): Array of time at which the frame should be regenerated
        trials_df (pd.DataFrame): Dataframe contains information for each trial.
        verbose (bool, optional): Print information. Defaults to True.

    Returns:
        frame_indices (np.array): Array of valid frame indices.
    """
    # for frames before and after the protocol, keep them 0s
    before = frame_times < trials_df.imaging_harptime_stim_start.iloc[0]
    after = frame_times > trials_df.imaging_harptime_stim_stop.iloc[-1]
    if verbose:
        print(
            "Ignoring %d frames before and %d after the stimulus presentation"
            % (np.sum(before), np.sum(after))
        )
    valid_frames = ~before & ~after

    trial_index = (
        trials_df.imaging_harptime_stim_start.searchsorted(frame_times, side="right")
        - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    trial_end = trials_df.loc[trial_index, "imaging_harptime_stim_stop"].values
    grey_time = frame_times - trial_end > 0
    if verbose:
        print(
            "Ignoring %d frames in grey inter-trial intervals"
            % np.sum(grey_time & valid_frames)
        )
    valid_frames = valid_frames & (~grey_time)
    frame_indices = np.where(valid_frames)[0]

    return frame_indices


def regenerate_frames(
    frame_times,
    trials_df,
    vs_df,
    param_logger,
    time_column="HarpTime",
    resolution=1,
    sphere_size=10,
    azimuth_limits=(-120, 120),
    elevation_limits=(-40, 40),
    verbose=True,
    output_datatype="int16",
    output=None,
):
    """Regenerate frames of sphere stimulus

    Args:
        frame_times (np.array): Array of time at which the frame should be regenerated.
        trials_df (pd.DataFrame): Dataframe contains information for each trial.
        vs_df (pd.DataFrame): Dataframe contains information for each monitor frame.
        param_logger (pd.DataFrame): Params saved by Bonsai logger
        time_column (str): Name of the column containing timing information in
                           dataframes (Default: 'HarpTime')
        resolution (float): size of a pixel in degrees
        sphere_size (float): size of a sphere in degrees
        azimuth_limits ([float, float]): Minimum and maximum azimuth of the display
        elevation_limits ([float, float]): Minimum and maximum elevation of the display
        verbose (bool): Print information
        output_datatype (type): datatype of the output. Use bool to have binary
                                sphere/no sphere output. int for seeing sphere overlap.
                                Not used if output is provided
        output (np.array): Array to add output. Will be done inplace

    Returns:
        virtual_screen (np.array): an array of [elevation, azimuth] with spheres added.
    """
    frame_times = np.array(frame_times, ndmin=1)
    mouse_pos_cm = (
        vs_df["eye_z"].values * 100
    )  # (np.array): position of the mouse in cm
    mouse_pos_time = vs_df[
        "monitor_harptime"
    ].values  # (np.array): time of each mouse_pos_cm sample

    out_shape = (
        len(frame_times),
        int((elevation_limits[1] - elevation_limits[0]) / resolution),
        int((azimuth_limits[1] - azimuth_limits[0]) / resolution),
    )
    if output is None:
        output = np.zeros(out_shape, dtype=output_datatype)
    else:
        assert output.shape == out_shape

    # Find frame indices that are not grey and within the imaging time.
    trial_index = (
        trials_df.imaging_harptime_stim_start.searchsorted(frame_times, side="right")
        - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    frame_indices = find_valid_frames(frame_times, trials_df, verbose=verbose)
    frame_times[-1] = frame_times[-2]
    mouse_position = mouse_pos_cm[mouse_pos_time.searchsorted(frame_times)]

    # now process the valid frames
    log_ends = param_logger[time_column].searchsorted(frame_times)
    for frame_index in tqdm(frame_indices):
        corridor = trials_df.loc[int(trial_index[frame_index])]
        logger = param_logger.iloc[
            corridor.param_log_start : np.max(
                [log_ends[frame_index], corridor.param_log_start + 1]
            )
        ]
        sphere_coordinates = np.array(logger[["X", "Y", "Z"]].values, dtype=float)
        sphere_coordinates[:, 2] = (
            sphere_coordinates[:, 2] - mouse_position[frame_index]
        )

        this_frame = draw_spheres(
            sphere_x=sphere_coordinates[:, 0],
            sphere_y=sphere_coordinates[:, 1],
            sphere_z=sphere_coordinates[:, 2],
            depth=float(corridor.depth) * 100,
            resolution=float(resolution),
            sphere_size=float(sphere_size),
            azimuth_limits=np.array(azimuth_limits, dtype=float),
            elevation_limits=np.array(elevation_limits, dtype=float),
        )
        if this_frame is None:
            this_frame = np.zeros((out_shape[1], out_shape[2]))
            print(f"Warning: failed to reconstruct frame {frame_index}")
        output[frame_index] = this_frame

    return output


def draw_spheres(
    sphere_x,
    sphere_y,
    sphere_z,
    depth,
    resolution=0.1,
    sphere_size=10,
    azimuth_limits=(-120, 120),
    elevation_limits=(-40, 40),
):
    """Recreate stimulus for a single frame from corrected sphere position

    Given the positions of the spheres relative to the mouse and the corridor depth,
    recreate a single frame

    Args:
        sphere_x (np.array): X positions for all spheres on the frame
        sphere_y (np.array): Y positions for all spheres on the frame
        sphere_z (np.array): Z positions for all spheres on the frame
        depth (float): Depth for that corridor. Used for size adjustement
        resolution (float): size of a pixel in degrees
        sphere_size (float): size of a sphere in degrees
        azimuth_limits ([float, float]): Minimum and maximum azimuth of the display
        elevation_limits ([float, float]): Minimum and maximum elevation of the display

    Returns:
        virtual_screen (np.array): an array of [elevation, azimuth] with spheres added.

    """

    radius, azimuth, elevation = cartesian_to_spherical(sphere_x, sphere_y, sphere_z)
    # we switch from trigo circle, counterclockwise with 0 on the right to azimuth,
    # clockwise with 0 in front
    az_compas = np.mod(-(azimuth - 90), 360)
    az_compas[az_compas > 180] = az_compas[az_compas > 180] - 360

    # now prepare output
    azi_n = int((azimuth_limits[1] - azimuth_limits[0]) / resolution)
    ele_n = int((elevation_limits[1] - elevation_limits[0]) / resolution)

    # find if the sphere is on the screen, that means in the -120 +120 azimuth range
    in_screen = (az_compas > azimuth_limits[0]) & (az_compas < azimuth_limits[1])
    # and in the -40, 40 elevation range
    in_screen = in_screen & (
        (elevation > elevation_limits[0]) & (elevation < elevation_limits[1])
    )
    if not np.any(in_screen):
        return

    # convert `in_screen` spheres in pixel space
    az_on_screen = (az_compas[in_screen] - azimuth_limits[0]) / resolution
    el_on_screen = (elevation[in_screen] - elevation_limits[0]) / resolution
    size = depth / radius[in_screen] * sphere_size / resolution

    xx, yy = np.meshgrid(np.arange(azi_n), np.arange(ele_n))
    xx = np.outer(xx.reshape(-1), np.ones(len(az_on_screen)))
    yy = np.outer(yy.reshape(-1), np.ones(len(el_on_screen)))
    ok = (xx - az_on_screen) ** 2 + (yy - el_on_screen) ** 2 - size**2
    ok = ok <= 0
    # When plotting output, the origin (for lowest azimuth and elevation) is at lower left
    return np.any(ok, axis=1).reshape((ele_n, azi_n))


def cartesian_to_spherical(x, y, z):
    """Transform cartesian X, Y, Z bonsai coordinate to spherical

    Args:
        x (np.array): x position from bonsai. Positive is to the right of the mouse
        y (np.array): y position from bonsai. Positive is above the mouse
        z (np.array): z position from bonsai. Positive is in front of the mouse

    Returns:
        radius (np.array): radius, same unit as x,y,z
        azimuth (np.array): azimuth angle in trigonometric coordinates (0 is to the
                            right of the mouse, positive is counterclockwise, towards
                            the nose)
        elevation (np.array): elevation angle. 0 is in front of the mouse, positive
                              towards the top.
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(z, x)
    elevation = np.arctan2(y, np.sqrt(x**2 + z**2))

    azimuth = np.degrees(azimuth)
    elevation = np.degrees(elevation)
    return radius, azimuth, elevation


def calculate_optic_flow_angle(r, r_new, distance):
    angle = np.arccos((r**2 + r_new**2 - distance**2) / (2 * r * r_new))
    return angle


def _meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j, k] = k  # change to x[k] if indexing xy
            yy[j, k] = j  # change to y[j] if indexing xy
    return xx, yy


def format_imaging_df(recording, imaging_df):
    """Format sphere params in imaging_df.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.

    Returns:
        DataFrame: contains information for each monitor frame and vis-stim.

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
    imaging_df.RS = imaging_df.mouse_z_harp.diff() / imaging_df.mouse_z_harptime.diff()
    # average RS eye for each imaging volume
    imaging_df.RS_eye = imaging_df.eye_z.diff() / imaging_df.monitor_harptime.diff()
    # depth for each imaging volume
    imaging_df[imaging_df["depth"] == -9999].depth = np.nan
    imaging_df.depth = imaging_df.depth / 100  # convert cm to m
    # OF for each imaging volume
    imaging_df["OF"] = imaging_df.RS_eye / imaging_df.depth
    return imaging_df


def generate_trials_df(recording, imaging_df):
    """Generate a DataFrame that contains information for each trial.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        imaging_df(pd.DataFrame): dataframe that contains info for each imaging volume.

    Returns:
        DataFrame: contains information for each trial.

    """

    trials_df = pd.DataFrame(
        columns=[
            "trial_no",
            "depth",
            "closed_loop",
            "imaging_harptime_stim_start",
            "imaging_harptime_stim_stop",
            "imaging_harptime_blank_start",
            "imaging_harptime_blank_stop",
            "imaging_stim_start",
            "imaging_stim_stop",
            "imaging_blank_start",
            "imaging_blank_stop",
            "RS_stim",  # actual running speed, m/s
            "RS_blank",
            "RS_eye_stim",  # virtual running speed, m/s
            "OF_stim",  # optic flow speed = RS/depth, rad/s
            "dff_stim",
            "dff_blank",
        ]
    )

    # Find the change of depth
    imaging_df["stim"] = np.nan
    imaging_df.loc[imaging_df.depth.notnull(), "stim"] = 1
    imaging_df.loc[imaging_df.depth < 0, "stim"] = 0
    imaging_df_simple = imaging_df[
        (imaging_df["stim"].diff() != 0) & (imaging_df["stim"]).notnull()
    ]
    imaging_df_simple.depth = np.round(imaging_df_simple.depth, 2)

    # Find frame or volume of imaging_df for trial start and stop
    # (depending on whether return_volume=True in generate_imaging_df)
    blank_time = 10
    start_volume_stim = imaging_df_simple[
        (imaging_df_simple["stim"] == 1)
    ].imaging_frame.values
    start_volume_blank = imaging_df_simple[
        (imaging_df_simple["stim"] == 0)
    ].imaging_frame.values
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
            (np.abs(imaging_df.imaging_frame - last_blank_stop_time)).idxmin(),
        )
    stop_volume_stim = start_volume_blank - 1

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

    if np.isnan(
        trials_df.imaging_blank_stop.iloc[-1]
    ):  # If the blank stop of last trial is beyond the number of imaging frames
        trials_df.imaging_blank_stop.iloc[-1] = len(imaging_df) - 1

    mask = trials_df.imaging_stim_start == trials_df.imaging_blank_stop.shift(
        1
    )  # Get rid of the overlap of imaging frame no. between different trials
    trials_df.loc[mask, "imaging_stim_start"] += 1

    # Assign protocol to trials_df
    if "Playback" in recording.name:
        trials_df.closed_loop = 0
    else:
        trials_df.closed_loop = 1

    # Assign RS array from imaging_df back to trials_df
    trials_df.RS_stim = trials_df.apply(
        lambda x: imaging_df.RS.loc[
            int(x.imaging_stim_start) : int(x.imaging_stim_stop)
        ].values,
        axis=1,
    )

    trials_df.RS_blank = trials_df.apply(
        lambda x: imaging_df.RS.loc[
            int(x.imaging_blank_start) : int(x.imaging_blank_stop)
        ].values,
        axis=1,
    )

    trials_df.RS_eye_stim = trials_df.apply(
        lambda x: imaging_df.RS_eye.loc[
            int(x.imaging_stim_start) : int(x.imaging_stim_stop)
        ].values,
        axis=1,
    )

    trials_df.OF_stim = trials_df.apply(
        lambda x: imaging_df.OF.loc[
            int(x.imaging_stim_start) : int(x.imaging_stim_stop)
        ].values,
        axis=1,
    )

    # Assign dffs array to trials_df
    trials_df.dff_stim = trials_df.apply(
        lambda x: np.stack(
            imaging_df.dffs.loc[int(x.imaging_stim_start) : int(x.imaging_stim_stop)]
        ).squeeze(),
        axis=1,
    )
    # nvolumes x ncells

    trials_df.dff_blank = trials_df.apply(
        lambda x: np.stack(
            imaging_df.dffs.loc[int(x.imaging_blank_start) : int(x.imaging_blank_stop)]
        ).squeeze(),
        axis=1,
    )
    # nvolumes x ncells

    # Rename
    trials_df = trials_df.drop(columns=["imaging_blank_start"])

    return trials_df


def search_param_log_trials(recording, trials_df, flexilims_session):
    """Add the start param logger row and stop param logger row to each trial. This is required for regenerate_spheres.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        trials_df (pd.DataFrane): Dataframe that contails information for each trial.
        flexilims_session (flexilims_session): flexilims session.

    Returns:
        Dataframe: Dataframe that contails information for each trial.
    """
    harp_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp",
        allow_multiple=False,
        return_dataseries=False,
    )
    if type(harp_ds.extra_attributes["csv_files"]) == str:
        harp_files = eval(harp_ds.extra_attributes["csv_files"])
    else:
        harp_files = harp_ds.extra_attributes["csv_files"]
    paramlog_path = harp_ds.path_full / harp_files["NewParams"]
    param_log = pd.read_csv(paramlog_path)
    # trial index for each row of param log
    start_idx = (
        trials_df.imaging_harptime_stim_start.searchsorted(param_log.HarpTime) - 1
    )
    start_idx = np.clip(start_idx, 0, len(trials_df) - 1)
    start_idx = pd.Series(start_idx)
    start_idx = start_idx[start_idx.diff() != 0].index.values
    trials_df["param_log_start"] = start_idx

    stop_idx = trials_df.imaging_harptime_stim_stop.searchsorted(param_log.HarpTime) - 1
    stop_idx = pd.Series(stop_idx)
    stop_idx = stop_idx[stop_idx.diff() != 0].index.values
    if stop_idx[0] == 0:
        stop_idx = stop_idx[1:]
    stop_idx = stop_idx[: len(start_idx)]
    trials_df["param_log_stop"] = stop_idx

    return trials_df


def sync_all_recordings(
    session_name,
    flexilims_session=None,
    project=None,
    filter_datasets=None,
    recording_type="two_photon",
    protocol_base="SpheresPermTubeReward",
    photodiode_protocol=5,
    return_volumes=True,
):
    """Concatenate synchronisation results for all recordings in a session.

    Args:
        session_name (str): {mouse}_{session}
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.
        filter_datasets (dict): dictionary of filter keys and values to filter for the desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        recording_type (str, optional): Type of the recording. Defaults to "two_photon".
        protocol_base (str, optional): Base of the protocol. Defaults to "SpheresPermTubeReward".
        photodiode_protocol (int): number of photodiode quad colors used for monitoring frame refresh.
            Either 2 or 5 for now. Defaults to 5.
        return_volumes (bool): if True, return only the first frame of each imaging volume. Defaults to True.

    Returns:
        (pd.DataFrame, pd.DataFrame): tuple of two dataframes, one concatenated vs_df for all recordings, one concatenated trials_df for all recordings.
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

    for i, recording_name in enumerate(recordings.name):
        recording = flz.get_entity(
            datatype="recording",
            name=recording_name,
            flexilims_session=flexilims_session,
        )

        print(f"Processing recording {i+1}/{len(recordings)}")
        vs_df = synchronisation.generate_vs_df(
            recording=recording,
            photodiode_protocol=photodiode_protocol,
            flexilims_session=flexilims_session,
            project=project,
        )

        imaging_df = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=recording,
            flexilims_session=flexilims_session,
            filter_datasets=filter_datasets,
            return_volumes=return_volumes,
        )

        imaging_df = format_imaging_df(recording=recording, imaging_df=imaging_df)

        trials_df = generate_trials_df(recording=recording, imaging_df=imaging_df)

        trials_df = search_param_log_trials(
            recording=recording,
            trials_df=trials_df,
            flexilims_session=flexilims_session,
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
    recording_type="two_photon",
    protocol_base="SpheresPermTubeReward",
    photodiode_protocol=5,
    return_volumes=True,
    resolution=5,
):
    """Concatenate regenerated frames for all recordings in a session.

    Args:
        session_name (str): {mouse}_{session}
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.
        filter_datasets (dict): dictionary of filter keys and values to filter for the desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        recording_type (str, optional): Type of the recording. Defaults to "two_photon".
        protocol_base (str, optional): Base of the protocol. Defaults to "SpheresPermTubeReward".
        photodiode_protocol (int): number of photodiode quad colors used for monitoring frame refresh.
            Either 2 or 5 for now. Defaults to 5.
        return_volumes (bool): if True, return only the first frame of each imaging volume. Defaults to True.
        resolution (float): size of a pixel in degrees

    Returns:
        (np.array, pd.DataFrame): tuple, one concatenated regenerated frames for all recordings (nframes * y * x), one concatenated imaging_df for all recordings.
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

    for i, recording_name in enumerate(recordings.name):
        recording = flz.get_entity(
            datatype="recording",
            name=recording_name,
            flexilims_session=flexilims_session,
        )

        # Generate vs_df, imaging_df, trials_df for this recording
        print(f"Regenerating frames for recording {i+1}/{len(recordings)}")
        vs_df = synchronisation.generate_vs_df(
            recording=recording,
            photodiode_protocol=photodiode_protocol,
            flexilims_session=flexilims_session,
            project=project,
        )

        imaging_df = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=recording,
            flexilims_session=flexilims_session,
            filter_datasets=filter_datasets,
            return_volumes=return_volumes,
        )

        imaging_df = format_imaging_df(recording=recording, imaging_df=imaging_df)

        trials_df = generate_trials_df(recording=recording, imaging_df=imaging_df)

        trials_df = search_param_log_trials(
            recording=recording,
            trials_df=trials_df,
            flexilims_session=flexilims_session,
        )

        # Load paramlog
        harp_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=recording.name,
            dataset_type="harp",
            allow_multiple=False,
            return_dataseries=False,
        )
        if type(harp_ds.extra_attributes["csv_files"]) == str:
            harp_files = eval(harp_ds.extra_attributes["csv_files"])
        else:
            harp_files = harp_ds.extra_attributes["csv_files"]
        paramlog_path = harp_ds.path_full / harp_files["NewParams"]
        param_log = pd.read_csv(paramlog_path)

        # Regenerate frames for this trial
        sphere_size = 10 * vs_df.OriginalSize.unique()[1] / 0.087
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
            verbose=True,
            output_datatype="int16",
            output=None,
            # flip_x=True,
        )

        if i == 0:
            frames_all = frames
            imaging_df_all = imaging_df
        else:
            frames_all = np.concatenate((frames_all, frames), axis=0)
            imaging_df_all = pd.concat([trials_df_all, trials_df], ignore_index=True)
    print(f"Finished concatenating regenerated frames and imaging_df")

    return frames_all, imaging_df_all


def laplace_matrix(nx, ny):
    Ls = []
    for x in range(nx):
        for y in range(ny):
            m = np.zeros((nx, ny))
            m[x, y] = 4
            if x > 0:
                m[x - 1, y] = -1
            if x < m.shape[0] - 1:
                m[x + 1, y] = -1
            if y > 0:
                m[x, y - 1] = -1
            if y < m.shape[1] - 1:
                m[x, y + 1] = -1
            Ls.append(m.flatten())
    L = np.stack(Ls, axis=0)
    return L


def fit_3d_rfs(
    imaging_df,
    frames,
    reg_xy=100,
    reg_depth=20,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    choose_rois=(),
    validation=False,
):
    """Fit 3D receptive fields using regularized least squares regression, with only one set of hyperparameters. 
    Runs on all ROIs in parallel.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): array of frames
        reg_xy (float): regularization constant for spatial regularization
        reg_depth (float): regularization constant for depth regularization
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.
        choose_rois (list): a list of ROI indices to fit. Defaults to [], which means fit all ROIs.
        validation (bool): whether to include a validation set for hyperparameter tuning. Defaults to False.

    Returns:
        coef (np.array): array of coefficients for each pixel, ndepths x (ndepths x nazi x nele + 1) x ncells
        r2 (list): list of arrays of r2 for each ROI for training, validation and test sets, ncells x 2

    """
    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    if len(choose_rois)>0:
        resps = resps[:, choose_rois]
    depths = imaging_df.depth.unique()
    depths = depths[~np.isnan(depths)]
    depths = depths[depths > 0]
    depths = np.sort(depths)
    L = laplace_matrix(frames.shape[1], frames.shape[2])
    Ls = []
    Ls_depth = []

    trial_idx = np.zeros_like(imaging_df.depth)
    trial_idx = np.cumsum(
        np.logical_and(np.abs(imaging_df.depth.diff()) > 0, imaging_df.depth > 0)
    )
    trial_idx[imaging_df.depth.isna()] = np.nan
    trial_idx[imaging_df.depth < 0] = np.nan
    imaging_df["trial_idx"] = trial_idx
    # get the depth of the first row for each trial
    depths_by_trial = imaging_df.groupby("trial_idx").first().depth
    # convert to categorical codes
    depths_by_trial.update(pd.Categorical(depths_by_trial).codes)
    # convert index to int
    depths_by_trial.index = depths_by_trial.index.astype(int)

    X = np.zeros((frames.shape[0], frames.shape[1] * frames.shape[2] * depths.shape[0]))
    for idepth, depth in enumerate(depths):
        depth_idx = imaging_df.depth == depth
        m = np.roll(np.reshape(frames, (frames.shape[0], -1)), shift_stim, axis=0)[
            depth_idx, :
        ]
        # place m in the right columns of X
        X[depth_idx, idepth * m.shape[1] : (idepth + 1) * m.shape[1]] = m
        # add regularization penalty on the second derivative of the coefficients
        # in X and Y
        L_xy = np.zeros((L.shape[0], X.shape[1]))
        L_xy[:, idepth * L.shape[1] : (idepth + 1) * L.shape[1]] = L
        Ls.append(L_xy)
        # add regularization penalty on the second derivative of the coefficients
        # along the depth axis
        L_depth = np.zeros((m.shape[1], X.shape[1]))
        L_depth[:, idepth * m.shape[1] : (idepth + 1) * m.shape[1]] = (
            np.identity(m.shape[1]) * 2
        )
        if idepth > 0:
            L_depth[:, (idepth - 1) * m.shape[1] : idepth * m.shape[1]] = -np.identity(
                m.shape[1]
            )
        if idepth < depths.shape[0] - 1:
            L_depth[
                :, (idepth + 1) * m.shape[1] : (idepth + 2) * m.shape[1]
            ] = -np.identity(m.shape[1])
        Ls_depth.append(L_depth)

    L = np.concatenate(Ls, axis=0)
    L = np.concatenate([L, np.zeros((L.shape[0], 1))], axis=1)
    L_depth = np.concatenate(Ls_depth, axis=0)
    L_depth = np.concatenate([L_depth, np.zeros((L_depth.shape[0], 1))], axis=1)
    # add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coefs = []
    # 0 for train and -1 for test, 1 for validation prediction
    n_splits = 3 if validation else 2
    Y_pred = np.zeros((resps.shape[0], resps.shape[1], n_splits)) * np.nan
    # randomly split trials into training and test sets
    stratified_kfold = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
    # Use validation set to select the best regularization parameters (train, val, test),
    # or use test set to evaluate performance (train, test)
    for train_trials, test_trials in stratified_kfold.split(
        depths_by_trial.index, depths_by_trial.values
    ):
        if validation:
            train_trials, validation_trials = train_test_split(
                train_trials,
                stratify=depths_by_trial.iloc[train_trials].values,
                test_size=(1 / (k_folds - 1)),
            )
            validation_idx = np.isin(imaging_df.trial_idx, validation_trials)
        train_idx = np.isin(imaging_df.trial_idx, train_trials)
        test_idx = np.isin(imaging_df.trial_idx, test_trials)

        X_train = np.concatenate(
            [X[train_idx, :], reg_xy * L, reg_depth * L_depth], axis=0
        )
        Q = np.linalg.inv(X_train.T @ X_train) @ X_train.T

        Y_train = np.concatenate(
            [
                resps[train_idx, :],
                np.zeros((L.shape[0], resps.shape[1])),
                np.zeros((L_depth.shape[0], resps.shape[1])),
            ],
            axis=0,
        )
        coef = Q @ Y_train
        coefs.append(coef)

        if validation:
            idxs = [train_idx, validation_idx, test_idx]
        else:
            idxs = [train_idx, test_idx]
        for isplit, idx in enumerate(idxs):
            Y_pred[idx, :, isplit] = X[idx, :] @ coef
    # calculate R2
    r2 = np.zeros((resps.shape[1], n_splits)) * np.nan
    for isplit in range(n_splits):
        use_idx = np.isfinite(Y_pred[:, 0, isplit])
        residual_var = np.sum(
            (Y_pred[use_idx, :, isplit] - resps[use_idx, :]) ** 2,
            axis=0,
        )
        total_var = np.sum(
            (resps[use_idx, :] - np.mean(resps[use_idx, :], axis=0)) ** 2, axis=0
        )
        r2[:, isplit] = 1 - residual_var / total_var
    return coefs, r2


def fit_3d_rfs_hyperparam_tuning(
    imaging_df,
    frames,
    reg_xys=[20, 40, 80, 160, 320],
    reg_depths=[20, 40, 80, 160, 320],
    shift_stims=2,
    use_col="dffs",
    k_folds=5,
    tune_separately=True,
    validation=False,
):
    """Fit 3D receptive fields using regularized least squares regression, with hyperparameter tuning.
    Runs on all ROIs in parallel.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): array of frames
        reg_xys (list): a list of regularization constant for spatial regularization
        reg_depths (list): a list of regularization constant for depth regularization
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.
        tune_separately (bool): whether to tune hyperparameters separately for each ROI. Defaults to True.
        validation (bool): whether to include a validation set for hyperparameter tuning. Defaults to False.

    Returns:
        coef (np.array): array of coefficients for each pixel, ndepths x (ndepths x nazi x nele + 1) x ncells
        r2 (list): list of arrays of r2 for each ROI for training, validation and test sets, ncells x 2
        best_reg_xys (np.array): array of best reg_xy for each ROI
        best_reg_depths (np.array): array of best reg_depth for each ROI

    """
    
    all_coef = []
    all_rs = []
    hyperparams = []
    good_neuron_percs = np.zeros((len(reg_xys), len(reg_depths)))
    for i, reg_xy in enumerate(reg_xys):
        for j, reg_depth in enumerate(reg_depths):
            print(f"fitting reg_xy: {reg_xy}, reg_depth: {reg_depth}")
            coef, r2 = fit_3d_rfs(imaging_df, frames, reg_xy=reg_xy, reg_depth=reg_depth, shift_stim=shift_stims, use_col=use_col,k_folds=k_folds, validation=validation)
            good_neuron_percs[i,j] = np.mean(r2[:,1]>.01)
            all_coef.append(coef)
            all_rs.append(r2)
            hyperparams.append([reg_xy, reg_depth])
    all_coef = np.stack(all_coef, axis=0)
    all_rs = np.stack(all_rs, axis=0)

    best_hyperparam_idxs = np.argmax(all_rs[:, :, 1], axis=0)

    if not tune_separately:
        max_idx = np.unravel_index(np.argmax(good_neuron_percs), good_neuron_percs.shape)
        best_reg_xy, best_reg_depth = reg_xys[max_idx[0]], reg_depths[max_idx[1]]
        print(f"Best param found for all ROIs: reg_xy: {best_reg_xy}, reg_depth: {best_reg_depth}, R2>0.01: {good_neuron_percs[max_idx]:.4f}")
        coef, r2 = fit_3d_rfs(imaging_df, frames, reg_xy=best_reg_xy, reg_depth=best_reg_depth, shift_stim=shift_stims, use_col=use_col,k_folds=k_folds, validation=validation)
        [best_reg_xy, best_reg_depth] = hyperparams[mode(best_hyperparam_idxs[0][0])]
        print(
            f"Best param found for all ROIs: reg_xy: {best_reg_xy}, reg_depth: {best_reg_depth}"
        )
        coef = np.stack(coef)
        best_reg_xys = np.ones(len(r2[:,1]))*best_reg_xy
        best_reg_depths = np.ones(len(r2[:,1]))*best_reg_depth
    else:
        best_reg_xys = np.zeros(len(r2[:,1]))
        best_reg_depths = np.zeros(len(r2[:,1]))
        for iparam in np.sort(np.unique(best_hyperparam_idxs)):
            [best_reg_xy, best_reg_depth] = hyperparams[iparam]
            fit_neurons = np.arange(imaging_df[use_col][0].shape[1])[best_hyperparam_idxs==iparam]
            print(f"Best param found for {len(fit_neurons)} neurons: reg_xy: {best_reg_xy}, reg_depth: {best_reg_depth}")
            coef_temp, r2_temp = fit_3d_rfs(imaging_df, frames, reg_xy=best_reg_xy, reg_depth=best_reg_depth, shift_stim=shift_stims, use_col=use_col,k_folds=k_folds, choose_rois=fit_neurons, validation=validation)
            coef = np.stack(coef)
            coef[:, :, fit_neurons] = np.stack(coef_temp)
            r2[fit_neurons,:] = r2_temp
            best_reg_xys[fit_neurons] = best_reg_xy
            best_reg_depths[fit_neurons] = best_reg_depth
             
    return coef, r2, best_reg_xys, best_reg_depths


def fit_3d_rfs_ipsi(
    imaging_df,
    frames,
    best_reg_xys,
    best_reg_depths,
    shift_stims=2,
    use_col="dffs",
    k_folds=5,
    validation=False,
):
    """Fit 3D receptive fields using the ipsilateral side of stimuli using regularized least squares regression, using the best set of hyperparameter of the contralateral side.
    Runs on all ROIs in parallel.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): array of frames
        best_reg_xys (list): a list of best regularization constant for spatial regularization from the contra side fitting.
        best_reg_depths (list): a list of best regularization constant for depth regularization from the contra side fitting.
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.
        validation (bool): whether to include a validation set for hyperparameter tuning. Defaults to False.

    Returns:
        coef (np.array): array of coefficients for each pixel, ndepths x (ndepths x nazi x nele + 1) x ncells
        r2 (list): list of arrays of r2 for each ROI for training, validation and test sets, ncells x 2

    """
    
    best_regs = np.stack([best_reg_xys, best_reg_depths], axis=1)
    coef_temp, r2_temp= fit_3d_rfs(imaging_df, frames, reg_xy=80, reg_depth=40, shift_stim=shift_stims, use_col=use_col,k_folds=k_folds, validation=validation)
    coef = np.zeros_like(np.stack(coef_temp))
    r2 = np.zeros_like(np.stack(r2_temp))
    for best_reg in np.unique(best_regs, axis=0):
        best_reg_neurons = np.where(np.all(best_reg==best_regs, axis=1))[0]
        print(f"Fit with best param for {len(best_reg_neurons)} neurons: reg_xy: {best_reg[0]}, reg_depth: {best_reg[1]}")
        coef_temp, r2_temp = fit_3d_rfs(imaging_df, frames, reg_xy=best_reg[0], reg_depth=best_reg[1], shift_stim=shift_stims, use_col=use_col,k_folds=k_folds, choose_rois=best_reg_neurons, validation=validation)
        coef[:, :, best_reg_neurons] = np.stack(coef_temp)
        r2[best_reg_neurons,:] = r2_temp
    return coef, r2
        

def find_sig_rfs(coef, coef_ipsi, n_std=5):
    '''Find the neurons with a significant RF (compared to ipsi side)

    Args:
        coef (_type_): _description_
        coef_ipsi (_type_): _description_
        n_std (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    '''
    coef_mean = np.mean(np.stack(coef, axis=2), axis=2)
    coef_ipsi_mean = np.mean(np.stack(coef_ipsi, axis=2), axis=2)

    threshold = n_std * np.std(coef_ipsi_mean[:-1, :], axis=0) + np.mean(coef_ipsi_mean[:-1, :], axis=0)
    sig = np.max(coef_mean[:-1, :], axis=0) > threshold
    sig_ipsi = np.max(coef_ipsi_mean[:-1, :], axis=0) > threshold
    
    return sig, sig_ipsi

# def fit_3d_rfs(
#     imaging_df,
#     frames,
#     reg_xy=100,
#     reg_depth=20,
#     shift_stim=2,
#     use_col="dffs",
#     k_folds=5,
# ):
#     """Fit 3D receptive fields using regularized least squares regression.
#     Runs on all ROIs in parallel.

#     Args:
#         imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
#         frames (np.array): array of frames
#         reg_xy (float): regularization constant for spatial regularization
#         reg_depth (float): regularization constant for depth regularization
#         shift_stim (int): number of frames to shift the stimulus by.
#             This is to account for the delay between the stimulus and the response.
#             Defaults to 2.
#         use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".

#     Returns:
#         coef (np.array): array of coefficients for each pixel
#         r2 (list): list of arrays of r2 for each ROI for training, validation and test sets
#     """
#     resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
#     depths = imaging_df.depth.unique()
#     depths = depths[~np.isnan(depths)]
#     depths = depths[depths > 0]
#     depths = np.sort(depths)
#     L = laplace_matrix(frames.shape[1], frames.shape[2])
#     Ls = []
#     Ls_depth = []

#     trial_idx = np.zeros_like(imaging_df.depth)
#     trial_idx = np.cumsum(
#         np.logical_and(np.abs(imaging_df.depth.diff()) > 0, imaging_df.depth > 0)
#     )
#     trial_idx[imaging_df.depth.isna()] = np.nan
#     trial_idx[imaging_df.depth < 0] = np.nan
#     imaging_df["trial_idx"] = trial_idx
#     # get the depth of the first row for each trial
#     depths_by_trial = imaging_df.groupby("trial_idx").first().depth
#     # convert to categorical codes
#     depths_by_trial.update(pd.Categorical(depths_by_trial).codes)
#     # convert index to int
#     depths_by_trial.index = depths_by_trial.index.astype(int)

#     X = np.zeros((frames.shape[0], frames.shape[1] * frames.shape[2] * depths.shape[0]))
#     for idepth, depth in enumerate(depths):
#         depth_idx = imaging_df.depth == depth
#         m = np.roll(np.reshape(frames, (frames.shape[0], -1)), shift_stim, axis=0)[
#             depth_idx, :
#         ]
#         # place m in the right columns of X
#         X[depth_idx, idepth * m.shape[1] : (idepth + 1) * m.shape[1]] = m
#         # add regularization penalty on the second derivative of the coefficients
#         # in X and Y
#         L_xy = np.zeros((L.shape[0], X.shape[1]))
#         L_xy[:, idepth * L.shape[1] : (idepth + 1) * L.shape[1]] = L
#         Ls.append(L_xy)
#         # add regularization penalty on the second derivative of the coefficients
#         # along the depth axis
#         L_depth = np.zeros((m.shape[1], X.shape[1]))
#         L_depth[:, idepth * m.shape[1] : (idepth + 1) * m.shape[1]] = (
#             np.identity(m.shape[1]) * 2
#         )
#         if idepth > 0:
#             L_depth[:, (idepth - 1) * m.shape[1] : idepth * m.shape[1]] = -np.identity(
#                 m.shape[1]
#             )
#         if idepth < depths.shape[0] - 1:
#             L_depth[
#                 :, (idepth + 1) * m.shape[1] : (idepth + 2) * m.shape[1]
#             ] = -np.identity(m.shape[1])
#         Ls_depth.append(L_depth)

#     L = np.concatenate(Ls, axis=0)
#     L = np.concatenate([L, np.zeros((L.shape[0], 1))], axis=1)
#     L_depth = np.concatenate(Ls_depth, axis=0)
#     L_depth = np.concatenate([L_depth, np.zeros((L_depth.shape[0], 1))], axis=1)
#     # add bias
#     X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
#     coefs = []

#     Y_pred = np.zeros((resps.shape[0], resps.shape[1], 2)) * np.nan
#     # randomly split trials into training, validation and test
#     stratified_kfold = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
#     for train_trials, test_trials in stratified_kfold.split(
#         depths_by_trial.index, depths_by_trial.values
#     ):
#         # train_trials, validation_trials = train_test_split(
#         #     train_trials,
#         #     stratify=depths_by_trial.iloc[train_trials].values,
#         #     test_size=(1 / (k_folds - 1)),
#         # )
#         test_idx = np.isin(imaging_df.trial_idx, test_trials)
#         train_idx = np.isin(imaging_df.trial_idx, train_trials)
#         # validation_idx = np.isin(imaging_df.trial_idx, validation_trials)
#         X_train = np.concatenate(
#             [X[train_idx, :], reg_xy * L, reg_depth * L_depth], axis=0
#         )
#         Q = np.linalg.inv(X_train.T @ X_train) @ X_train.T

#         Y_train = np.concatenate(
#             [
#                 resps[train_idx, :],
#                 np.zeros((L.shape[0], resps.shape[1])),
#                 np.zeros((L_depth.shape[0], resps.shape[1])),
#             ],
#             axis=0,
#         )
#         coef = Q @ Y_train
#         coefs.append(coef)
#         # compute predictions for this fold
#         for isplit, idx in enumerate([train_idx, test_idx]):
#             Y_pred[idx, :, isplit] = X[idx, :] @ coef
#     use_idx = np.isfinite(Y_pred[:, 0, 0])
#     residual_var = np.sum(
#         (Y_pred[use_idx, :, :] - resps[use_idx, :, np.newaxis]) ** 2,
#         axis=0,
#     )
#     total_var = np.sum(
#         (resps[use_idx, :] - np.mean(resps[use_idx, :], axis=0)) ** 2, axis=0
#     )
#     r2 = 1 - residual_var / total_var[:, np.newaxis]

#     return coefs, r2


def fit_3d_rfs_parametric(coef, nx, ny, nz, model="gaussian"):
    (zs, ys, xs) = np.meshgrid(
        np.arange(nz),
        np.arange(ny),
        np.arange(nx),
        indexing="ij",
    )
    if model == "gaussian":
        func = partial(gaussian_3d_rf, min_sigma=0.25)
    else:
        func = partial(gabor_3d_rf, min_sigma=0.25)

    coef_fit = coef.copy()
    params = []
    # lower_bounds = Gaussian3DRFParams(
    #     log_amplitude=-np.inf,
    #     x0=0,
    #     y0=0,
    #     log_sigma_x2=-np.inf,
    #     log_sigma_y2=-np.inf,
    #     theta=0,
    #     offset=-np.inf,
    #     z0=0,
    #     log_sigma_z=-np.inf,
    # )
    # upper_bounds = Gaussian3DRFParams(
    #     log_amplitude=np.inf,
    #     x0=nx,
    #     y0=ny,
    #     log_sigma_x2=np.inf,
    #     log_sigma_y2=np.inf,
    #     theta=np.pi / 2,
    #     offset=np.inf,
    #     z0=nz,
    #     log_sigma_z=np.inf,
    # )
    # TODO using bounds currently is not working well
    for roi in tqdm(range(coef.shape[1])):
        c = np.reshape(coef[:-1, roi], (nz, ny, nx))
        # get the index of the maximum of c
        idepth, iy, ix = np.unravel_index(np.argmax(c), c.shape)
        if model == "gaussian":
            p0 = Gaussian3DRFParams(
                log_amplitude=np.log(c.max()),
                x0=ix,
                y0=iy,
                log_sigma_x2=0,
                log_sigma_y2=0,
                theta=0,
                offset=0,
                z0=idepth,
                log_sigma_z=0,
            )
        else:
            p0 = Gabor3DRFParams(
                log_amplitude=np.log(c.max()),
                x0=ix,
                y0=iy,
                log_sigma_x2=0,
                log_sigma_y2=0,
                theta=0,
                offset=0,
                log_sf=0,
                alpha=0,
                phase=0,
                z0=idepth,
                log_sigma_z=0,
            )
        try:
            popt = curve_fit(
                func,
                (xs.flatten(), ys.flatten(), zs.flatten()),
                c.flatten(),
                p0=p0,
            )[0]
        except RuntimeError:
            print(f"Warning: failed to fit gaussian to ROI {roi}")
            popt = p0
        coef_fit[:-1, roi] = func((xs.flatten(), ys.flatten(), zs.flatten()), *popt)
        params.append(popt)
    return coef_fit, params
