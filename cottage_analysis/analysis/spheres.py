import numpy as np
import pandas as pd
from tqdm import tqdm
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.imaging.common.find_frames import find_imaging_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers
import flexiznam as flz
from pathlib import Path


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
    before = frame_times < trials_df.harptime_stim_start.iloc[0]
    after = frame_times > trials_df.harptime_stim_stop.iloc[-1]
    if verbose:
        print(
            "Ignoring %d frames before and %d after the stimulus presentation"
            % (np.sum(before), np.sum(after))
        )
    valid_frames = ~before & ~after

    trial_index = (
        trials_df.harptime_stim_start.searchsorted(frame_times, side="right") - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    trial_end = trials_df.loc[trial_index, "harptime_stim_stop"].values
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
        "onset_harptime"
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
        trials_df.harptime_stim_start.searchsorted(frame_times, side="right") - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    frame_indices = find_valid_frames(frame_times, trials_df, verbose=verbose)
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


def format_vs_df_params(recording, vs_df):
    """Format sphere params in vs_df.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        vs_df(pd.DataFrame): dataframe that contains info for each monitor frame.

    Returns:
        DataFrame: contains information for each monitor frame and vis-stim.

    """

    if "Radius" in vs_df.columns:
        vs_df = vs_df.rename(columns={"Radius": "depth"})
    elif "Depth" in vs_df.columns:
        vs_df = vs_df.rename(columns={"Depth": "depth"})
    if "depth" in vs_df.columns:
        vs_df["depth"] = vs_df["depth"] / 100  # convert cm to m
        if np.isnan(vs_df["depth"].iloc[-1]):
            vs_df = vs_df[:-1]

    # Indicate whether it's a closed loop or open loop session
    if "Playback" in recording.name:
        vs_df["closed_loop"] = 0
    else:
        vs_df["closed_loop"] = 1

    return vs_df


def init_neurons_df(
    recording,
    filter_datasets=None,
    flexilims_session=None,
    project=None,
    conflicts="skip",
):
    """Initialize a dataframe containing basic information about rois in each session. This dataframe will be saved.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        filter_datasets (dict): dictionary of filter keys and values to filter for the desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.
        conflicts (str): how to deal with conflicts when updating flexilims. Defaults to "skip".

    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    neurons_df = pd.DataFrame(
        columns=[
            "roi",  # ROI number
            "plane_no",  # plane number, which plane this roi is located in
        ]
    )
    neurons_ds = flz.Dataset.from_origin(
        origin_id=recording["id"],
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        conflicts="skip",
    )
    neurons_ds.path = neurons_ds.path.parent / f"neurons_df.pickle"

    if (neurons_ds.flexilims_status() != "not online") and (conflicts == "skip"):
        print("Loading existing neurons_df file...")
        return np.load(neurons_ds.path_full), neurons_ds

    suite2p_traces = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_id=recording.origin_id,
        dataset_type="suite2p_rois",
        filter_datasets=None,
        allow_multiple=False,
        return_dataseries=False,
    )
    nplanes = suite2p_traces.extra_attributes["nplanes"]
    for iplane in range(nplanes):
        F = np.load(
            suite2p_traces.path_full / f"plane{iplane}/F.npy", allow_pickle=True
        )
        append_neurons = pd.DataFrame(
            {
                "roi": np.arange(F.shape[0]),
                "plane_no": np.repeat(iplane, F.shape[0]),
            }
        )
        neurons_df = pd.concat([neurons_df, append_neurons], ignore_index=True)

    # save neurons_df
    neurons_ds.path_full.parent.mkdir(parents=True, exist_ok=True)
    neurons_df.to_pickle(neurons_ds.path_full)

    # update flexilims
    neurons_ds.update_flexilims(mode="overwrite")

    return neurons_df, neurons_ds


def generate_imaging_df(
    recording, vs_df, filter_datasets=None, flexilims_session=None, project=None
):
    """Generate a DataFrame that contains information for each imaging volume.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        vs_df(pd.DataFrame): dataframe that contains info for each monitor frame.
        filter_datasets (dict): dictionary of filter keys and values to filter for the desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.

    Returns:
        DataFrame: contains information for each imaging volume.

    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    # Imaging_df: to find the RS/OF array for each imaging frame
    imaging_df = pd.DataFrame(
        columns=[
            "imaging_volume",
            "harptime_imaging_trigger",
            "depth",
            "is_stim",
            "RS",  # actual running speed, m/s
            "RS_eye",  # virtual running speed, m/s
            "OF",  # optic flow speed = RS/depth, rad/s
            "dffs",
            "closed_loop",
        ]
    )

    # Add vis-stim parameters to vs_df
    vs_df = format_vs_df_params(recording=recording, vs_df=vs_df)

    suite2p_traces = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="suite2p_traces",
        filter_datasets=filter_datasets,
        allow_multiple=False,
        return_dataseries=False,
    )
    # find imaging frame number logged in bonsai
    if "nframes" in suite2p_traces.extra_attributes:
        frame_number = int(suite2p_traces.extra_attributes["nframes"])
    else:
        frame_number = int(
            np.load(suite2p_traces.path_full / "plane0" / "dff_ast.npy").shape[1]
        )
    nplanes = suite2p_traces.extra_attributes["nplanes"]

    # Find imaging frame trigger time
    harp_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp_npz",
        allow_multiple=False,
        return_dataseries=False,
    )
    img_frame_logger = format_loggers.format_img_frame_logger(
        harpmessage_file=harp_ds.path_full, register_address=32
    )
    img_frame_logger = find_imaging_frames(
        harp_message=img_frame_logger,
        frame_number=frame_number * nplanes,
        frame_period=0.0324 * 2,
        register_address=32,
        frame_period_tolerance=0.001,
    )

    # find imaging volume number
    max_frame_in_vs_df = np.nanmax(vs_df.imaging_frame)
    if frame_number != (max_frame_in_vs_df + 1):
        print(
            f"WARNING: Last {(frame_number-1-max_frame_in_vs_df)} imaging frames might be dropped. Check vs_df!"
        )
    imaging_df.imaging_volume = np.floor(
        (np.arange(max_frame_in_vs_df + 1) / nplanes)
    ).astype(int)

    # Assign dff to each imaging volume
    all_dff = []
    for iplane in range(nplanes):
        dff = np.load(suite2p_traces.path_full / f"plane{iplane}" / "dff_ast.npy")
        dff = dff[:frame_number]
        all_dff.append(dff)
    all_dff = np.vstack(all_dff)  # nrois (for all planes) x nvolume
    imaging_df.dffs = all_dff.T.tolist()

    # average RS for each imaging volume
    grouped_vs_df = vs_df.groupby("imaging_volume")
    imaging_df.RS = grouped_vs_df.apply(
        lambda x: (x["mouse_z_harp"].iloc[-1] - x["mouse_z_harp"].iloc[0])
        / (x["onset_harptime"].iloc[-1] - x["onset_harptime"].iloc[0])
    )
    imaging_df.loc[0, "RS"] = 0

    # average RS eye for each imaging volume
    imaging_df.RS_eye = grouped_vs_df.apply(
        lambda x: (x["eye_z"].iloc[-1] - x["eye_z"].iloc[0])
        / (x["onset_harptime"].iloc[-1] - x["onset_harptime"].iloc[0])
    )
    imaging_df.loc[0, "RS_eye"] = 0

    # depth for each imaging volume
    imaging_df.depth = grouped_vs_df.depth.min()
    imaging_df.is_stim = imaging_df.apply(lambda x: int(x.depth > 0), axis=1)
    imaging_df.loc[imaging_df["depth"].isna(), "depth"] = 0
    imaging_df.loc[imaging_df["depth"] < 0, "depth"] = np.nan
    imaging_df["depth"] = imaging_df.depth.fillna(method="ffill")
    imaging_df.loc[imaging_df["depth"] == 0, "depth"] = np.nan

    # fill in missing imaging volume due to frame drop
    imaging_df = synchronisation.fill_missing_imaging_volumes(imaging_df)

    # OF for each imaging volume
    imaging_df["OF"] = imaging_df.RS_eye / imaging_df.depth
    imaging_df.loc[imaging_df.is_stim == 0, "OF"] = np.nan

    # closed loop status for each imaging frame
    if "Playback" in recording.name:
        imaging_df.closed_loop = 0
    else:
        imaging_df.closed_loop = 1

    return imaging_df


def generate_trials_df(
    recording, vs_df, filter_datasets=None, flexilims_session=None, project=None
):
    """Generate a DataFrame that contains information for each trial.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        vs_df(pd.DataFrame): dataframe that contains info for each monitor frame.
        filter_datasets (dict): dictionary of filter keys and values to filter for the desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.

    Returns:
        DataFrame: contains information for each trial.

    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    imaging_df = generate_imaging_df(
        recording=recording,
        vs_df=vs_df,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets=filter_datasets,
    )

    # Add vis-stim parameters to vs_df
    vs_df = format_vs_df_params(recording=recording, vs_df=vs_df)

    # trials_df
    trials_df = pd.DataFrame(
        columns=[
            "trial_no",
            "depth",
            "harptime_stim_start",
            "harptime_stim_stop",
            "harptime_blank_start",
            "harptime_blank_stop",
            "imaging_volume_stim_start",
            "imaging_volume_stim_stop",
            "imaging_volume_blank_start",
            "imaging_volume_blank_stop",
            "param_log_start",  # which row of param log does this trial start
            "param_log_stop",  # which row of param log does this trial stop
            "RS_stim",  # actual running speed, m/s
            "RS_blank",
            "RS_eye_stim",  # virtual running speed, m/s
            "OF_stim",  # optic flow speed = RS/depth, rad/s
            "dff_stim",
            "dff_blank",
            # "spheres_no",
            "closed_loop",
        ]
    )

    # Find the start and stop of each trial
    blank_time = 10  # s

    vs_df["stim"] = np.nan
    vs_df.loc[vs_df.depth.notnull(), "stim"] = 1
    vs_df.loc[vs_df.depth < 0, "stim"] = 0
    vs_df_simple = vs_df[(vs_df["stim"].diff() != 0) & (vs_df["stim"].notnull())]
    vs_df_simple.depth = np.round(vs_df_simple.depth, 2)

    start_idx_stim = vs_df_simple[(vs_df_simple["stim"] == 1)].index
    start_idx_blank = vs_df_simple[(vs_df_simple["stim"] == 0)].index
    if len(start_idx_stim) != len(start_idx_blank):
        if (len(start_idx_stim) - len(start_idx_blank)) == 1:
            stop_idx_blank = start_idx_stim[1:] - 1
            start_idx_stim = start_idx_stim[: len(start_idx_blank)]
        else:
            print("Warning: incorrect stimulus trial structure! Double check!")
    else:
        stop_idx_blank = start_idx_stim[1:] - 1
        last_blank_stop_time = (
            vs_df.loc[start_idx_blank[-1]].onset_harptime + blank_time
        )
        stop_idx_blank = np.append(
            stop_idx_blank,
            (np.abs(vs_df["onset_harptime"] - last_blank_stop_time)).idxmin(),
        )
    stop_idx_stim = start_idx_blank - 1

    # Assign trial no, depth, start/stop time, start/stop imaging volume to trials_df
    # Harptime for starts and stops are harptime for monitor frames, not corresponding to imaging trigger harptime
    trials_df.trial_no = np.arange(len(start_idx_stim))
    trials_df.depth = vs_df.loc[start_idx_stim].depth
    trials_df.harptime_stim_start = vs_df.loc[start_idx_stim].onset_harptime.values
    trials_df.harptime_stim_stop = vs_df.loc[stop_idx_stim].onset_harptime.values
    trials_df.harptime_blank_start = vs_df.loc[start_idx_blank].onset_harptime.values
    trials_df.harptime_blank_stop = vs_df.loc[stop_idx_blank].onset_harptime.values

    trials_df.imaging_volume_stim_start = vs_df.loc[
        start_idx_stim
    ].imaging_volume.values
    trials_df.imaging_volume_blank_start = vs_df.loc[
        start_idx_blank
    ].imaging_volume.values
    trials_df.imaging_volume_blank_stop = vs_df.loc[
        stop_idx_blank
    ].imaging_volume.values
    if np.isnan(
        trials_df.imaging_volume_blank_stop.iloc[-1]
    ):  # If the blank stop of last trial is beyond the number of imaging frames
        trials_df.imaging_volume_blank_stop.iloc[-1] = len(imaging_df) - 1
    trials_df.imaging_volume_stim_stop = trials_df.imaging_volume_blank_start - 1

    mask = (
        trials_df.imaging_volume_stim_start
        == trials_df.imaging_volume_blank_stop.shift(1)
    )  # Get rid of the overlap of imaging frame no. between different trials
    trials_df.loc[mask, "imaging_volume_stim_start"] += 1

    if "Playback" in recording.name:
        trials_df.closed_loop = 0
    else:
        trials_df.closed_loop = 1

    # Assign RS array from imaging_df back to trials_df
    trials_df.RS_stim = trials_df.apply(
        lambda x: imaging_df.RS.loc[
            int(x.imaging_volume_stim_start) : int(x.imaging_volume_stim_stop)
        ].values,
        axis=1,
    )

    trials_df.RS_blank = trials_df.apply(
        lambda x: imaging_df.RS.loc[
            int(x.imaging_volume_blank_start) : int(x.imaging_volume_blank_stop)
        ].values,
        axis=1,
    )

    trials_df.RS_eye_stim = trials_df.apply(
        lambda x: imaging_df.RS_eye.loc[
            int(x.imaging_volume_stim_start) : int(x.imaging_volume_stim_stop)
        ].values,
        axis=1,
    )

    trials_df.OF_stim = trials_df.apply(
        lambda x: imaging_df.OF.loc[
            int(x.imaging_volume_stim_start) : int(x.imaging_volume_stim_stop)
        ].values,
        axis=1,
    )

    # Assign dffs array to trials_df
    trials_df.dff_stim = trials_df.apply(
        lambda x: np.stack(
            imaging_df.dffs.loc[
                int(x.imaging_volume_stim_start) : int(x.imaging_volume_stim_stop)
            ]
        ),
        axis=1,
    )
    # nvolumes x ncells

    trials_df.dff_blank = trials_df.apply(
        lambda x: np.stack(
            imaging_df.dffs.loc[
                int(x.imaging_volume_blank_start) : int(x.imaging_volume_blank_stop)
            ]
        ),
        axis=1,
    )
    # nvolumes x ncells

    # Add the start param logger row and stop param logger row to each trial
    harp_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp",
        allow_multiple=False,
        return_dataseries=False,
    )
    paramlog_path = harp_ds.path_full / harp_ds.csv_files["NewParams"]
    param_log = pd.read_csv(paramlog_path)
    # trial index for each row of param log
    start_idx = trials_df.harptime_stim_start.searchsorted(param_log.HarpTime) - 1
    start_idx = np.clip(start_idx, 0, len(trials_df) - 1)
    start_idx = pd.Series(start_idx)
    start_idx = start_idx[start_idx.diff() != 0].index.values
    trials_df["param_log_start"] = start_idx

    stop_idx = trials_df.harptime_stim_stop.searchsorted(param_log.HarpTime) - 1
    stop_idx = pd.Series(stop_idx)
    stop_idx = stop_idx[stop_idx.diff() != 0].index.values
    if stop_idx[0] == 0:
        stop_idx = stop_idx[1:]
    stop_idx = stop_idx[: len(start_idx)]
    trials_df["param_log_stop"] = stop_idx

    # Rename
    trials_df = trials_df.drop(columns=["imaging_volume_blank_start"])

    return trials_df
