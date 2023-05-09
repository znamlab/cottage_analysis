import jax
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


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

    # Find frame indices that are not grey or within the imaging time.
    trial_index = (
        trials_df.harptime_stim_start.searchsorted(frame_times, side="right") - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    frame_indices = find_valid_frames(frame_times, trials_df, verbose=verbose)
    mouse_position = mouse_pos_cm[mouse_pos_time.searchsorted(frame_times)]

    # now process the valid frames
    log_ends = param_logger[time_column].searchsorted(frame_times)
    draw_sph_jit = jax.jit(draw_spheres)
    for frame_index in tqdm(frame_indices):
        corridor = trials_df.loc[int(trial_index[frame_index])]
        logger = param_logger.iloc[corridor.param_log_start : log_ends[frame_index]]
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
