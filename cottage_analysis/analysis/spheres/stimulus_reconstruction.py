from functools import partial
import numpy as np

from cottage_analysis.analysis.spheres.rf_fitting import find_valid_frames
from tqdm import tqdm

print = partial(print, flush=True)


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
        return None, 0

    # convert `in_screen` spheres in pixel space
    az_on_screen = (az_compas[in_screen] - azimuth_limits[0]) / resolution
    el_on_screen = (elevation[in_screen] - elevation_limits[0]) / resolution
    if np.any(radius[in_screen] < depth * 0.95):  # allow 5% of rounding error
        raise ValueError("Radius values are too small compared to depth")
    size = depth / radius[in_screen] * sphere_size / resolution

    xx, yy = np.meshgrid(np.arange(azi_n), np.arange(ele_n))
    xx = np.outer(xx.reshape(-1), np.ones(len(az_on_screen)))
    yy = np.outer(yy.reshape(-1), np.ones(len(el_on_screen)))
    ok = (xx - az_on_screen) ** 2 + (yy - el_on_screen) ** 2 - size**2
    ok = ok <= 0
    # When plotting output, the origin (for lowest azimuth and elevation) is at lower left
    frame = np.any(ok, axis=1).reshape((ele_n, azi_n))
    return frame, np.sum(in_screen)


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
    verbose=False,
    output_datatype="int16",
    output=None,
    return_sphere_number=False,
    separate_depths=None,
):
    """Regenerate frames of sphere stimulus

    `frame_times` is usually the imaging frame time, not the monitor frame time.


    Args:
        frame_times (np.array): Array of time at which the frame should be regenerated.
        trials_df (pd.DataFrame): Dataframe contains information for each trial.
        vs_df (pd.DataFrame): Dataframe containing mouse position information for each
            monitor frame.
        param_logger (pd.DataFrame): Params saved by Bonsai logger
        time_column (str): Name of the column containing timing information in
            dataframes (Default: 'HarpTime')
        resolution (float): size of a pixel in degrees
        sphere_size (float): size of a sphere in degrees
        azimuth_limits ([float, float]): Minimum and maximum azimuth of the display
        elevation_limits ([float, float]): Minimum and maximum elevation of the display
        verbose (bool): Print information
        output_datatype (type): datatype of the output. Use bool to have binary
            sphere/no sphere output. int for seeing sphere overlap. Not used if output
            is provided
        output (np.array): Array to add output. Will be done inplace
        return_sphere_number (bool): If True, return the number of spheres on screen for
            each frame. Defaults to False.
        separate_depths (list): List of depths to separate. If None, all depths are
            put on the same frame. Defaults to None.


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
    if separate_depths is not None:
        ndepths = len(separate_depths)
        out_shape = (ndepths,) + out_shape
    if output is None:
        output = np.zeros(out_shape, dtype=output_datatype)
    else:
        assert output.shape == out_shape
    if return_sphere_number:
        if separate_depths is not None:
            n_spheres_per_frame = np.zeros(
                (len(separate_depths), len(frame_times)), dtype=int
            )
        else:
            n_spheres_per_frame = np.zeros(len(frame_times), dtype=int)
    # Find frame indices that are not grey and within the imaging time.
    assert trials_df.imaging_harptime_stim_start.is_monotonic_increasing
    trial_index = (
        trials_df.imaging_harptime_stim_start.searchsorted(frame_times, side="right")
        - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    frame_indices = find_valid_frames(frame_times, trials_df, verbose=verbose)
    # If the imaging frame is after the last found monitor frame, we cannot find the
    # position of the mouse for that frame. We will assume that the frame is gray
    # and use the last found position for the searchsorted to avoid crashing.
    delayed_frame = np.where(frame_times > mouse_pos_time[-1])[0]
    if len(delayed_frame) > 0:
        if verbose:
            print(
                f"WARNING: {len(delayed_frame)} imaging frames are after the last found "
                + "monitor frame.\nWe will assume that they are gray."
            )
        frame_indices = frame_indices[
            : np.searchsorted(frame_indices, delayed_frame[0])
        ]
        frame_times[delayed_frame] = frame_times[delayed_frame[0] - 1]
    mouse_position = mouse_pos_cm[mouse_pos_time.searchsorted(frame_times)]

    # now process the valid frames
    log_ends = param_logger[time_column].searchsorted(frame_times)
    for frame_index in tqdm(frame_indices):
        # find the trial in which the frame is
        corridor = trials_df.loc[int(trial_index[frame_index])]
        # load the logger from trial start until the time of the frame. This is the list
        # of all the spheres as they appear. Some/most of the spheres might already be
        # far behind the mouse.
        full_logger = param_logger.iloc[
            corridor.param_log_start : np.max(
                [log_ends[frame_index], corridor.param_log_start + 1]
            )
        ]
        depth_col = "Radius" if "Radius" in full_logger.columns else "Depth"
        for depth, logger in full_logger.groupby(depth_col):
            if separate_depths is not None:
                if depth == -9999:
                    continue
                idepth = separate_depths.index(depth)
            # remove the spheres that are behind the mouse
            if "Radius" in logger.columns:
                logger = logger[logger.Radius > 0]
            elif "Depth" in logger.columns:
                logger = logger[logger.Depth > 0]
            else:
                raise ValueError("Neither Radius nor Depth in param_logger columns")
            sphere_coordinates = np.array(logger[["X", "Y", "Z"]].values, dtype=float)
            sphere_coordinates[:, 2] = (
                sphere_coordinates[:, 2] - mouse_position[frame_index]
            )

            this_frame, n_on_screen = draw_spheres(
                sphere_x=sphere_coordinates[:, 0],
                sphere_y=sphere_coordinates[:, 1],
                sphere_z=sphere_coordinates[:, 2],
                depth=depth,
                resolution=float(resolution),
                sphere_size=float(sphere_size),
                azimuth_limits=np.array(azimuth_limits, dtype=float),
                elevation_limits=np.array(elevation_limits, dtype=float),
            )
            if return_sphere_number:
                if separate_depths is not None:
                    n_spheres_per_frame[idepth, frame_index] = n_on_screen
                else:
                    n_spheres_per_frame[frame_index] = n_on_screen

            if this_frame is None:
                if verbose:
                    # Some frames are not reconstructed. This is likely due to the fact that
                    # that the trial just started and sphere had not time to appear. We will
                    # complain only if at least 10ms have passed since the start of the trial.
                    trial_start = corridor.imaging_harptime_stim_start
                    t2start = frame_times[frame_index] - trial_start
                    if verbose and (n_on_screen == 0) and (t2start > 0.01):
                        print(
                            f"Warning: failed to reconstruct frame {frame_index}"
                            + f" ({n_on_screen} spheres on screen, {t2start:.3f}s after trial start)"
                        )
                # 0 in case output was provided
                if separate_depths is not None:
                    output[idepth, frame_index] *= 0
                else:
                    output[frame_index] *= 0
            elif separate_depths is not None:
                output[idepth, frame_index] = this_frame.astype(output.dtype)
            else:
                output[frame_index] = this_frame.astype(output.dtype)
    if return_sphere_number:
        return output, n_spheres_per_frame
    return output
