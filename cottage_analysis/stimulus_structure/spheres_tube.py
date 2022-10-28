import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trial_structure(param_logger, time_column='HarpTime'):
    """Generate a corridor dataframe

    The dataframe has one entry per corridor with depth, start/end sample and
    start/end time

    Args:
        param_logger (pd.DataFrame): DataFrame read from params logger of bonsai
        time_column (str): Name of the column containing timing information

    Returns:
        corridor_df (pd.DataFrame): trial structure dataframe
    """
    depth = np.round(param_logger.Depth, 2)
    trials_border = np.diff(np.hstack([-9999, depth]))
    trials_onset = np.where(trials_border > 5000)[0]
    trials_offset = np.where(trials_border < -5000)[0]
    n_corridors = len(trials_onset)
    if len(trials_offset) == n_corridors - 1:  # the last corridor is cut
        trials_offset = np.hstack([trials_offset, len(depth) - 1])
    elif len(trials_offset) != n_corridors:
        raise IOError('Found %d corridor starts but %d ends... ' % (n_corridors,
                                                                    len(trials_offset)))

    corridor_df = dict(trial=np.arange(n_corridors), depth=depth[trials_onset].values,
                       start_sample=np.array(trials_onset, dtype=int),
                       end_sample=np.array(trials_offset, dtype=int),
                       start_time=param_logger.loc[trials_onset, time_column].values,
                       end_time=param_logger.loc[trials_offset, time_column].values)
    return pd.DataFrame(corridor_df)


def regenerate_frames(frame_times, param_logger, mouse_pos_cm, mouse_pos_time,
                      corridor_df=None, time_column='HarpTime', resolution=0.1,
                      sphere_size=10, azimuth_limits=(-120, 120),
                      elevation_limits=(-40, 40),
                      verbose=True, output_datatype=bool, output=None):
    """Regenerate frames of sphere stimulus

    Args:
        frame_times (np.array): Array of time at which the frame should be regenerated
        param_logger (pd.DataFrame): Params saved by Bonsai logger
        mouse_pos_cm (np.array): position of the mouse in cm
        mouse_pos_time (np.array): time of each mouse_pos_cm sample
        corridor_df (pd.DataFrame): trial structure dataframe. One line per corridor
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

    if corridor_df is None:
        corridor_df = trial_structure(param_logger, time_column)
    out_shape = (len(frame_times), int(np.diff(elevation_limits) / resolution),
                 int(np.diff(azimuth_limits) / resolution))
    if output is None:
        output = np.zeros(out_shape, dtype=output_datatype)
    else:
        assert output.shape == out_shape

    # for frames before and after the protocol, keep them 0s
    before = frame_times < corridor_df.start_time.iloc[0]
    after = frame_times > corridor_df.end_time.iloc[-1]
    if verbose:
        print('Ignoring %d frames before and %d after the stimulus presentation'
              % (np.sum(before), np.sum(after)))
    valid_frames = ~before & ~after

    corridor_index = corridor_df.start_time.searchsorted(frame_times, side='right') - 1
    corridor_index = np.clip(corridor_index, 0, len(corridor_df) - 1)
    corridor_end = corridor_df.loc[corridor_index, 'end_time']
    grey_time = frame_times - corridor_end > 0
    if verbose:
        print('Ignoring %d frames in grey inter-trial intervals' % np.sum(grey_time &
                                                                          valid_frames))
    valid_frames = valid_frames & (~grey_time)
    mouse_position = mouse_pos_cm[mouse_pos_time.searchsorted(frame_times)]
    frame_indices = np.where(valid_frames)[0]
    for frame_index in frame_indices:
        corridor = corridor_df.loc[corridor_index[frame_index]]
        frame_time = frame_times[frame_index]
        # keep the spheres appearing from the beginning of the corridor until frame time
        logger = param_logger.iloc[int(corridor.start_sample):
                                   param_logger[time_column].searchsorted(frame_time)]
        sphere_coordinates = np.array(logger[['X', 'Y', 'Z']], dtype=float)
        sphere_coordinates[:, 2] -= mouse_position[frame_index]
        output[frame_index] = draw_spheres(*sphere_coordinates.T, corridor.depth,
                                           resolution=resolution,
                                           sphere_size=sphere_size,
                                           azimuth_limits=azimuth_limits,
                                           elevation_limits=elevation_limits,
                                           output_datatype=output_datatype,
                                           output=output[frame_index])
    return output


def draw_spheres(sphere_x, sphere_y, sphere_z, depth, resolution=0.1, sphere_size=10,
                 azimuth_limits=(-120, 120),
                 elevation_limits=(-40, 40), output_datatype=bool, output=None):
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
        output_datatype (type): datatype of the output. Use bool to have binary
                                sphere/no sphere output. int for seeing sphere overlap.
                                Not used if output is provided
        output (np.array or None): if provided, must be the correct shape to match the
                                   output image. Will be modified in place.

    Returns:
        virtual_screen (np.array): an array of [elevation, azimuth] with spheres added.
    """

    radius, azimuth, elevation = cartesian_to_spherical(sphere_x, sphere_y, sphere_z)
    # we switch from trigo circle, counterclockwise with 0 on the right to azimuth,
    # clockwise with 0 in front
    az_compas = np.mod(-(azimuth - 90), 360)
    az_compas[az_compas > 180] -= 360

    # now plot the things
    azi_n = int(np.diff(azimuth_limits) / resolution)
    ele_n = int(np.diff(elevation_limits) / resolution)

    # find if the sphere is on the screen, that means in the -120 +120 azimuth range
    in_screen = (az_compas > azimuth_limits[0]) & (az_compas < azimuth_limits[1])
    # and in the -40, 40 elevation range
    in_screen = in_screen & ((elevation > elevation_limits[0]) &
                             (elevation < elevation_limits[1]))

    # convert sphere in pixel space
    sphere_on_screen = np.vstack([az_compas[in_screen],
                                  elevation[in_screen]]).T
    sphere_on_screen -= np.array([azimuth_limits[0], elevation_limits[0]])
    sphere_on_screen = sphere_on_screen / resolution
    size = depth / radius * sphere_size / resolution
    sphere_max_px = sphere_size / resolution
    footprint = np.meshgrid(*[np.arange(sphere_max_px + 1) - sphere_max_px / 2] * 2)
    if output is None:
        virtual_screen = np.zeros((ele_n, azi_n), dtype=output_datatype)
    else:
        assert output.shape == (ele_n, azi_n)
        virtual_screen = output
    for center, s in zip(sphere_on_screen, size):
        # note that np.round will round 0.5 to the nearest even value. We need floor or
        # ceil to avoid duplicating values
        az, el = [np.array(np.floor(f + c), dtype=int) for f, c in zip(footprint, center)]
        circle = (footprint[0] ** 2 + footprint[1] ** 2) <= (s / 2) ** 2
        valid = (az >= 0) & (az < azi_n) & (el >= 0) & (el < ele_n)
        virtual_screen[el[valid], az[valid]] += circle[valid]
    return virtual_screen


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
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(z, x)
    elevation = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))

    azimuth = np.degrees(azimuth)
    elevation = np.degrees(elevation)
    return radius, azimuth, elevation


def calculate_optic_flow_angle(r, r_new, distance):
    angle = np.arccos((r ** 2 + r_new ** 2 - distance ** 2) / (2 * r * r_new))
    return angle
