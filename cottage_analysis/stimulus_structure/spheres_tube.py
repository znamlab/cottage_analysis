import jax
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp


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
    depth = jnp.round(param_logger.Depth.values, 2)
    trials_border = jnp.diff(jnp.hstack([-9999, depth]))
    trials_onset = jnp.where(trials_border > 5000)[0]
    trials_offset = jnp.where(trials_border < -5000)[0]
    n_corridors = len(trials_onset)
    if len(trials_offset) == n_corridors - 1:  # the last corridor is cut
        trials_offset = jnp.hstack([trials_offset, len(depth) - 1])
    elif len(trials_offset) != n_corridors:
        raise IOError('Found %d corridor starts but %d ends... ' % (n_corridors,
                                                                    len(trials_offset)))

    corridor_df = dict(trial=jnp.arange(n_corridors), depth=depth[trials_onset],
                       start_sample=jnp.array(trials_onset, dtype=int),
                       end_sample=jnp.array(trials_offset, dtype=int),
                       start_time=param_logger.loc[trials_onset, time_column].values,
                       end_time=param_logger.loc[trials_offset, time_column].values)
    return pd.DataFrame(corridor_df)


def regenerate_frames(frame_times, param_logger, mouse_pos_cm, mouse_pos_time,
                      corridor_df=None, time_column='HarpTime', resolution=1,
                      sphere_size=10, azimuth_limits=(-120, 120),
                      elevation_limits=(-40, 40),
                      verbose=True, output_datatype='int16', output=None):
    """Regenerate frames of sphere stimulus

    Args:
        frame_times (jnp.array): Array of time at which the frame should be regenerated
        param_logger (pd.DataFrame): Params saved by Bonsai logger
        mouse_pos_cm (jnp.array): position of the mouse in cm
        mouse_pos_time (jnp.array): time of each mouse_pos_cm sample
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
        output (jnp.array): Array to add output. Will be done inplace

    Returns:
        virtual_screen (jnp.array): an array of [elevation, azimuth] with spheres added.
    """
    frame_times = jnp.array(frame_times, ndmin=1)

    if corridor_df is None:
        corridor_df = trial_structure(param_logger, time_column)
    out_shape = (len(frame_times), 
                 int((elevation_limits[1]-elevation_limits[0]) / resolution),
                 int((azimuth_limits[1] - azimuth_limits[0]) / resolution))
    if output is None:
        output = jnp.zeros(out_shape, dtype=output_datatype)
    else:
        assert output.shape == out_shape

    # for frames before and after the protocol, keep them 0s
    before = frame_times < corridor_df.start_time.iloc[0]
    after = frame_times > corridor_df.end_time.iloc[-1]
    if verbose:
        print('Ignoring %d frames before and %d after the stimulus presentation'
              % (jnp.sum(before), jnp.sum(after)))
    valid_frames = ~before & ~after

    corridor_index = corridor_df.start_time.searchsorted(frame_times, side='right') - 1
    corridor_index = jnp.clip(corridor_index, 0, len(corridor_df) - 1)
    corridor_end = corridor_df.loc[corridor_index, 'end_time'].values
    grey_time = frame_times - corridor_end > 0
    if verbose:
        print('Ignoring %d frames in grey inter-trial intervals' % jnp.sum(grey_time &
                                                                          valid_frames))
    valid_frames = valid_frames & (~grey_time)
    mouse_position = mouse_pos_cm[mouse_pos_time.searchsorted(frame_times)]
    frame_indices = jnp.where(valid_frames)[0]
    # now process the valid frames
    log_ends = param_logger[time_column].searchsorted(frame_times)
    draw_sph_jit = jax.jit(draw_spheres)
    for frame_index in frame_indices:
        corridor = corridor_df.loc[int(corridor_index[frame_index])]
        logger = param_logger.iloc[int(corridor.start_sample): log_ends[frame_index]]
        sphere_coordinates = jnp.array(logger[['X', 'Y', 'Z']].values, dtype=float)
        sphere_coordinates = sphere_coordinates.at[:, 2].set(sphere_coordinates[:, 2] - mouse_position[frame_index])
        this_frame = draw_spheres(
                     sphere_x=sphere_coordinates[:, 0],
                     sphere_y=sphere_coordinates[:, 1],
                     sphere_z=sphere_coordinates[:, 2],
                     depth=float(corridor.depth),
                     resolution=float(resolution),
                     sphere_size=float(sphere_size),
                     azimuth_limits=jnp.array(azimuth_limits, dtype=float),
                     elevation_limits=jnp.array(elevation_limits, dtype=float))
        output[frame_index] = this_frame
    return output




def draw_spheres(sphere_x, sphere_y, sphere_z, depth, resolution=0.1,
                 sphere_size=10, azimuth_limits=(-120, 120), elevation_limits=(-40, 40)):
    """Recreate stimulus for a single frame from corrected sphere position

    Given the positions of the spheres relative to the mouse and the corridor depth,
    recreate a single frame

    Args:
        sphere_x (jnp.array): X positions for all spheres on the frame
        sphere_y (jnp.array): Y positions for all spheres on the frame
        sphere_z (jnp.array): Z positions for all spheres on the frame
        depth (float): Depth for that corridor. Used for size adjustement
        resolution (float): size of a pixel in degrees
        sphere_size (float): size of a sphere in degrees
        azimuth_limits ([float, float]): Minimum and maximum azimuth of the display
        elevation_limits ([float, float]): Minimum and maximum elevation of the display


    Returns:
        virtual_screen (jnp.array): an array of [elevation, azimuth] with spheres added.
    """

    radius, azimuth, elevation = cartesian_to_spherical(sphere_x, sphere_y, sphere_z)
    # we switch from trigo circle, counterclockwise with 0 on the right to azimuth,
    # clockwise with 0 in front
    az_compas = jnp.mod(-(azimuth - 90), 360)
    az_compas = az_compas.at[az_compas > 180].set(az_compas[az_compas > 180] - 360)

    # now prepare output
    azi_n = int((azimuth_limits[1]-azimuth_limits[0]) / resolution)
    ele_n = int((elevation_limits[1]-elevation_limits[0]) / resolution)

    # find if the sphere is on the screen, that means in the -120 +120 azimuth range
    in_screen = (az_compas > azimuth_limits[0]) & (az_compas < azimuth_limits[1])
    # and in the -40, 40 elevation range
    in_screen = in_screen & ((elevation > elevation_limits[0]) &
                             (elevation < elevation_limits[1]))
    if not jnp.any(in_screen):
        return

    # convert `in_screen` spheres in pixel space
    az_on_screen = (az_compas[in_screen] - azimuth_limits[0]) / resolution
    el_on_screen = (elevation[in_screen] - elevation_limits[0]) / resolution
    size = depth / radius[in_screen] * sphere_size / resolution

    xx, yy = jnp.meshgrid(jnp.arange(azi_n), jnp.arange(ele_n))
    xx = jnp.outer(xx.reshape(-1), jnp.ones(len(az_on_screen)))
    yy = jnp.outer(yy.reshape(-1), jnp.ones(len(el_on_screen)))
    ok = (xx - az_on_screen)**2 + (yy - el_on_screen)**2 - size**2
    ok = ok <= 0
    return jnp.any(ok, axis=1).reshape((ele_n, azi_n))



def cartesian_to_spherical(x, y, z):
    """Transform cartesian X, Y, Z bonsai coordinate to spherical

    Args:
        x (jnp.array): x position from bonsai. Positive is to the right of the mouse
        y (jnp.array): y position from bonsai. Positive is above the mouse
        z (jnp.array): z position from bonsai. Positive is in front of the mouse

    Returns:
        radius (jnp.array): radius, same unit as x,y,z
        azimuth (jnp.array): azimuth angle in trigonometric coordinates (0 is to the
                            right of the mouse, positive is counterclockwise, towards
                            the nose)
        elevation (jnp.array): elevation angle. 0 is in front of the mouse, positive
                              towards the top.
    """
    radius = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = jnp.arctan2(z, x)
    elevation = jnp.arctan2(y, jnp.sqrt(x ** 2 + z ** 2))

    azimuth = jnp.degrees(azimuth)
    elevation = jnp.degrees(elevation)
    return radius, azimuth, elevation


def calculate_optic_flow_angle(r, r_new, distance):
    angle = jnp.arccos((r ** 2 + r_new ** 2 - distance ** 2) / (2 * r * r_new))
    return angle



def _meshgrid(x, y):
    xx = jnp.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = jnp.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j, k] = k  # change to x[k] if indexing xy
            yy[j, k] = j  # change to y[j] if indexing xy
    return xx, yy
