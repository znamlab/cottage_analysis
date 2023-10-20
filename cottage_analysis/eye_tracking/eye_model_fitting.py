"""
Fitting of eye tracking results


Code adapted from the C++ version https://github.com/LeszekSwirski/singleeyefitter
"""
import warnings
import math
import numpy as np
from numba import jit, njit, prange
from numba_progress import ProgressBar
import pandas as pd
from tqdm import tqdm
from skimage.measure import EllipseModel
import pandas as pd
import numpy as np
import flexiznam as flz
from cottage_analysis import eye_tracking
from cottage_analysis.eye_tracking import diagnostics
import cottage_analysis.eye_tracking.analysis as analeyesis


def fit_ellipses(dlc_res_file, likelihood_threshold=None):
    """Fit an ellipse to DLC set of points

    This is the first post-dlc step. Simply find the best ellipse throught the 12 points
    tracked on the pupil border

    Args:
        dlc_res_file (pandas.DataFrame or str): DLC data or path to the file containing
            them
        likelihood_threshold (float, optional): Threshold on likelihood to include
            points in fit. Defaults to None.

    Returns:
        pandas.DataFrame: Ellipse dataframe with a line per frame. Failed fit have all
            their parameters to NaN
    """
    if isinstance(dlc_res_file, pd.DataFrame):
        dlc_res = dlc_res_file
    else:
        dlc_res = pd.read_hdf(dlc_res_file)
    ellipse = EllipseModel()
    ellipse_fits = []
    for frame_id, track in tqdm(dlc_res.iterrows(), total=len(dlc_res)):
        # remove the model name
        track = track.copy()
        track.index = track.index.droplevel(0)
        xdata = track.loc[[("eye_{0}".format(pos), "x") for pos in range(1, 13)]]
        ydata = track.loc[[("eye_{0}".format(pos), "y") for pos in range(1, 13)]]
        likelihood = track.loc[
            [("eye_{0}".format(pos), "likelihood") for pos in range(1, 13)]
        ]
        if likelihood_threshold is not None:
            ok = likelihood > likelihood_threshold
            xdata = xdata[ok]
            ydata = ydata[ok]

        xy = np.vstack([xdata.values, ydata.values]).T
        success = ellipse.estimate(xy)
        if not success:
            print("Failed to fit %s" % frame_id, flush=True)
            ellipse_fits.append(
                dict(
                    centre_x=np.nan,
                    centre_y=np.nan,
                    angle=np.nan,
                    major_radius=np.nan,
                    minor_radius=np.nan,
                    error=np.nan,
                    rsquare=np.nan,
                )
            )
            continue
        xc, yc, a, b, theta = ellipse.params
        # It's a mess. see:
        # https://github.com/scikit-image/scikit-image/issues/2646
        if a < b:
            if theta < np.pi / 2:
                theta += np.pi / 2
            else:
                theta -= np.pi / 2
            a, b = b, a
        else:
            if theta < 0:
                theta += np.pi
            else:
                pass  # that's good
        residuals = ellipse.residuals(xy)
        ss_res = np.sum(residuals**2)
        error = ss_res / len(residuals)
        ss_tot = np.sum((xy - np.mean(xy, axis=0)) ** 2)
        rsquare = 1 - ss_res / ss_tot
        ellipse_fits.append(
            dict(
                centre_x=xc,
                centre_y=yc,
                angle=theta,
                major_radius=a,
                minor_radius=b,
                error=error,
                rsquare=rsquare,
            )
        )
    return pd.DataFrame(ellipse_fits)


def estimate_eye_centre(binned_frames, verbose=True):
    """Estimate the eye centre and f/z0 from binned ellipse parameters

    Args:
        binned_frames (pandas.DataFrame): Binned ellipse parameters
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        tuple: (eye_centre, f_z0)
    """
    if verbose:
        print("Find eye centre", flush=True)
    p = np.vstack([binned_frames[f"pupil_{a}"].values for a in "xy"])
    n = np.vstack(
        [np.cos(binned_frames.angle.values), np.sin(binned_frames.angle.values)]
    )
    intercept_minor = pts_intersection(p, n)
    n = np.vstack(
        [
            np.cos(binned_frames.angle + np.pi / 2),
            np.sin(binned_frames.angle + np.pi / 2),
        ]
    )
    axes_ratio = binned_frames.minor_radius.values / binned_frames.major_radius.values
    eye_centre_binned = intercept_minor.flatten()

    delta_pts = (
        np.vstack([binned_frames.pupil_x, binned_frames.pupil_y])
        - eye_centre_binned[:, np.newaxis]
    )
    sum_sqrt_ratio = np.sum(
        np.sqrt(1 - axes_ratio**2) * np.linalg.norm(delta_pts, axis=0)
    )
    sum_sq_ratio = np.sum(1 - axes_ratio**2)
    f_z0_binned = sum_sqrt_ratio / sum_sq_ratio
    if verbose:
        print(rf"Eye centre: {eye_centre_binned}. f/z0: {f_z0_binned}")
    return eye_centre_binned, f_z0_binned


def reproject_ellipses(
    camera_ds,
    target_ds,
    phi0=0,
    theta0=0,
    likelihood_threshold=0.88,
    rsquare_threshold=0.99,
    error_threshold=3,
    plot=True,
):
    """Run the reproject_eye function on a camera dataset

    DLC and ellipse fitting must have been done first

    Args:
        camera_ds (flexiznam.schema.camera_data.CameraData): Camera dataset
        target_ds (flexiznam.schema.datasets.Dataset): Target dataset
        phi0 (int, optional): Initial guess for the phi angle. Defaults to 0.
        theta0 (float, optional): Initial guess for the theta angle. Defaults to 0.
        likelihood_threshold (float, optional): Threshold on likelihood to include
            points in fit. Defaults to 0.88.
        rsquare_threshold (float, optional): Threshold on rsquare to include
            points in fit. Defaults to 0.99.
        error_threshold (float, optional): Threshold on error to include points in fit.
            Defaults to 3.
        plot (bool, optional): Plot results. Defaults to True.
    """

    # get the data
    flm_sess = camera_ds.flexilims_session
    dlc_res, data = analeyesis.get_data(
        camera_ds,
        flexilims_session=flm_sess,
        likelihood_threshold=likelihood_threshold,
        rsquare_threshold=rsquare_threshold,
        error_threshold=error_threshold,
    )
    save_folder = target_ds.path_full.parent
    # make bins of ellipse centre position
    print("Bin data", flush=True)
    elli = pd.DataFrame(data[data.valid], copy=True)
    nbins = (25, 25)
    count, bin_edges_x, bin_edges_y = np.histogram2d(
        elli.pupil_x, elli.pupil_y, bins=nbins
    )
    elli["bin_id_x"] = bin_edges_x.searchsorted(elli.pupil_x.values)
    elli["bin_id_y"] = bin_edges_y.searchsorted(elli.pupil_y.values)
    binned_ellipses = elli.groupby(["bin_id_x", "bin_id_y"])
    ns = binned_ellipses.valid.aggregate(len)
    binned_ellipses = binned_ellipses.aggregate(np.nanmedian)
    enough_frames = binned_ellipses[ns > 10]

    # PLOT
    if plot:
        dlc_tracks = eye_tracking.eye_tracking.get_tracking_datasets(
            camera_ds, flexilims_session=flm_sess
        )
        dlc_ds = dlc_tracks["cropped"]
        cropping = dlc_ds.extra_attributes["cropping"]

        diagnostics.plot_binned_ellipse_params(
            binned_ellipses,
            ns,
            save_folder,
            min_frame_cutoff=10,
            fig_title=camera_ds.full_name,
            camera_ds=camera_ds,
            cropping=cropping,
        )
    eye_centre_binned, f_z0_binned = estimate_eye_centre(enough_frames)

    # plot it
    if plot:
        diagnostics.plot_eye_centre_estimate(
            eye_centre_binned,
            f_z0_binned,
            camera_ds,
            binned_frames=enough_frames,
            cropping=cropping,
            save_folder=save_folder,
            example_frame=1000,
        )

    # fit median eye position with fine grid
    print("Fit median position", flush=True)
    most_frequent_bin = ns.idxmax()
    params_most_frequent_bin = binned_ellipses.loc[
        most_frequent_bin,
        ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"],
    ]
    p0 = (float(phi0), float(theta0), 1.0)
    params_med, e = minimise_reprojection_error(
        tuple(params_most_frequent_bin.values),
        p0,
        eye_centre_binned,
        f_z0_binned,
        p_range=(np.pi / 3, np.pi / 3, 0.5),
        grid_size=20,
        niter=5,
        reduction_factor=5,
    )
    phi, theta, radius = params_med
    # Plot fit of median position
    if plot:
        initial_model = reproj_ellipse(
            phi=phi,
            theta=theta,
            r=radius,
            eye_centre=eye_centre_binned,
            f_z0=f_z0_binned,
        )

        diagnostics.plot_reprojection(
            eye_centre_binned,
            f_z0_binned,
            dlc_res,
            fitted_params=params_most_frequent_bin,
            fitted_model=initial_model,
            cropping=cropping,
            camera_ds=camera_ds,
            target_file=save_folder / f"initial_reprojection_median_eye_position.png",
        )

    # optimise for all binned positions
    print("Reproject binned data", flush=True)
    eye_rotation_initial = np.zeros((len(enough_frames), 3))

    for i_pos, (pos, s) in tqdm(
        enumerate(enough_frames.iterrows()), total=len(enough_frames)
    ):
        ellipse_params = s[
            ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
        ].values
        p, e = minimise_reprojection_error(
            ellipse_params,
            p0=params_med,
            eye_centre=eye_centre_binned,
            f_z0=f_z0_binned,
            p_range=(np.pi / 3, np.pi / 3, 0.5),
            grid_size=10,
            niter=5,
            reduction_factor=5,
        )
        eye_rotation_initial[i_pos] = p

    if plot:
        diagnostics.plot_gaze_fit(
            binned_ellipses=enough_frames,
            eye_rotation=eye_rotation_initial,
            save_folder=save_folder,
            nbins=nbins,
        )

    # Now optimise eye_centre and f_z0
    print("Optimise eye parameters", flush=True)
    # skip to use about 40 frames to go a bit faster
    if len(enough_frames) < 40:
        skip = 1
    else:
        skip = int(np.ceil(len(enough_frames) / 40))
    source_ellipses = (
        enough_frames[::skip]
        .loc[:, ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]]
        .values
    )
    gazes = eye_rotation_initial[::skip]
    (x, y, f_z0), err = optimise_eye_parameters(
        ellipses=source_ellipses,
        gazes=gazes,
        p0=(*eye_centre_binned, f_z0_binned),
        p_range=(70.0, 70.0, 50.0),
        grid_size=7,
        niter=3,
        reduction_factor=3,
        verbose=True,
    )
    eye_centre = np.array([x, y])

    # Refit median eye position with new eye
    # needed because we want to limit the search 60 degrees around that
    params_med, e = minimise_reprojection_error(
        params_most_frequent_bin.values,
        params_med,
        eye_centre,
        f_z0,
        p_range=(np.pi / 2, np.pi / 2, 0.5),
        grid_size=20,
        niter=5,
        reduction_factor=5,
    )

    if plot:
        # replot median eye posisiton with better eye
        new_model = reproj_ellipse(*params_med, eye_centre=eye_centre, f_z0=f_z0)

        diagnostics.plot_reprojection(
            eye_centre_binned,
            f_z0_binned,
            dlc_res,
            fitted_params=params_most_frequent_bin,
            fitted_model=new_model,
            cropping=cropping,
            camera_ds=camera_ds,
            target_file=save_folder / f"optimised_reprojection_median_eye_position.png",
            initial_model=initial_model,
        )

    # SAVE Eye parameters
    print("Saving eye parameters", flush=True)
    np.savez(
        save_folder / f"eye_parameters.npz",
        eye_centre=eye_centre,
        f_z0=f_z0,
        median_eye_position_parameters=params_med,
    )

    # optimise for all frames
    print("Fitting all frames", flush=True)
    # create a list of all ellipse params
    parameters = data[
        ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
    ].values
    is_valid = data.valid.values
    with ProgressBar(total=len(parameters)) as progress:
        eye_rotation = minimise_all(
            parameters,
            is_valid,
            p0,
            eye_centre,
            f_z0,
            progress,
            p_range=(1.0, 1.0, 0.5),
            grid_size=10,
            niter=3,
            reduction_factor=3,
        )
    np.save(target_ds.path_full, eye_rotation)
    print("Done!")


@njit(parallel=True, nogil=True)
def minimise_all(
    parameters,
    is_valid,
    p0,
    eye_centre,
    f_z0,
    progress_proxy,
    p_range=(1.0, 1.0, 0.5),
    grid_size=10,
    niter=3,
    reduction_factor=3,
):
    """Run minimisation of reprojection error on all frames in parallel

    Args:

        parameters (numpy.array): N x 5 array of ellipse parameters
        is_valid (numpy.array): N array of bool indicating if ellipse is valid
        p0 (tuple): Starting estimates of parameter (phi, theta, radius), centre of grid
        eye_centre (numpy.array): x,y of eye centre in camera coordinate
        f_z0 (float): scale factor
        progress_proxy (tqdm.tqdm): Progress bar
        p_range (tuple, optional): range of grid for the 3 parameters. Defaults to
            (1, 1, 0.5)
        grid_size (int, optional): number of values for each level of the grid.
            Defaults to 10.
        niter (int, optional): number of iteration. Defaults to 3
        reduction_factor (int, optional): reduction of p_range at each iteration.
            Defaults to 3
    """
    nframes = len(parameters)
    eye_rotation = np.zeros((nframes, 3))
    eye_rotation[~is_valid] += np.nan
    for ipos in prange(nframes):
        progress_proxy.update(1)
        if not is_valid[ipos]:
            continue
        ellipse_params = parameters[ipos]
        pa, e = minimise_reprojection_error(
            ellipse_params,
            p0,
            eye_centre,
            f_z0,
            p_range,
            grid_size,
            niter,
            reduction_factor,
        )
        eye_rotation[ipos] = pa
    return eye_rotation


@njit
def minimise_reprojection_error(
    ellipse,
    p0,
    eye_centre,
    f_z0,
    p_range=(1.0, 1.0, 0.5),
    grid_size=10,
    niter=3,
    reduction_factor=3,
):
    """Iterative grid search of best gaze vector to minimize reprojection error

    Args:
        ellipse (tuple): Ellipse to fit, provided either as (x,y, major, minor, angle)
            tuple of parameters
        p0 (tuple): Starting estimates of parameter (phi, theta, radius), centre of grid
        eye_centre (numpy.array): x,y of eye centre in camera coordinate
        f_z0 (float): scale factor
        p_range (tuple, optional): range of grid for the 3 parameters. Defaults to
            (1, 1, 0.5)
        grid_size (int, optional): number of values for each level of the grid.
            Defaults to 10.
        niter (int, optional): number of iteration. Defaults to 3
        reduction_factor (int, optional): reduction of p_range at each iteration.
            Defaults to 5
        verbose (bool, optional): Print progress. Default to True.

    Returns:
        parameters (tuple): Best gaze parameters (phi, theta, radius)
        min_ind (tuple): Index of minimal error in grid for (phi, theta, radius)
        error (numpy array): len(grid_phi) x len(grid_theta) x len(grid_radius) array
            of reprojection errors
    """

    for i_iter in range(niter):
        grids = []
        for p, r in zip(p0, p_range):
            grids.append(np.linspace(-r, r, grid_size) + p)
        params, error = grid_search_best_gaze(
            ellipse,
            eye_centre=eye_centre,
            f_z0=f_z0,
            grid_phi=grids[0],
            grid_theta=grids[1],
            grid_radius=grids[2],
        )

        p_range = (
            p_range[0] / reduction_factor,
            p_range[1] / reduction_factor,
            p_range[2] / reduction_factor,
        )
    return params, error


def optimise_eye_parameters(
    ellipses,
    gazes,
    p0,
    p_range=(50.0, 50.0, 30.0),
    grid_size=10,
    niter=5,
    reduction_factor=3,
    verbose=True,
    inner_search_kwargs=None,
):
    params = tuple(p0)
    if verbose:
        p_display = np.round(params, 2)
        print(f"Initial eye parameters: {p_display}.", flush=True)
    for i_iter in range(niter):
        if verbose:
            print(f"Iteration {i_iter + 1}", flush=True)

        grids = [np.linspace(-r, r, grid_size) + p for r, p in zip(p_range, params)]
        params, error = grid_search_best_eye(
            ellipses, gazes, *grids, inner_search_kwargs
        )
        if verbose:
            p_display = np.round(params, 2)
            print(
                f"    Best eye parameters: {p_display}. Error: {error:.0f}",
                flush=True,
            )
        p_range = [p / reduction_factor for p in p_range]
    return params, error


def get_gaze_vector(phi, theta):
    """Get the gaze vector from phi and theta

    Args:
        phi (float): Angle in radians
        theta (float): Angle in radians

    Returns:
        numpy.array: 3-D vector of gaze direction in camera coordinates
    """
    return np.array(
        [np.sin(theta), np.sin(phi) * np.cos(theta), -np.cos(phi) * np.cos(theta)]
    )


def convert_to_world(gaze_vec, rmat, is_flipped=True):
    """Convert gaze vectors from camera to world coordinates

    This include weird adhoc transformation because the camera images where flipped
    vertically

    Args:
        gaze_vec (numpy.array): N x 3 array of gaze in camera coordinate
        rmat (numpy.array): 3x3 rotation matrix
        is_flipped (bool, optional): Is the image flipped vertically. Defaults to True.

    Returns:
        numpy array: N x 3 array
    """
    flipped_gaze = np.array(gaze_vec, copy=True)
    flipped_gaze[:, 1] *= -1  # to have back y going up instead of down
    rotated_gaze_vec = (rmat @ flipped_gaze.T).T
    if is_flipped:
        rotated_gaze_vec = rotated_gaze_vec[
            :, [0, 2, 1]
        ]  # because of camera mirror made it a lefthand coordinate system
    else:
        raise NotImplementedError(
            "You need to check that. Not sure if the flipped is needed"
        )

    return rotated_gaze_vec


def gaze_to_azel(gaze_vector, zero_median=False):
    """Transform gaze vectors in world coordinates to Azimuth and Elevation

    Args:
        gaze_vector (numpy.array): N x 3 array of gaze
        zero_median (bool, optional): Subtract the median. Defaults to False.

    Returns:
        azimuth (numpy.array): len(N) array of azimuth in radians
        elevation (numpy.array): len(N) array of elevation in radians
    """
    azimuth = np.arctan2(gaze_vector[:, 1], gaze_vector[:, 0])
    elevation = np.arctan2(gaze_vector[:, 2], np.sum(gaze_vector[:, :2] ** 2, axis=1))
    # zero the median pos
    if zero_median:
        azimuth -= np.nanmedian(azimuth)
        elevation -= np.nanmedian(elevation)
        # put back in -pi pi
        azimuth = np.mod(azimuth + np.pi, 2 * np.pi) - np.pi
        elevation = np.mod(elevation + np.pi, 2 * np.pi) - np.pi
    return azimuth, elevation


@jit(nopython=True)
def reproj_centre(phi, theta, eye_centre, f_z0):
    """Reproject ellipse centre on camera frame

    Wallace and Kerr method.

    There is an extra minus 1 in the y of the centre reprojection compared to their
    methods to have the camera y axis pointing down

    Args:
        phi (float): Vertical angle in radians
        theta (float): Horizontal angle in radians
        eye_centre (numpy.array): x,y position of eye centre
        f_z0 (float): Scale factor

    Returns:
        numpy.array: X, Y of pupil centre in camera coordinates
    """

    return f_z0 * np.array([np.sin(theta), -np.sin(phi) * np.cos(theta)]) + eye_centre


@njit
def reproj_ellipse(phi, theta, r, eye_centre, f_z0):
    """Reproject ellipse on camera frame

    Wallace and Kerr method

    Args:
        phi (float): Vertical angle in radians
        theta (float): Horizontal angle in radians
        r (float): Radius of pupil in units of f_z0
        eye_centre (numpy.array): x,y position of eye centre
        f_z0 (float): Scale factor

    Returns:
        EllipseModel: Ellipse in camera coordinates
    """
    w3 = -np.cos(phi) * np.cos(theta)
    major = r * f_z0
    minor = np.abs(w3) * major
    # from Wallace et al:
    if np.sin(phi) != 0:
        angle = np.arctan(np.tan(theta) / np.sin(phi))
    else:
        angle = np.pi / 2 * np.sign(np.tan(theta))
    centre = reproj_centre(phi, theta, eye_centre, f_z0)
    if False:
        # one could also look at the angle to centre
        vect = centre - eye_centre
        angle = np.arcsin(vect[0] / np.linalg.norm(vect))

    # params are xc, yc, a, b, theta
    ellipse = (centre[0], centre[1], major / 2, minor / 2, angle)
    return ellipse


@njit
def ellipse_distance(model1, model2, ev_pts=None):
    """Compute the distance between two ellipses

    This is done by summing the distances of points along the border

    Args:
        model1 (tuple): First ellipse
        model2 (tuple): Second ellipse
        ev_pts (numpy.array, optional): Angles to use for comparison. If None will do
            a full circle in pi/6 increament. Defaults to None.

    Returns:
        float: Error as sum of distances
    """
    if ev_pts is None:
        ev_pts = np.arange(0, 2 * np.pi, np.pi / 6)
    xc, yc, a, b, theta = model1
    pts1 = predict_xy(ev_pts, xc, yc, a, b, theta)
    xc, yc, a, b, theta = model2
    pts2 = predict_xy(ev_pts, xc, yc, a, b, theta)
    error = np.sum(np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1)))
    return error


def pts_intersection(pts, normals):
    """Find best interesection of lines in 2D

    See:
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_two_dimensions_2

    Args:
        pts (numpy.array): 2 x N array of points on the lines
        normals (numpy.array): 2 x N array of normals to the lines

    Returns:
        numpy.array: (x, y) of least-square solution
    """
    n_nt = normals.T[:, :, np.newaxis] @ normals.T[:, np.newaxis, :]
    inv_sum = np.linalg.inv(np.sum(n_nt, axis=0))
    direct_sum = np.sum(n_nt @ pts.T[:, :, np.newaxis], axis=0)
    return inv_sum @ direct_sum


@njit
def grid_search_best_gaze(
    source_ellipse, eye_centre, f_z0, grid_phi, grid_theta, grid_radius
):
    """Grid search of best gaze vector to minimize reprojection error

    Args:
        source_ellipse (tuple): Ellipse to fit, (x,y, major, minor, angle) tuple of
            parameters
        eye_centre (numpy.array): x,y of eye centre in camera coordinate
        f_z0 (float): scale factor
        grid_phi (numpy.array): Values of phi for grid search
        grid_theta (numpy.array): Values of theta for grid search
        grid_radius (numpy.array): Values of radius for grid search

    Returns:
        parameters (tuple): Best gaze parameters (phi, theta, radius)
        min_ind (tuple): Index of minimal error in grid for (phi, theta, radius)
        error (numpy array): len(grid_phi) x len(grid_theta) x len(grid_radius) array
            of reprojection errors
    """
    params = (0, 0, 0)
    error = np.inf
    for phi in grid_phi:
        for theta in grid_theta:
            for r in grid_radius:
                el = reproj_ellipse(phi, theta, r, eye_centre=eye_centre, f_z0=f_z0)
                dst = ellipse_distance(source_ellipse, el)
                if dst < error:
                    error = dst
                    params = (phi, theta, r)
    return params, error


def grid_search_best_eye(
    source_ellipses,
    ellipse_fits,
    grid_eye_x,
    grid_eye_y,
    grid_f_z0,
    inner_search_kwargs=None,
):
    """Optimise eye parameters by grid search

    Grid search on eye parameters (center x, y and f/z0 scale). For each combination,
    optimise phi/theta/radius for all source_ellipses and sum reprojection errors

    Args:
        source_ellipses (list): List of ellipses or ellipse parameter, input data
        ellipse_fits (list): List of phi/theta/radius parameters to initial search for
            each source_ellipse
        grid_eye_x (numpy.array): List of x values to test
        grid_eye_y (numpy.array): List of y values to test
        grid_f_z0 (numpy.array): List of f_z0 values to test
        inner_search_kwargs (dict, optional): Parameters of inner search. If None will
            use: p_range=(np.deg2rad(30), np.deg2rad(30), 0.2), niter=3, and grid_size=5
            Defaults to None.

    Returns:
        params (tuple): Best (x, y, f_z0) eye parameters
        index (tuple): Index of best parameter in grid
        errors (numpy.array): Matrix of error for each position in the grid
    """

    inner_search_params = dict(
        p_range=(np.deg2rad(30), np.deg2rad(30), 0.2),
        niter=3,
        grid_size=5,
    )
    if inner_search_kwargs is not None:
        for k, v in inner_search_kwargs.items():
            if k not in inner_search_params:
                warnings.warn(f"Unknown parameter for inner loop: {k}")
            else:
                inner_search_kwargs[k] = v

    best_eye = (0, 0, 0)
    best_error = np.inf
    for x in grid_eye_x:
        for y in grid_eye_y:
            for fz in grid_f_z0:
                error = 0
                for ellipse, fit_params in zip(source_ellipses, ellipse_fits):
                    _, single_error = minimise_reprojection_error(
                        ellipse,
                        p0=fit_params,
                        eye_centre=np.array([x, y]),
                        f_z0=fz,
                        **inner_search_params,
                    )
                    error += single_error
                if error < best_error:
                    best_error = error
                    best_eye = (x, y, fz)
    return best_eye, best_error


@njit
def predict_xy(t, xc, yc, a, b, theta):
    """Predict x- and y-coordinates using the estimated model.

    This is extracted from EllipseModel to avoid unessesary checks

    Parameters
    ----------
    t : array
        Angles in circle in radians. Angles start to count from positive
        x-axis to positive y-axis in a right-handed system.
    params : (5, ) array, optional
        Optional custom parameter set.

    Returns
    -------
    xy : (..., 2) array
        Predicted x- and y-coordinates.

    """

    ct = np.cos(t)
    st = np.sin(t)
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    x = xc + a * ctheta * ct - b * stheta * st
    y = yc + a * stheta * ct + b * ctheta * st

    return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)
