"""
Fitting of eye tracking results


Code adapted from the C++ version https://github.com/LeszekSwirski/singleeyefitter
"""
import warnings
import numpy as np
import pandas as pd
from skimage.measure import EllipseModel


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
    for frame_id, track in dlc_res.iterrows():
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


def minimise_reprojection_error(
    ellipse,
    p0,
    eye_centre,
    f_z0,
    p_range=(1, 1, 0.5),
    grid_size=10,
    niter=3,
    reduction_factor=3,
    verbose=True,
):
    """Iterative grid search of best gaze vector to minimize reprojection error

    Args:
        ellipse (EllipseModel or tuple): Ellipse to fit, provided either as model
            or as its (x,y, major, minor, angle) tuple of parameters
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
    if not isinstance(ellipse, EllipseModel):
        model1 = EllipseModel()
        model1.params = ellipse
    else:
        model1 = ellipse

    params = tuple(p0)
    for i_iter in range(niter):
        if verbose:
            print(f"Iteration {i_iter + 1}")
        grids = [np.linspace(-r, r, grid_size) + p for p, r in zip(params, p_range)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            params, ind, errors = grid_search_best_gaze(
                ellipse,
                eye_centre=eye_centre,
                f_z0=f_z0,
                grid_phi=grids[0],
                grid_theta=grids[1],
                grid_radius=grids[2],
            )
        if verbose:
            p_display = np.round(params, 2)
            print(f"    Best gaze: {p_display}. Error: {errors[ind]:.0f}")
        p_range = [p / reduction_factor for p in p_range]
    return params, ind, errors


def optimise_eye_parameters(
    ellipses,
    gazes,
    p0,
    p_range=(50, 50, 30),
    grid_size=10,
    niter=5,
    reduction_factor=3,
    verbose=True,
    inner_search_kwargs=None,
):
    source_ellipses = list(ellipses)
    for i in range(len(source_ellipses)):
        source_ellipse = source_ellipses[i]
        if not isinstance(source_ellipse, EllipseModel):
            model1 = EllipseModel()
            p = tuple(source_ellipse)
            assert len(p) == 5
            model1.params = p
        else:
            model1 = source_ellipse
        source_ellipses[i] = model1

    params = tuple(p0)
    if verbose:
        p_display = np.round(params, 2)
        print(f"Initial eye parameters: {p_display}.", flush=True)
    for i_iter in range(niter):
        if verbose:
            print(f"Iteration {i_iter + 1}", flush=True)

        grids = [np.linspace(-r, r, grid_size) + p for r, p in zip(p_range, params)]
        params, ind, errors = grid_search_best_eye(
            source_ellipses, gazes, *grids, inner_search_kwargs
        )
        if verbose:
            p_display = np.round(params, 2)
            print(
                f"    Best eye parameters: {p_display}. Error: {errors[ind]:.0f}",
                flush=True,
            )
        p_range = [p / reduction_factor for p in p_range]
    return params, ind, errors


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
    ellipse = EllipseModel()
    # params are xc, yc, a, b, theta
    ellipse.params = (centre[0], centre[1], major / 2, minor / 2, angle)
    return ellipse


def ellipse_distance(model1, model2, ev_pts=None):
    """Compute the distance between two ellipses

    This is done by summing the distances of points along the border

    Args:
        model1 (EllipseModel): First ellipse
        model2 (EllipseModel): Second ellipse
        ev_pts (numpy.array, optional): Angles to use for comparison. If None will do
            a full circle in pi/6 increament. Defaults to None.

    Returns:
        float: Error as sum of distances
    """
    if ev_pts is None:
        ev_pts = np.arange(0, 2 * np.pi, np.pi / 6)
    pts1 = model1.predict_xy(ev_pts)
    pts2 = model2.predict_xy(ev_pts)
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


def grid_search_best_gaze(
    source_ellipse, eye_centre, f_z0, grid_phi, grid_theta, grid_radius
):
    """Grid search of best gaze vector to minimize reprojection error

    Args:
        source_ellipse (EllipseModel or tuple): Ellipse to fit, provided either as model
            or as its (x,y, major, minor, angle) tuple of parameters
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
    if not isinstance(source_ellipse, EllipseModel):
        model1 = EllipseModel()
        model1.params = source_ellipse
    else:
        model1 = source_ellipse
    out = np.zeros((len(grid_phi), len(grid_theta), len(grid_radius)))
    for ip, phi in enumerate(grid_phi):
        for it, theta in enumerate(grid_theta):
            for ir, r in enumerate(grid_radius):
                el = reproj_ellipse(phi, theta, r, eye_centre=eye_centre, f_z0=f_z0)
                out[ip, it, ir] = ellipse_distance(model1, el)
    ind = np.unravel_index(np.nanargmin(out, axis=None), out.shape)
    phi = grid_phi[ind[0]]
    theta = grid_theta[ind[1]]
    radius = grid_radius[ind[2]]
    return (phi, theta, radius), ind, out


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
        verbose=False,
    )
    if inner_search_kwargs is not None:
        for k, v in inner_search_kwargs.items():
            if k not in inner_search_params:
                warnings.warn(f"Unknown parameter for inner loop: {k}")
            else:
                inner_search_kwargs[k] = v

    source_ellipses = list(source_ellipses)
    for i in range(len(source_ellipses)):
        source_ellipse = source_ellipses[i]
        if not isinstance(source_ellipse, EllipseModel):
            model1 = EllipseModel()
            model1.params = source_ellipse
        else:
            model1 = source_ellipse
        source_ellipses[i] = model1

    out = np.zeros((len(grid_eye_x), len(grid_eye_y), len(grid_f_z0)))
    for ix, x in enumerate(grid_eye_x):
        for iy, y in enumerate(grid_eye_y):
            for ifz, fz in enumerate(grid_f_z0):
                error = 0
                for ellipse, fit_params in zip(source_ellipses, ellipse_fits):
                    _, ind, errs = minimise_reprojection_error(
                        ellipse,
                        p0=fit_params,
                        eye_centre=np.array([x, y]),
                        f_z0=fz,
                        **inner_search_params,
                    )
                    error += errs[ind]
                out[ix, iy, ifz] = error
    ind = np.unravel_index(np.nanargmin(out, axis=None), out.shape)
    x = grid_eye_x[ind[0]]
    y = grid_eye_y[ind[1]]
    f_z0 = grid_f_z0[ind[2]]
    return (x, y, f_z0), ind, out