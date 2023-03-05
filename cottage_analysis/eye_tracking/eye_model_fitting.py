"""
Fitting of eye tracking results


Code adapted from the C++ version https://github.com/LeszekSwirski/singleeyefitter
"""
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
            ))
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


def reproj_centre(phi, theta, eye_centre, f_z0):
    """Reproject ellipse centre on camera frame

    Wallace and Kerr method.

    Args:
        phi (float): Angle in radians
        theta (float): Angle in radians
        eye_centre (numpy.array): x,y position of eye centre
        f_z0 (float): Scale factor

    Returns:
        numpy.array: X, Y of pupil centre in camera coordinates
    """
    return f_z0 * np.array([np.sin(theta), np.sin(phi)+np.cos(theta)]) + eye_centre

def reproj_ellipse(phi, theta, r, eye_centre, f_z0):
    """Reproject ellipse on camera frame

    Wallace and Kerr method

    Args:
        phi (float): Angle in radians
        theta (float): Angle in radians
        r (float): Radius of pupil in units of f_z0
        eye_centre (numpy.array): x,y position of eye centre
        f_z0 (float): Scale factor

    Returns:
        EllipseModel: Ellipse in camera coordinates
    """
    w3 = -np.cos(phi)*np.cos(theta)
    major = r * f_z0
    minor = np.abs(w3) * major
    # from Wallace et al: not sure what it is
    # angle = np.arctan(np.tan(theta)/np.sin(phi))
    centre = reproj_centre(phi, theta, eye_centre, f_z0)
    vect = centre - eye_centre
    angle = np.arcsin(vect[0]/np.linalg.norm(vect))
    ellipse = EllipseModel()
    # params are xc, yc, a, b, theta
    ellipse.params = (centre[0], centre[1], major/2, minor/2, angle)
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
        ev_pts = np.arange(0, 2*np.pi, np.pi/6)
    pts1 = model1.predict_xy(ev_pts)
    pts2 = model2.predict_xy(ev_pts)
    error = np.sum(np.sqrt(np.sum((pts1 - pts2)**2, axis=1)))
    return error

def grid_fit_ellipse(source_ellipse, eye_centre, f_z0, grid_phi, grid_theta, grid_radius):
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
