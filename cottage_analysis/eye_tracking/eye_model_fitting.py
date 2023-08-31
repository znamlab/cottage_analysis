"""
Fitting of eye tracking results


Code adapted from the C++ version https://github.com/LeszekSwirski/singleeyefitter
"""
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import EllipseModel
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import flexiznam as flz
from cottage_analysis import eye_tracking
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


def reproject_ellipses(camera_ds, target_ds, phi0=0, theta0=0, plot=True):
    """Run the reproject_eye function on a camera dataset

    DLC and ellipse fitting must have been done first

    Args:
        camera_ds (flexiznam.schema.camera_data.CameraData): Camera dataset
        target_ds (flexiznam.schema.datasets.Dataset): Target dataset
        theta0 (float, optional): Initial guess for the theta angle. Defaults to 0.
        phi0 (int, optional): Initial guess for the phi angle. Defaults to 0.
    """

    # get the data
    flm_sess = camera_ds.flexilims_session
    dlc_res, data = analeyesis.get_data(
        camera_ds,
        flexilims_session=flm_sess,
        likelihood_threshold=0.88,
        rsquare_threshold=0.99,
        error_threshold=3,
    )
    save_folder = target_ds.path_full.parent
    save_folder.mkdir(exist_ok=True)

    # make bins of ellipse centre position
    print("Bin data", flush=True)
    elli = pd.DataFrame(data[data.valid], copy=True)
    count, bin_edges_x, bin_edges_y = np.histogram2d(
        elli.pupil_x, elli.pupil_y, bins=(25, 25)
    )
    elli["bin_id_x"] = bin_edges_x.searchsorted(elli.pupil_x.values)
    elli["bin_id_y"] = bin_edges_y.searchsorted(elli.pupil_y.values)
    binned_ellipses = elli.groupby(["bin_id_x", "bin_id_y"])
    ns = binned_ellipses.valid.aggregate(len)
    binned_ellipses = binned_ellipses.aggregate(np.nanmedian)
    enough_frames = binned_ellipses[ns > 10]

    # PLOT
    if plot:
        mat = np.zeros((len(ns.index.levels[0]), len(ns.index.levels[1]))) + np.nan
        fig = plt.figure(figsize=(15, 5))
        for ip, p in enumerate(["angle", "minor_radius", "major_radius"]):
            mat[
                enough_frames.index.get_level_values(0),
                enough_frames.index.get_level_values(1),
            ] = enough_frames[p]
            lim = np.nanquantile(mat, [0.01, 0.99])
            ax = fig.add_subplot(1, 3, ip + 1)
            img = ax.imshow(mat, vmin=lim[0], vmax=lim[1])
            fig.colorbar(img, ax=ax)
            ax.set_title(p)
        fig.suptitle(camera_ds.dataset_name)
        fig.savefig(save_folder / f"binned_pupil_params.png", dpi=600)
        plt.close(fig)

    # Find eye centre
    print("Find eye centre", flush=True)
    p = np.vstack([enough_frames[f"pupil_{a}"].values for a in "xy"])
    n = np.vstack(
        [np.cos(enough_frames.angle.values), np.sin(enough_frames.angle.values)]
    )
    intercept_minor = pts_intersection(p, n)
    n = np.vstack(
        [
            np.cos(enough_frames.angle + np.pi / 2),
            np.sin(enough_frames.angle + np.pi / 2),
        ]
    )
    axes_ratio = enough_frames.minor_radius.values / enough_frames.major_radius.values
    eye_centre_binned = intercept_minor.flatten()

    delta_pts = (
        np.vstack([enough_frames.pupil_x, enough_frames.pupil_y])
        - eye_centre_binned[:, np.newaxis]
    )
    sum_sqrt_ratio = np.sum(
        np.sqrt(1 - axes_ratio**2) * np.linalg.norm(delta_pts, axis=0)
    )
    sum_sq_ratio = np.sum(1 - axes_ratio**2)
    f_z0_binned = sum_sqrt_ratio / sum_sq_ratio
    print(rf"Eye centre: {eye_centre_binned}. f/z0: {f_z0_binned}")
    # plot it
    if plot:
        start_frame = 1000
        video_file = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
        dlc_tracks = eye_tracking.eye_tracking.get_tracking_datasets(
            camera_ds, flexilims_session=flm_sess
        )
        dlc_ds = dlc_tracks["cropped"]
        cropping = dlc_ds.extra_attributes["cropping"]
        cam_data = cv2.VideoCapture(str(video_file))
        cam_data.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        ret, frame = cam_data.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[cropping[2] : cropping[3], cropping[0] : cropping[1]]
        cam_data.release()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        img = ax.imshow(gray, cmap="gray")
        fig.colorbar(img, ax=ax)
        track = dlc_res.loc[start_frame]
        track.index = track.index.droplevel(["scorer"])

        for i, series in enough_frames.iterrows():
            origin = np.array([series.pupil_x, series.pupil_y])
            ref = np.array([series.reflection_x, series.reflection_y])
            n_v = np.array(
                [np.cos(series.angle + np.pi / 2), np.sin(series.angle + np.pi / 2)]
            )
            rng = np.array([-200, 200])
            ax.plot(
                *[(origin[a] + ref[a] + n_v[a] * rng) for a in range(2)],
                color="purple",
                alpha=0.1,
                lw=1,
            )
        ax.plot(*(eye_centre_binned + ref), color="g", marker="o")
        eye_binned = mpl.patches.Circle(
            xy=(eye_centre_binned + ref),
            radius=f_z0_binned,
            facecolor="none",
            edgecolor="g",
        )
        ax.add_artist(eye_binned)
        ax.set_xlim(0, gray.shape[1])
        _ = ax.set_ylim(gray.shape[0], 0)
        fig.savefig(save_folder / f"eye_centre_estimate.png")
        plt.close(fig)

    # fit median eye position with fine grid
    print("Fit median position", flush=True)
    most_frequent_bin = ns.idxmax()
    params_most_frequent_bin = binned_ellipses.loc[
        most_frequent_bin,
        ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"],
    ]
    p0 = (phi0, theta0, 1)
    params_med, i, e = minimise_reprojection_error(
        params_most_frequent_bin,
        p0,
        eye_centre_binned,
        f_z0_binned,
        p_range=(np.pi / 3, np.pi / 3, 0.5),
        grid_size=20,
        niter=5,
        reduction_factor=5,
        verbose=True,
    )
    phi, theta, radius = params_med
    # Plot fit of median position
    if plot:
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(gray, cmap="gray")
        source_model = EllipseModel()
        source_model.params = params_most_frequent_bin
        circ_coord = source_model.predict_xy(
            np.arange(0, 2 * np.pi, 0.1)
        ) + ref.reshape(1, 2)
        ax.plot(circ_coord[:, 0], circ_coord[:, 1], label="DLC fit", color="lightblue")
        ax.plot(*(eye_centre_binned + ref), color="g", marker="o", label="Eye centre")
        eye_binned = mpl.patches.Circle(
            xy=(eye_centre_binned + ref),
            radius=f_z0_binned,
            facecolor="none",
            edgecolor="g",
            label=r"$\frac{f}{z_0}$",
        )
        ax.add_artist(eye_binned)

        fitted_model = reproj_ellipse(
            phi=phi,
            theta=theta,
            r=radius,
            eye_centre=eye_centre_binned,
            f_z0=f_z0_binned,
        )
        circ_coord = fitted_model.predict_xy(
            np.arange(0, 2 * np.pi, 0.1)
        ) + ref.reshape(1, 2)
        ax.plot(
            circ_coord[:, 0],
            circ_coord[:, 1],
            label="Reprojection",
            color="purple",
            ls="--",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig(save_folder / f"initial_reprojection_median_eye_position.png")
        plt.close(fig)

    # optimise for all binned positions
    print("Reproject binned data", flush=True)
    eye_rotation_initial = np.zeros((len(enough_frames), 3))

    for i_pos, (pos, s) in enumerate(enough_frames.iterrows()):
        ellipse_params = s[
            ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
        ]
        p, i, e = minimise_reprojection_error(
            ellipse_params,
            p0=params_med,
            eye_centre=eye_centre_binned,
            f_z0=f_z0_binned,
            p_range=(np.pi / 3, np.pi / 3, 0.5),
            grid_size=10,
            niter=5,
            reduction_factor=5,
            verbose=False,
        )
        eye_rotation_initial[i_pos] = p
    if plot:
        mat = (
            np.zeros(
                (
                    len(binned_ellipses.index.levels[0]),
                    len(binned_ellipses.index.levels[1]),
                    3,
                )
            )
            + np.nan
        )
        for i_pos, (pos, _) in enumerate(enough_frames.iterrows()):
            mat[pos[0], pos[1]] = eye_rotation_initial[i_pos]
        fig = plt.figure(figsize=(15, 4))
        labels = ["phi", "theta", "radius"]
        for i in range(3):
            plt.subplot(1, 3, 1 + i)
            if i < 2:
                d = np.rad2deg(mat[..., i])
            else:
                d = mat[..., i]
            plt.imshow(d)
            plt.title(labels[i])
            plt.colorbar()
        fig.savefig(save_folder / f"initial_gaze_fit.png")
        plt.close(fig)

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
    gazes = eye_rotation_initial[::4]
    (x, y, f_z0), ind, err = optimise_eye_parameters(
        ellipses=source_ellipses,
        gazes=gazes,
        p0=(*eye_centre_binned, f_z0_binned),
        p_range=(70, 70, 50),
        grid_size=7,
        niter=3,
        reduction_factor=3,
        verbose=True,
    )
    eye_centre = np.array([x, y])

    # Refit median eye position with new eye
    # needed because we want to limit the search 60 degrees around that
    params_med, i, e = minimise_reprojection_error(
        params_most_frequent_bin,
        params_med,
        eye_centre,
        f_z0,
        p_range=(np.pi / 2, np.pi / 2, 0.5),
        grid_size=20,
        niter=5,
        reduction_factor=5,
        verbose=True,
    )

    if plot:
        # replot median eye posisiton with better eye
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        ax.imshow(gray, cmap="gray")
        source_model = EllipseModel()
        source_model.params = params_most_frequent_bin
        circ_coord = source_model.predict_xy(
            np.arange(0, 2 * np.pi, 0.1)
        ) + ref.reshape(1, 2)
        ax.scatter(*(eye_centre_binned + ref), color="g", label="Eye centre (initial)")
        ax.scatter(
            *(eye_centre + ref), ec="purple", label="Eye centre (optimised)", fc="None"
        )
        eye_binned = mpl.patches.Circle(
            xy=(eye_centre_binned + ref),
            radius=f_z0_binned,
            facecolor="none",
            edgecolor="g",
            label=r"$\frac{f}{z_0}$ (initial)",
        )
        ax.add_artist(eye_binned)

        eye_binned = mpl.patches.Circle(
            xy=(eye_centre + ref),
            radius=f_z0,
            facecolor="none",
            edgecolor="purple",
            ls="--",
            label=r"$\frac{f}{z_0}$ (optimised)",
        )
        ax.add_artist(eye_binned)

        ax.plot(circ_coord[:, 0], circ_coord[:, 1], label="DLC fit", color="lightblue")
        # use phi/theta/radius to get original estimate
        fitted_model = reproj_ellipse(
            *(phi, theta, radius), eye_centre=eye_centre_binned, f_z0=f_z0_binned
        )
        circ_coord = fitted_model.predict_xy(
            np.arange(0, 2 * np.pi, 0.1)
        ) + ref.reshape(1, 2)
        ax.plot(
            circ_coord[:, 0],
            circ_coord[:, 1],
            label="Original reprojection",
            color="orange",
            ls="--",
        )

        fitted_model = reproj_ellipse(*params_med, eye_centre=eye_centre, f_z0=f_z0)
        circ_coord = fitted_model.predict_xy(
            np.arange(0, 2 * np.pi, 0.1)
        ) + ref.reshape(1, 2)
        ax.plot(
            circ_coord[:, 0],
            circ_coord[:, 1],
            label="Optimised reprojection",
            color="purple",
            ls=":",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        fig.savefig(save_folder / f"optimised_reprojection_median_eye_position.png")
        plt.close(fig)

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
    eye_rotation = np.zeros((len(data), 3))
    eye_rotation[~data.valid] += np.nan

    for i_pos, series in data.iterrows():
        if np.mod(i_pos, 1000) == 0:
            percent = i_pos / len(eye_rotation) * 100
            print(f"{percent:.2f}%", flush=True)
        if not series.valid:
            continue
        ellipse_params = series[
            ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
        ]
        pa, i, e = minimise_reprojection_error(
            ellipse_params,
            p0=params_med,
            eye_centre=eye_centre,
            f_z0=f_z0,
            p_range=(np.pi / 3, np.pi / 3, 0.5),
            grid_size=10,
            niter=3,
            reduction_factor=5,
            verbose=False,
        )
        eye_rotation[i_pos] = pa
    np.save(ds.path_full, eye_rotation)
    print("Done!")


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
