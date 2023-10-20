"""Functions to save diagnostic plots for eye tracking"""
import cv2
import yaml
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import EllipseModel

import flexiznam as flz
from znamutils import slurm_it

from cottage_analysis.eye_tracking import analysis
import cottage_analysis.eye_tracking.eye_tracking as cottage_tracking


def check_cropping(dlc_ds, camera_ds, rotate180=False, conflicts="skip"):
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    dlc_results = pd.read_hdf(dlc_file)

    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if not crop_file.exists() or conflicts != "skip":
        print(f"Creating {crop_file}")
        cottage_tracking.create_crop_file(camera_ds, dlc_ds, conflicts=conflicts)

    with open(crop_file, "r") as f:
        crop_info = yaml.safe_load(f)

    cam_data = cv2.VideoCapture(str(video_path))
    fid = int(cam_data.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
    cam_data.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
    ret, frame = cam_data.read()
    cam_data.release()
    if dlc_ds.extra_attributes["cropping"] is not None:
        crop = dlc_ds.extra_attributes["cropping"]
        if isinstance(crop, list):
            frame = frame[crop[2] : crop[3], crop[0] : crop[1]]
        else:
            print(f"Cropping is not a list. It is {crop}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(frame)
    ax.set_title(f"{camera_ds.dataset_name}, frame {fid}")
    frame_tracking = dlc_results.median(axis=0)
    frame_tracking.index = frame_tracking.index.droplevel("scorer")
    if "eye_0" not in frame_tracking:
        eye_tracking = frame_tracking["eye_12"]
    else:
        eye_tracking = frame_tracking["eye_0"]
    sc = ax.scatter(eye_tracking.x, eye_tracking.y, s=20, label="Pupil")

    corners = [
        "temporal_eye_corner",
        "nasal_eye_corner",
        "top_eye_lid",
        "bottom_eye_lid",
    ]
    for color, corner in zip("rgbk", corners):
        data = frame_tracking[corner]
        ax.plot(data["x"], data["y"], marker="o", ls="none", color=color, label=corner)

    rec = plt.Rectangle(
        (crop_info["xmin"], crop_info["ymin"]),
        crop_info["xmax"] - crop_info["xmin"],
        crop_info["ymax"] - crop_info["ymin"],
        fill=False,
        color="orange",
        lw=3,
    )
    ax.add_patch(rec)
    if rotate180:
        ax.invert_yaxis()
        ax.invert_xaxis()
    ax.legend()

    target = dlc_ds.path_full / "diagnostic_cropping.png"
    fig.savefig(target, dpi=300)
    return fig


def plot_dlc_tracking(camera_ds, dlc_ds, likelihood_threshold=None):
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()


@slurm_it(
    conda_env="cottage_analysis",
    slurm_options=dict(
        ntasks=1,
        time="72:00:00",
        mem="16G",
        partition="cpu",
    ),
)
def plot_ellipse_fit(
    camera_ds_name,
    project,
    likelihood_threshold=None,
    start_frame=None,
    duration=None,
    playback_speed=4,
    vmin=None,
    vmax=None,
):
    """Plot ellipse fit for a given camera dataset

    The generated movie
    Args:
        camera_ds_name (str): Name of the camera dataset
        project (str): Name of the project
        likelihood_threshold (float, optional): Likelihood threshold. Defaults to None.
        start_frame (int, optional): Frame to start plotting from.  If None,
            use the middle of the movie. Defaults to None.
        duration (int, optional): Duration of output movie in seconds. Defaults to None.
        playback_speed (int, optional): Playback speed, relative to original speed.
            Defaults to 4 times faster.
        vmin (float, optional): Minimum value for the colormap. Defaults to None.
        vmax (float, optional): Maximum value for the colormap. Defaults to None.

    Returns:
        None"""
    flm_sess = flz.get_flexilims_session(project_id=project)
    camera_ds = flz.Dataset.from_flexilims(
        name=camera_ds_name, flexilims_session=flm_sess
    )
    ds_dict = cottage_tracking.get_tracking_datasets(
        camera_ds, flexilims_session=flm_sess
    )
    if ds_dict["cropped"] is None:
        raise IOError("No cropped dataset found")
    dlc_ds = ds_dict["cropped"]
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if likelihood_threshold is None:
        if "likelihood_threshold" in dlc_ds.extra_attributes:
            likelihood_threshold = dlc_ds.extra_attributes["likelihood_threshold"]
        else:
            likelihood_threshold = 1
    if start_frame is None:
        start_frame = frame_count // 2 - 30
    analysis.plot_movie(
        camera=camera_ds,
        target_file=dlc_ds.path_full / "ellipse_fit.mp4",
        start_frame=start_frame,
        duration=duration,
        dlc_res=None,
        ellipse=None,
        vmax=vmax,
        vmin=vmin,
        adapt_alpha=False,
        playback_speed=playback_speed,
        crop_border=dlc_ds.extra_attributes["cropping"],
        recrop=False,
        likelihood_threshold=likelihood_threshold,
    )


def plot_binned_ellipse_params(
    binned_ellipses,
    ns,
    save_folder,
    min_frame_cutoff=10,
    fig_title=None,
    camera_ds=None,
    cropping=None,
):
    """Plot binned ellipse parameters

    Args:
        binned_elipses (pd.DataFrame): Binned ellipse parameters
        ns (pd.DataFrame): Number of frames per bin
        save_folder (Path): Folder to save the figure to
        min_frame_cutoff (int, optional): Minimum number of frames to include a bin.
            Defaults to 10.
        fig_title (str, optional): Title of the figure. Defaults to None.
        camera_ds (flexiznam.Dataset, optional): Camera dataset. Defaults to None.
        cropping (list, optional): Cropping info for image. Defaults to None.

    Returns:
        None
    """
    if camera_ds is not None:
        # get one frame in the middle of the recording to use as background
        video_file = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
        cam_data = cv2.VideoCapture(str(video_file))
        cam_data.set(
            cv2.CAP_PROP_POS_FRAMES, int(cam_data.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        )
        ret, frame = cam_data.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cropping is not None:
            gray = gray[cropping[2] : cropping[3], cropping[0] : cropping[1]]
        cam_data.release()

    enough_frames = binned_ellipses[ns > min_frame_cutoff].copy()
    enough_frames["eccentricity"] = np.sqrt(
        1 - (enough_frames["minor_radius"] ** 2 / enough_frames["major_radius"] ** 2)
    )
    mat = np.zeros((len(ns.index.levels[0]), len(ns.index.levels[1]))) + np.nan
    extent = (
        binned_ellipses.centre_x.min(),
        binned_ellipses.centre_x.max(),
        binned_ellipses.centre_y.max(),
        binned_ellipses.centre_y.min(),
    )

    fig = plt.figure(figsize=(7, 6))
    for ip, p in enumerate(["angle", "eccentricity", "minor_radius", "major_radius"]):
        mat[
            enough_frames.index.get_level_values(0),
            enough_frames.index.get_level_values(1),
        ] = enough_frames[p]
        if ip:
            lim = np.nanquantile(mat, [0.01, 0.99])
        else:
            lim = (0, np.pi)
        ax = fig.add_subplot(2, 2, ip + 1)
        divider = make_axes_locatable(ax)
        if camera_ds is not None:
            ax.imshow(gray, cmap="gray")
        cmap = "viridis" if ip else "twilight"
        img = ax.imshow(mat.T, vmin=lim[0], vmax=lim[1], cmap=cmap, extent=extent)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, orientation="vertical")
        ax.set_title(p)
        if camera_ds is not None:
            ax.set_xlim(0, gray.shape[1])
            ax.set_ylim(gray.shape[0], 0)
    fig.suptitle(fig_title)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1, top=0.8)
    fig.savefig(save_folder / f"binned_pupilparams.png", dpi=600)
    plt.close(fig)


def plot_eye_centre_estimate(
    eye_centre_binned,
    f_z0_binned,
    camera_ds,
    binned_frames,
    cropping,
    save_folder,
    example_frame=1000,
):
    video_file = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cam_data = cv2.VideoCapture(str(video_file))
    cam_data.set(cv2.CAP_PROP_POS_FRAMES, example_frame - 1)
    ret, frame = cam_data.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[cropping[2] : cropping[3], cropping[0] : cropping[1]]
    cam_data.release()
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10 * gray.shape[0] / gray.shape[1])
    divider = make_axes_locatable(ax)
    img = ax.imshow(gray, cmap="gray")

    # add eccentricity color coded
    binned_frames = binned_frames.copy()
    binned_frames["eccentricity"] = np.sqrt(
        1 - (binned_frames["minor_radius"] ** 2 / binned_frames["major_radius"] ** 2)
    )
    mat = (
        np.zeros(
            (len(binned_frames.index.levels[0]), len(binned_frames.index.levels[1]))
        )
        + np.nan
    )
    mat[
        binned_frames.index.get_level_values(0),
        binned_frames.index.get_level_values(1),
    ] = binned_frames["eccentricity"]
    minx, maxx = binned_frames.centre_x.min(), binned_frames.centre_x.max()
    miny, maxy = binned_frames.centre_y.min(), binned_frames.centre_y.max()
    img = ax.imshow(mat, cmap="RdBu_r", alpha=0.9, extent=[minx, maxx, maxy, miny])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax, orientation="vertical")

    for i, series in binned_frames.iterrows():
        origin = np.array([series.pupil_x, series.pupil_y])
        ref = np.array([series.reflection_x, series.reflection_y])
        n_v = np.array(
            [np.cos(series.angle + np.pi / 2), np.sin(series.angle + np.pi / 2)]
        )
        rng = np.array([-200, 200])
        ax.plot(
            *[(origin[a] + ref[a] + n_v[a] * rng) for a in range(2)],
            color="purple",
            alpha=0.2,
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


def plot_reprojection(
    eye_centre,
    f_z0,
    dlc_res,
    fitted_params,
    fitted_model,
    camera_ds,
    cropping,
    target_file,
    initial_model=None,
):
    """Plot reprojection of fitted ellipse on example frame

    The frame is selected to be as close as possible to the fitted eye centre.

    Args:
        eye_centre (np.array): Estimated eye centre
        f_z0 (float): Estimated f/z0
        dlc_res (pd.DataFrame): DLC results
        fitted_params (tuple): Parameters of the fitted ellipse.
        fitted_model (EllipseModel): Output of the fit.
        camera_ds (flexiznam.Dataset): Camera dataset
        cropping (list): Cropping info for image
        save_folder (Path): Folder to save the figure to

    Returns:
        None"""
    # find example frame with eye close to median position
    dlc_res = dlc_res.copy()
    if "scorer" in dlc_res.columns.names:
        dlc_res.columns = dlc_res.columns.droplevel("scorer")
    eye_labels = [f"eye_{i}" for i in range(1, 13)]
    ref = dlc_res["reflection"]
    eye_track = dlc_res[eye_labels]
    eye_track.columns = eye_track.columns.droplevel("bodyparts")
    eye_track = eye_track - ref
    eyex = eye_track["x"].median(axis=1)
    eyey = eye_track["y"].median(axis=1)
    dst = (eyex - fitted_params.pupil_x).abs() + (eyey - fitted_params.pupil_y).abs()
    example_frame = dst.idxmin()

    series = dlc_res.loc[example_frame]
    reflection = np.array([series.reflection.x, series.reflection.y])
    video_file = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    cam_data = cv2.VideoCapture(str(video_file))
    cam_data.set(cv2.CAP_PROP_POS_FRAMES, example_frame - 1)
    ret, frame = cam_data.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[cropping[2] : cropping[3], cropping[0] : cropping[1]]
    cam_data.release()

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(gray, cmap="gray")
    circ_coord = EllipseModel.predict_xy(
        None, np.arange(0, 2 * np.pi, 0.1), fitted_params
    ) + reflection.reshape(1, 2)
    ax.plot(circ_coord[:, 0], circ_coord[:, 1], label="DLC fit", color="lightblue")
    ax.plot(*(eye_centre + reflection), color="g", marker="o", label="Eye centre")
    eye_binned = mpl.patches.Circle(
        xy=(eye_centre + reflection),
        radius=f_z0,
        facecolor="none",
        edgecolor="g",
        label=r"$\frac{f}{z_0}$",
    )
    ax.add_artist(eye_binned)

    circ_coord = EllipseModel.predict_xy(
        None, np.arange(0, 2 * np.pi, 0.1), fitted_model
    ) + reflection.reshape(1, 2)
    ax.plot(
        circ_coord[:, 0],
        circ_coord[:, 1],
        label="Reprojection",
        color="purple",
        ls="--",
    )
    if initial_model is not None:
        circ_coord = EllipseModel.predict_xy(
            None, np.arange(0, 2 * np.pi, 0.1), initial_model
        ) + reflection.reshape(1, 2)
        ax.plot(
            circ_coord[:, 0],
            circ_coord[:, 1],
            label="Initial fit",
            color="purple",
            ls=":",
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(target_file)
    plt.close(fig)


def plot_gaze_fit(binned_ellipses, eye_rotation, save_folder, nbins):
    """Plot gaze fit on a grid

    Args:
        binned_ellipses (pd.DataFrame): Binned ellipse parameters
        eye_rotation (np.array): Estimated eye rotation ()
        save_folder (Path): Folder to save the figure to
        nbins (tuple): Number of bins in the grid

    Returns:
        None
    """
    mat = np.zeros((nbins[0] + 1, nbins[1] + 1, 3)) + np.nan
    for i_pos, (pos, _) in enumerate(binned_ellipses.iterrows()):
        mat[pos[0], pos[1]] = eye_rotation[i_pos]
    fig = plt.figure(figsize=(15, 4))
    labels = ["phi", "theta", "radius"]
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        if i < 2:
            d = np.rad2deg(mat[..., i])
            cmap = "twilight"
        else:
            d = mat[..., i]
            cmap = "viridis"
        plt.imshow(
            d, cmap=cmap, vmin=np.nanquantile(d, 0.01), vmax=np.nanquantile(d, 0.99)
        )
        plt.title(labels[i])
        plt.colorbar()
    fig.savefig(save_folder / f"initial_gaze_fit.png")
    plt.close(fig)
