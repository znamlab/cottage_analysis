"""This script is intended to be copied and edited by eye_model_fitting.reproject_pupils"""
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
import flexiznam as flz
from cottage_analysis.eye_tracking import analysis as analeyesis
from cottage_analysis.eye_tracking import eye_model_fitting as emf


PROJECT = "XXX_PROJECT_XXX"
CAMERA_DATASET_ID = "XXX_CAMERA_DS_ID_XXX"
save_folder = "XXX_TARGET_FOLDER_XXX"
phi0 = "XXX_PHI0_XXX"
theta0 = "XXX_THETA0_XXX"
PLOT = True

# get the data
flm_sess = flz.get_flexilims_session(project_id=PROJECT)
camera = flz.Dataset.from_flexilims(flexilims_session=flm_sess, id=CAMERA_DATASET_ID)
dlc_res, data = analeyesis.get_data(
    camera,
    flexilims_session=flm_sess,
    likelihood_threshold=0.88,
    rsquare_threshold=0.99,
    error_threshold=3,
)
basename = camera.dataset_name
processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
save_folder = Path(save_folder)
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
if PLOT:
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
    fig.suptitle(CAMERA_DATASET_NAME)
    fig.savefig(save_folder / f"{basename}_binned_pupil_params.png", dpi=600)
    plt.close(fig)

# Find eye centre
print("Find eye centre", flush=True)
p = np.vstack([enough_frames[f"pupil_{a}"].values for a in "xy"])
n = np.vstack([np.cos(enough_frames.angle.values), np.sin(enough_frames.angle.values)])
intercept_minor = emf.pts_intersection(p, n)
n = np.vstack(
    [np.cos(enough_frames.angle + np.pi / 2), np.sin(enough_frames.angle + np.pi / 2)]
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
if PLOT:
    start_frame = 1000
    video_file = camera.path_full / camera.extra_attributes["video_file"]
    dlc_ds_name = "_".join(
        list(camera.genealogy[:-1]) + ["dlc_tracking", camera.dataset_name, "data", "0"]
    )
    dlc_ds = flz.Dataset.from_flexilims(name=dlc_ds_name, flexilims_session=flm_sess)
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
    fig.savefig(save_folder / f"{basename}_eye_centre_estimate.png")
    plt.close(fig)


# fit median eye position with fine grid
print("Fit median position", flush=True)
most_frequent_bin = ns.idxmax()
params_most_frequent_bin = binned_ellipses.loc[
    most_frequent_bin, ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
]
p0 = (phi0, theta0, 1)
params_med, i, e = emf.minimise_reprojection_error(
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
if PLOT:
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(gray, cmap="gray")
    source_model = EllipseModel()
    source_model.params = params_most_frequent_bin
    circ_coord = source_model.predict_xy(np.arange(0, 2 * np.pi, 0.1)) + ref.reshape(
        1, 2
    )
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

    fitted_model = emf.reproj_ellipse(
        phi=phi, theta=theta, r=radius, eye_centre=eye_centre_binned, f_z0=f_z0_binned
    )
    circ_coord = fitted_model.predict_xy(np.arange(0, 2 * np.pi, 0.1)) + ref.reshape(
        1, 2
    )
    ax.plot(
        circ_coord[:, 0],
        circ_coord[:, 1],
        label="Reprojection",
        color="purple",
        ls="--",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(
        save_folder / f"{basename}_initial_reprojection_median_eye_position.png"
    )
    plt.close(fig)

# optimise for all binned positions
print("Reproject binned data", flush=True)
eye_rotation_initial = np.zeros((len(enough_frames), 3))
grid_angles = np.deg2rad(np.arange(0, 360, 5))
grid_radius = np.arange(0.8, 1.2, 0.1)
for i_pos, (pos, s) in enumerate(enough_frames.iterrows()):
    ellipse_params = s[["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]]
    p, i, e = emf.minimise_reprojection_error(
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
if PLOT:
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
    fig.savefig(save_folder / f"{basename}_initial_gaze_fit.png")
    plt.close(fig)

# Now optimise eye_centre and f_z0
print("Optimise eye parameters", flush=True)
# skip to use about 20 frames to go a bit faster
skip = int(np.ceil(len(enough_frames) / 20))
source_ellipses = (
    enough_frames[::4]
    .loc[:, ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]]
    .values
)
gazes = eye_rotation_initial[::4]
(x, y, f_z0), ind, err = emf.optimise_eye_parameters(
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
params_med, i, e = emf.minimise_reprojection_error(
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

if PLOT:
    # replot median eye posisiton with better eye
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(gray, cmap="gray")
    source_model = EllipseModel()
    source_model.params = params_most_frequent_bin
    circ_coord = source_model.predict_xy(np.arange(0, 2 * np.pi, 0.1)) + ref.reshape(
        1, 2
    )
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
    fitted_model = emf.reproj_ellipse(
        *(phi, theta, radius), eye_centre=eye_centre_binned, f_z0=f_z0_binned
    )
    circ_coord = fitted_model.predict_xy(np.arange(0, 2 * np.pi, 0.1)) + ref.reshape(
        1, 2
    )
    ax.plot(
        circ_coord[:, 0],
        circ_coord[:, 1],
        label="Original reprojection",
        color="orange",
        ls="--",
    )

    fitted_model = emf.reproj_ellipse(*params_med, eye_centre=eye_centre, f_z0=f_z0)
    circ_coord = fitted_model.predict_xy(np.arange(0, 2 * np.pi, 0.1)) + ref.reshape(
        1, 2
    )
    ax.plot(
        circ_coord[:, 0],
        circ_coord[:, 1],
        label="Optimised reprojection",
        color="purple",
        ls=":",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(
        save_folder / f"{basename}_optimised_reprojection_median_eye_position.png"
    )
    plt.close(fig)


# SAVE Eye parameters
print("Saving eye parameters", flush=True)
np.savez(
    save_folder / f"{basename}_eye_parameters.npz",
    eye_centre=eye_centre,
    f_z0=f_z0,
    median_eye_position_parameters=params_med,
)


# optimise for all frames
print("Fitting all frames", flush=True)
eye_rotation = np.zeros((len(data), 3))
eye_rotation[~data.valid] += np.nan
grid_angles = np.deg2rad(np.arange(0, 360, 5))
grid_radius = np.arange(0.8, 1.2, 0.1)
for i_pos, series in data.iterrows():
    if np.mod(i_pos, 1000) == 0:
        percent = i_pos / len(eye_rotation) * 100
        print(f"{percent:.2f}%", flush=True)
    if not series.valid:
        continue
    ellipse_params = series[
        ["pupil_x", "pupil_y", "major_radius", "minor_radius", "angle"]
    ]
    pa, i, e = emf.minimise_reprojection_error(
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
np.save(save_folder / f"{basename}_eye_rotation_by_frame.npy", eye_rotation)
print("Done!")
