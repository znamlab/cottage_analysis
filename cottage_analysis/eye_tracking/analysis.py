import flexiznam as flz
import pandas as pd
import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from pathlib import Path
from tqdm import tqdm
import cv2
import pickle
from cottage_analysis.utilities.plot_utils import get_img_from_fig, write_fig_to_video


def get_data(
    camera,
    flexilims_session=None,
    likelihood_threshold=0.88,
    rsquare_threshold=0.99,
    error_threshold=4,
    ds_is_cropped=True,
):
    """Get eye tracking data from camera dataset

    Args:
        camera (flexiznam.schema.camera_data.CameraData): Camera dataset
        flexilims_session (flexilims.Session): Flexilims session to interact with
            database. Must have the proper project. If None, use
            `camera.flexilims_sessions`. Default to None.
        likelihood_threshold (float, optional): Threshold on average DLC likelihood.
            Defaults to 0.88.
        rsquare_threshold (float, optional): Threshold on rsquare of ellipse fit.
            Defaults to 0.99.
        error_threshold (float, optional): Threshold on error of ellipse fit, in px.
            Defaults to 4.
        ds_is_cropped (bool, optional): Whether the dataset is cropped. Defaults to
            True.

    Returns:
        panda.DataFrame: DLC results
        panda.DataFrame: Ellipse fits
    """

    rec_ds = flz.get_children(
        parent_id=camera.origin_id,
        flexilims_session=flexilims_session,
        children_datatype="dataset",
    )
    cam_analysis = rec_ds[rec_ds.name.map(lambda x: camera.dataset_name in x)]
    dlc = cam_analysis[cam_analysis.dataset_type == "dlc_tracking"]
    if ds_is_cropped:
        dlc = dlc[[(c is not None) for c in dlc.cropping]]
    else:
        dlc = dlc[[(c is None) for c in dlc.cropping]]
    assert len(dlc) == 1
    dlc = flz.Dataset.from_dataseries(dlc.iloc[0], flexilims_session=flexilims_session)
    dlc_res = pd.read_hdf(dlc.path_full / dlc.extra_attributes["dlc_file"])
    # Get ellipse fits
    ellipse_csv = list(dlc.path_full.glob("*ellipse_fits.csv"))
    assert len(ellipse_csv) == 1
    ellipse = pd.read_csv(ellipse_csv[0])
    # add dlc likelihood

    dlc_like = dlc_res.xs("likelihood", axis="columns", level=2)
    dlc_like.columns = dlc_like.columns.droplevel("scorer")
    reflection_like = dlc_like["reflection"]
    to_drop = [c for c in dlc_like.columns if c[-1].isalpha()]
    dlc_like = dlc_like.drop(axis="columns", labels=to_drop).mean(axis="columns")
    ellipse["dlc_avg_likelihood"] = dlc_like
    valid = (
        (ellipse.dlc_avg_likelihood > likelihood_threshold)
        & (ellipse.rsquare > rsquare_threshold)
        & (ellipse.error < error_threshold)
        & (reflection_like > likelihood_threshold)
    )
    ellipse["valid"] = valid

    reflection = dlc_res.xs(axis="columns", level=1, key="reflection")
    reflection.columns = reflection.columns.droplevel("scorer")
    ellipse["reflection_x"] = reflection.x.values
    ellipse["reflection_y"] = reflection.y.values
    ellipse["pupil_x"] = ellipse.centre_x - ellipse.reflection_x
    ellipse["pupil_y"] = ellipse.centre_y - ellipse.reflection_y
    ellipse.loc[~ellipse.valid, "x"] = np.nan
    ellipse.loc[~ellipse.valid, "y"] = np.nan

    return dlc_res, ellipse


def plot_movie(
    camera,
    target_file,
    start_frame=0,
    duration=None,
    dlc_res=None,
    ellipse=None,
    vmax=None,
    vmin=None,
    playback_speed=4,
    crop_border=None,
    use_original_encoding=False,
    recrop=False,
    likelihood_threshold=0.88,
    adapt_alpha=True,
):
    """Plot a movie of raw video, video with dlc tracking and video with ellipse fit

    Args:
        camera (flexiznam.schema.camera_data.CameraData): Camera dataset
        target_file (str): Full path to video output
        start_frame (int, optional): First frame to plot. Defaults to 0.
        duration (float, optional): Duration of video, in seconds. If None, plot until
            the end. Defaults to None.
        dlc_res (pandas.DataFrame, optional): DLC results. Will be loaded if None.
            Defaults to None.
        ellipse (pandas.DataFrame, optional): Ellipse fit. Will be loaded if None.
            Defaults to None.
        vmax (int, optional): vmax for video grayscale image. Defaults to None
        vmin (int, optional): vmin for video grayscale image. Defaults to None
        playback_speed (float, optional): playback speed, relative to original video
            speed (which might not be real time). Default to 4.
        crop_border (list, optional): Border to crop video. Defaults to None.
        use_original_encoding (bool, optional): Whether to use original video encoding
            (might not be supported by opencv). Defaults to False.
        recrop (bool, optional): Whether to recrop video. Defaults to False.
        likelihood_threshold (float, optional): Threshold on DLC likelihood use for
            scatter color.
        adapt_alpha (bool, optional): Whether to adapt alpha of scatter points with
            likelihood. Defaults to True.
    """

    if dlc_res is None or ellipse is None:
        dlc_res, ellipse = get_data(camera, flexilims_session=camera.flexilims_session)
    # Find DLC crop area
    if recrop:
        borders = np.zeros((4, 2))
        for iw, w in enumerate(
            ("left_eye_corner", "right_eye_corner", "top_eye_lid", "bottom_eye_lid")
        ):
            vals = dlc_res.xs(w, level=1, axis=1)
            vals.columns = vals.columns.droplevel("scorer")
            v = np.nanmedian(vals[["x", "y"]].values, axis=0)
            borders[iw, :] = v

        borders = np.vstack([np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)])
        borders += ((np.diff(borders, axis=0) * 0.1).T @ np.array([[-1, 1]])).T
        borders = borders.astype(int)
    video_file = camera.path_full / camera.extra_attributes["video_file"]
    ellipse_model = EllipseModel()

    fig = plt.figure()
    fig.set_size_inches((9, 3))

    img = get_img_from_fig(fig)
    cam_data = cv2.VideoCapture(str(video_file))
    fps = cam_data.get(cv2.CAP_PROP_FPS)
    if use_original_encoding:
        fcc = int(cam_data.get(cv2.CAP_PROP_FOURCC))
        fcc = (
            chr(fcc & 0xFF)
            + chr((fcc >> 8) & 0xFF)
            + chr((fcc >> 16) & 0xFF)
            + chr((fcc >> 24) & 0xFF)
        )
    else:
        fcc = "mp4v"
    output = cv2.VideoWriter(
        str(target_file),
        cv2.VideoWriter_fourcc(*fcc),
        fps * playback_speed,
        (img.shape[1], img.shape[0]),
    )

    if duration is None:
        nframes = cam_data.get(cv2.CAP_PROP_FRAME_COUNT) - start_frame
    else:
        nframes = int(fps * duration)
    cam_data.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    for frame_id in tqdm(np.arange(nframes) + start_frame):
        track = dlc_res.loc[frame_id]
        # plot
        fig.clear()
        ax_img = fig.add_subplot(1, 3, 1)
        ax_track = fig.add_subplot(1, 3, 2)
        ax_fit = fig.add_subplot(1, 3, 3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        ret, frame = cam_data.read()
        if crop_border is not None:
            frame = frame[
                crop_border[2] : crop_border[3], crop_border[0] : crop_border[1]
            ]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for ax in [ax_img, ax_fit, ax_track]:
            img = gray[slice(*borders[:, 1]), slice(*borders[:, 0])] if recrop else gray
            ax.imshow(
                img,
                cmap="gray",
                vmax=vmax,
                vmin=vmin,
            )
            ax.set_yticks([])
            ax.set_xticks([])

        track.index = track.index.droplevel(["scorer"])
        xdata = track.loc[[(f"eye_{i}", "x") for i in np.arange(1, 13)]]
        ydata = track.loc[[(f"eye_{i}", "y") for i in np.arange(1, 13)]]
        likelihood = track.loc[[(f"eye_{i}", "likelihood") for i in np.arange(1, 13)]]
        if recrop:
            xs, ys = borders[0, 0], borders[0, 1]
        else:
            xs, ys = 0, 0
        ax_track.scatter(
            xdata - xs,
            ydata - ys,
            s=likelihood * 10,
            alpha=likelihood if adapt_alpha else 1,
            color=["g" if l > likelihood_threshold else "r" for l in likelihood],
        )
        ax_track.scatter(
            track.loc[("reflection", "x")] - xs,
            track.loc[("reflection", "y")] - ys,
        )
        # params are xc, yc, a, b, theta
        params = ellipse.loc[
            frame_id, ["centre_x", "centre_y", "major_radius", "minor_radius", "angle"]
        ]
        ellipse_model.params = params.values
        circ_coord = ellipse_model.predict_xy(np.arange(0, 2 * np.pi, 0.1))
        ax_fit.plot(circ_coord[:, 0] - xs, circ_coord[:, 1] - ys)
        write_fig_to_video(fig, output)

    cam_data.release()
    output.release()
    print(f"Saved in {target_file}")


def add_behaviour(
    camera, dlc_res, ellipse, speed_threshold=0.01, log_speeds=False, verbose=True
):
    """Add running speed, optic flow and depth to ellipse dataframe

    This assumes that there can be a few triggers after the end of the scanimage session
    and cuts them (up to 5)
    Args:
        camera (flexiznam.CameraDataset): Camera dataset, used for finding data and flexilims interaction
        dlc_res (pandas.DataFrame): DLC results
        ellipse (pandas.DataFrame): Ellipse fits
        speed_threshold (float, optional): Threshold to cut running speeds. Defaults to
            0.01.
        log_speeds (bool, optional): If True, speeds at log10, otherwise raw. Defaults
            to False.
        verbose (bool, optional): If True tell how many frames are cut. Defaults to True

    Returns:
        pandas.DataFrame: Combined dataframe, copy of ellipse with speeds
    """
    # get data
    flm_sess = camera.flexilims_session
    assert flm_sess is not None
    recording = flz.get_entity(id=camera.origin_id, flexilims_session=flm_sess)

    sess_ds = flz.get_children(
        parent_id=recording.origin_id,
        flexilims_session=flm_sess,
        children_datatype="dataset",
    )
    suite_2p = sess_ds[sess_ds.dataset_type == "suite2p_rois"]
    assert len(suite_2p) == 1
    suite_2p = flz.Dataset.from_dataseries(suite_2p.iloc[0], flexilims_session=flm_sess)

    ops = np.load(
        suite_2p.path_full / "suite2p" / "plane0" / "ops.npy", allow_pickle=True
    ).item()
    processed = Path(flz.PARAMETERS["data_root"]["processed"])

    with open(processed / recording.path / "img_VS.pickle", "rb") as handle:
        param_logger = pickle.load(handle)
    with open(processed / recording.path / "stim_dict.pickle", "rb") as handle:
        stim_dict = pickle.load(handle)

    sampling = ops["fs"]

    vrs = np.array(param_logger.EyeZ.diff() / param_logger.HarpTime.diff(), dtype=float)
    vrs = np.clip(vrs, speed_threshold, None)
    if "MouseZ" in param_logger.columns:
        rs = np.array(
            param_logger.MouseZ.diff() / param_logger.HarpTime.diff(), dtype=float
        )
        rs = np.clip(rs, speed_threshold, None)
    else:
        warn(f"No MouseZ for {recording.path}")
        assert not recording.protocol.lower().endswith("playback")
        rs = np.array(vrs)
    depth = np.array(param_logger.Depth, copy=True, dtype=float)
    depth[depth < 0] = np.nan
    of = np.degrees(vrs / depth)
    # convert to cm
    depth *= 100
    rs *= 100
    vrs *= 100
    if log_speeds:
        rs = np.log10(rs)
        vrs = np.log10(vrs)
        of = np.log10(of)
    if verbose:
        print(f"Running speed with {len(rs)}, vs {len(ellipse)} frames")
    ntocut = len(ellipse) - len(rs)
    if ntocut > 5:
        raise ValueError("{ntocut} more frames in video than SI trggers")
    elif ntocut > 0:
        if verbose:
            print(f"Cutting the last {ntocut} frames")
        data = pd.DataFrame(ellipse.iloc[:-ntocut, :], copy=True)
    else:
        raise NotImplementedError

    data["running_speed"] = rs
    data["optic_flow"] = of
    data["virtual_running_speed"] = vrs
    data["depth"] = np.round(depth, 0)
    return data, sampling
