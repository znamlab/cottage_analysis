"""Functions to save diagnostic plots for eye tracking"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import cottage_analysis.eye_tracking.eye_tracking as cottage_tracking
from cottage_analysis.eye_tracking import analysis
import yaml
import flexiznam as flz
import numpy as np


def check_cropping(dlc_ds, camera_ds, rotate180=False):
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    dlc_results = pd.read_hdf(dlc_file)

    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if not crop_file.exists():
        print(f"File {crop_file} does not exist")
        cottage_tracking.create_crop_file(camera_ds, dlc_ds)

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
    eye_tracking = frame_tracking[[f"eye_{i + 1}" for i in range(12)]]
    xs = eye_tracking.xs("x", level="coords").values
    ys = eye_tracking.xs("y", level="coords").values
    sc = ax.scatter(xs, ys, s=20, label="Pupil")

    corners = ["left_eye_corner", "right_eye_corner", "top_eye_lid", "bottom_eye_lid"]
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


def plot_ellipse_fit(camera_ds_name, project, likelihood_threshold):
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

    analysis.plot_movie(
        camera=camera_ds,
        target_file=dlc_ds.path_full / "ellipse_fit.mp4",
        start_frame=frame_count // 2 - 30,
        duration=60,
        dlc_res=None,
        ellipse=None,
        vmax=None,
        vmin=None,
        playback_speed=4,
        crop_border=dlc_ds.extra_attributes["cropping"],
        recrop=False,
    )
