import functools

print = functools.partial(print, flush=True)

import os
import sys
import defopt
import pickle
import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon
import random
import itertools
import flexiznam as flz
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42  # save text as text not outlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from cottage_analysis.filepath import generate_filepaths
from cottage_analysis.depth_analysis.depth_preprocess import process_params
from cottage_analysis.analysis import common_utils


# ----- FUNCS -----

MIN_SIGMA = 0.5


def gaussian_func(x, a, x0, log_sigma, b):
    a = a
    sigma = np.exp(log_sigma) + MIN_SIGMA
    return (a * np.exp(-((x - x0) ** 2)) / (2 * sigma**2)) + b


MIN_SIGMA_blob = 0.1


def twoD_Gaussian(
    xy_tuple, log_amplitude, xo, yo, log_sigma_x2, log_sigma_y2, theta, offset
):
    (x, y) = xy_tuple
    sigma_x_sq = np.exp(log_sigma_x2) + MIN_SIGMA_blob  # 0.25
    sigma_y_sq = np.exp(log_sigma_y2) + MIN_SIGMA_blob  # 0.25
    amplitude = np.exp(log_amplitude)
    a = (np.cos(theta) ** 2) / (2 * sigma_x_sq) + (np.sin(theta) ** 2) / (
        2 * sigma_y_sq
    )
    b = (np.sin(2 * theta)) / (4 * sigma_x_sq) - (np.sin(2 * theta)) / (4 * sigma_y_sq)
    c = (np.sin(theta) ** 2) / (2 * sigma_x_sq) + (np.cos(theta) ** 2) / (
        2 * sigma_y_sq
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g


def gaussian_blob_fit_crossval(project, mouse, session):
    """
    :param str project: project name (determines the root directory for raw data)
    :param str mouse: mouse name
    :param str session: session name,Sdate
    :return: None
    """
    # ----- SETUPS -----
    frame_rate = 15
    speed_thr_cal = (
        0.2  # m/s, threshold for running speed when calculating depth neurons
    )
    speed_thr = 0.01  # m/s
    depth_min = 2
    depth_max = 2000
    rs_min = 0.5
    rs_max = 500
    of_min = 0.03
    of_max = 3000
    batch_num = 5

    # ----- PATHS -----
    flexilims_session = flz.get_flexilims_session(project_id=project)
    sess_children = generate_filepaths.get_session_children(
        project=project,
        mouse=mouse,
        session=session,
        flexilims_session=flexilims_session,
    )
    if len(sess_children[sess_children.name.str.contains("Playback")]) > 0:
        protocols = ["SpheresPermTubeReward", "SpheresPermTubeRewardPlayback"]
    else:
        protocols = ["SpheresPermTubeReward"]
    root = Path(flz.PARAMETERS["data_root"]["processed"])

    # Only do this for closed loop
    for protocol in [protocols[0]]:
        print(
            f"---------Process protocol {protocol}/{len(protocols)}---------",
            flush=True,
        )
        print("---STEP 1: Load files...---", "\n", flush=True)
        # ----- STEP1: Generate file path -----
        session_protocol_folder = root / project / mouse / session / protocol
        session_analysis_folder = session_protocol_folder
        session_analysis_folder_old_original = (
            root / project / mouse / session / protocols[0]
        )
        (
            _,
            _,
            _,
            suite2p_folder,
            _,
        ) = generate_filepaths.generate_file_folders(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            all_protocol_recording_entries=None,
            recording_no=0,
            flexilims_session=None,
        )

        save_prefix = "plane0/gaussian_blob_crossval"
        save_folder = session_analysis_folder / save_prefix
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Load suite2p files
        ops = np.load(suite2p_folder / "ops.npy", allow_pickle=True)
        ops = ops.item()
        iscell = np.load(suite2p_folder / "iscell.npy", allow_pickle=True)[:, 0]

        # All_rois
        which_rois = (np.arange(iscell.shape[0]))[iscell.astype("bool")]

        with open(session_protocol_folder / "plane0/img_VS_all.pickle", "rb") as handle:
            img_VS_all = pickle.load(handle)
        with open(
            session_protocol_folder / "plane0/stim_dict_all.pickle", "rb"
        ) as handle:
            stim_dict_all = pickle.load(handle)
        dffs_ast_all = np.load(session_protocol_folder / "plane0/dffs_ast_all.npy")

        depth_neurons = np.load(
            session_analysis_folder_old_original / "plane0/depth_neurons.npy"
        )
        max_depths = np.load(
            session_analysis_folder_old_original / "plane0/max_depths_index.npy"
        )

        depth_list = img_VS_all["Depth"].unique()
        depth_list = np.round(depth_list, 2)
        depth_list = depth_list[~np.isnan(depth_list)].tolist()
        depth_list.remove(-99.99)
        depth_list.sort()
        if len(depth_list) == 5:
            depth_list = [0.06, 0.19, 0.6, 1.9, 6]

        # cmap = cm.cool.reversed()
        # line_colors = []
        # norm = matplotlib.colors.Normalize(vmin=np.log(min(depth_list)), vmax=np.log(max(depth_list)))
        # for depth in depth_list:
        #     rgba_color = cmap(norm(np.log(depth)),bytes=True)
        #     rgba_color = tuple(it/255 for it in rgba_color)
        #     line_colors.append(rgba_color)

        print("---STEP 1 FINISHED.---", "\n", flush=True)

        # fit gaussian depth
        print(
            "---START STEP 2---",
            "\n",
            "Closed loop - Fit gaussian depth tuning...",
            flush=True,
        )
        print("MIN SIGMA", str(MIN_SIGMA), flush=True)
        gaussian_depth_fit_df = pd.DataFrame(
            columns=[
                "ROI",
                "preferred_depth_idx",
                "a",
                "x0_logged",
                "log_sigma",
                "b",
                "r_sq",
            ]
        )
        gaussian_depth_fit_df.ROI = depth_neurons
        gaussian_depth_fit_df.preferred_depth_idx = max_depths
        speeds = img_VS_all.MouseZ.diff() / img_VS_all.HarpTime.diff()  # m/s
        speeds[0] = 0
        # speed_thr = 0.01
        # speeds_thred = thr(speeds, speed_thr)
        speed_arr, _ = process_params.create_speed_arr(
            speeds=speeds,
            depth_list=depth_list,
            stim_dict=stim_dict_all,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        (ndepths, ntrials, nframes) = speed_arr.shape

        # Generate list of trials half and half, one half for Gaussian depth fitting, one half for RSOF blob fitting
        np.random.seed(int(session[1:]))
        trial_list1 = np.random.choice(int(ntrials), int(ntrials // 2), replace=False)
        trial_list2 = np.array(
            [i for i in np.arange(int(ntrials)) if i not in trial_list1]
        )

        gaussian_depth_fit_df = pd.DataFrame(
            columns=[
                "ROI",
                "preferred_depth_max",
                "preferred_depth",
                "a",
                "x0_logged",
                "log_sigma",
                "b",
                "r_sq",
            ]
        )
        gaussian_depth_fit_df.ROI = depth_neurons
        gaussian_depth_fit_df.preferred_depth_idx = max_depths

        for iroi, choose_roi in enumerate(depth_neurons):
            roi = choose_roi
            trace_arr, _ = process_params.create_trace_arr_per_roi(
                which_roi=roi,
                dffs=dffs_ast_all,
                depth_list=depth_list,
                stim_dict=stim_dict_all,
                mode="sort_by_depth",
                protocol="fix_length",
                blank_period=0,
                frame_rate=frame_rate,
            )
            trace_arr_half = trace_arr[:, trial_list1, :]
            speed_arr_half = speed_arr[:, trial_list1, :]
            trace_arr_half[speed_arr_half < speed_thr_cal] = np.nan
            trace_arr_mean_eachtrial_half = np.nanmean(trace_arr_half, axis=2)

            # When open loop, there will be nan values in this array because of low running speed sometimes
            trace_arr_mean_eachtrial_half = np.nan_to_num(trace_arr_mean_eachtrial_half)
            x = np.log(
                np.repeat(
                    np.array(depth_list) * 100, trace_arr_mean_eachtrial_half.shape[1]
                )
            )
            roi_number = np.where(depth_neurons == roi)[0][0]
            popt_arr = []
            r_sq_arr = []
            for ibatch in range(batch_num):
                np.random.seed(ibatch)
                p0 = np.concatenate(
                    (
                        np.abs(np.random.normal(size=1)),
                        np.atleast_1d(
                            np.log(
                                np.array(depth_list[int(max_depths[roi_number])]) * 100
                            )
                        ),
                        np.abs(np.random.normal(size=1)),
                        np.random.normal(size=1),
                    )
                ).flatten()
                popt, pcov = curve_fit(
                    gaussian_func,
                    x,
                    trace_arr_mean_eachtrial_half.flatten(),
                    p0=p0,
                    maxfev=100000,
                    bounds=(
                        [0, np.log(depth_min), 0, -np.inf],
                        [np.inf, np.log(depth_max), np.inf, np.inf],
                    ),
                )

                y_pred = gaussian_func(x, *popt)
                r_sq = common_utils.calculate_r_squared(
                    trace_arr_mean_eachtrial_half.flatten(), y_pred
                )
                popt_arr.append(popt)
                r_sq_arr.append(r_sq)
            idx_best = np.argmax(r_sq_arr)
            popt_best = popt_arr[idx_best]
            rsq_best = r_sq_arr[idx_best]

            gaussian_depth_fit_df.iloc[iroi, 3:-1] = popt_best
            gaussian_depth_fit_df.iloc[iroi, -1] = rsq_best
            gaussian_depth_fit_df.loc[iroi, "preferred_depth_max"] = (
                np.array(depth_list[int(max_depths[roi_number])]) * 100
            )
            gaussian_depth_fit_df.loc[iroi, "preferred_depth"] = np.exp(popt_best[1])
            preferred_depth_close = gaussian_depth_fit_df.preferred_depth
            preferred_depth_max_close = gaussian_depth_fit_df.preferred_depth_max

            if iroi % 10 == 0:
                print(roi, flush=True)
        save_filename = save_folder / (
            "gaussian_depth_fit_" + str(MIN_SIGMA) + ".pickle"
        )
        with open(save_filename, "wb") as handle:
            pickle.dump(gaussian_depth_fit_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # For both close loop and open loop --> fit gaussian blob
        print(
            "---START STEP 3---", "\n", "Fit gaussian depth tuning blob...", flush=True
        )
        # ----- STEP3: Process params -----
        if "Playback" in protocol:
            is_actual_running_list = [True, False]
        else:
            is_actual_running_list = [True]
        for is_actual_running in is_actual_running_list:
            if is_actual_running:
                sfx = "actual_running"
            else:
                sfx = "virtual_running"
            # Running speed
            # Running speed is thresholded with a small threshold to get rid of non-zero values (default threshold 0.01)
            speeds = (
                img_VS_all.MouseZ.diff() / img_VS_all.HarpTime.diff()
            )  # CHANGE TO MOUSEZ AFTERWARDS!!!
            speeds[0] = 0
            speeds = np.clip(speeds, a_min=speed_thr, a_max=None)
            speed_arr, _ = process_params.create_speed_arr(
                speeds,
                depth_list,
                stim_dict_all,
                mode="sort_by_depth",
                protocol="fix_length",
                blank_period=0,
                frame_rate=frame_rate,
            )
            (ndepths, ntrials, nframes) = speed_arr.shape
            # OF (Unit: rad/s)
            if "Playback" in protocol:
                speeds_eye = (
                    img_VS_all.EyeZ.diff() / img_VS_all.HarpTime.diff()
                )  # EyeZ is how the perspective of animal moves
                speeds_eye[0] = 0
                speeds_eye = np.clip(speeds_eye, a_min=speed_thr, a_max=None)
                speed_eye_arr, _ = process_params.create_speed_arr(
                    speeds_eye,
                    depth_list,
                    stim_dict_all,
                    mode="sort_by_depth",
                    protocol="fix_length",
                    blank_period=0,
                    frame_rate=frame_rate,
                )
                optics = process_params.calculate_OF(
                    rs=speeds_eye, img_VS=img_VS_all, mode="no_RF"
                )

            else:
                optics = process_params.calculate_OF(
                    rs=speeds, img_VS=img_VS_all, mode="no_RF"
                )
            of_arr, _ = process_params.create_speed_arr(
                optics,
                depth_list,
                stim_dict_all,
                mode="sort_by_depth",
                protocol="fix_length",
                blank_period=0,
                frame_rate=frame_rate,
            )

            if is_actual_running:
                print("Fit actual running speed...", flush=True)
                rs_arr = speed_arr
            else:
                print("Fit virtual running speed...", flush=True)
                rs_arr = speed_eye_arr
            X = rs_arr[:, trial_list2, :].flatten()
            Y = of_arr[:, trial_list2, :].flatten()

            results = pd.DataFrame(
                columns=[
                    "ROI",
                    "preferred_depth_idx",
                    "preferred_depth_max",
                    "preferred_depth",
                    "log_amplitude",
                    "xo_logged",
                    "yo_logged",
                    "log_sigma_x2",
                    "log_sigma_y2",
                    "theta",
                    "offset",
                    "r_sq",
                ]
            )
            results["ROI"] = depth_neurons
            results["preferred_depth_idx"] = max_depths
            results["preferred_depth"] = preferred_depth_close
            results["preferred_depth_max"] = preferred_depth_max_close

            # Loop through all rois
            for iroi, choose_roi in enumerate(depth_neurons):
                roi = choose_roi
                trace_arr, _ = process_params.create_trace_arr_per_roi(
                    roi,
                    dffs_ast_all,
                    depth_list,
                    stim_dict_all,
                    mode="sort_by_depth",
                    protocol="fix_length",
                    blank_period=0,
                    frame_rate=frame_rate,
                )
                Z = trace_arr[:, trial_list2, :].flatten()

                Z_ = Z[~np.isnan(Z)]
                X_ = X[~np.isnan(Z)]
                Y_ = Y[~np.isnan(Z)]
                log_X_ = np.log(X_ * 100)
                log_Y_ = np.log(np.degrees(Y_))
                popt_arr = []
                rsq_arr = []
                for ibatch in range(batch_num):
                    mu0 = 0
                    sigma0 = 1

                    p0 = np.concatenate(
                        (
                            np.random.normal(mu0, sigma0, size=1),
                            np.atleast_1d(
                                np.min(
                                    [
                                        np.abs(np.random.normal(mu0, sigma0, size=1)),
                                        np.log(rs_min),
                                    ]
                                )
                            ),
                            np.atleast_1d(
                                [
                                    np.min(
                                        [
                                            np.abs(
                                                np.random.normal(mu0, sigma0, size=1)
                                            )
                                            + np.log(of_min)
                                        ]
                                    )
                                ]
                            ),
                            np.random.normal(mu0, sigma0, size=2),
                            np.atleast_1d(
                                np.max(
                                    [
                                        np.abs(np.random.normal(mu0, sigma0, size=1)),
                                        np.radians(90),
                                    ]
                                )
                            ),
                            np.random.normal(mu0, sigma0, size=1),
                        )
                    )
                    popt, pcov = curve_fit(
                        twoD_Gaussian,
                        (log_X_, log_Y_),
                        Z_,
                        maxfev=100000,
                        bounds=(
                            [
                                -np.inf,
                                np.log(rs_min),
                                np.log(of_min),
                                -np.inf,
                                -np.inf,
                                0,
                                -np.inf,
                            ],
                            [
                                np.inf,
                                np.log(rs_max),
                                np.log(of_max),
                                np.inf,
                                np.inf,
                                np.radians(90),
                                np.inf,
                            ],
                        ),
                    )

                    Z_fit_ = twoD_Gaussian(np.array([log_X_, log_Y_]), *popt)
                    Z_fit = np.empty(Z.shape)
                    Z_fit[:] = np.NaN
                    Z_fit[~np.isnan(Z)] = Z_fit_
                    mse = np.mean((Z_fit_ - Z_) ** 2)
                    r_sq = common_utils.calculate_r_squared(Z_, Z_fit_)
                    popt_arr.append(popt)
                    rsq_arr.append(r_sq)
                idx_best = np.argmax(np.array(rsq_arr))
                popt_best = popt_arr[idx_best]
                Z_fit_best_ = twoD_Gaussian(np.array([log_X_, log_Y_]), *popt_best)
                Z_fit_best = np.empty(Z.shape)
                Z_fit_best[:] = np.NaN
                Z_fit_best[~np.isnan(Z)] = Z_fit_best_
                mse = np.mean((Z_fit_best_ - Z_) ** 2)
                r_sq_best = common_utils.calculate_r_squared(Z_, Z_fit_best_)

                results.iloc[iroi, 4:-1] = popt_best
                results.iloc[iroi, -1] = r_sq_best
                # print(str(roi), np.round(np.exp(popt_best[1:3]),2), flush=True)
                if iroi % 10 == 0:
                    print(roi, flush=True)
                iroi += 1

            save_filename = save_folder / (
                "gaussian_blob_fit_" + str(MIN_SIGMA_blob) + "_" + sfx + ".pickle"
            )
            with open(save_filename, "wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


project = "hey2_3d-vision_foodres_20220101"
mouses = ["PZAH8.2i", "PZAH8.2f"]  # "PZAH6.4b","PZAG3.4f",
V1_sessions = [
    # ["S20220519", "S20220524"],
    # ["S20220520","S20220523", "S20220524", "S20220526", "S20220527"],
    # ["S20230203", "S20230209", "S20230216", "S20230220"],
    ["S20230220"],
    ["S20230206", "S20230210", "S20230213", "S20230214"],
]
for imouse, mouse in enumerate(mouses):
    for session in V1_sessions[imouse]:
        print(f"Process {mouse} {session}...")
        gaussian_blob_fit_crossval(project, mouse, session)
        print(f"Finished")
print(f"Finished all.")
