from functools import partial
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from scipy.optimize import curve_fit
import flexiznam as flz
from cottage_analysis.analysis import common_utils

print = partial(print, flush=True)


def gaussian_2d(
    xy_tuple,
    log_amplitude,
    xo,
    yo,
    log_sigma_x2,
    log_sigma_y2,
    theta,
    offset,
    min_sigma,
):
    (x, y) = xy_tuple
    sigma_x_sq = np.exp(log_sigma_x2) + min_sigma
    sigma_y_sq = np.exp(log_sigma_y2) + min_sigma
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


def gaussian_2d_fit(X, y, lower_bounds, upper_bounds, min_sigma, niter=5):
    """Fit a 2D gaussian to the data.

    Args:
        X: tuple of x and y coordinates
        y: response
        lower_bounds (list): lower bounds for the parameters
        upper_bounds (list): upper bounds for the parameters
        min_sigma (float): minimum sigma for the gaussian
        niter (int): number of iterations to run the fitting.
            The best fit is chosen based on the R squared value. Defaults to 5.

    Returns:
        popt_best (list): best fit parameters
        rsq_best (float): R squared value of the best fit

    """
    popt_arr = []
    rsq_arr = []
    np.random.seed(42)
    for _ in range(niter):
        gaussian_2d_ = partial(gaussian_2d, min_sigma=min_sigma)
        popt, _ = curve_fit(
            gaussian_2d_,
            X,
            y,
            maxfev=100000,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
        )

        dff_fit = gaussian_2d(np.array(X), *popt)
        r_sq = common_utils.calculate_r_squared(y, dff_fit)
        popt_arr.append(popt)
        rsq_arr.append(r_sq)
    idx_best = np.argmax(np.array(rsq_arr))
    popt_best = popt_arr[idx_best]
    rsq_best = rsq_arr[idx_best]
    return popt_best, rsq_best


def analyze_rs_of_tuning(
    project,
    mouse,
    session,
    protocol="SpheresPermTubeReward",
    rs_thr=0.01,
    param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
    niter=5,
    min_sigma=0.25,
):
    # Load files
    root = Path(flz.PARAMETERS["data_root"]["processed"])
    session_folder = root / project / mouse / session

    with open(session_folder / "plane0/trials_df.pickle", "rb") as handle:
        trials_df = pickle.load(handle)
    with open(session_folder / "plane0/neurons_df.pickle", "rb") as handle:
        neurons_df = pickle.load(handle)
    neurons_df = neurons_df.assign(
        preferred_RS_closed_loop=np.nan,
        preferred_OF_closed_loop=np.nan,
        gaussian_blob_popt_closed_loop=[[np.nan]] * len(neurons_df),
        gaussian_blob_rsq_closed_loop=np.nan,
        preferred_RS_open_loop_actual=np.nan,
        preferred_OF_open_loop_actual=np.nan,
        gaussian_blob_popt_open_loop_actual=[[np.nan]] * len(neurons_df),
        gaussian_blob_rsq_open_loop_actual=np.nan,
        preferred_RS_open_loop_virtual=np.nan,
        preferred_OF_open_loop_virtual=np.nan,
        gaussian_blob_popt_open_loop_virtual=[[np.nan]] * len(neurons_df),
        gaussian_blob_rsq_open_loop_virtual=np.nan,
    )

    # Determine whether this session has open loop or not
    if len(trials_df.closed_loop.unique()) == 2:
        protocols = [protocol, f"{protocol}Playback"]
    elif len(trials_df.closed_loop.unique()) == 1:
        protocols = [protocol]

    # Loop through all protocols
    for iprotocol, protocol in enumerate(protocols):
        print(f"---------Process protocol {iprotocol+1}/{len(protocols)}---------")
        if "Playback" in protocol:
            is_closedloop = 0
            protocol_sfx = "open_loop"
        else:
            is_closedloop = 1
            protocol_sfx = "closed_loop"
        trials_df_protocol = trials_df[trials_df.closed_loop == is_closedloop]

        # Concatenate arrays of RS/OF/dff from all trials together
        rs = np.concatenate(trials_df_protocol["RS_stim"].values)
        rs_eye = np.concatenate(trials_df_protocol["RS_eye_stim"].values)
        of = np.concatenate(trials_df_protocol["OF_stim"].values)
        dff = np.concatenate(trials_df_protocol["dff_stim"].values, axis=1)

        # Take out the values where running is below a certain threshold
        running = (
            (rs > rs_thr) & (rs_eye > rs_thr) & (~np.isnan(of))
        )  # !!! OF has a small number of frame = nan, investigate synchronisation.py
        rs = rs[running]
        rs_eye = rs_eye[running]
        of = of[running]
        dff = dff[:, running]

        # Fit data to 2D gaussian function
        if is_closedloop:
            rs_arrays = [np.log(rs * 100)]  # m-->cm
        else:
            rs_arrays = [np.log(rs * 100), np.log(rs_eye * 100)]  # m-->cm
        of = np.log(np.degrees(of))  # rad-->deg
        rs_min = param_range["rs_min"] * 100  # m-->cm
        rs_max = param_range["rs_max"] * 100  # m-->cm
        of_min = param_range["of_min"]  # degrees/s
        of_max = param_range["of_max"]  # degrees/s
        lower_bounds = [
            -np.inf,
            np.log(rs_min),
            np.log(of_min),
            -np.inf,
            -np.inf,
            0,
            -np.inf,
        ]
        upper_bounds = [
            np.inf,
            np.log(rs_max),
            np.log(of_max),
            np.inf,
            np.inf,
            np.radians(90),
            np.inf,
        ]
        for i_rs, rs_to_use in enumerate(rs_arrays):
            if is_closedloop:
                rs_type = ""
            else:
                if i_rs == 0:
                    rs_type = "_actual"
                else:
                    rs_type = "_virtual"
            print(f"Fitting {protocol_sfx}{rs_type} running...")
            for iroi in tqdm(range(dff.shape[0])):
                popt, rsq = gaussian_2d_fit(
                    (rs_to_use, of),
                    dff[iroi, :],
                    lower_bounds,
                    upper_bounds,
                    min_sigma,
                    niter,
                )

                neurons_df.loc[iroi, f"preferred_RS_{protocol_sfx}{rs_type}"] = (
                    np.exp(popt[1]) / 100
                )  # m
                neurons_df.loc[
                    iroi, f"preferred_OF_{protocol_sfx}{rs_type}"
                ] = np.radians(
                    np.exp(popt[2])
                )  # rad/s
                neurons_df[f"gaussian_blob_popt_{protocol_sfx}{rs_type}"].iloc[
                    iroi
                ] = popt  # !! Calculated with RS in cm and OF in degrees/s
                neurons_df.loc[iroi, f"gaussian_blob_rsq_{protocol_sfx}{rs_type}"] = rsq
    neurons_df.to_pickle(session_folder / "plane0/neurons_df.pickle")

    return neurons_df
