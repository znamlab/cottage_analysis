from functools import partial
import numpy as np
from pathlib import Path
from tqdm import tqdm
import flexiznam as flz
from cottage_analysis.analysis import common_utils
import pandas as pd

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
    trials_df = pd.read_pickle(session_folder / "plane0/trials_df.pickle")
    neurons_df = pd.read_pickle(session_folder / "plane0/neurons_df.pickle")

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
    lower_bounds = [
        -np.inf,
        np.log(param_range["rs_min"]),
        np.log(param_range["of_min"]),
        -np.inf,
        -np.inf,
        0,
        -np.inf,
    ]
    upper_bounds = [
        np.inf,
        np.log(param_range["rs_max"]),
        np.log(param_range["of_max"]),
        np.inf,
        np.inf,
        np.radians(90),
        np.inf,
    ]
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
            rs_arrays = [np.log(rs)]
        else:
            rs_arrays = [np.log(rs), np.log(rs_eye)]
        of = np.log(np.degrees(of))  # rad-->deg
        for i_rs, rs_to_use in enumerate(rs_arrays):
            if is_closedloop:
                rs_type = ""
            elif i_rs == 0:
                rs_type = "_actual"
            else:
                rs_type = "_virtual"
            print(f"Fitting {protocol_sfx}{rs_type} running...")
            for iroi in tqdm(range(dff.shape[0])):
                gaussian_2d_ = partial(gaussian_2d, min_sigma=min_sigma)
                popt, rsq = common_utils.iterate_fit(
                    gaussian_2d_,
                    (rs_to_use, of),
                    dff[iroi, :],
                    lower_bounds,
                    upper_bounds,
                    niter=niter,
                )

                neurons_df.loc[iroi, f"preferred_RS_{protocol_sfx}{rs_type}"] = np.exp(
                    popt[1]
                )
                neurons_df.loc[
                    iroi, f"preferred_OF_{protocol_sfx}{rs_type}"
                ] = np.radians(
                    np.exp(popt[2])
                )  # rad/s
                neurons_df[f"gaussian_blob_popt_{protocol_sfx}{rs_type}"].iloc[
                    iroi
                ] = popt  # !! Calculated with RS in m and OF in degrees/s
                neurons_df.loc[iroi, f"gaussian_blob_rsq_{protocol_sfx}{rs_type}"] = rsq
    neurons_df.to_pickle(session_folder / "plane0/neurons_df.pickle")

    return neurons_df
