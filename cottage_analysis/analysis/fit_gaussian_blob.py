import numpy as np
from pathlib import Path
from tqdm import tqdm
import flexiznam as flz
from cottage_analysis.analysis import common_utils
from collections import namedtuple
import pandas as pd
from functools import partial

print = partial(print, flush=True)

Gaussian2DParams = namedtuple(
    "Gaussian2DParams",
    ["log_amplitude", "x0", "y0", "log_sigma_x2", "log_sigma_y2", "theta", "offset"],
)

GratingParams = namedtuple(
    "GratingParams",
    [
        "log_amplitude",
        "sf0",
        "tf0",
        "log_sigma_x2",
        "log_sigma_y2",
        "theta",
        "offset",
        "alpha0",
        "log_kappa",
        "dsi",
    ],
)


def gaussian_2d(
    xy_tuple,
    log_amplitude,
    x0,
    y0,
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
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )
    return g


def direction_tuning(alpha, alpha0, log_kappa, dsi):
    """Direction tuning function based on von mises distribution.

    Args:
        alpha (float): grating direction in radians
        alpha0 (float): preferred grating direction
        log_kappa (float): shape parameter, defines tuning width
        dsi: direction selectivity index, between 0 and 1

    Returns:
        tuning: tuning value

    """
    kappa = np.exp(log_kappa)
    resp = np.exp(kappa * np.cos(alpha - alpha0)) + (1 - dsi) * np.exp(
        kappa * np.cos(alpha - alpha0 - np.pi)
    )
    peak_resp = np.exp(kappa) + (1 - dsi) * np.exp(-kappa)
    return resp / peak_resp


def grating_tuning(
    stim_tuple,
    log_amplitude,
    x0,
    y0,
    log_sigma_x2,
    log_sigma_y2,
    theta,
    offset,
    alpha0,
    log_kappa,
    dsi,
    min_sigma,
):
    """2D gaussian function with direction tuning."""
    (sf, tf, alpha) = stim_tuple
    gaussian = gaussian_2d(
        (sf, tf),
        log_amplitude,
        x0,
        y0,
        log_sigma_x2,
        log_sigma_y2,
        theta,
        offset,
        min_sigma,
    )
    tuning = direction_tuning(alpha, alpha0, log_kappa, dsi)
    return gaussian * tuning


def fit_rs_of_tuning(
    trials_df,
    neurons_df,
    neurons_ds,
    rs_thr=0.01,
    param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
    niter=5,
    min_sigma=0.25,
    conflicts="skip",
):
    # session paths
    session_folder = neurons_ds.path_full.parent

    if conflicts == "skip":
        return neurons_df, neurons_ds

    # Initialize neurons_df with columns for RS/OF tuning
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

    # Set bounds for gaussian fit params
    lower_bounds = Gaussian2DParams(
        log_amplitude=-np.inf,
        x0=np.log(param_range["rs_min"]),
        y0=np.log(param_range["of_min"]),
        log_sigma_x2=-np.inf,
        log_sigma_y2=-np.inf,
        theta=0,
        offset=-np.inf,
    )
    upper_bounds = Gaussian2DParams(
        log_amplitude=np.inf,
        x0=np.log(param_range["rs_max"]),
        y0=np.log(param_range["of_max"]),
        log_sigma_x2=np.inf,
        log_sigma_y2=np.inf,
        theta=np.pi / 2,
        offset=np.inf,
    )

    def p0_func():
        # edit the code below to use a namedtupled instead of a list
        return Gaussian2DParams(
            log_amplitude=np.random.normal(),
            x0=np.random.uniform(
                np.log(param_range["rs_min"]), np.log(param_range["rs_max"])
            ),
            y0=np.random.uniform(
                np.log(param_range["of_min"]), np.log(param_range["of_max"])
            ),
            log_sigma_x2=np.random.normal(),
            log_sigma_y2=np.random.normal(),
            theta=np.random.uniform(0, 0.5 * np.pi),
            offset=np.random.normal(),
        )

    # Loop through all protocols
    for iprotocol, is_closedloop in enumerate(trials_df.closed_loop.unique()):
        print(
            f"---------Process protocol {iprotocol+1}/{len(trials_df.closed_loop.unique())}---------"
        )
        if is_closedloop:
            protocol_sfx = "closed_loop"
        else:
            protocol_sfx = "open_loop"
        trials_df_protocol = trials_df[trials_df.closed_loop == is_closedloop]

        # Concatenate arrays of RS/OF/dff from all trials together
        rs = np.concatenate(trials_df_protocol["RS_stim"].values)
        rs_eye = np.concatenate(trials_df_protocol["RS_eye_stim"].values)
        of = np.concatenate(trials_df_protocol["OF_stim"].values)
        dff = np.concatenate(trials_df_protocol["dff_stim"].values, axis=0)

        # Take out the values where running is below a certain threshold
        running = (
            (rs > rs_thr) & (rs_eye > rs_thr) & (~np.isnan(of))
        )  # !!! OF has a small number of frame = nan, investigate synchronisation.py
        rs = rs[running]
        rs_eye = rs_eye[running]
        of = of[running]
        dff = dff[running, :]

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
            for iroi in tqdm(range(dff.shape[1])):
                gaussian_2d_ = partial(gaussian_2d, min_sigma=min_sigma)
                popt, rsq = common_utils.iterate_fit(
                    gaussian_2d_,
                    (rs_to_use, of),
                    dff[:, iroi],
                    lower_bounds,
                    upper_bounds,
                    niter=niter,
                    p0_func=p0_func,
                )

                neurons_df.at[iroi, f"preferred_RS_{protocol_sfx}{rs_type}"] = np.exp(
                    popt[1]
                )
                neurons_df.at[
                    iroi, f"preferred_OF_{protocol_sfx}{rs_type}"
                ] = np.radians(
                    np.exp(popt[2])
                )  # rad/s
                # !! Calculated with RS in m and OF in degrees/s
                neurons_df.at[
                    iroi, f"gaussian_blob_popt_{protocol_sfx}{rs_type}"
                ] = popt
                neurons_df.loc[iroi, f"gaussian_blob_rsq_{protocol_sfx}{rs_type}"] = rsq

    # save neurons_df
    neurons_df.to_pickle(session_folder / "neurons_df.pickle")

    # update flexilims
    neurons_ds.extra_attributes["fit_RSOF_rs_min"] = param_range["rs_min"]
    neurons_ds.extra_attributes["fit_RSOF_rs_max"] = param_range["rs_max"]
    neurons_ds.extra_attributes["fit_RSOF_of_min"] = param_range["of_min"]
    neurons_ds.extra_attributes["fit_RSOF_of_max"] = param_range["of_max"]
    neurons_ds.extra_attributes["fit_RSOF_min_sigma"] = min_sigma
    neurons_ds.update_flexilims(mode="overwrite")

    return neurons_df, neurons_ds


def fit_sftf_tuning(trials_df, niter=5, min_sigma=0.25):
    """
    Fit spatial frequency and temporal frequency tuning with 2d gaussian function.

    Args:
        trials_df (pd.DataFrame): dataframe with `SpatialFrequency`, `TemporalFrequency` and `Angle` columns
            and integer column names for each ROI.
        niter (int, optional): Number of iterations for fitting the gaussian function. Defaults to 5.
        min_sigma (float, optional): Minimum value for sigma. Defaults to 0.25.

    Returns:
        neurons_df (DataFrame): A dataframe that contains the analysed properties for each ROI.

    """
    trials_df["log_SF"] = np.log(trials_df["SpatialFrequency"])
    trials_df["log_TF"] = np.log(trials_df["TemporalFrequency"])
    trials_df["Angle_rad"] = np.deg2rad(trials_df["Angle"])
    X = trials_df[["log_SF", "log_TF", "Angle_rad"]].to_numpy()
    lower_bounds = GratingParams(
        log_amplitude=-np.inf,
        sf0=trials_df["log_SF"].min() - 1,
        tf0=trials_df["log_TF"].min() - 1,
        log_sigma_x2=-np.inf,
        log_sigma_y2=-np.inf,
        theta=0,
        offset=-np.inf,
        alpha0=0,
        log_kappa=-np.inf,
        dsi=0,
    )
    upper_bounds = GratingParams(
        log_amplitude=np.inf,
        sf0=trials_df["log_SF"].max() + 1,
        tf0=trials_df["log_TF"].max() + 1,
        log_sigma_x2=np.inf,
        log_sigma_y2=np.inf,
        theta=0.5 * np.pi,
        offset=np.inf,
        alpha0=2 * np.pi,
        log_kappa=np.inf,
        dsi=1,
    )

    def p0_func():
        # edit the code below to use a namedtupled instead of a list
        return GratingParams(
            log_amplitude=np.random.normal(),
            sf0=trials_df.groupby("log_SF")[roi].mean().idxmax(),
            tf0=trials_df.groupby("log_TF")[roi].mean().idxmax(),
            log_sigma_x2=np.random.normal(),
            log_sigma_y2=np.random.normal(),
            theta=np.random.uniform(0, 0.5 * np.pi),
            offset=np.random.normal(),
            alpha0=trials_df.groupby("Angle_rad")[roi].mean().idxmax(),
            log_kappa=np.random.normal(),
            dsi=np.random.uniform(0, 1),
        )

    grating_tuning_ = partial(grating_tuning, min_sigma=min_sigma)
    params = []
    rsqs = []
    # int type columns correspond to ROIs
    int_cols = [type(col) == int for col in trials_df.columns]
    trials_df.columns[int_cols]
    for roi in tqdm(trials_df.columns[int_cols]):
        popt, rsq = common_utils.iterate_fit(
            grating_tuning_,
            X.T,
            trials_df[roi],
            lower_bounds,
            upper_bounds,
            niter=niter,
            p0_func=p0_func,
            verbose=False,
        )
        params.append(GratingParams(*popt))
        rsqs.append(rsq)

    neurons_df = pd.DataFrame(params)
    neurons_df["rsq"] = rsqs
    return neurons_df
