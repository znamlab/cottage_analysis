import numpy as np
from pathlib import Path
from tqdm import tqdm
import flexiznam as flz
from cottage_analysis.analysis import common_utils, find_depth_neurons
from collections import namedtuple
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

print = partial(print, flush=True)

# TODO:
# 1. check iteration round number for RS OF fit (try on a few cells and see how many rounds are needed)

Gaussian2DParams = namedtuple(
    "Gaussian2DParams",
    ["log_amplitude", "x0", "y0", "log_sigma_x2", "log_sigma_y2", "theta", "offset"],
)

GaussianAdditiveParams = namedtuple(
    "GaussianAdditiveParams",
    [
        "log_amplitude_x",
        "log_amplitude_y",
        "x0",
        "y0",
        "log_sigma_x2",
        "log_sigma_y2",
        "offset",
    ],
)

Gaussian1DParams = namedtuple(
    "Gaussian1DParams",
    ["log_amplitude", "x0", "log_sigma_x2", "offset"],
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

Gaussian3DRFParams = namedtuple(
    "Gaussian3DRFParams",
    [
        "log_amplitude",
        "x0",
        "y0",
        "log_sigma_x2",
        "log_sigma_y2",
        "theta",
        "offset",
        "z0",
        "log_sigma_z",
    ],
)

Gabor3DRFParams = namedtuple(
    "Gabor3DRFParams",
    [
        "log_amplitude",
        "x0",
        "y0",
        "log_sigma_x2",
        "log_sigma_y2",
        "theta",
        "offset",
        "log_sf",
        "alpha",
        "phase",
        "z0",
        "log_sigma_z",
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


def gaussian_1d(
    x,
    log_amplitude,
    x0,
    log_sigma_x2,
    offset,
    min_sigma,
):
    sigma_x_sq = np.exp(log_sigma_x2) + min_sigma
    amplitude = np.exp(log_amplitude)
    g = offset + amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma_x_sq))
    return g


def gaussian_additive(
    xy_tuple,
    log_amplitude_x,
    log_amplitude_y,
    x0,
    y0,
    log_sigma_x2,
    log_sigma_y2,
    offset,
    min_sigma,
):
    (x, y) = xy_tuple
    sigma_x_sq = np.exp(log_sigma_x2) + min_sigma
    sigma_y_sq = np.exp(log_sigma_y2) + min_sigma
    amplitude_x = np.exp(log_amplitude_x)
    amplitude_y = np.exp(log_amplitude_y)
    g = (
        offset
        + amplitude_x * np.exp(-((x - x0) ** 2) / (2 * sigma_x_sq))
        + amplitude_y * np.exp(-((y - y0) ** 2) / (2 * sigma_y_sq))
    )
    return g


def gabor_2d(
    xy_tuple,
    log_amplitude,
    x0,
    y0,
    log_sigma_x2,
    log_sigma_y2,
    theta,
    offset,
    min_sigma,
    log_sf,
    alpha,
    phase,
):
    """2D Gabor function.

    Args:
        xy_tuple (tuple): (x, y) tuple
        log_amplitude (float): log amplitude
        x0 (float): x center
        y0 (float): y center
        log_sigma_x2 (float): log sigma x squared
        log_sigma_y2 (float): log sigma y squared
        theta (float): orientation
        offset (float): offset
        min_sigma (float): minimum sigma
        log_sf (float): spatial frequency
        alpha (float): grating direction in radians
        phase (float): preferred grating direction

    Returns:
        gabor: gabor value

    """
    (x, y) = xy_tuple
    sine_grating = np.sin(
        2 * np.pi * np.exp(log_sf) * (x * np.cos(alpha) + y * np.sin(alpha) - phase)
    )

    gabor = (
        gaussian_2d(
            xy_tuple,
            log_amplitude,
            x0,
            y0,
            log_sigma_x2,
            log_sigma_y2,
            theta,
            0,
            min_sigma,
        )
        * sine_grating
    )
    return gabor + offset


def gabor_3d_rf(
    xy_tuple,
    log_amplitude,
    x0,
    y0,
    log_sigma_x2,
    log_sigma_y2,
    theta,
    offset,
    log_sf,
    alpha,
    phase,
    z0,
    log_sigma_z,
    min_sigma,
):
    """3D Gabor function.

    Args:
        xy_tuple (tuple): (x, y) tuple
        log_amplitude (float): log amplitude
        x0 (float): x center
        y0 (float): y center
        log_sigma_x2 (float): log sigma x squared
        log_sigma_y2 (float): log sigma y squared
        theta (float): orientation
        offset (float): offset
        min_sigma (float): minimum sigma
        log_sf (float): spatial frequency
        alpha (float): grating direction in radians
        phase (float): grating phase
        z0 (float): z center
        log_sigma_z (float): log sigma z

    Returns:
        gabor: gabor value

    """
    (x, y, z) = xy_tuple
    gabor = gabor_2d(
        (x, y),
        log_amplitude,
        x0,
        y0,
        log_sigma_x2,
        log_sigma_y2,
        theta,
        0,
        min_sigma,
        log_sf,
        alpha,
        phase,
    )
    depth_tuning = np.exp(-((z - z0) ** 2) / (2 * np.exp(log_sigma_z) ** 2))
    return gabor * depth_tuning + offset


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


def gaussian_3d_rf(
    stim_tuple,
    log_amplitude,
    x0,
    y0,
    log_sigma_x2,
    log_sigma_y2,
    theta,
    offset,
    z0,
    log_sigma_z,
    min_sigma,
):
    (x, y, z) = stim_tuple
    rf = gaussian_2d(
        (x, y),
        log_amplitude,
        x0,
        y0,
        log_sigma_x2,
        log_sigma_y2,
        theta,
        0,
        min_sigma,
    )
    depth_tuning = np.exp(-((z - z0) ** 2) / (2 * np.exp(log_sigma_z) ** 2))
    return rf * depth_tuning + offset


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
        0,
        min_sigma,
    )
    tuning = direction_tuning(alpha, alpha0, log_kappa, dsi)
    return gaussian * tuning + offset


def fit_rs_of_tuning(
    trials_df,
    model="gaussian_2d",
    choose_trials=None,
    closedloop_only=False,
    rs_thr=0.01,
    param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
    niter=5,
    min_sigma=0.25,
    k_folds=1,
):
    # Set bounds for gaussian fit params
    if model == "gaussian_2d":
        model_sfx = "_g2d"
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

    elif model == "gaussian_additive":
        model_sfx = "_gadd"
        lower_bounds = GaussianAdditiveParams(
            log_amplitude_x=-np.inf,
            log_amplitude_y=-np.inf,
            x0=np.log(param_range["rs_min"]),
            y0=np.log(param_range["of_min"]),
            log_sigma_x2=-np.inf,
            log_sigma_y2=-np.inf,
            offset=-np.inf,
        )
        upper_bounds = GaussianAdditiveParams(
            log_amplitude_x=np.inf,
            log_amplitude_y=np.inf,
            x0=np.log(param_range["rs_max"]),
            y0=np.log(param_range["of_max"]),
            log_sigma_x2=np.inf,
            log_sigma_y2=np.inf,
            offset=np.inf,
        )

        def p0_func():
            return GaussianAdditiveParams(
                log_amplitude_x=np.random.normal(),
                log_amplitude_y=np.random.normal(),
                x0=np.random.uniform(
                    np.log(param_range["rs_min"]), np.log(param_range["rs_max"])
                ),
                y0=np.random.uniform(
                    np.log(param_range["of_min"]), np.log(param_range["of_max"])
                ),
                log_sigma_x2=np.random.normal(),
                log_sigma_y2=np.random.normal(),
                offset=np.random.normal(),
            )

    elif model == "gaussian_OF":
        model_sfx = "_gof"
        lower_bounds = Gaussian1DParams(
            log_amplitude=-np.inf,
            x0=np.log(param_range["of_min"]),
            log_sigma_x2=-np.inf,
            offset=-np.inf,
        )
        upper_bounds = Gaussian1DParams(
            log_amplitude=np.inf,
            x0=np.log(param_range["of_max"]),
            log_sigma_x2=np.inf,
            offset=np.inf,
        )

        def p0_func():
            return Gaussian1DParams(
                log_amplitude=np.random.normal(),
                x0=np.random.uniform(
                    np.log(param_range["of_min"]), np.log(param_range["of_max"])
                ),
                log_sigma_x2=np.random.normal(),
                offset=np.random.normal(),
            )

    # Initialize neurons_df with columns for ROI number
    neurons_df_temp = pd.DataFrame(
        columns=['roi'], data=np.arange(trials_df["dff_stim"].iloc[0].shape[1]))

    # Loop through all protocols (closed loop and open loop)
    if k_folds > 1:
        closedloop_only = True
    if closedloop_only:
        all_protocols = [1]
    else:
        all_protocols = trials_df.closed_loop.unique()
        assert len(all_protocols) <= 2, "More than 2 protocols detected!"
        
    for iprotocol, is_closedloop in enumerate(all_protocols):
        print(
            f"Process protocol {iprotocol+1}/{len(trials_df.closed_loop.unique())}..."
        )
        if is_closedloop:
            protocol_sfx = "closedloop"
        else:
            protocol_sfx = "openloop"
        trials_df_protocol = trials_df[trials_df.closed_loop == is_closedloop]
        trials_df_fit, choose_trial_nums, sfx = common_utils.choose_trials_subset(
            trials_df_protocol, choose_trials
        )

        # Concatenate arrays of RS/OF/dff from all trials together
        rs = np.concatenate(trials_df_fit["RS_stim"].values)
        rs_eye = np.concatenate(trials_df_fit["RS_eye_stim"].values)
        of = np.concatenate(trials_df_fit["OF_stim"].values)
        dff = np.concatenate(trials_df_fit["dff_stim"].values, axis=0)
        
        # give class labels to each depth
        depth_list = find_depth_neurons.find_depth_list(trials_df)
        log_depth_list = np.log(depth_list)
        log_depth_list = np.append(
            log_depth_list, (log_depth_list[-1] + log_depth_list[1] - log_depth_list[0])
        )
        log_depth_list = np.insert(
            log_depth_list, 0, (log_depth_list[0] - log_depth_list[1] + log_depth_list[0])
        )
        depth_list_expand = np.exp(log_depth_list)
        bins = (depth_list_expand[1:] + depth_list_expand[:-1]) / 2
        trials_df_fit["depth_label"] = pd.cut(trials_df_fit["depth"], bins=bins, labels=np.arange(len(depth_list)))
        trials_df_fit["depth_label"] = [[x]*len(y) for x, y in zip(trials_df_fit["depth_label"], trials_df_fit["RS_stim"])]
        depth_labels = np.concatenate(trials_df_fit["depth_label"].values)

        # Take out the values where running is below a certain threshold
        running = (
            (rs > rs_thr) & (rs_eye > rs_thr) & (~np.isnan(of))
        )  # !!! OF has a small number of frame = nan, investigate synchronisation.py
        rs = rs[running]
        rs_eye = rs_eye[running]
        of = of[running]
        dff = dff[running, :]
        depth_labels = depth_labels[running]

        # Fit data to 2D gaussian function
        if (is_closedloop) or (model == "gaussian_OF"):
            rs_arrays = [np.log(rs)]
        else:
            rs_arrays = [np.log(rs), np.log(rs_eye)]
        of = np.log(np.degrees(of))  # rad-->deg
        for i_rs, rs_to_use in enumerate(rs_arrays):
            if (is_closedloop) or (model == "gaussian_OF"):
                rs_type = ""
            elif i_rs == 0:
                rs_type = "_actual"
            else:
                rs_type = "_virtual"
            print(f"Fitting {protocol_sfx}{rs_type} running...")

            if (model == "gaussian_2d") or (model == "gaussian_additive"):
                if model == "gaussian_2d":
                    model_func_ = partial(gaussian_2d, min_sigma=min_sigma)
                elif model == "gaussian_additive":
                    model_func_ = partial(gaussian_additive, min_sigma=min_sigma)
                    
                # Initialize columns with nan
                neurons_df_temp[f"preferred_RS_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                neurons_df_temp[f"preferred_OF_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                neurons_df_temp[f"rsof_popt_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = [[np.nan]] * len(neurons_df_temp)
                neurons_df_temp[f"rsof_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                
                # If k_folds = 1, fit for all data
                if k_folds == 1:
                    # Fit for each neuron
                    for roi in tqdm(range(dff.shape[1])):
                        popt, rsq = common_utils.iterate_fit(
                            model_func_,
                            (rs_to_use, of),
                            dff[:, roi],
                            lower_bounds,
                            upper_bounds,
                            niter=niter,
                            p0_func=p0_func,
                        )

                        neurons_df_temp.at[
                            roi, f"preferred_RS_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = np.exp(popt[1])
                        neurons_df_temp.at[
                            roi, f"preferred_OF_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = np.radians(
                            np.exp(popt[2])
                        )  # rad/s
                        # !! Calculated with RS in m and OF in degrees/s
                        neurons_df_temp.at[
                            roi, f"rsof_popt_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = popt
                        neurons_df_temp.loc[
                            roi, f"rsof_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = rsq
                        
                # If k_folds > 1, fit for each fold, then get a test rsq for each neuron
                if k_folds > 1:
                    print("Fit with cross-validation...")
                    neurons_df_temp[f"rsof_test_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    neurons_df_temp[f"rsof_test_spearmanr_rval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    neurons_df_temp[f"rsof_test_spearmanr_pval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                    # Loop through each roi
                    for roi in tqdm(range(dff.shape[1])):
                        # loop through the folds
                        dff_pred_all = []
                        dff_test_all = []
                        for fold, (train_index, test_index) in enumerate(
                            stratified_kfold.split(dff[:, roi], depth_labels)):
                            dff_train, dff_test = dff[train_index, roi], dff[test_index, roi]
                            rs_train, rs_test = rs_to_use[train_index], rs_to_use[test_index]
                            of_train, of_test = of[train_index], of[test_index]
                            dff_test_all.append(dff_test)

                            popt, _ = common_utils.iterate_fit(
                                model_func_,
                                (rs_train, of_train),
                                dff_train,
                                lower_bounds,
                                upper_bounds,
                                niter=niter,
                                p0_func=p0_func,
                            )
                            dff_pred = model_func_((rs_test, of_test), *popt)
                            dff_pred_all.append(dff_pred)
                        rsq = common_utils.calculate_r_squared(
                            np.concatenate(dff_test_all), np.concatenate(dff_pred_all)
                        )
                        rval, pval = spearmanr(
                            np.concatenate(dff_test_all), np.concatenate(dff_pred_all)
                        )
                        neurons_df_temp.at[roi, f"rsof_test_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = rsq
                        neurons_df_temp.at[
                            roi, f"rsof_test_spearmanr_rval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = rval
                        neurons_df_temp.at[
                            roi, f"rsof_test_spearmanr_pval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = pval

            elif model == "gaussian_OF":
                model_func_ = partial(gaussian_1d, min_sigma=min_sigma)
                if k_folds == 1:
                    # Initialize columns with nan
                    neurons_df_temp[f"preferred_RS_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    neurons_df_temp[f"preferred_OF_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    neurons_df_temp[f"rsof_popt_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = [[np.nan]] * len(neurons_df_temp)
                    neurons_df_temp[f"rsof_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    
                    for roi in tqdm(range(dff.shape[1])):
                        popt, rsq = common_utils.iterate_fit(
                            model_func_,
                            of,
                            dff[:, roi],
                            lower_bounds,
                            upper_bounds,
                            niter=niter,
                            p0_func=p0_func,
                        )

                        neurons_df_temp.at[
                            roi, f"preferred_OF_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = np.radians(np.exp(popt[1]))
                        # !! Calculated with OF in degrees/s
                        neurons_df_temp.at[
                            roi, f"rsof_popt_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = popt
                        neurons_df_temp.loc[
                            roi, f"rsof_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = rsq
                
                # If k_folds > 1, fit for each fold, then get a test rsq for each neuron
                elif k_folds > 1:
                    print("Fit with cross-validation...")
                    neurons_df_temp[f"rsof_test_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    neurons_df_temp[f"rsof_test_spearmanr_rval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    neurons_df_temp[f"rsof_test_spearmanr_pval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = np.nan
                    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                    # Loop through each roi
                    for roi in tqdm(range(dff.shape[1])):
                        # loop through the folds
                        dff_pred_all = []
                        dff_test_all = []
                        for fold, (train_index, test_index) in enumerate(
                            stratified_kfold.split(dff[:, roi], depth_labels)
                        ):
                            dff_train, dff_test = dff[train_index, roi], dff[test_index, roi]
                            of_train, of_test = of[train_index], of[test_index]
                            dff_test_all.append(dff_test)

                            popt, _ = common_utils.iterate_fit(
                                model_func_,
                                of_train,
                                dff_train,
                                lower_bounds,
                                upper_bounds,
                                niter=niter,
                                p0_func=p0_func,
                            )
                            dff_pred = model_func_(of_test, *popt)
                            dff_pred_all.append(dff_pred)
                        rsq = common_utils.calculate_r_squared(
                            np.concatenate(dff_test_all), np.concatenate(dff_pred_all)
                        )
                        rval, pval = spearmanr(
                            np.concatenate(dff_test_all), np.concatenate(dff_pred_all)
                        )
                        neurons_df_temp.at[roi, f"rsof_test_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}"] = rsq
                        neurons_df_temp.at[
                            roi, f"rsof_test_spearmanr_rval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = rval
                        neurons_df_temp.at[
                            roi, f"rsof_test_spearmanr_pval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"
                        ] = pval
                        
        if k_folds > 1:
            neurons_df_temp = neurons_df_temp[["roi", 
                                               f"rsof_test_rsq_{protocol_sfx}{rs_type}{sfx}{model_sfx}", 
                                               f"rsof_test_spearmanr_rval_{protocol_sfx}{rs_type}{sfx}{model_sfx}",
                                               f"rsof_test_spearmanr_pval_{protocol_sfx}{rs_type}{sfx}{model_sfx}"]]
            

    return neurons_df_temp


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
