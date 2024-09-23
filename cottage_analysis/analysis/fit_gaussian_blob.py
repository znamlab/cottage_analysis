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


def gaussian_OF(
    xy_tuple,
    log_amplitude,
    x0,
    log_sigma_x2,
    offset,
    min_sigma,
):
    (rs, of) = xy_tuple
    x = of
    g = gaussian_1d(x,
                    log_amplitude,
                    x0,
                    log_sigma_x2,
                    offset,
                    min_sigma,
                )
    return g

def gaussian_RS(
    xy_tuple,
    log_amplitude,
    x0,
    log_sigma_x2,
    offset,
    min_sigma,
):
    (rs, of) = xy_tuple
    x = rs
    g = gaussian_1d(x,
                    log_amplitude,
                    x0,
                    log_sigma_x2,
                    offset,
                    min_sigma,
                )
    return g


def gaussian_ratio(
    xy_tuple,
    log_amplitude,
    x0,
    log_sigma_x2,
    offset,
    min_sigma,
):
    (rs, of) = xy_tuple
    x = rs - of # ratio of logged rs/of
    g = gaussian_1d(x,
                    log_amplitude,
                    x0,
                    log_sigma_x2,
                    offset,
                    min_sigma,
                )
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


def initial_fit_conditions(
    model, 
    param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
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
            
    elif model == "gaussian_RS":
        model_sfx = "_grs"
        lower_bounds = Gaussian1DParams(
            log_amplitude=-np.inf,
            x0=np.log(param_range["rs_min"]),
            log_sigma_x2=-np.inf,
            offset=-np.inf,
        )
        upper_bounds = Gaussian1DParams(
            log_amplitude=np.inf,
            x0=np.log(param_range["rs_max"]),
            log_sigma_x2=np.inf,
            offset=np.inf,
        )

        def p0_func():
            return Gaussian1DParams(
                log_amplitude=np.random.normal(),
                x0=np.random.uniform(
                    np.log(param_range["rs_min"]), np.log(param_range["rs_max"])
                ),
                log_sigma_x2=np.random.normal(),
                offset=np.random.normal(),
            )
            
    elif model == "gaussian_ratio":
        model_sfx = "_gratio"
        lower_bounds = Gaussian1DParams(
            log_amplitude=-np.inf,
            x0=np.log(param_range["rs_min"]/param_range["of_max"]),
            log_sigma_x2=-np.inf,
            offset=-np.inf,
        )
        upper_bounds = Gaussian1DParams(
            log_amplitude=np.inf,
            x0=np.log(param_range["rs_max"]/param_range["of_min"]),
            log_sigma_x2=np.inf,
            offset=np.inf,
        )

        def p0_func():
            return Gaussian1DParams(
                log_amplitude=np.random.normal(),
                x0=np.random.uniform(
                    np.log(param_range["rs_min"]/param_range["of_max"]),
                    np.log(param_range["rs_max"]/param_range["of_min"]),
                ),
                log_sigma_x2=np.random.normal(),
                offset=np.random.normal(),
            )
            
    return model_sfx, lower_bounds, upper_bounds, p0_func


def depth_class_labels(trials_df):
        '''Give class labels to each depth'''
        
        # expand the depth list to one more log distance at both ends
        depth_list = find_depth_neurons.find_depth_list(trials_df)
        log_depth_list = np.log(depth_list)
        log_depth_list = np.append(
            log_depth_list, (log_depth_list[-1] + log_depth_list[1] - log_depth_list[0])
        )
        log_depth_list = np.insert(
            log_depth_list,
            0,
            (log_depth_list[0] - log_depth_list[1] + log_depth_list[0]),
        )
        depth_list_expand = np.exp(log_depth_list)
        
        # find the corresponding depth label according to depth bins for each trial
        bins = (depth_list_expand[1:] + depth_list_expand[:-1]) / 2
        trials_df["depth_label"] = pd.cut(
            trials_df["depth"], bins=bins, labels=np.arange(len(depth_list))
        )
        
        # copy depth label as many times as the number of frames in each trial
        trials_df["depth_labels"] = [
            [x] * len(y)
            for x, y in zip(trials_df["depth_label"], trials_df["RS_stim"])
        ]
        return trials_df
   

def fit_rs_of_tuning(
    trials_df,
    model="gaussian_2d",
    choose_trials=None,
    trial_sfx="",
    rs_thr=0.01,
    param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
    niter=5,
    min_sigma=0.25,
    k_folds=1,
    random_state=42,
    run_closedloop_only=False,
    run_openloop_only=False,
):
    def process_rs_of_for_fit(trials_df, trial_list=[], rs_thr=0.01):
        # take a subset of trials
        trials_df_part = trials_df.iloc[trial_list] if len(trial_list)>0 else trials_df
        
        # take the rs, of, dff, depth_labels from those trials
        rs = np.concatenate(trials_df_part["RS_stim"].values)
        rs_eye = np.concatenate(trials_df_part["RS_eye_stim"].values)
        of = np.concatenate(trials_df_part["OF_stim"].values)
        dff = np.concatenate(trials_df_part["dff_stim"].values, axis=0)
        depth_labels = np.concatenate(trials_df_part["depth_labels"].values)
        
        # choose frames that are above a certain running speed threshold
        running = (rs > rs_thr) & (rs_eye > rs_thr) & (~np.isnan(of))
        rs = np.log(rs[running])
        rs_eye = np.log(rs_eye[running])
        of = np.log(np.degrees(of[running]))  # fit using of in deg
        dff = dff[running, :]
        depth_labels = depth_labels[running]
        
        return rs, rs_eye, of, dff, depth_labels


    # Set bounds for gaussian fit params
    model_sfx, lower_bounds, upper_bounds, p0_func = initial_fit_conditions(model=model, param_range=param_range,)

    # initialize neurons_df with columns for ROI number
    neurons_df_temp = pd.DataFrame(
        columns=["roi"], data=np.arange(trials_df["dff_stim"].iloc[0].shape[1])
    )

    # Choose trials
    if choose_trials is not None and isinstance(
        choose_trials, list
    ): # choose a list of trials from all trials (including openloop and closed loop)
        trials_df_select, choose_trial_nums, trial_sfx = common_utils.choose_trials_subset(
            trials_df, choose_trials, sfx=trial_sfx,
        )
    else: # Otherwise, if choose_trials is "even" or "odd", choose trials within a certain protocol below
        trials_df_select = trials_df
    
    # Loop through all protocols (closed loop and open loop)
    if run_closedloop_only:
        all_protocols = [1]
        print("Running closed loop fitting only...")
    elif run_openloop_only:
        if 0 in trials_df_select.closed_loop.unique():
            all_protocols = [0]
            print("Running open loop fitting only...")
        else:
            all_protocols = []
            print("ERROR:Open loop protocol not found!")
    else:
        all_protocols = [1] if (k_folds > 1) else trials_df_select.closed_loop.unique()
    assert len(all_protocols) <= 2, "More than 2 protocols detected!"
    for is_closedloop in all_protocols:
        protocol_sfx = "closedloop" if is_closedloop else "openloop"
        print(
            f"Process protocol {protocol_sfx}/{len(trials_df_select.closed_loop.unique())}..."
        )

        trials_df_fit = trials_df_select[trials_df_select.closed_loop == is_closedloop].copy()
        if choose_trials is not None and isinstance(
            choose_trials, list
        ): # a list of trials from all trials have already been chosen
            # choose only closed loop or open loop trials
            trials_df_fit = trials_df_fit
        else: # Otherwise, if choose_trials is "even" or "odd", choose trials within a certain protocol
            trials_df_fit, choose_trial_nums, trial_sfx = common_utils.choose_trials_subset(
                trials_df_fit, choose_trials, sfx=trial_sfx,
            )

        # give class labels to each depth
        trials_df_fit = depth_class_labels(trials_df_fit)
        depth_label = trials_df_fit["depth_label"].values
        depth_labels = np.concatenate(trials_df_fit["depth_labels"].values)

        # initialize a model function
        model_func_ = partial(globals()[model], min_sigma=min_sigma)
        
        rs_types_openloop = ["_actual", "_virtual"]
        # if k_folds = 1, fit for all data
        if k_folds == 1: 
            # process data for fitting (rs, rs_eye, of are all logged)
            rs, rs_eye, of, dff, depth_labels = process_rs_of_for_fit(trials_df_fit, trial_list=[], rs_thr=rs_thr)

            # loop between actual and virtual running speeds
            # rs_arrays = [rs] if ((is_closedloop) or model == "gaussian_OF") else [rs, rs_eye] # only use virtual running speed if it's openloop and fits for models other than gaussian_OF
            rs_arrays = [rs] # don't fit with virtual running speed as it's never been used
    
            for i_rs, rs_to_use in enumerate(rs_arrays):
                rs_type = "" if is_closedloop else rs_types_openloop[i_rs]    
                print(f"Fitting {protocol_sfx}{rs_type} running...")

                # initialize columns to save
                neurons_df_temp[
                    f"rsof_popt_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                ] = [[np.nan]] * len(neurons_df_temp)
                neurons_df_temp[
                    f"rsof_minSigma_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                ] = min_sigma
                # if choose_trials is not None:
                #     neurons_df_temp[
                #         f"rsof_chooseTrials_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                #     ] = choose_trials

                # fit for each neuron
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

                    # Assign values to neurons_df_temp
                    if (model == "gaussian_additive") or (model == "gaussian_2d"):
                        neurons_df_temp.at[
                            roi, f"preferred_RS_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                        ] = np.exp(popt[1])
                        
                        neurons_df_temp.at[
                            roi, f"preferred_OF_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                        ] = np.radians(
                            np.exp(popt[2]))  # rad/s
                        
                    elif model == "gaussian_OF":
                        neurons_df_temp.at[
                            roi, f"preferred_OF_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                        ] = np.radians(np.exp(popt[1]))   # rad/s
                        
                    elif model == "gaussian_RS":
                        neurons_df_temp.at[
                            roi, f"preferred_RS_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                        ] = np.exp(popt[1])   # m/s
                        
                    elif model == "gaussian_ratio":
                        neurons_df_temp.at[
                            roi, f"preferred_RSOFratio_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                        ] = np.degrees(np.exp(popt[1]))  # m/deg --> m/deg * deg/rad = m/rad 
                    
                    neurons_df_temp.at[
                        roi, f"rsof_popt_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                    ] = popt  # !! calculated with RS in m and OF in degrees/s
                    neurons_df_temp.at[
                        roi, f"rsof_rsq_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                    ] = rsq
                    
                    dff_pred = model_func_((rs_to_use, of), *popt)
                    rval, pval = spearmanr(dff[:, roi], dff_pred)
                    neurons_df_temp.at[
                        roi, f"rsof_spearmanr_rval_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                    ] = rval
                    neurons_df_temp.at[
                        roi, f"rsof_spearmanr_pval_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                    ] = pval
                    

        # If k_folds > 1, fit for each fold, then get a test rsq for each neuron
        if k_folds > 1:
            print(f"Fit with {k_folds} fold cross-validation...")
            # train/test split based on trials 
            stratified_kfold = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=random_state,
                )
    
            # make a list of arrays for all folds
            data_all = {
                "train": {"rs": [], "rs_eye": [], "of": [], "dff": [], "depth_labels": []},
                "test": {"rs": [], "rs_eye": [], "of": [], "dff": [], "depth_labels": []}
            }
            for fold, (train_index, test_index) in enumerate(
                stratified_kfold.split(np.repeat(1,len(depth_label)), depth_label)
            ):
                for data_type, data_idx in [("train", train_index), ("test", test_index)]:
                    rs, rs_eye, of, dff, depth_labels = process_rs_of_for_fit(trials_df_fit, trial_list=data_idx, rs_thr=rs_thr)
                    data_all[data_type]["rs"].append(rs)
                    data_all[data_type]["rs_eye"].append(rs_eye)
                    data_all[data_type]["of"].append(of)
                    data_all[data_type]["dff"].append(dff)
                    data_all[data_type]["depth_labels"].append(depth_labels)
                
            # take actual or virtual running speeds
            # rs_arrays_train = [data_all["train"]["rs"]] if ((is_closedloop) or model == "gaussian_OF") else [data_all["train"]["rs"], data_all["train"]["rs_eye"]]
            # rs_arrays_test = [data_all["test"]["rs"]] if ((is_closedloop) or model == "gaussian_OF") else [data_all["test"]["rs"], data_all["test"]["rs_eye"]]
            rs_arrays_train = [data_all["train"]["rs"]] # don't fit with virtual running speed as it's never been used
            rs_arrays_test = [data_all["test"]["rs"]] # don't fit with virtual running speed as it's never been used

            for i_rs, (rs_to_use_train_all, rs_to_use_test_all) in enumerate(zip(rs_arrays_train, rs_arrays_test)):
                rs_type = "" if is_closedloop else rs_types_openloop[i_rs] 
                print(f"Fitting {protocol_sfx}{rs_type} running...")
                
                # initialize columns to save with nan
                for param in ["rsq", "spearmanr_rval", "spearmanr_pval"]:
                    neurons_df_temp[
                        f"rsof_train_{param}_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                    ] = [[np.nan]] * len(neurons_df_temp)
                neurons_df_temp[
                    f"rsof_train_popt_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                ] = [[[np.nan]]] * len(neurons_df_temp)
                neurons_df_temp[
                    f"rsof_minSigma_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                ] = min_sigma   
                neurons_df_temp[f"rsof_randomState_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"] = random_state
                neurons_df_temp[f"rsof_kFolds_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"] = k_folds
                # if choose_trials is not None:
                #     neurons_df_temp[
                #         f"rsof_chooseTrials_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                #     ] = choose_trials   
                
                # Loop through each roi
                for roi in tqdm(range(dff.shape[1])):
                    # loop through the folds
                    dff_pred_all, rsq_train, rval_train, pval_train, popt_train = [], [], [], [], []
                    
                    # Loop through each fold of cross validation
                    for rs_to_use_train, rs_to_use_test, of_train, of_test, dff_train, in zip(
                        rs_to_use_train_all,
                        rs_to_use_test_all,
                        data_all["train"]["of"],
                        data_all["test"]["of"],
                        data_all["train"]["dff"],
                        ): 

                        popt, rsq = common_utils.iterate_fit(
                            model_func_,
                            (rs_to_use_train, of_train),
                            dff_train[:, roi],
                            lower_bounds,
                            upper_bounds,
                            niter=niter,
                            p0_func=p0_func,
                        )
                        dff_pred = model_func_((rs_to_use_test, of_test), *popt)
                        dff_pred_all.append(dff_pred)
                        
                        # calculate r-square for train set
                        dff_pred_train = model_func_((rs_to_use_train, of_train), *popt)
                        rval, pval = spearmanr(dff_train[:, roi], dff_pred_train)
                        rsq_train.append(rsq)
                        rval_train.append(rval)
                        pval_train.append(pval)
                        popt_train.append(popt)
                        
                    rsq = common_utils.calculate_r_squared(
                        np.concatenate(data_all["test"]["dff"])[:, roi], np.concatenate(dff_pred_all)
                    )
                    rval, pval = spearmanr(
                        np.concatenate(data_all["test"]["dff"])[:, roi], np.concatenate(dff_pred_all)
                    )
                    
                    # Save values to neurons_df_temp
                    for param, value in zip(["train_rsq",
                                             "train_popt", 
                                             "train_spearmanr_rval",
                                             "train_spearmanr_pval",
                                             "test_rsq",
                                             "test_spearmanr_rval",
                                             "test_spearmanr_pval"],
                                            [rsq_train,
                                             popt_train,
                                             rval_train,
                                             pval_train,
                                             rsq,
                                             rval,
                                             pval]):
                        neurons_df_temp.at[
                            roi, f"rsof_{param}_{protocol_sfx}{rs_type}{trial_sfx}{model_sfx}"
                        ] = value

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
