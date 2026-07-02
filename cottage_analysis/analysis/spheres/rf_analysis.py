"""RF post-analysis: functions that operate on already-fitted RF coefficients.

Includes RF center finding, preferred-depth fitting, data loading and
aggregation, and RF gradient computation.
"""

from functools import partial
import os
import warnings

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from cottage_analysis.analysis import common_utils
import cottage_analysis.analysis.fit_gaussian_blob as fit_gaussian_blob
from cottage_analysis.analysis.spheres.rf_fitting import find_sig_rfs


def find_rf_centers(
    neurons_df,
    ndepths=8,
    frame_shape=(16, 24),
    is_closed_loop=1,
    resolution=5,
    coef=None,
):
    """Find the spatial center and best depth of receptive fields.

    Calculates the azimuth, elevation, and depth index corresponding to the
    maximum value of the fitted RF coefficients averaged across folds.

    Args:
        neurons_df (pd.DataFrame): DataFrame containing RF coefficients.
        ndepths (int, optional): Number of depths used in stimulus. Defaults to 8.
        frame_shape (tuple, optional): (n_ele, n_azi) shape of stimulus frames.
            Defaults to (16, 24).
        is_closed_loop (int, optional): Whether to use closed-loop coefficients.
            Defaults to 1.
        resolution (int, optional): Degrees per pixel. Defaults to 5.
        coef (np.ndarray, optional): Pre-stacked coefficients (n_neurons, n_folds,
            n_features). If None, loaded from neurons_df.

    Returns:
        tuple: (rf_azi, rf_ele, rf_idepth, coef)
            - rf_azi: Array of azimuth centers in degrees.
            - rf_ele: Array of elevation centers in degrees.
            - rf_idepth: Array of best depth indices.
            - coef: Staked coefficient array used for calculation.
    """
    if is_closed_loop:
        sfx = "_closedloop"
    else:
        sfx = "_openloop"
    if coef is None:
        coef = np.stack(neurons_df[f"rf_coef{sfx}"].values)
    coef_ = (coef[:, :, :-1]).reshape(
        coef.shape[0], coef.shape[1], ndepths, frame_shape[0], frame_shape[1]
    )
    # Suppress warning emitted by all-NaN ROIs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coef_mean = np.nanmean(coef_, axis=1)

    # Find the center (index of maximum value of fitted RF)
    is_all_nan = np.all(np.isnan(coef_mean), axis=(1, 2, 3))
    max_idx = np.zeros((coef_mean.shape[0], 3), dtype=int)
    for i in range(coef_mean.shape[0]):
        if not is_all_nan[i]:
            max_idx[i] = np.unravel_index(
                np.nanargmax(coef_mean[i, :, :, :]), coef_mean[0, :, :, :].shape
            )

    def index_to_deg(idx, resolution=resolution, n_ele=80):
        azi = (idx[:, 2] + 0.5) * resolution
        ele = (idx[:, 1] + 0.5 - n_ele / 2) * resolution
        return azi, ele

    azi, ele = index_to_deg(max_idx, n_ele=frame_shape[0])
    idepth = max_idx[:, 0].astype(float)
    azi[is_all_nan] = np.nan
    ele[is_all_nan] = np.nan
    idepth[is_all_nan] = np.nan
    neurons_df["rf_azi"] = azi
    neurons_df["rf_ele"] = ele
    return azi, ele, idepth, coef


def fit_rf_preferred_depth(
    neurons_df,
    depths,
    frame_shape=(16, 24),
    is_closed_loop=1,
    niter=10,
    min_sigma=0.5,
    depth_bounds=(np.log(0.02), np.log(20)),
    use_multidepth=False,
    suffix=None,
):
    """Fit a 1D Gaussian across depths at the best azimuth/elevation pixel.

    For each neuron, this function:
      1. Finds the (azi, ele) pixel with the maximum mean RF coefficient
         (same as find_rf_centers).
      2. Extracts the depth tuning curve at that pixel.
      3. Fits a 1D Gaussian in log-depth space to estimate a continuous
         preferred depth.

    Args:
        neurons_df (pd.DataFrame): DataFrame with RF coefficient columns.
        depths (array-like): Depth values in meters.
        frame_shape (tuple): (n_ele, n_azi) spatial shape. Default (16, 24).
        is_closed_loop (int): Whether to use closed loop coefficients.
            Default 1.
        niter (int): Number of fitting iterations to avoid local minima.
            Default 10.
        min_sigma (float): Minimum sigma for the Gaussian. Default 0.5.
        depth_bounds (tuple): Tuple of (min_depth, max_depth) in log(meters).
            Default (np.log(0.02), np.log(20)).
        use_multidepth (bool): Whether to use multidepth coefficients.
            Default False.
        suffix (str, optional): Custom suffix for the columns. Default None.

    Returns:
        tuple: (rf_preferred_depth, rf_depth_popt, rf_depth_rsq)
            - rf_preferred_depth: array of fitted preferred depths (meters)
            - rf_depth_popt: list of fit parameters per neuron
            - rf_depth_rsq: array of R-squared values for the fit
    """
    depths = np.asarray(depths)
    ndepths = len(depths)
    log_depths = np.log(depths)

    # Ensure depth_bounds covers the range of log_depths to avoid ValueError in curve_fit
    if depth_bounds is None:
        depth_bounds = (log_depths.min(), log_depths.max())
    else:
        depth_bounds = (
            min(depth_bounds[0], log_depths.min()),
            max(depth_bounds[1], log_depths.max()),
        )

    # Load coef from neurons_df (n_neurons, n_folds, n_features)
    # with n_features = ndepths * n_ele * n_azi + 1 (bias term)

    if suffix is None:
        suffix = "_closedloop" if is_closed_loop else "_openloop"
        if use_multidepth:
            suffix += "_multidepth"
    coef = np.stack(neurons_df[f"rf_coef{suffix}"].values)

    # Drop bias term and reshape to (n_neurons, n_folds, ndepths, n_ele, n_azi)
    coef_ = (coef[:, :, :-1]).reshape(
        coef.shape[0], coef.shape[1], ndepths, frame_shape[0], frame_shape[1]
    )
    # Suppress warning emitted by all-NaN ROIs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coef_mean = np.nanmean(coef_, axis=1)  # average over folds
    is_all_nan = np.all(np.isnan(coef_mean), axis=(1, 2, 3))

    # Find the (depth, ele, azi) index of the maximum response per neuron
    max_idx = np.zeros((coef_mean.shape[0], 3), dtype=int)
    for i in range(coef_mean.shape[0]):
        if not is_all_nan[i]:
            max_idx[i] = np.unravel_index(
                np.nanargmax(coef_mean[i]), coef_mean[i].shape
            )

    n_neurons = coef_mean.shape[0]
    rf_preferred_depth = np.full(n_neurons, np.nan)
    rf_depth_popt = [np.full(4, np.nan)] * n_neurons
    rf_depth_rsq = np.full(n_neurons, np.nan)

    # Gaussian fitting setup (same pattern as fit_preferred_depth)
    gaussian_func_ = partial(fit_gaussian_blob.gaussian_1d, min_sigma=min_sigma)
    lower_bounds = [-np.inf, depth_bounds[0], -np.inf, -np.inf]
    upper_bounds = [np.inf, depth_bounds[1], np.inf, np.inf]

    for i in tqdm(range(n_neurons), desc="Fitting RF depth tuning"):
        if is_all_nan[i]:
            continue

        best_depth_idx, best_ele, best_azi = max_idx[i]
        y = coef_mean[i, :, best_ele, best_azi]  # depth tuning curve

        if np.all(np.isnan(y)):
            continue

        # Initial guess: peak near the argmax depth
        def p0_func(peak=log_depths[best_depth_idx]):
            return np.concatenate(
                (
                    np.random.normal(size=1),
                    np.atleast_1d(peak),
                    np.random.normal(size=1),
                    np.random.normal(size=1),
                )
            ).flatten()

        try:
            popt, rsq = common_utils.iterate_fit(
                func=gaussian_func_,
                X=log_depths,
                y=y,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                niter=niter,
                p0_func=p0_func,
            )
            rf_preferred_depth[i] = np.exp(popt[1])
            rf_depth_popt[i] = popt
            rf_depth_rsq[i] = rsq
        except RuntimeError:
            # curve_fit failed to converge for this neuron
            pass

    neurons_df[f"rf_preferred_depth{suffix}"] = rf_preferred_depth
    neurons_df[f"rf_depth_popt{suffix}"] = rf_depth_popt
    neurons_df[f"rf_depth_rsq{suffix}"] = rf_depth_rsq

    return rf_preferred_depth, rf_depth_popt, rf_depth_rsq


def get_rf_results(project, sessions, is_closed_loop=1):
    """Load and aggregate RF results from specified sessions.

    Args:
        project (str): Flexilims project ID.
        sessions (list): List of session names.
        is_closed_loop (int, optional): Whether to load closed-loop coefficients.
            Defaults to 1.

    Returns:
        pd.DataFrame: Aggregated results DataFrame with ROI, session, preferred
            depth, and coefficients.
    """
    import flexiznam as flz
    from cottage_analysis.pipelines import pipeline_utils
    from cottage_analysis.io_module import suite2p as s2p_io

    if is_closed_loop:
        sfx = "_closedloop"
    else:
        sfx = "_openloop"
    for i, session_name in enumerate(sessions):
        flexilims_session = flz.get_flexilims_session(project_id=project)
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        results = pd.DataFrame(
            {
                "session": np.nan,
                "roi": np.nan,
                "iscell": np.nan,
                "preferred_depth": np.nan,
                "preferred_depth_rsq": np.nan,
                "coef": [[np.nan]] * len(neurons_df),
                "coef_ipsi": [[np.nan]] * len(neurons_df),
            }
        )

        # Add roi, preferred depth, iscell to results
        results["roi"] = np.arange(len(neurons_df))
        results["session"] = session_name
        results["preferred_depth"] = neurons_df["preferred_depth_closedloop"]
        results["preferred_depth_rsq"] = neurons_df["depth_tuning_test_rsq_closedloop"]
        exp_session = flz.get_entity(
            datatype="session", name=session_name, flexilims_session=flexilims_session
        )
        suite2p_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=exp_session.name,
            dataset_type="suite2p_rois",
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            return_dataseries=False,
        )
        iscell = s2p_io.load_is_cell(suite2p_ds.path_full)
        results["iscell"] = iscell

        # Add coef to results
        results[f"rf_coef{sfx}"] = neurons_df[f"rf_coef{sfx}"]
        results[f"rf_coef_ipsi{sfx}"] = neurons_df[f"rf_coef_ipsi{sfx}"]

        if i == 0:
            results_all = results
        else:
            results_all = pd.concat([results_all, results], axis=0, ignore_index=True)

    return results_all


def load_sig_rf(
    flexilims_session,
    session_list,
    use_cols=[
        "roi",
        "is_depth_neuron",
        "best_depth",
        "preferred_depth_closedloop",
        "preferred_depth_closedloop_crossval",
        "depth_tuning_test_rsq_closedloop",
        "depth_tuning_test_spearmanr_rval_closedloop",
        "depth_tuning_test_spearmanr_pval_closedloop",
        "rf_coef_closedloop",
        "rf_coef_ipsi_closedloop",
        "rf_rsq_closedloop",
        "rf_rsq_ipsi_closedloop",
        "preferred_RS_closedloop_g2d",
        "preferred_RS_closedloop_crossval_g2d",
        "preferred_OF_closedloop_g2d",
        "preferred_OF_closedloop_crossval_g2d",
        "rsof_test_rsq_closedloop_g2d",
        "rsof_rsq_closedloop_g2d",
        "rsof_popt_closedloop_g2d",
    ],
    n_std=6,
    verbose=1,
    filter_datasets=None,
    use_multidepth=False,
    sphere_presentation_mask=None,
):
    """
    Load significant RFs for each session in session_list.

    Args:
        flexilims_session (FlexiLimsSession): Object to interact with FlexiLims DB.
        session_list (list): List of session names to process.
        use_cols (list, optional): Columns to include in loaded data.
            Defaults to a predefined list.
        n_std (int, optional): Number of standard deviations for significant RFs.
            Defaults to 5.
        verbose (int, optional): Verbosity level for logging. Defaults to 1.
        filter_datasets (dict, optional): Dictionary to filter datasets.
            Defaults to `{"anatomical_only": 3}`.
        use_multidepth (bool, optional): If True, uses `_closedloop_multidepth`
            suffix for RF coefficients. Defaults to False.
            Note: `multidepth` can be used for selecting RF significance,
            but `_closedloop` is always used for depth significance.
        sphere_presentation_mask (np.ndarray, optional): Mask to use for
            filtering out non-significant RFs. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - all_sig (list): List of significant RFs.
            - all_sig_ipsi (list): List of significant ipsilateral RFs.
        - neurons_df_all (pd.DataFrame): Concatenated DataFrame of neurons
            from all sessions.

    """
    import flexiznam as flz
    from cottage_analysis.pipelines import pipeline_utils
    from cottage_analysis.io_module import suite2p as s2p_io
    from cottage_analysis.analysis import roi_location

    if filter_datasets is None:
        filter_datasets = {"anatomical_only": 3}

    if use_multidepth and use_cols is not None:
        sfx = "_closedloop_multidepth"
        # add the columns
        use_cols += [
            "rf_coef_closedloop_multidepth",
            "rf_coef_ipsi_closedloop_multidepth",
            "rf_rsq_closedloop_multidepth",
            "rf_rsq_ipsi_closedloop_multidepth",
        ]
    else:
        sfx = "_closedloop"

    all_sig = []
    all_sig_ipsi = []
    isess = 0
    neurons_df_all = []
    for session in session_list:
        if ("PZAH6.4b" in session) or ("PZAG3.4f" in session):
            ndepths = 5
        else:
            ndepths = 8
        # get session
        session_series = flz.get_entity(
            datatype="session", name=session, flexilims_session=flexilims_session
        )
        if (
            "exclude_reason" in session_series
            and session_series["exclude_reason"] == "not V1"
        ):
            v1 = False
        else:
            v1 = True
        # Load neurons_df
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        try:
            neurons_df = pd.read_pickle(neurons_ds.path_full)
        except FileNotFoundError:
            print(f"ERROR: SESSION {session}: neurons_df not found")
            continue

        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df.copy()
            else:
                neurons_df = neurons_df[use_cols].copy()

            # Load iscell
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=session,
                dataset_type="suite2p_rois",
                filter_datasets=filter_datasets,
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = s2p_io.load_is_cell(suite2p_ds.path_full)
            neurons_df["iscell"] = iscell
            neurons_df["session"] = session
            roi_location.determine_roi_locations(
                neurons_df, flexilims_session, session, suite2p_ds
            )
            # Load RF significant %
            coef = np.stack(neurons_df[f"rf_coef{sfx}"].values)
            coef_ipsi = np.stack(neurons_df[f"rf_coef_ipsi{sfx}"].values)
            if sphere_presentation_mask is not None:
                mid_az = int(sphere_presentation_mask.shape[-1] // 2)
                mask_ipsi_2d = sphere_presentation_mask[:, :mid_az].flatten()
                mask_contra_2d = sphere_presentation_mask[:, mid_az:].flatten()

                # Tile masks across ndepths and add True for the bias term
                mask_ipsi = np.concatenate(
                    [np.tile(mask_ipsi_2d, ndepths), [True]]
                ).astype(bool)
                mask_contra = np.concatenate(
                    [np.tile(mask_contra_2d, ndepths), [True]]
                ).astype(bool)

                coef[..., ~mask_contra] = np.nan
                coef_ipsi[..., ~mask_ipsi] = np.nan
                neurons_df[f"rf_coef{sfx}"] = list(coef)
                neurons_df[f"rf_coef_ipsi{sfx}"] = list(coef_ipsi)
            if coef_ipsi.ndim == 3:
                sig, sig_ipsi = find_sig_rfs(
                    np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
                    np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
                    n_std=n_std,
                )
                neurons_df["rf_sig"] = sig
                neurons_df["rf_sig_ipsi"] = sig_ipsi
                select_neurons = (
                    (neurons_df["iscell"] == 1)
                    & (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
                    & (neurons_df["depth_tuning_test_spearmanr_rval_closedloop"] > 0.1)
                )
                sig = sig[select_neurons]
                sig_ipsi = sig_ipsi[select_neurons]
                all_sig.append(np.mean(sig))
                all_sig_ipsi.append(np.mean(sig_ipsi))

                azi, ele, idepth, _ = find_rf_centers(
                    neurons_df,
                    ndepths=ndepths,
                    frame_shape=(16, 24),
                    is_closed_loop=1,
                    resolution=5,
                    coef=coef,
                )
                neurons_df["rf_azi"] = azi
                neurons_df["rf_ele"] = ele
                neurons_df["rf_idepth"] = idepth
                neurons_df["v1"] = v1
                neurons_df_all.append(neurons_df)
                if verbose:
                    print(f"SESSION {session} concatenated")
                isess += 1
            else:
                print(
                    f"ERROR: SESSION {session}: rf_coef_closedloop and rf_coef_ipsi_closedloop not all 3D"
                )

        else:
            print(f"ERROR: SESSION {session}: specified cols not all in neurons_df")
    neurons_df_all = pd.concat(neurons_df_all, axis=0, ignore_index=True)
    return all_sig, all_sig_ipsi, neurons_df_all


def calculate_rf_gradient(
    flexilims_session,
    neurons_ds,
    neurons_df,
    session_name,
    n_std=6,
):
    """Calculate spatial gradients for azimuth, elevation, and depth.

    Computes the linear regression slope of RF parameters against ROI
    spatial coordinates (center_x, center_y).

    Args:
        flexilims_session (FlexilimsSession): Flexilims session object.
        neurons_ds (DataSeries): DataSeries for the neurons file.
        neurons_df (pd.DataFrame): DataFrame containing ROI data and RF centers.
        session_name (str): Name of the session.
        n_std (int, optional): Significance threshold for RFs. Defaults to 6.

    Returns:
        pd.DataFrame: DataFrame containing the calculated slopes.
    """
    import flexiznam as flz
    from cottage_analysis.io_module import suite2p as s2p_io

    # load iscell
    suite2p_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        allow_multiple=False,
        return_dataseries=False,
    )
    iscell = s2p_io.load_is_cell(suite2p_ds.path_full)
    neurons_df["iscell"] = iscell

    # calculate gradients of azimuth and elevation
    session_df = pd.DataFrame([session_name], columns=["session_name"])
    coef = np.stack(neurons_df[f"rf_coef_closedloop"].values)
    coef_ipsi = np.stack(neurons_df[f"rf_coef_ipsi_closedloop"].values)
    sig, _ = find_sig_rfs(
        np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
        np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
        n_std=n_std,
    )
    select_neurons = neurons_df[(sig == 1) & (neurons_df["iscell"] == 1)]
    null_neurons = neurons_df[(sig == 0) & (neurons_df["iscell"] == 1)]
    # find the gradient of col w.r.t. center_x and center_y
    for neurons, sfx in zip([select_neurons, null_neurons], ["rf_sig", "rf_null"]):
        for col, colname in zip(
            ["rf_azi", "rf_ele", "preferred_depth_closedloop"],
            ["azi", "ele", "depth"],
        ):
            slope_x = scipy.stats.linregress(
                x=neurons["center_x"], y=neurons[col]
            ).slope
            slope_y = scipy.stats.linregress(
                x=neurons["center_y"], y=neurons[col]
            ).slope
            norm = np.linalg.norm(np.array([slope_x, slope_y]))
            slope_x /= norm
            slope_y /= norm
            session_df[f"slope_x_{colname}_{sfx}"] = slope_x
            session_df[f"slope_y_{colname}_{sfx}"] = slope_y

    # save file
    session_df.to_pickle(neurons_ds.path_full.parent / "rf_gradients.pkl")
    return session_df


def calculate_rf_gradient_all_sessions(
    flexilims_session,
    session_list,
    neurons_df_all_aligned,
    filename="rf_gradients.pkl",
    conflicts="skip",
    verbose=False,
):
    """Aggregate RF gradient calculations across multiple sessions.

    Args:
        flexilims_session (FlexilimsSession): Flexilims session object.
        session_list (list): List of session names.
        neurons_df_all_aligned (pd.DataFrame): DataFrame with data from all sessions.
        filename (str, optional): Name of the pickle file to save/load.
            Defaults to "rf_gradients.pkl".
        conflicts (str, optional): Handling of existing files ("skip", "overwrite").
            Defaults to "skip".
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        pd.DataFrame: Concatenated gradients from all sessions.
    """
    from cottage_analysis.pipelines import pipeline_utils

    for isess, session in enumerate(session_list):
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        if os.path.exists(neurons_ds.path_full.parent / filename) and (
            conflicts == "skip"
        ):
            session_df = pd.read_pickle(neurons_ds.path_full.parent / filename)
        else:
            neurons_df = neurons_df_all_aligned[
                neurons_df_all_aligned.session == session
            ]
            session_df = calculate_rf_gradient(
                flexilims_session=flexilims_session,
                neurons_ds=neurons_ds,
                neurons_df=neurons_df,
                session_name=session,
                n_std=6,
            )
        if isess == 0:
            session_df_all = session_df
        else:
            session_df_all = pd.concat([session_df_all, session_df], ignore_index=True)
        if verbose:
            print(f"Finished concat rf gradient from session {session}")

    return session_df_all
