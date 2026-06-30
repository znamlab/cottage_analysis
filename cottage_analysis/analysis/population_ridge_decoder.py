"""Population decoder for continuous variables (optic flow, running speed).

Decodes continuous variables from population neural activity (dff_stim)
using Ridge regression with k-fold cross-validation.

Mirrors the structure of ``population_depth_decoder.py`` but adapted for
regression (continuous targets) rather than classification (discrete depth
labels).
"""

import functools

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from cottage_analysis.analysis.population_depth_decoder import (
    rolling_average,
    downsample,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def downsample_for_regression(
    trials_df,
    target_col,
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
):
    """Apply rolling average + downsampling to dff and a continuous target.

    Args:
        trials_df (pd.DataFrame): Must contain ``dff_stim`` and *target_col*.
        target_col (str): Column name of the continuous target variable
            (e.g. ``"OF_stim"`` or ``"RS_stim"``).
        rolling_window (float): Rolling-average window in seconds.
        frame_rate (float): Imaging frame rate in Hz.
        downsample_window (float): Downsampling window in seconds.

    Returns:
        pd.DataFrame: *trials_df* with added columns
            ``dff_stim_rolling``, ``dff_stim_downsample``,
            ``{target_col}_rolling``, ``{target_col}_downsample``.
    """
    win = round(rolling_window * frame_rate)
    ds_factor = round(downsample_window * frame_rate)

    # Neural data
    trials_df["dff_stim_rolling"] = trials_df["dff_stim"].apply(
        lambda x: rolling_average(x, window=win, axis=0)
    )
    trials_df["dff_stim_downsample"] = trials_df["dff_stim_rolling"].apply(
        lambda x: downsample(x, factor=ds_factor, mode="average")
    )

    # Target variable
    trials_df[f"{target_col}_rolling"] = trials_df[target_col].apply(
        lambda x: rolling_average(x, window=win, axis=0)
    )
    trials_df[f"{target_col}_downsample"] = trials_df[f"{target_col}_rolling"].apply(
        lambda x: downsample(x, factor=ds_factor, mode="average")
    )

    return trials_df


def _prepare_target(
    target_arr,
    log_transform=True,
):
    """Prepare the target vector, optionally log-transforming.

    Frames where the target is <= 0 (invalid for log) are set to NaN.

    Args:
        target_arr (np.ndarray): 1-D target vector.
        log_transform (bool): Whether to log-transform the target.

    Returns:
        np.ndarray: Processed target vector (may contain NaN where invalid).
    """
    target = target_arr.copy().astype(float)
    if log_transform:
        invalid = target <= 0
        target[invalid] = np.nan
        target = np.log(target)
    return target


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def split_train_test(
    trials_df,
    dff_col="dff_stim_downsample",
    target_col_ds="OF_stim_downsample",
    k_folds=5,
    random_state=42,
    log_transform=True,
    rs_thr=None,
    max_rs2motor_diff=None,
):
    """Split trials into k train/test folds.

    Unlike the depth decoder (StratifiedKFold on discrete labels), we use
    plain KFold because the target is continuous.

    Args:
        trials_df (pd.DataFrame): DataFrame with *dff_col* and *target_col_ds*.
        dff_col (str): Column with downsampled dff arrays.
        target_col_ds (str): Column with downsampled target arrays.
        k_folds (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        log_transform (bool): Whether to log-transform the target.
        rs_thr (float or None): If not None, exclude frames where RS_stim
            (or RS_stim_downsample if present) is below this threshold.
        max_rs2motor_diff (float or None): If not None, exclude frames where
            max_abs_rs2motor_diff_ratio (or downsample) is above or equal to this.

    Returns:
        dict: With keys ``dff_train``, ``dff_test``, ``y_train``, ``y_test``,
            ``test_frame_indices`` — each a list of length *k_folds*.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    result = {
        "dff_train": [],
        "dff_test": [],
        "y_train": [],
        "y_test": [],
        "test_frame_indices": [],
    }

    trial_indices = np.arange(len(trials_df))

    for train_idx, test_idx in kf.split(trial_indices):
        for split_name, idx in [("train", train_idx), ("test", test_idx)]:
            dff = np.vstack(trials_df.iloc[idx][dff_col].values)
            target_raw = np.hstack(trials_df.iloc[idx][target_col_ds].values)
            target = _prepare_target(target_raw, log_transform=log_transform)

            # Build validity mask
            valid = ~np.isnan(target)
            if np.any(np.isnan(dff)):
                valid &= ~np.any(np.isnan(dff), axis=1)

            # Optional running-speed filter
            if rs_thr is not None:
                rs_col_ds = "RS_stim_downsample"
                if rs_col_ds in trials_df.columns:
                    rs_arr = np.hstack(trials_df.iloc[idx][rs_col_ds].values)
                else:
                    rs_arr = np.hstack(trials_df.iloc[idx]["RS_stim"].values)
                valid &= rs_arr >= rs_thr

            # Optional rs2motor diff filter
            if max_rs2motor_diff is not None:
                ratio_col_ds = "max_abs_rs2motor_diff_ratio_downsample"
                if ratio_col_ds in trials_df.columns:
                    ratio_arr = np.hstack(trials_df.iloc[idx][ratio_col_ds].values)
                elif "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns:
                    ratio_arr = np.hstack(
                        trials_df.iloc[idx]["max_abs_rs2motor_diff_ratio_stim"].values
                    )
                else:
                    ratio_arr = None

                if ratio_arr is not None:
                    valid &= ratio_arr < max_rs2motor_diff

            dff_valid = dff[valid]
            target_valid = target[valid]

            result[f"dff_{split_name}"].append(dff_valid)
            result[f"y_{split_name}"].append(target_valid)

            if split_name == "test":
                # Track valid test frame indices for reconstruction
                trial_frame_lens = trials_df[dff_col].apply(len).values
                frame_offsets = np.concatenate([[0], np.cumsum(trial_frame_lens)[:-1]])
                test_frame_idx = np.hstack(
                    [
                        np.arange(frame_offsets[i], frame_offsets[i] + n)
                        for i, n in zip(idx, trials_df.iloc[idx][dff_col].apply(len))
                    ]
                )
                result["test_frame_indices"].append(test_frame_idx[valid])

    return result


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def ridge_hyperparam_tuning(
    X_train,
    y_train,
    X_test,
    y_test,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    verbose=True,
):
    """Tune Ridge alpha on a held-out set.

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
        y_train (np.ndarray): Training target (n_samples,).
        X_test (np.ndarray): Validation features.
        y_test (np.ndarray): Validation target.
        alphas (tuple): Candidate regularisation strengths.
        verbose (bool): Whether to print hyperparameter tuning messages.

    Returns:
        float: Best alpha value.
    """
    best_r2 = -np.inf
    best_alpha = alphas[0]
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
    if verbose:
        print(f"  Best alpha: {best_alpha}  (R² = {best_r2:.4f})")
    return best_alpha


def fit_ridge_fold(
    X_train,
    y_train,
    X_test,
    y_test,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    zscore_dff=True,
    verbose=True,
):
    """Tune hyperparameters and evaluate on test set for one fold.

    Uses 80/20 split within the training set for hyperparameter tuning,
    then retrains on the full training set with the best alpha.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target.
        alphas (tuple): Candidate regularisation strengths.
        zscore_dff (bool): Whether to z-score X_train and X_test per fold
            using X_train statistics.
        verbose (bool): Whether to print hyperparameter tuning messages.

    Returns:
        dict: ``y_pred``, ``best_alpha``, ``model``, ``r2``, ``pearson_r``,
            ``mse``.
    """
    if zscore_dff:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    # Internal split for hyperparameter tuning (80/20 of training data)
    n = len(y_train)
    n_val = max(1, int(0.2 * n))
    rng = np.random.RandomState(42)
    val_idx = rng.choice(n, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    best_alpha = ridge_hyperparam_tuning(
        X_train[train_idx],
        y_train[train_idx],
        X_train[val_idx],
        y_train[val_idx],
        alphas=alphas,
        verbose=verbose,
    )

    # Retrain on full training set
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    r_val, _ = pearsonr(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "y_pred": y_pred,
        "best_alpha": best_alpha,
        "model": model,
        "r2": r2,
        "pearson_r": r_val,
        "mse": mse,
        "mae": mae,
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def continuous_decoder(
    trials_df,
    target_col="OF_stim",
    closed_loop=1,
    rolling_window=None,
    frame_rate=15,
    downsample_window=None,
    log_transform=True,
    rs_thr=None,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    k_folds=5,
    random_state=42,
    shuffle_control=False,
    zscore_dff=True,
    verbose=True,
    max_rs2motor_diff=None,
):
    """Decode a continuous variable from population neural activity.

    This is the main entry point. It preprocesses the data, runs k-fold
    cross-validated Ridge regression, and returns aggregated results.

    Args:
        trials_df (pd.DataFrame): Must contain ``dff_stim``, *target_col*,
            and ``closed_loop`` columns.
        target_col (str): Column to decode. Typically ``"OF_stim"`` for optic
            flow or ``"RS_stim"`` for running speed.
        closed_loop (int): 1 for closed-loop, 0 for open-loop.
        rolling_window (float): Rolling-average window in seconds. Set to
            *None* to skip smoothing.
        frame_rate (float): Imaging frame rate in Hz.
        downsample_window (float): Downsampling window in seconds. Set to
            *None* to skip downsampling.
        log_transform (bool): Whether to log-transform the target variable.
        rs_thr (float or None): Running-speed threshold (m/s). Frames below
            this are excluded. Default *None* (no filtering).
        alphas (tuple): Candidate Ridge regularisation strengths.
        k_folds (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        shuffle_control (bool): If True, randomly shuffle the target variable
            to estimate chance-level performance.

    Returns:
        dict: Results with keys:
            - ``r2``: Overall R² across all test folds.
            - ``pearson_r``: Overall Pearson correlation.
            - ``mse``: Overall mean squared error.
            - ``y_test``: Concatenated true test values.
            - ``y_pred``: Concatenated predicted test values.
            - ``fold_results``: List of per-fold result dicts.
            - ``target_col``: Which variable was decoded.
            - ``log_transform``: Whether log-transform was applied.
            - ``k_folds``: Number of folds used.
            - ``alphas_used``: Per-fold best alphas.
    """
    # Filter by protocol
    if closed_loop is not None and "closed_loop" in trials_df.columns:
        df_filtered = trials_df[trials_df.closed_loop == closed_loop].copy()
    else:
        df_filtered = trials_df.copy()

    original_indices = df_filtered.index.values
    df = df_filtered.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No trials found with closed_loop={closed_loop}. "
            f"Available values: {trials_df.closed_loop.unique()}"
        )

    # Check if target_col contains scalar values (e.g. depth)
    # If so, expand it to a 1D array of the same length as dff_stim for each trial.
    first_val = df[target_col].iloc[0]
    if not isinstance(first_val, (np.ndarray, list, pd.Series)):
        df[target_col] = df.apply(
            lambda row: np.repeat(row[target_col], len(row["dff_stim"])), axis=1
        )

    # Check if we have any all-NaN ROIs
    dff_col_to_check = "dff_stim"
    all_dff = np.vstack(df[dff_col_to_check].values)
    all_nan_rois = np.all(np.isnan(all_dff), axis=0)
    valid_rois_idx = np.where(~all_nan_rois)[0]
    n_total_neurons = all_nan_rois.shape[0]

    if np.any(all_nan_rois):
        print(
            f"continuous_decoder: Excluding {np.sum(all_nan_rois)} ROIs that contain only NaN values."
        )
        df["dff_stim"] = df["dff_stim"].apply(lambda x: x[:, valid_rois_idx])

    # Preprocessing: rolling average + downsample
    if rolling_window is not None and downsample_window is not None:
        df = downsample_for_regression(
            df,
            target_col=target_col,
            rolling_window=rolling_window,
            frame_rate=int(frame_rate),
            downsample_window=downsample_window,
        )
        # Also downsample RS if we have an rs_thr and target is not RS itself
        if rs_thr is not None and target_col != "RS_stim":
            df["RS_stim_rolling"] = df["RS_stim"].apply(
                lambda x: rolling_average(
                    x, window=round(rolling_window * frame_rate), axis=0
                )
            )
            df["RS_stim_downsample"] = df["RS_stim_rolling"].apply(
                lambda x: downsample(
                    x, factor=round(downsample_window * frame_rate), mode="average"
                )
            )
        # Also downsample max_abs_rs2motor_diff_ratio_stim if we have a max_rs2motor_diff
        if max_rs2motor_diff is not None:
            ratio_col = "max_abs_rs2motor_diff_ratio_stim"
            if ratio_col in df.columns:
                df["max_abs_rs2motor_diff_ratio_rolling"] = df[ratio_col].apply(
                    lambda x: rolling_average(
                        x, window=round(rolling_window * frame_rate), axis=0
                    )
                )
                df["max_abs_rs2motor_diff_ratio_downsample"] = df[
                    "max_abs_rs2motor_diff_ratio_rolling"
                ].apply(
                    lambda x: downsample(
                        x, factor=round(downsample_window * frame_rate), mode="average"
                    )
                )
        dff_col = "dff_stim_downsample"
        target_col_ds = f"{target_col}_downsample"
    else:
        dff_col = "dff_stim"
        target_col_ds = target_col

    # Shuffle control: permute target arrays across trials and resample to match destination trial lengths
    if shuffle_control:
        rng = np.random.RandomState(random_state)
        shuffled_idx = rng.permutation(len(df))
        shuffled_arrays = []
        for i in range(len(df)):
            orig_target = df[target_col_ds].iloc[shuffled_idx[i]]
            target_len = len(df[dff_col].iloc[i])
            if len(orig_target) == target_len:
                resampled = orig_target
            else:
                x_orig = np.linspace(0, 1, len(orig_target))
                x_new = np.linspace(0, 1, target_len)
                resampled = np.interp(x_new, x_orig, orig_target)
            shuffled_arrays.append(resampled)
        df[target_col_ds] = shuffled_arrays

    # Train/test split
    splits = split_train_test(
        df,
        dff_col=dff_col,
        target_col_ds=target_col_ds,
        k_folds=k_folds,
        random_state=random_state,
        log_transform=log_transform,
        rs_thr=rs_thr,
        max_rs2motor_diff=max_rs2motor_diff,
    )

    # Fit each fold
    fold_results = []
    for i in range(k_folds):
        if verbose:
            print(f"Fitting fold {i + 1}/{k_folds}...")
        res = fit_ridge_fold(
            X_train=splits["dff_train"][i],
            y_train=splits["y_train"][i],
            X_test=splits["dff_test"][i],
            y_test=splits["y_test"][i],
            alphas=alphas,
            zscore_dff=zscore_dff,
            verbose=verbose,
        )
        if np.any(all_nan_rois):
            coef_full = np.full(n_total_neurons, np.nan)
            coef_full[valid_rois_idx] = res["model"].coef_
            res["model"].coef_ = coef_full
        fold_results.append(res)

    # Aggregate across folds
    y_test_all = np.concatenate([splits["y_test"][i] for i in range(k_folds)])
    y_pred_all = np.concatenate([fr["y_pred"] for fr in fold_results])
    test_indices_all = np.concatenate(
        [splits["test_frame_indices"][i] for i in range(k_folds)]
    )

    # Sort chronologically to match original trial/frame order
    sort_idx = np.argsort(test_indices_all)
    y_test_all = y_test_all[sort_idx]
    y_pred_all = y_pred_all[sort_idx]
    test_indices_sorted = test_indices_all[sort_idx]

    overall_r2 = r2_score(y_test_all, y_pred_all)
    overall_r, _ = pearsonr(y_test_all, y_pred_all)
    overall_mse = mean_squared_error(y_test_all, y_pred_all)
    overall_mae = mean_absolute_error(y_test_all, y_pred_all)

    # Reconstruct full-length per-trial arrays containing NaNs for invalid frames
    total_frames = sum(df[dff_col].apply(len).values)
    y_pred_full = np.full(total_frames, np.nan)
    y_test_full = np.full(total_frames, np.nan)
    y_pred_full[test_indices_sorted] = y_pred_all
    y_test_full[test_indices_sorted] = y_test_all

    trial_lens = df[dff_col].apply(len).values
    frame_offsets = np.concatenate([[0], np.cumsum(trial_lens)])

    y_pred_series = pd.Series(index=trials_df.index, dtype=object)
    y_test_series = pd.Series(index=trials_df.index, dtype=object)
    for i, orig_idx in enumerate(original_indices):
        y_pred_series.at[orig_idx] = y_pred_full[
            frame_offsets[i] : frame_offsets[i + 1]
        ]
        y_test_series.at[orig_idx] = y_test_full[
            frame_offsets[i] : frame_offsets[i + 1]
        ]

    if verbose:
        print(f"\nOverall results for {target_col}:")
        print(f"  R²       = {overall_r2:.4f}")
        print(f"  Pearson r = {overall_r:.4f}")
        print(f"  MSE       = {overall_mse:.4f}")
        print(f"  MAE       = {overall_mae:.4f}")

    return {
        "r2": overall_r2,
        "pearson_r": overall_r,
        "mse": overall_mse,
        "mae": overall_mae,
        "y_test": y_test_all,
        "y_pred": y_pred_all,
        "y_test_trials": y_test_series,
        "y_pred_trials": y_pred_series,
        "fold_results": fold_results,
        "target_col": target_col,
        "log_transform": log_transform,
        "k_folds": k_folds,
        "alphas_used": [fr["best_alpha"] for fr in fold_results],
    }


def of_decoder(trials_df, **kwargs):
    """Decode optic flow (OF_stim) from population activity.

    Convenience wrapper around :func:`continuous_decoder` with
    ``target_col="OF_stim"``.

    All keyword arguments are forwarded to :func:`continuous_decoder`.
    See its docstring for the full parameter list.

    Returns:
        dict: Decoder results. See :func:`continuous_decoder`.
    """
    kwargs.setdefault("target_col", "OF_stim")
    kwargs.setdefault("log_transform", True)
    return continuous_decoder(trials_df, **kwargs)


def rs_decoder(trials_df, **kwargs):
    """Decode running speed (RS_stim) from population activity.

    Convenience wrapper around :func:`continuous_decoder` with
    ``target_col="RS_stim"``.

    All keyword arguments are forwarded to :func:`continuous_decoder`.
    See its docstring for the full parameter list.

    Returns:
        dict: Decoder results. See :func:`continuous_decoder`.
    """
    kwargs.setdefault("target_col", "RS_stim")
    kwargs.setdefault("log_transform", True)
    return continuous_decoder(trials_df, **kwargs)


def depth_decoder(trials_df, **kwargs):
    """Decode depth (depth) from population activity using Ridge regression.

    Convenience wrapper around :func:`continuous_decoder` with
    ``target_col="depth"``.

    All keyword arguments are forwarded to :func:`continuous_decoder`.
    See its docstring for the full parameter list.

    Returns:
        dict: Decoder results. See :func:`continuous_decoder`.
    """
    kwargs.setdefault("target_col", "depth")
    kwargs.setdefault("log_transform", True)
    return continuous_decoder(trials_df, **kwargs)


def rsof_product_decoder(trials_df, **kwargs):
    """Decode the depth-orthogonal speed-flow product from population activity.

    Depth is essentially the *ratio* of running speed to optic flow
    (``OF = RS / depth``), so in log space ``log(depth) ~ log(RS) - log(OF)``.
    The axis orthogonal to depth is therefore ``log(RS) + log(OF) = log(RS * OF)``.
    This decoder targets that orthogonal dimension, i.e. the elementwise product
    ``RS * OF`` (stored in the ``"rsof_product_stim"`` column).

    Convenience wrapper around :func:`continuous_decoder` with
    ``target_col="rsof_product_stim"``.

    All keyword arguments are forwarded to :func:`continuous_decoder`.
    See its docstring for the full parameter list.

    Returns:
        dict: Decoder results. See :func:`continuous_decoder`.
    """
    kwargs.setdefault("target_col", "rsof_product_stim")
    kwargs.setdefault("log_transform", True)
    return continuous_decoder(trials_df, **kwargs)


def decode_with_neuron_subsets(
    trials_df,
    decoder_func,
    subset_sizes=None,
    n_resamples="auto",
    random_state=42,
    **decoder_kwargs,
):
    """Evaluate decoding performance as a function of the number of neurons included.

    Args:
        trials_df (pd.DataFrame): DataFrame containing 'dff_stim' and target columns.
        decoder_func (callable): Convenience wrapper (e.g., `of_decoder`,
            `rs_decoder`, or `depth_decoder`).
        subset_sizes (list of int or None): List of neuron subset sizes to test.
            If None, defaults to a standard list up to the total number of neurons.
        n_resamples (int): Number of random neuron subsampling iterations per size.
        random_state (int): Seed for reproducibility.
        **decoder_kwargs: Additional keyword arguments forwarded to the decoder.

    Returns:
        dict: A dictionary containing:
            - ``subset_sizes``: List of subset sizes evaluated.
            - ``r2_mean``, ``r2_std``: Mean and standard deviation of R² per size.
            - ``pearson_r_mean``, ``pearson_r_std``: Mean and standard deviation of Pearson r.
            - ``mse_mean``, ``mse_std``: Mean and std of Mean Squared Error.
            - ``raw_results``: Dict mapping each size to its list of individual resample run dicts.
    """
    # Exclude all-NaN ROIs before subset analysis
    all_dff = np.vstack(trials_df["dff_stim"].values)
    all_nan_rois = np.all(np.isnan(all_dff), axis=0)
    valid_rois_idx = np.where(~all_nan_rois)[0]
    if np.any(all_nan_rois):
        print(
            f"decode_with_neuron_subsets: Excluding {np.sum(all_nan_rois)} ROIs that contain only NaN values."
        )
        trials_df = trials_df.copy()
        trials_df["dff_stim"] = trials_df["dff_stim"].apply(
            lambda x: x[:, valid_rois_idx]
        )

    n_total_neurons = trials_df["dff_stim"].iloc[0].shape[1]

    if subset_sizes is None:
        steps = [5, 10, 20, 50, 100, 200, 500, 1000]
        subset_sizes = [s for s in steps if s < n_total_neurons]
        if n_total_neurons not in subset_sizes:
            subset_sizes.append(n_total_neurons)
        subset_sizes.append("inf")

    # Convert 'inf' (string or float) to n_total_neurons
    parsed_sizes = []
    for s in subset_sizes:
        if (
            s == "inf"
            or s == float("inf")
            or (isinstance(s, (float, np.floating)) and np.isinf(s))
        ):
            parsed_sizes.append(n_total_neurons)
        else:
            parsed_sizes.append(int(s))

    # Sort and filter subset sizes to be unique and within valid range
    subset_sizes = sorted(
        list(set([s for s in parsed_sizes if 0 < s <= n_total_neurons]))
    )

    # Calculate the number of runs for each size (supports dict, list, int, or "auto")
    runs_per_size = {}
    for size in subset_sizes:
        if size == n_total_neurons:
            runs_per_size[size] = 1
        elif n_resamples == "auto":
            prop = size / n_total_neurons
            if prop <= 0.1:
                runs_per_size[size] = 20
            elif prop <= 0.3:
                runs_per_size[size] = 10
            elif prop <= 0.6:
                runs_per_size[size] = 5
            else:
                runs_per_size[size] = 3
        elif isinstance(n_resamples, dict):
            runs_per_size[size] = n_resamples.get(size, 5)
        elif isinstance(n_resamples, (list, tuple, np.ndarray)):
            idx = subset_sizes.index(size)
            runs_per_size[size] = n_resamples[idx]
        else:
            runs_per_size[size] = n_resamples

    # Calculate total steps for the progress bar
    total_steps = sum(runs_per_size.values())

    # Force child decoders to be silent
    decoder_kwargs["verbose"] = False

    results = {
        "subset_sizes": subset_sizes,
        "n_resamples": [],
        "r2_mean": [],
        "r2_std": [],
        "pearson_r_mean": [],
        "pearson_r_std": [],
        "mse_mean": [],
        "mse_std": [],
        "mae_mean": [],
        "mae_std": [],
        "raw_results": {},
    }

    rng = np.random.RandomState(random_state)

    # Initialize tqdm progress bar
    target_name = decoder_kwargs.get(
        "target_col", decoder_func.__name__.split("_")[0]
    ).upper()
    pbar = tqdm(total=total_steps, desc=f"Subsampling ({target_name})")

    for size in subset_sizes:
        r2s, rs, mses, maes = [], [], [], []
        size_raw = []

        # Get the number of runs for this size
        runs = runs_per_size[size]

        for run in range(runs):
            pbar.set_description(
                f"Subsampling ({target_name}) - Size {size} (run {run + 1}/{runs})"
            )
            # Subsample neurons
            if size == n_total_neurons:
                neuron_idx = np.arange(n_total_neurons)
            else:
                neuron_idx = rng.choice(n_total_neurons, size=size, replace=False)

            # Create a copy of the dataframe and slice dff_stim
            df_subset = trials_df.copy()
            df_subset["dff_stim"] = df_subset["dff_stim"].apply(
                lambda x: x[:, neuron_idx]
            )

            # Run the decoder
            res = decoder_func(df_subset, **decoder_kwargs)

            r2s.append(res["r2"])
            rs.append(res["pearson_r"])
            mses.append(res["mse"])
            maes.append(res["mae"])
            size_raw.append(res)

            pbar.update(1)

        results["n_resamples"].append(runs)
        results["r2_mean"].append(np.mean(r2s))
        results["r2_std"].append(np.std(r2s) if len(r2s) > 1 else 0.0)
        results["pearson_r_mean"].append(np.mean(rs))
        results["pearson_r_std"].append(np.std(rs) if len(rs) > 1 else 0.0)
        results["mse_mean"].append(np.mean(mses))
        results["mse_std"].append(np.std(mses) if len(mses) > 1 else 0.0)
        results["mae_mean"].append(np.mean(maes))
        results["mae_std"].append(np.std(maes) if len(maes) > 1 else 0.0)
        results["raw_results"][size] = size_raw

    pbar.close()
    return results
