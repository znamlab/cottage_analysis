from functools import partial
import gc
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from cottage_analysis.analysis.fit_gaussian_blob import (
    Gabor3DRFParams,
    Gaussian3DRFParams,
    gabor_3d_rf,
    gaussian_3d_rf,
)

from cottage_analysis.analysis.spheres.stimulus_reconstruction import (
    find_valid_frames,
)

print = partial(print, flush=True)

# default (reg_xy, reg_depth) candidates of the hyperparameter search
DEFAULT_REG_GRID = np.geomspace(2.5, 10240, 13)


def _second_difference(n):
    """(n, n) second-difference matrix: 2 on the diagonal, -1 next to it."""
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def laplace_matrix(nx, ny):
    """Discrete Laplacian of an (nx, ny) image flattened in C order.

    Each row has 4 on the diagonal and -1 for each 4-neighbour of the pixel (the
    diagonal stays 4 on the border).

    Args:
        nx (int): number of rows of the image.
        ny (int): number of columns of the image.

    Returns:
        np.array: (nx * ny, nx * ny) Laplacian.
    """
    return np.kron(_second_difference(nx), np.eye(ny)) + np.kron(
        np.eye(nx), _second_difference(ny)
    )


def _penalties(ndepths, nelev, nazim):
    """Spatial and depth penalty operators of the 3D RF fits.

    Coefficients are ordered (depth, elevation, azimuth), followed by a bias that is
    not penalised.

    Args:
        ndepths (int): number of depths.
        nelev (int): number of elevation bins.
        nazim (int): number of azimuth bins.

    Returns:
        L (np.array): Laplacian of each depth, (ndepths * npix, nfeatures + 1).
        L_depth (np.array): second difference along depth of each pixel,
            (ndepths * npix, nfeatures + 1).
    """
    L = np.kron(np.eye(ndepths), laplace_matrix(nelev, nazim))
    L_depth = np.kron(_second_difference(ndepths), np.eye(nelev * nazim))
    # zero column for the bias
    return np.pad(L, ((0, 0), (0, 1))), np.pad(L_depth, ((0, 0), (0, 1)))


def _single_depth_design(imaging_df, frames, shift_stim):
    """Design matrix and penalty operators of single-depth RF fits (one depth per trial).

    The stimulus of each frame goes in the block of the depth shown on that frame.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): stimulus frames, nframes x nelevation x nazimuth.
        shift_stim (int): shift to account for response lag.

    Returns:
        X (np.array): design matrix with a bias column, (nframes, nfeatures + 1).
        L (np.array): spatial Laplacian penalty, (npenalties, nfeatures + 1).
        L_depth (np.array): depth second-difference penalty, (npenalties, nfeatures + 1).
    """
    depths = imaging_df.depth.unique()
    depths = depths[~np.isnan(depths)]
    depths = depths[depths > 0]
    depths = np.sort(depths)
    nframes, nelev, nazim = frames.shape
    npix = nelev * nazim

    # shift to account for response lag
    lagged = np.roll(frames.reshape(nframes, npix), shift_stim, axis=0)
    X = np.zeros((nframes, npix * len(depths)))
    for idepth, depth in enumerate(depths):
        depth_idx = (imaging_df.depth == depth).values
        # stimulus in the columns of the depth shown on that frame
        X[depth_idx, idepth * npix : (idepth + 1) * npix] = lagged[depth_idx]
    L, L_depth = _penalties(len(depths), nelev, nazim)
    # add bias
    X = np.concatenate([X, np.ones((nframes, 1))], axis=1)
    return X, L, L_depth


def _single_depth_trial_index(imaging_df):
    """Trial index of each frame of a single-depth session, NaN outside trials.

    A new trial starts at each change to a positive depth. Frames with a negative or
    NaN depth (gray screen, outside the protocol) are NaN.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.

    Returns:
        np.array: trial index of each frame, NaN outside trials.
    """
    depth = imaging_df.depth
    trial_start = (np.abs(depth.diff()) > 0) & (depth > 0)
    trial_idx = np.cumsum(trial_start.values).astype(float)
    trial_idx[(depth.isna() | (depth < 0)).values] = np.nan
    return trial_idx


def single_depth_cv_folds(imaging_df, k_folds=5):
    """Test fold of each frame in the single-depth cross-validation, NaN if unused.

    Folds come from `StratifiedKFold` over trials, stratified by depth.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        k_folds (int): number of folds. Defaults to 5.

    Returns:
        np.array: test fold of each frame.
    """
    trial_idx = _single_depth_trial_index(imaging_df)
    depths_by_trial = imaging_df.depth.groupby(trial_idx).first()
    categorical = pd.Categorical(depths_by_trial).codes
    stratified_kfold = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
    fold = np.full(len(imaging_df), np.nan)
    for ifold, (_, test_trials) in enumerate(
        stratified_kfold.split(depths_by_trial.index, categorical)
    ):
        # positions in depths_by_trial, converted to trial indices
        fold[np.isin(trial_idx, depths_by_trial.index[test_trials])] = ifold
    return fold


def _trial_index(imaging_df):
    """Trial index of each frame of a multidepth session, NaN outside trials.

    Trials are runs of `stim == 1`, set by `find_stim_time` from the protocol trial
    times. `depth` cannot be used: in multidepth protocols it is the radius of the
    last logged sphere, which is negative for sphere removals, also during trials.
    A trial still running at the end of the recording is excluded, as in
    `trials_df`.

    If imaging_df has a boolean `rf_use_frame` column (e.g. running frames only),
    frames where it is False are also set to NaN. Trials are numbered before this
    selection, so cross-validation folds still split whole trials.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor
            frame, with a `stim` column.

    Returns:
        np.array: trial index of each frame, counting from 1, NaN outside trials.
    """
    if "stim" not in imaging_df.columns:
        raise ValueError(
            "imaging_df has no `stim` column, needed to find multidepth trials"
        )
    in_trial = (imaging_df.stim == 1).values.copy()
    if in_trial[-1]:
        # the recording stopped during the last trial, which is excluded from
        # trials_df and so not reconstructed in the stimulus frames
        last_start = np.flatnonzero(~in_trial)[-1] + 1 if not in_trial.all() else 0
        in_trial[last_start:] = False
    trial_start = np.hstack([in_trial[0], in_trial[1:] & ~in_trial[:-1]])
    trial_idx = np.cumsum(trial_start).astype(float)
    trial_idx[~in_trial] = np.nan
    if "rf_use_frame" in imaging_df.columns:
        trial_idx[~imaging_df.rf_use_frame.values.astype(bool)] = np.nan
    return trial_idx


def multidepth_cv_folds(imaging_df, k_folds=5):
    """Test fold of each frame in the multidepth cross-validation.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.
        k_folds (int): number of folds. Defaults to 5.

    Returns:
        np.array: test fold of each frame, NaN outside trials.
    """
    trial_idx = _trial_index(imaging_df)
    trials = pd.Series(trial_idx).dropna().unique()
    kfold = KFold(n_splits=k_folds, random_state=42, shuffle=True)
    fold = np.full(len(trial_idx), np.nan)
    for ifold, (_, test_trials) in enumerate(kfold.split(trials)):
        fold[np.isin(trial_idx, trials[test_trials])] = ifold
    return fold


def _multidepth_design(imaging_df, frames, shift_stim):
    """Design matrix and penalty operators for multidepth RF fits.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.
        frames (np.array): stimulus frames (depth, nframes, ele, azi).
        shift_stim (int): shift to account for response lag.

    Returns:
        X (np.array): design matrix with a bias column, (nframes, nfeatures + 1).
        L (np.array): spatial Laplacian penalty, (npenalties, nfeatures + 1).
        L_depth (np.array): depth second-difference penalty, (npenalties, nfeatures + 1).
    """
    ndepths, nframes, nelev, nazim = frames.shape
    depths = imaging_df.depth.unique()
    depths = depths[~np.isnan(depths)]
    depths = depths[depths > 0]
    depths = np.sort(depths)

    assert depths.shape[0] == frames.shape[0]
    # Shift to account for response lag
    X = np.roll(frames, shift_stim, axis=1)
    X = np.swapaxes(X, 0, 1)  # put back frame number as first axis
    # (now we have frame, depth, ele, azi)
    X = X.reshape(X.shape[0], -1)  # flatten

    L, L_depth = _penalties(ndepths, nelev, nazim)
    # add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X, L, L_depth


def _ridge_cv(
    X, Y, L, L_depth, fold, k_folds, reg_grid=None, roi_regs=None, compute_coefs=True
):
    """Cross-validated penalised regression of the 3D RF fits.

    Solves, for each fold, (X_train' X_train + reg_xy^2 L'L + reg_depth^2 D'D) c =
    X_train' Y_train for every ROI. For each fold and reg_xy, one generalised
    eigendecomposition of the penalised normal equations gives the solution for every
    reg_depth and ROI, so the whole grid is searched without refitting.

    Either selects, for each ROI, the (reg_xy, reg_depth) of `reg_grid` with the best
    held-out R2 (pooled over folds), or uses the given `roi_regs`.

    Args:
        X (np.array): design matrix, (nframes, nfeatures).
        Y (np.array): responses, (nframes, nrois).
        L (np.array): spatial penalty, (npenalties, nfeatures).
        L_depth (np.array): depth penalty, (npenalties, nfeatures).
        fold (np.array): test fold of each frame, NaN for frames not used.
        k_folds (int): number of folds.
        reg_grid (np.array): (ngrid, 2) candidate (reg_xy, reg_depth) pairs.
        roi_regs (np.array): (nrois, 2) (reg_xy, reg_depth) of each ROI, instead of
            the search.
        compute_coefs (bool): if False, stop after the search of `reg_grid` and
            return None for `coefs` and `r2`. Defaults to True.

    Returns:
        coefs (np.array): coefficients of each fold, (k_folds, nfeatures, nrois).
        r2 (np.array): train and test R2 of each ROI, (nrois, 2). The train
            prediction of a frame is from the last fold in which it is trained on.
        best_idx (np.array): index of the selected pair in `reg_grid` (None if
            `roi_regs` is given).
        best_r2 (np.array): held-out R2 of the selected pair (as r2[:, 1]).
    """
    from scipy.linalg import eigh

    used = np.isfinite(fold)
    X, Y, fold = X[used], Y[used], fold[used]
    test_masks = [fold == ifold for ifold in range(k_folds)]
    G = np.stack([X[m].T @ X[m] for m in test_masks])
    B = np.stack([X[m].T @ Y[m] for m in test_masks])
    G_all, B_all = G.sum(axis=0), B.sum(axis=0)
    LtL = L.T @ L
    DtD = L_depth.T @ L_depth
    total_var = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    nrois = Y.shape[1]

    best_idx = None
    if roi_regs is None:
        reg_xys = np.unique(reg_grid[:, 0])
        reg_depths = np.unique(reg_grid[:, 1])
        yy = np.stack([np.sum(Y[m] ** 2, axis=0) for m in test_masks])
        # held-out sum of squared errors, pooled over folds
        sse = np.zeros((len(reg_xys), len(reg_depths), nrois))
        for ifold in range(k_folds):
            for ixy, reg_xy in enumerate(reg_xys):
                # V' M V = I and V' D'D V = diag(s), so the solution for
                # M + reg_depth^2 D'D is V diag(1 / (1 + reg_depth^2 s)) V'
                M = G_all - G[ifold] + reg_xy**2 * LtL
                s, V = eigh(DtD, M)
                P = V.T @ (B_all - B[ifold])
                Q = V.T @ B[ifold]
                W = V.T @ G[ifold] @ V
                for idepth, reg_depth in enumerate(reg_depths):
                    Z = P / (1 + reg_depth**2 * s)[:, None]
                    sse[ixy, idepth] += (
                        yy[ifold]
                        - 2 * np.sum(Z * Q, axis=0)
                        + np.sum(Z * (W @ Z), axis=0)
                    )
                gc.collect()
        r2_grid = 1 - sse / total_var
        # best pair, in the order of reg_grid
        ixy = np.searchsorted(reg_xys, reg_grid[:, 0])
        idepth = np.searchsorted(reg_depths, reg_grid[:, 1])
        r2_pairs = r2_grid[ixy, idepth]
        best_idx = np.argmax(r2_pairs, axis=0)
        if not compute_coefs:
            return None, None, best_idx, r2_pairs[best_idx, np.arange(nrois)]
        roi_regs = reg_grid[best_idx]

    # coefficients of each fold at the reg of each ROI
    coefs = np.zeros((k_folds, X.shape[1], nrois))
    for ifold in range(k_folds):
        for reg_xy in np.unique(roi_regs[:, 0]):
            rois = roi_regs[:, 0] == reg_xy
            M = G_all - G[ifold] + reg_xy**2 * LtL
            s, V = eigh(DtD, M)
            P = V.T @ (B_all[:, rois] - B[ifold][:, rois])
            Z = P / (1 + roi_regs[rois, 1][None, :] ** 2 * s[:, None])
            coefs[ifold][:, rois] = V @ Z
        gc.collect()

    # test predictions from the fold of the frame, train from the last fold in
    # which the frame is in the training set
    pred_test = np.zeros_like(Y)
    pred_train = np.full_like(Y, np.nan)
    for ifold, m in enumerate(test_masks):
        pred_test[m] = X[m] @ coefs[ifold]
        pred_train[~m] = X[~m] @ coefs[ifold]
    r2 = np.zeros((nrois, 2))
    # any, not all: a silent ROI (NaN after zscore) must not drop every frame
    train_ok = np.any(np.isfinite(pred_train), axis=1)
    Yt = Y[train_ok]
    r2[:, 0] = 1 - np.sum((pred_train[train_ok] - Yt) ** 2, axis=0) / np.sum(
        (Yt - Yt.mean(axis=0)) ** 2, axis=0
    )
    r2[:, 1] = 1 - np.sum((pred_test - Y) ** 2, axis=0) / total_var
    return coefs, r2, best_idx, r2[:, 1].copy()


def _design_and_folds(imaging_df, frames, shift_stim, k_folds):
    """Design matrix, penalties and test folds of the single- or multi-depth fit."""
    if frames.ndim == 4:
        X, L, L_depth = _multidepth_design(imaging_df, frames, shift_stim)
        fold = multidepth_cv_folds(imaging_df, k_folds)
    elif frames.ndim == 3:
        X, L, L_depth = _single_depth_design(imaging_df, frames, shift_stim)
        fold = single_depth_cv_folds(imaging_df, k_folds)
    else:
        raise ValueError("frames must be 3D or 4D")
    return X, L, L_depth, fold


def fit_3d_rfs(
    imaging_df,
    frames,
    reg_xy=100,
    reg_depth=20,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    choose_rois=(),
):
    """Fit 3D receptive fields using regularized least squares regression, with only one
    set of hyperparameters.

    Runs on all ROIs in parallel. Works for single-depth (3D) and multidepth (4D)
    frames.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): stimulus frames, (nframes, ele, azi) or
            (depth, nframes, ele, azi).
        reg_xy (float): regularization constant for spatial regularization
        reg_depth (float): regularization constant for depth regularization
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.
        choose_rois (list): a list of ROI indices to fit. Defaults to [], which means
            fit all ROIs.

    Returns:
        coef (np.array): coefficients of each fold, k_folds x (ndepths x nele x nazi
            + 1) x ncells
        r2 (np.array): train and test R2 of each ROI, ncells x 2

    """
    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    if len(choose_rois) > 0:
        resps = resps[:, choose_rois]
    X, L, L_depth, fold = _design_and_folds(imaging_df, frames, shift_stim, k_folds)
    roi_regs = np.tile([float(reg_xy), float(reg_depth)], (resps.shape[1], 1))
    coef, r2, _, _ = _ridge_cv(X, resps, L, L_depth, fold, k_folds, roi_regs=roi_regs)
    return coef, r2


fit_3d_rfs_multidepth = fit_3d_rfs


def fit_3d_rfs_grid_search(
    imaging_df,
    frames,
    reg_xys=DEFAULT_REG_GRID,
    reg_depths=DEFAULT_REG_GRID,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    choose_rois=(),
):
    """Best cross-validated R2 of each ROI over a (reg_xy, reg_depth) grid.

    Gives the test R2 and hyperparameters of `fit_3d_rfs_hyperparam_tuning`, without
    computing the coefficients. Works for single-depth (3D) and multidepth (4D)
    frames.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.
        frames (np.array): stimulus frames, (depth, nframes, ele, azi) or
            (nframes, ele, azi).
        reg_xys (np.array): spatial regularization constants. Defaults to
            DEFAULT_REG_GRID.
        reg_depths (np.array): depth regularization constants. Defaults to
            DEFAULT_REG_GRID.
        shift_stim (int): shift to account for response lag. Defaults to 2.
        use_col (str): column to use for the response. Defaults to "dffs".
        k_folds (int): number of folds for cross-validation. Defaults to 5.
        choose_rois (tuple): indices of the ROIs to use. Defaults to ().

    Returns:
        best_r2 (np.array): best test R2 over the grid, (nrois,).
        best_reg (np.array): index of the best (reg_xy, reg_depth) in the grid,
            ordered as in `fit_3d_rfs_hyperparam_tuning`, (nrois,).
    """
    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    if len(choose_rois) > 0:
        resps = resps[:, choose_rois]
    X, L, L_depth, fold = _design_and_folds(imaging_df, frames, shift_stim, k_folds)
    grid = np.array([[a, b] for a in reg_xys for b in reg_depths])
    _, _, best_idx, best_r2 = _ridge_cv(
        X, resps, L, L_depth, fold, k_folds, reg_grid=grid, compute_coefs=False
    )
    return best_r2, best_idx


def fit_3d_rfs_hyperparam_tuning(
    imaging_df,
    frames,
    reg_xys=DEFAULT_REG_GRID,
    reg_depths=DEFAULT_REG_GRID,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    tune_separately=True,
    validation=False,
    **kwargs,
):
    """Fit 3D receptive fields using regularized least squares regression, with
    hyperparameter tuning for each ROI.

    Runs on all ROIs in parallel. Each ROI gets the (reg_xy, reg_depth) of the grid
    with the best cross-validated test R2. The grid is searched with `_ridge_cv`:
    one generalised eigendecomposition per fold and reg_xy serves every reg_depth
    and ROI (see docs/rf_fitting_solver.pdf).

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): stimulus frames, (nframes, ele, azi) or
            (depth, nframes, ele, azi).
        reg_xys (np.array): spatial regularization constants. Defaults to
            DEFAULT_REG_GRID.
        reg_depths (np.array): depth regularization constants. Defaults to
            DEFAULT_REG_GRID.
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.

    Returns:
        coef (np.array): coefficients of each fold, k_folds x (ndepths x nele x nazi
            + 1) x ncells
        r2 (np.array): train and test R2 of each ROI, ncells x 2
        best_reg_xys (np.array): array of best reg_xy for each ROI
        best_reg_depths (np.array): array of best reg_depth for each ROI

    """
    if not tune_separately:
        warnings.warn(
            "tune_separately=False is deprecated and not supported by the generalized "
            "eigensolver. Hyperparameters are tuned per-ROI.",
            FutureWarning,
            stacklevel=2,
        )
    if validation:
        warnings.warn(
            "validation=True is deprecated and not supported by the generalized "
            "eigensolver. K-fold cross-validation evaluates held-out test folds directly.",
            FutureWarning,
            stacklevel=2,
        )

    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    X, L, L_depth, fold = _design_and_folds(imaging_df, frames, shift_stim, k_folds)
    grid = np.array([[a, b] for a in reg_xys for b in reg_depths], dtype=float)
    coef, r2, best_idx, _ = _ridge_cv(
        X, resps, L, L_depth, fold, k_folds, reg_grid=grid
    )
    best_reg_xys, best_reg_depths = grid[best_idx].T
    return coef, r2, best_reg_xys, best_reg_depths


def fit_3d_rfs_ipsi(
    imaging_df,
    frames,
    best_reg_xys,
    best_reg_depths,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    validation=False,
    **kwargs,
):
    """Fit 3D receptive fields using the ipsilateral side of stimuli using regularized least squares regression, using the best set of hyperparameter of the contralateral side.
    Runs on all ROIs in parallel.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): array of frames
        best_reg_xys (list): a list of best regularization constant for spatial regularization from the contra side fitting.
        best_reg_depths (list): a list of best regularization constant for depth regularization from the contra side fitting.
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.

    Returns:
        coef (np.array): coefficients of each fold, k_folds x (ndepths x nele x nazi
            + 1) x ncells
        r2 (np.array): train and test R2 of each ROI, ncells x 2

    """
    if validation:
        warnings.warn(
            "validation=True is deprecated and not supported by the generalized "
            "eigensolver.",
            FutureWarning,
            stacklevel=2,
        )

    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    X, L, L_depth, fold = _design_and_folds(imaging_df, frames, shift_stim, k_folds)
    roi_regs = np.stack([best_reg_xys, best_reg_depths], axis=1).astype(float)
    coef, r2, _, _ = _ridge_cv(X, resps, L, L_depth, fold, k_folds, roi_regs=roi_regs)
    return coef, r2


def find_sig_rfs(coef, coef_ipsi, n_std=6):
    """Find neurons with a significant RF compared to the ipsilateral side.

    A neuron is significant if the peak of its mean contralateral RF exceeds
    n_std standard deviations above the mean of the ipsilateral RF.
    ROIs that are all-NaN across folds are marked as not significant.

    Args:
        coef (list of np.ndarray or np.ndarray): Contralateral RF coefficients per
            fold: a list of (n_features, n_rois) arrays, or a (k_folds, n_features,
            n_rois) array as returned by the fit functions. Coefficients stored per
            ROI in neurons_df, (n_rois, k_folds, n_features), must be reordered
            first with `np.moveaxis(coef, 0, -1)`.
        coef_ipsi (list of np.ndarray or np.ndarray): Ipsilateral RF coefficients,
            same format as `coef`.
        n_std (float, optional): Number of standard deviations above the
            ipsilateral mean to use as the significance threshold. Defaults to 6.

    Returns:
        sig (np.ndarray): Boolean array of shape (n_rois,), True if the
            contralateral RF is significant.
        sig_ipsi (np.ndarray): Boolean array of shape (n_rois,), True if the
            ipsilateral RF exceeds its own threshold (sanity check).
    """
    coef_stacked = np.stack(coef, axis=2)
    coef_ipsi_stacked = np.stack(coef_ipsi, axis=2)
    nrois = coef_stacked.shape[1]

    # ROIs that are all-NaN across folds get False
    valid = ~np.all(np.isnan(coef_stacked), axis=(0, 2))
    sig = np.zeros(nrois, dtype=bool)
    sig_ipsi = np.zeros(nrois, dtype=bool)

    if np.any(valid):
        coef_mean = np.nanmean(coef_stacked[:, valid, :], axis=2)
        coef_ipsi_mean = np.nanmean(coef_ipsi_stacked[:, valid, :], axis=2)

        threshold = n_std * np.nanstd(coef_ipsi_mean[:-1, :], axis=0) + np.nanmean(
            coef_ipsi_mean[:-1, :], axis=0
        )
        sig[valid] = np.nanmax(coef_mean[:-1, :], axis=0) > threshold
        sig_ipsi[valid] = np.nanmax(coef_ipsi_mean[:-1, :], axis=0) > threshold

    return sig, sig_ipsi


def fit_3d_rfs_parametric(coef, nx, ny, nz, model="gaussian"):
    (zs, ys, xs) = np.meshgrid(
        np.arange(nz),
        np.arange(ny),
        np.arange(nx),
        indexing="ij",
    )
    if model == "gaussian":
        func = partial(gaussian_3d_rf, min_sigma=0.25)
    else:
        func = partial(gabor_3d_rf, min_sigma=0.25)

    coef_fit = coef.copy()
    params = []
    # lower_bounds = Gaussian3DRFParams(
    #     log_amplitude=-np.inf,
    #     x0=0,
    #     y0=0,
    #     log_sigma_x2=-np.inf,
    #     log_sigma_y2=-np.inf,
    #     theta=0,
    #     offset=-np.inf,
    #     z0=0,
    #     log_sigma_z=-np.inf,
    # )
    # upper_bounds = Gaussian3DRFParams(
    #     log_amplitude=np.inf,
    #     x0=nx,
    #     y0=ny,
    #     log_sigma_x2=np.inf,
    #     log_sigma_y2=np.inf,
    #     theta=np.pi / 2,
    #     offset=np.inf,
    #     z0=nz,
    #     log_sigma_z=np.inf,
    # )
    # TODO using bounds currently is not working well
    for roi in tqdm(range(coef.shape[1])):
        c = np.reshape(coef[:-1, roi], (nz, ny, nx))
        # get the index of the maximum of c
        idepth, iy, ix = np.unravel_index(np.argmax(c), c.shape)
        if model == "gaussian":
            p0 = Gaussian3DRFParams(
                log_amplitude=np.log(c.max()),
                x0=ix,
                y0=iy,
                log_sigma_x2=0,
                log_sigma_y2=0,
                theta=0,
                offset=0,
                z0=idepth,
                log_sigma_z=0,
            )
        else:
            p0 = Gabor3DRFParams(
                log_amplitude=np.log(c.max()),
                x0=ix,
                y0=iy,
                log_sigma_x2=0,
                log_sigma_y2=0,
                theta=0,
                offset=0,
                log_sf=0,
                alpha=0,
                phase=0,
                z0=idepth,
                log_sigma_z=0,
            )
        try:
            popt = curve_fit(
                func,
                (xs.flatten(), ys.flatten(), zs.flatten()),
                c.flatten(),
                p0=p0,
            )[0]
        except RuntimeError:
            print(f"Warning: failed to fit gaussian to ROI {roi}")
            popt = p0
        coef_fit[:-1, roi] = func((xs.flatten(), ys.flatten(), zs.flatten()), *popt)
        params.append(popt)
    return coef_fit, params
