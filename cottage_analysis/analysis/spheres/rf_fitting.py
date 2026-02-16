from functools import partial
import gc
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

from cottage_analysis.analysis.fit_gaussian_blob import (
    Gabor3DRFParams,
    Gaussian3DRFParams,
    gabor_3d_rf,
    gaussian_3d_rf,
)

print = partial(print, flush=True)


def laplace_matrix(nx, ny):
    Ls = []
    for x in range(nx):
        for y in range(ny):
            m = np.zeros((nx, ny))
            m[x, y] = 4
            if x > 0:
                m[x - 1, y] = -1
            if x < m.shape[0] - 1:
                m[x + 1, y] = -1
            if y > 0:
                m[x, y - 1] = -1
            if y < m.shape[1] - 1:
                m[x, y + 1] = -1
            Ls.append(m.flatten())
    L = np.stack(Ls, axis=0)
    return L


def fit_3d_rfs(
    imaging_df,
    frames,
    reg_xy=100,
    reg_depth=20,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    choose_rois=(),
    validation=False,
):
    """Fit 3D receptive fields using regularized least squares regression, with only one
    set of hyperparameters.

    Runs on all ROIs in parallel.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): array of frames, nframes x nelevation x nazimuth
        reg_xy (float): regularization constant for spatial regularization
        reg_depth (float): regularization constant for depth regularization
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.
        choose_rois (list): a list of ROI indices to fit. Defaults to [], which means
            fit all ROIs.
        validation (bool): whether to include a validation set for hyperparameter
            tuning. Defaults to False.

    Returns:
        coef (np.array): array of coefficients for each pixel, ndepths x (ndepths x
            nazi x nele + 1) x ncells
        r2 (list): list of arrays of r2 for each ROI for training, validation and test
            sets, ncells x 2

    """
    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    if len(choose_rois) > 0:
        resps = resps[:, choose_rois]
    depths = imaging_df.depth.unique()
    depths = depths[~np.isnan(depths)]
    depths = depths[depths > 0]
    depths = np.sort(depths)
    L = laplace_matrix(frames.shape[1], frames.shape[2])
    Ls = []
    Ls_depth = []

    trial_idx = np.zeros_like(imaging_df.depth)
    trial_idx = np.cumsum(
        np.logical_and(np.abs(imaging_df.depth.diff()) > 0, imaging_df.depth > 0)
    )
    trial_idx[imaging_df.depth.isna()] = np.nan
    trial_idx[imaging_df.depth < 0] = np.nan
    imaging_df["trial_idx"] = trial_idx
    # get the depth of the first row for each trial
    depths_by_trial = imaging_df.groupby("trial_idx").first().depth
    # convert to categorical codes
    categorical = pd.Categorical(depths_by_trial).codes
    depths_by_trial.update(pd.Series(categorical, index=depths_by_trial.index))
    depths_by_trial = depths_by_trial.astype(categorical.dtype)
    # convert index to int
    depths_by_trial.index = depths_by_trial.index.astype(int)

    X = np.zeros((frames.shape[0], frames.shape[1] * frames.shape[2] * depths.shape[0]))
    for idepth, depth in enumerate(depths):
        depth_idx = imaging_df.depth == depth
        m = np.roll(np.reshape(frames, (frames.shape[0], -1)), shift_stim, axis=0)[
            depth_idx, :
        ]
        # place m in the right columns of X
        X[depth_idx, idepth * m.shape[1] : (idepth + 1) * m.shape[1]] = m
        # add regularization penalty on the second derivative of the coefficients
        # in X and Y
        L_xy = np.zeros((L.shape[0], X.shape[1]))
        L_xy[:, idepth * L.shape[1] : (idepth + 1) * L.shape[1]] = L
        Ls.append(L_xy)
        # add regularization penalty on the second derivative of the coefficients
        # along the depth axis
        L_depth = np.zeros((m.shape[1], X.shape[1]))
        L_depth[:, idepth * m.shape[1] : (idepth + 1) * m.shape[1]] = (
            np.identity(m.shape[1]) * 2
        )
        if idepth > 0:
            L_depth[:, (idepth - 1) * m.shape[1] : idepth * m.shape[1]] = -np.identity(
                m.shape[1]
            )
        if idepth < depths.shape[0] - 1:
            L_depth[
                :, (idepth + 1) * m.shape[1] : (idepth + 2) * m.shape[1]
            ] = -np.identity(m.shape[1])
        Ls_depth.append(L_depth)

    L = np.concatenate(Ls, axis=0)
    L = np.concatenate([L, np.zeros((L.shape[0], 1))], axis=1)
    L_depth = np.concatenate(Ls_depth, axis=0)
    L_depth = np.concatenate([L_depth, np.zeros((L_depth.shape[0], 1))], axis=1)
    # add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coefs = []
    # 0 for train and -1 for test, 1 for validation prediction
    n_splits = 3 if validation else 2
    Y_pred = np.zeros((resps.shape[0], resps.shape[1], n_splits)) * np.nan
    # randomly split trials into training and test sets
    stratified_kfold = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
    # Use validation set to select the best regularization parameters (train, val, test),
    # or use test set to evaluate performance (train, test)
    for train_trials, test_trials in stratified_kfold.split(
        depths_by_trial.index, depths_by_trial.values
    ):
        if validation:
            train_trials, validation_trials = train_test_split(
                train_trials,
                stratify=depths_by_trial.iloc[train_trials].values,
                test_size=(1 / (k_folds - 1)),
            )
            validation_idx = np.isin(imaging_df.trial_idx, validation_trials)
        train_idx = np.isin(imaging_df.trial_idx, train_trials)
        test_idx = np.isin(imaging_df.trial_idx, test_trials)

        X_train = np.concatenate(
            [X[train_idx, :], reg_xy * L, reg_depth * L_depth], axis=0
        )
        Q = np.linalg.inv(X_train.T @ X_train) @ X_train.T

        Y_train = np.concatenate(
            [
                resps[train_idx, :],
                np.zeros((L.shape[0], resps.shape[1])),
                np.zeros((L_depth.shape[0], resps.shape[1])),
            ],
            axis=0,
        )
        coef = Q @ Y_train
        coefs.append(coef)

        if validation:
            idxs = [train_idx, validation_idx, test_idx]
        else:
            idxs = [train_idx, test_idx]
        for isplit, idx in enumerate(idxs):
            Y_pred[idx, :, isplit] = X[idx, :] @ coef
    # calculate R2
    r2 = np.zeros((resps.shape[1], n_splits)) * np.nan
    for isplit in range(n_splits):
        use_idx = np.isfinite(Y_pred[:, 0, isplit])
        residual_var = np.sum(
            (Y_pred[use_idx, :, isplit] - resps[use_idx, :]) ** 2,
            axis=0,
        )
        total_var = np.sum(
            (resps[use_idx, :] - np.mean(resps[use_idx, :], axis=0)) ** 2, axis=0
        )
        r2[:, isplit] = 1 - residual_var / total_var
    return coefs, r2


def fit_3d_rfs_multidepth(
    imaging_df,
    frames,
    reg_xy=100,
    reg_depth=20,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    choose_rois=(),
    validation=False,
):
    """Fit 3D receptive fields with multiple depths.

    Adapted version of fit_3d_rfs to fit data where mutliple depth are present on the
    same trial

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.
        frames (np.array): imaging data (nframes, depth, ele, azi).
        reg_xy (float): regularization penalty on the first derivative of the coefficients
            along the x and y axes. Defaults to 100.
        reg_depth (float): regularization penalty on the first derivative of the coefficients
            along the depth axis. Defaults to 20.
        shift_stim (int): shift to account for response lag. Defaults to 2.
        use_col (str): column to use for the response. Defaults to "dffs".
        k_folds (int): number of folds for cross-validation. Defaults to 5.
        choose_rois (tuple): indices of the ROIs to use. Defaults to ().
        validation (bool): if True, use validation set to select the best regularization
            parameters (train, val, test). Defaults to False.

    Returns:
        np.array: 3D receptive fields (depth, ele, azi).
        np.array: R2 values for each ROI and split.
    """
    ndepths, nframes, nelev, nazim = frames.shape

    resps = zscore(np.concatenate(imaging_df[use_col]), axis=0)
    if len(choose_rois) > 0:
        resps = resps[:, choose_rois]
    depths = imaging_df.depth.unique()
    depths = depths[~np.isnan(depths)]
    depths = depths[depths > 0]
    depths = np.sort(depths)

    is_stim = imaging_df.depth > 0
    trial_start_stop = np.diff(is_stim.astype(int))
    trial_idx = np.cumsum(np.hstack([0, trial_start_stop == 1])).astype(float)
    trial_idx[imaging_df.depth.isna()] = np.nan
    trial_idx[imaging_df.depth < 0] = np.nan
    imaging_df["trial_idx"] = trial_idx

    assert depths.shape[0] == frames.shape[0]
    # Shift to account for response lag
    X = np.roll(frames, shift_stim, axis=1)
    X = np.swapaxes(X, 0, 1)  # put back frame number as first axis
    # (now we have frame, depth, ele, azi)
    X = X.reshape(X.shape[0], -1)  # flatten

    L = laplace_matrix(nelev, nazim)
    Ls = []
    Ls_depth = []
    for idepth, depth in enumerate(depths):
        L_xy = np.zeros((L.shape[0], X.shape[1]))
        L_xy[:, idepth * L.shape[1] : (idepth + 1) * L.shape[1]] = L
        Ls.append(L_xy)
        # add regularization penalty on the second derivative of the coefficients
        # along the depth axis
        L_depth = np.zeros((L.shape[1], X.shape[1]))
        L_depth[:, idepth * L.shape[1] : (idepth + 1) * L.shape[1]] = (
            np.identity(L.shape[1]) * 2
        )
        if idepth > 0:
            L_depth[:, (idepth - 1) * L.shape[1] : idepth * L.shape[1]] = -np.identity(
                L.shape[1]
            )
        if idepth < depths.shape[0] - 1:
            L_depth[
                :, (idepth + 1) * L.shape[1] : (idepth + 2) * L.shape[1]
            ] = -np.identity(L.shape[1])
        Ls_depth.append(L_depth)
    L = np.concatenate(Ls, axis=0)
    L = np.concatenate([L, np.zeros((L.shape[0], 1))], axis=1)
    L_depth = np.concatenate(Ls_depth, axis=0)
    L_depth = np.concatenate([L_depth, np.zeros((L_depth.shape[0], 1))], axis=1)
    # add bias
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coefs = []
    # 0 for train and -1 for test, 1 for validation prediction
    n_splits = 3 if validation else 2
    Y_pred = np.zeros((resps.shape[0], resps.shape[1], n_splits)) * np.nan
    # randomly split trials into training and test sets
    kfold = KFold(n_splits=k_folds, random_state=42, shuffle=True)
    # Use validation set to select the best regularization parameters (train, val, test),
    # or use test set to evaluate performance (train, test)
    trials = imaging_df.trial_idx.dropna().unique()
    for train_trials, test_trials in kfold.split(trials):
        if validation:
            train_trials, validation_trials = train_test_split(
                train_trials,
                test_size=(1 / (k_folds - 1)),
            )
            validation_idx = np.isin(imaging_df.trial_idx, validation_trials)
        train_idx = np.isin(imaging_df.trial_idx, train_trials)
        test_idx = np.isin(imaging_df.trial_idx, test_trials)

        X_train = np.concatenate(
            [X[train_idx, :], reg_xy * L, reg_depth * L_depth], axis=0
        )
        Q = np.linalg.inv(X_train.T @ X_train) @ X_train.T

        Y_train = np.concatenate(
            [
                resps[train_idx, :],
                np.zeros((L.shape[0], resps.shape[1])),
                np.zeros((L_depth.shape[0], resps.shape[1])),
            ],
            axis=0,
        )
        coef = Q @ Y_train
        coefs.append(coef)

        if validation:
            idxs = [train_idx, validation_idx, test_idx]
        else:
            idxs = [train_idx, test_idx]
        for isplit, idx in enumerate(idxs):
            Y_pred[idx, :, isplit] = X[idx, :] @ coef
    # calculate R2
    r2 = np.zeros((resps.shape[1], n_splits)) * np.nan
    for isplit in range(n_splits):
        use_idx = np.isfinite(Y_pred[:, 0, isplit])
        residual_var = np.sum(
            (Y_pred[use_idx, :, isplit] - resps[use_idx, :]) ** 2,
            axis=0,
        )
        total_var = np.sum(
            (resps[use_idx, :] - np.mean(resps[use_idx, :], axis=0)) ** 2, axis=0
        )
        r2[:, isplit] = 1 - residual_var / total_var
    return coefs, r2


def fit_3d_rfs_hyperparam_tuning(
    imaging_df,
    frames,
    reg_xys=[20, 40, 80, 160, 320],
    reg_depths=[20, 40, 80, 160, 320],
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    tune_separately=True,
    validation=True,
    r2_threshold=0.01,
):
    """Fit 3D receptive fields using regularized least squares regression, with hyperparameter tuning.
    Runs on all ROIs in parallel.

    Args:
        imaging_df (pd.DataFrame): dataframe that contains info for each imaging volume.
        frames (np.array): array of frames
        reg_xys (list): a list of regularization constant for spatial regularization
        reg_depths (list): a list of regularization constant for depth regularization
        shift_stim (int): number of frames to shift the stimulus by.
            This is to account for the delay between the stimulus and the response.
            Defaults to 2.
        use_col (str): column in imaging_df to use for fitting. Defaults to "dffs".
        k_folds (int): number of folds for cross validation. Defaults to 5.
        tune_separately (bool): whether to tune hyperparameters separately for each ROI. Defaults to True.
        validation (bool): whether to include a validation set for hyperparameter tuning. Defaults to False.
        r2_threshold (float): threshold for the minimum R2 for a ROI to be considered good. Defaults to 0.01.

    Returns:
        coef (np.array): array of coefficients for each pixel, ndepths x (ndepths x nazi x nele + 1) x ncells
        r2 (list): list of arrays of r2 for each ROI for training, validation and test sets, ncells x 2
        best_reg_xys (np.array): array of best reg_xy for each ROI
        best_reg_depths (np.array): array of best reg_depth for each ROI

    """
    depth_list = imaging_df.depth.dropna().unique()
    depth_list = np.sort(depth_list[depth_list > 0])
    all_coef = np.zeros(
        (
            len(reg_xys) * len(reg_depths),
            k_folds,
            frames.shape[-2] * frames.shape[-1] * len(depth_list) + 1,
            imaging_df.loc[0, "dffs"].shape[1],
        )
    )
    all_r2s = np.zeros(
        (len(reg_xys) * len(reg_depths), imaging_df.loc[0, "dffs"].shape[1], 2)
    )
    hyperparams = np.zeros((len(reg_xys) * len(reg_depths), 2))
    good_neuron_percs = np.zeros((len(reg_xys), len(reg_depths)))
    nrois = imaging_df.loc[0, "dffs"].shape[1]
    if frames.ndim == 4:
        fit_func = fit_3d_rfs_multidepth
    elif frames.ndim == 3:
        fit_func = fit_3d_rfs
    else:
        raise ValueError("frames must be 3D or 4D")
    idx = 0
    for i, reg_xy in enumerate(reg_xys):
        for j, reg_depth in enumerate(reg_depths):
            print(f"fitting reg_xy: {reg_xy}, reg_depth: {reg_depth}")
            coef, r2 = fit_func(
                imaging_df,
                frames,
                reg_xy=reg_xy,
                reg_depth=reg_depth,
                shift_stim=shift_stim,
                use_col=use_col,
                k_folds=k_folds,
                validation=validation,
            )
            gc.collect()
            good_neuron_percs[i, j] = np.mean(r2[:, 1] > r2_threshold)
            all_coef[idx] = np.stack(coef)
            all_r2s[idx] = r2
            hyperparams[idx] = [reg_xy, reg_depth]
            # all_coef.append(np.stack(coef))
            # all_r2s.append(r2)
            # hyperparams.append([reg_xy, reg_depth])
            idx += 1
    if not tune_separately:
        max_idx = np.argmax(good_neuron_percs)
        best_reg_xy, best_reg_depth = hyperparams[max_idx]
        print(
            f"Best param found for all ROIs: "
            f"reg_xy: {best_reg_xy}, "
            f"reg_depth: {best_reg_depth}, "
            f"R2>{r2_threshold}: {good_neuron_percs[max_idx]:.4f}"
        )
        coef = all_coef[max_idx]
        best_reg_xys = np.ones(nrois) * best_reg_xy
        best_reg_depths = np.ones(nrois) * best_reg_depth
    else:
        coef = np.zeros_like(all_coef[0])
        best_hyperparam_idxs = np.argmax(np.stack(all_r2s, axis=0)[:, :, 1], axis=0)
        best_reg_xys = np.zeros(nrois)
        best_reg_depths = np.zeros(nrois)
        for iroi in range(nrois):
            [best_reg_xy, best_reg_depth] = hyperparams[best_hyperparam_idxs[iroi]]
            print(
                f"Best param found for ROI {iroi}: "
                f"reg_xy: {best_reg_xy}, "
                f"reg_depth: {best_reg_depth}"
            )
            best_reg_xys[iroi] = best_reg_xy
            best_reg_depths[iroi] = best_reg_depth
            coef[:, :, iroi] = all_coef[best_hyperparam_idxs[iroi]][:, :, iroi]
            r2[iroi, :] = all_r2s[best_hyperparam_idxs[iroi]][iroi, :]
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
        validation (bool): whether to include a validation set for hyperparameter tuning. Defaults to False.

    Returns:
        coef (np.array): array of coefficients for each pixel, ndepths x (ndepths x nazi x nele + 1) x ncells
        r2 (list): list of arrays of r2 for each ROI for training, validation and test sets, ncells x 2

    """
    if frames.ndim == 4:
        fit_func = fit_3d_rfs_multidepth
    elif frames.ndim == 3:
        fit_func = fit_3d_rfs
    else:
        raise ValueError("frames must be 3D or 4D")
    best_regs = np.stack([best_reg_xys, best_reg_depths], axis=1)
    coef_temp, r2_temp = fit_func(
        imaging_df,
        frames,
        reg_xy=80,
        reg_depth=40,
        shift_stim=shift_stim,
        use_col=use_col,
        k_folds=k_folds,
        validation=validation,
    )
    coef = np.zeros_like(np.stack(coef_temp))
    r2 = np.zeros_like(np.stack(r2_temp))
    for best_reg in np.unique(best_regs, axis=0):
        best_reg_neurons = np.where(np.all(best_reg == best_regs, axis=1))[0]
        print(
            f"Fit with best param for {len(best_reg_neurons)} neurons: reg_xy: {best_reg[0]}, reg_depth: {best_reg[1]}"
        )
        coef_temp, r2_temp = fit_func(
            imaging_df,
            frames,
            reg_xy=best_reg[0],
            reg_depth=best_reg[1],
            shift_stim=shift_stim,
            use_col=use_col,
            k_folds=k_folds,
            choose_rois=best_reg_neurons,
            validation=validation,
        )
        gc.collect()
        coef[:, :, best_reg_neurons] = np.stack(coef_temp)
        r2[best_reg_neurons, :] = r2_temp
    return coef, r2


def find_sig_rfs(coef, coef_ipsi, n_std=6):
    """Find the neurons with a significant RF (compared to ipsi side)

    Args:
        coef (_type_): _description_
        coef_ipsi (_type_): _description_
        n_std (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    coef_mean = np.nanmean(np.stack(coef, axis=2), axis=2)
    coef_ipsi_mean = np.nanmean(np.stack(coef_ipsi, axis=2), axis=2)

    threshold = n_std * np.nanstd(coef_ipsi_mean[:-1, :], axis=0) + np.nanmean(
        coef_ipsi_mean[:-1, :], axis=0
    )
    sig = np.nanmax(coef_mean[:-1, :], axis=0) > threshold
    sig_ipsi = np.nanmax(coef_ipsi_mean[:-1, :], axis=0) > threshold

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


def find_valid_frames(frame_times, trials_df, verbose=True):
    """Find frame numbers that are valid (not gray period, or not before or after the
    imaging frames) and used for regenerating sphere stimuli.

    Args:
        frame_times (np.array): Array of time at which the frame should be regenerated
        trials_df (pd.DataFrame): Dataframe contains information for each trial.
        verbose (bool, optional): Print information. Defaults to True.

    Returns:
        frame_indices (np.array): Array of valid frame indices.
    """
    # for frames before and after the protocol, keep them 0s
    before = frame_times < trials_df.imaging_harptime_stim_start.iloc[0]
    after = frame_times > trials_df.imaging_harptime_stim_stop.iloc[-1]
    if verbose:
        print(
            "Ignoring %d frames before and %d after the stimulus presentation"
            % (np.sum(before), np.sum(after))
        )
    valid_frames = ~before & ~after

    trial_index = (
        trials_df.imaging_harptime_stim_start.searchsorted(frame_times, side="right")
        - 1
    )
    trial_index = np.clip(trial_index, 0, len(trials_df) - 1)
    trial_end = trials_df.loc[trial_index, "imaging_harptime_stim_stop"].values
    grey_time = frame_times - trial_end > 0
    if verbose:
        print(
            "Ignoring %d frames in grey inter-trial intervals"
            % np.sum(grey_time & valid_frames)
        )
    valid_frames = valid_frames & (~grey_time)
    frame_indices = np.where(valid_frames)[0]

    return frame_indices
