import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

from cottage_analysis.analysis.spheres import rf_fitting as rf
from cottage_analysis.analysis.spheres import stimulus_reconstruction

REG_XYS = np.array([0.5, 5.0, 50.0])
REG_DEPTHS = np.array([1.0, 10.0])


def reference_fit(imaging_df, frames, reg_xy, reg_depth, k_folds=5):
    """Exhaustive fit at one (reg_xy, reg_depth), as the solver before `_ridge_cv`.

    Solves the stacked least squares [X_train; reg_xy L; reg_depth D] c = [Y_train; 0; 0]
    with an explicit inverse for each fold, on the design and folds of rf_fitting.
    """
    Y = zscore(np.concatenate(imaging_df["dffs"]), axis=0)
    X, L, D, fold = rf._design_and_folds(imaging_df, frames, 2, k_folds)
    coefs = []
    pred_train = np.full_like(Y, np.nan)
    pred_test = np.full_like(Y, np.nan)
    for ifold in range(k_folds):
        train, test = np.isfinite(fold) & (fold != ifold), fold == ifold
        A = np.concatenate([X[train], reg_xy * L, reg_depth * D])
        Y_aug = np.concatenate(
            [Y[train], np.zeros((L.shape[0] + D.shape[0], Y.shape[1]))]
        )
        coef = np.linalg.inv(A.T @ A) @ A.T @ Y_aug
        coefs.append(coef)
        pred_train[train] = X[train] @ coef
        pred_test[test] = X[test] @ coef
    r2 = np.zeros((Y.shape[1], 2))
    for isplit, pred in enumerate([pred_train, pred_test]):
        ok = np.isfinite(pred[:, 0])
        sse = np.sum((pred[ok] - Y[ok]) ** 2, axis=0)
        r2[:, isplit] = 1 - sse / np.sum((Y[ok] - Y[ok].mean(axis=0)) ** 2, axis=0)
    return np.stack(coefs), r2


def reference_tuning(imaging_df, frames, reg_xys, reg_depths):
    """Exhaustive grid search: refit at every pair, best test R2 per ROI."""
    grid = np.array([[a, b] for a in reg_xys for b in reg_depths])
    fits = [reference_fit(imaging_df.copy(), frames, a, b) for a, b in grid]
    best = np.argmax(np.stack([r2[:, 1] for _, r2 in fits]), axis=0)
    coef = np.stack([fits[b][0][:, :, i] for i, b in enumerate(best)], axis=2)
    r2 = np.stack([fits[b][1][i] for i, b in enumerate(best)])
    return coef, r2, grid[best, 0], grid[best, 1]


def make_session(seed=1):
    """Small multidepth session: 3 tuned and 3 untuned ROIs, 30 trials."""
    rng = np.random.default_rng(seed)
    ndepths, nele, nazi, ncells = 3, 3, 4, 6
    depth = []
    for _ in range(30):
        depth += [np.nan] * 3 + [1.0] * rng.integers(15, 30)
    depth = np.array(depth)
    frames = (rng.random((ndepths, len(depth), nele, nazi)) > 0.7).astype(float)
    frames[:, np.isnan(depth)] = 0
    stim = ~np.isnan(depth)
    depth[stim] = 1 + (np.cumsum(stim)[stim] % ndepths)
    weights = rng.normal(size=(ndepths * nele * nazi, ncells)) * (np.arange(ncells) < 3)
    drive = np.roll(frames, 2, axis=1).swapaxes(0, 1).reshape(len(depth), -1)
    resp = drive @ weights + rng.normal(size=(len(depth), ncells))
    imaging_df = pd.DataFrame(
        {"depth": depth, "stim": stim.astype(int), "dffs": list(resp[:, None, :])}
    )
    return imaging_df, frames


def test_grid_search_matches_hyperparam_tuning():
    imaging_df, frames = make_session()
    _, r2, best_xy, best_depth = reference_tuning(
        imaging_df, frames, REG_XYS, REG_DEPTHS
    )
    best_r2, best_reg = rf.fit_3d_rfs_grid_search(
        imaging_df.copy(), frames, reg_xys=REG_XYS, reg_depths=REG_DEPTHS
    )
    grid = np.array([[a, b] for a in REG_XYS for b in REG_DEPTHS])
    np.testing.assert_allclose(best_r2, r2[:, 1], atol=1e-10)
    np.testing.assert_array_equal(grid[best_reg, 0], best_xy)
    np.testing.assert_array_equal(grid[best_reg, 1], best_depth)


def make_single_depth_session(seed=2, start_in_trial=False):
    """Small single-depth session: one depth per trial, gray (depth -99.99) between trials."""
    rng = np.random.default_rng(seed)
    depths, nele, nazi, ncells = [0.1, 0.4, 1.6], 3, 4, 6
    depth = [] if start_in_trial else [-99.99] * 2
    for itrial in range(30):
        depth += [depths[itrial % 3]] * rng.integers(15, 30) + [-99.99] * 3
    depth = np.array(depth)
    frames = (rng.random((len(depth), nele, nazi)) > 0.7).astype(float)
    frames[depth < 0] = 0
    drive = np.zeros((len(depth), len(depths) * nele * nazi))
    lagged = np.roll(frames, 2, axis=0).reshape(len(depth), -1)
    for i, d in enumerate(depths):
        m = depth == d
        drive[m, i * nele * nazi : (i + 1) * nele * nazi] = lagged[m]
    weights = rng.normal(size=(drive.shape[1], ncells)) * (np.arange(ncells) < 3)
    resp = drive @ weights + rng.normal(size=(len(depth), ncells))
    imaging_df = pd.DataFrame({"depth": depth, "dffs": list(resp[:, None, :])})
    return imaging_df, frames


@pytest.mark.parametrize(
    "maker",
    [
        make_session,
        make_single_depth_session,
        lambda: make_single_depth_session(start_in_trial=True),
    ],
    ids=["multidepth", "single_depth", "single_depth_trial_at_start"],
)
def test_hyperparam_tuning_matches_exhaustive_search(maker):
    imaging_df, frames = maker()
    fast = rf.fit_3d_rfs_hyperparam_tuning(
        imaging_df.copy(), frames, reg_xys=REG_XYS, reg_depths=REG_DEPTHS, k_folds=5
    )
    slow = reference_tuning(imaging_df, frames, REG_XYS, REG_DEPTHS)
    for quick, ref in zip(fast, slow):
        np.testing.assert_allclose(quick, ref, atol=1e-8)


@pytest.mark.parametrize(
    "maker",
    [make_session, make_single_depth_session],
    ids=["multidepth", "single_depth"],
)
def test_fits_at_given_regs_match_reference(maker):
    imaging_df, frames = maker()
    # fit_3d_rfs: one pair for all ROIs, on a subset of ROIs
    coef, r2 = rf.fit_3d_rfs(
        imaging_df.copy(), frames, reg_xy=5.0, reg_depth=10.0, choose_rois=[1, 4]
    )
    coef_ref, r2_ref = reference_fit(imaging_df.copy(), frames, 5.0, 10.0)
    np.testing.assert_allclose(coef, coef_ref[:, :, [1, 4]], atol=1e-8)
    np.testing.assert_allclose(r2, r2_ref[[1, 4]], atol=1e-8)
    # fit_3d_rfs_ipsi: a different pair for each ROI
    ncells = r2_ref.shape[0]
    best_xy = REG_XYS[np.arange(ncells) % len(REG_XYS)]
    best_depth = REG_DEPTHS[np.arange(ncells) % len(REG_DEPTHS)]
    coef, r2 = rf.fit_3d_rfs_ipsi(imaging_df.copy(), frames, best_xy, best_depth)
    for iroi in range(ncells):
        coef_ref, r2_ref = reference_fit(
            imaging_df.copy(), frames, best_xy[iroi], best_depth[iroi]
        )
        np.testing.assert_allclose(coef[:, :, iroi], coef_ref[:, :, iroi], atol=1e-8)
        np.testing.assert_allclose(r2[iroi], r2_ref[iroi], atol=1e-8)


@pytest.mark.parametrize("start_in_trial", [False, True])
def test_single_depth_folds_use_every_trial_once(start_in_trial):
    imaging_df, _ = make_single_depth_session(start_in_trial=start_in_trial)
    fold = rf.single_depth_cv_folds(imaging_df, k_folds=5)
    trial_idx = rf._single_depth_trial_index(imaging_df)
    np.testing.assert_array_equal(np.isfinite(fold), np.isfinite(trial_idx))
    per_trial = pd.Series(fold).groupby(trial_idx).nunique()
    assert np.all(per_trial == 1)
    # stratified: each depth is spread over the folds
    depth_by_trial = imaging_df.depth.groupby(trial_idx).first()
    fold_by_trial = pd.Series(fold).groupby(trial_idx).first()
    counts = pd.crosstab(depth_by_trial, fold_by_trial)
    assert counts.values.min() >= 1 and counts.values.max() - counts.values.min() <= 1


def test_trial_index_uses_stim_column():
    # multidepth: depth is negative for sphere removals, also inside trials
    stim = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    depth = np.array([-1, -1, 0.2, -1, 1.6, 3.2, -1, -1, -1, 0.4, -1, -1])
    imaging_df = pd.DataFrame({"stim": stim, "depth": depth})
    expected = np.array([np.nan] * 2 + [1] * 4 + [np.nan] * 2 + [2] * 3 + [np.nan])
    np.testing.assert_array_equal(rf._trial_index(imaging_df), expected)
    with pytest.raises(ValueError):
        rf._trial_index(imaging_df.drop(columns="stim"))
    # a trial still running at the end of the recording is excluded
    unfinished = pd.DataFrame(
        {"stim": np.r_[stim, 1, 1], "depth": np.r_[depth, 0.8, -1]}
    )
    np.testing.assert_array_equal(
        rf._trial_index(unfinished), np.r_[expected, np.nan, np.nan]
    )


def test_cv_folds_use_every_trial_once():
    imaging_df, _ = make_session()
    trial_idx = rf._trial_index(imaging_df)
    fold = rf.multidepth_cv_folds(imaging_df, k_folds=5)
    np.testing.assert_array_equal(np.isfinite(fold), np.isfinite(trial_idx))
    per_trial = pd.Series(fold).groupby(trial_idx).nunique()
    assert np.all(per_trial == 1)
    assert set(np.unique(fold[np.isfinite(fold)])) == set(range(5))


def test_silent_roi_does_not_break_other_rois():
    imaging_df, frames = make_session()
    resp = np.concatenate(imaging_df.dffs)
    resp[:, 5] = 0  # silent ROI: NaN after zscore
    imaging_df["dffs"] = list(resp[:, None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        _, r2, _, _ = rf.fit_3d_rfs_hyperparam_tuning(
            imaging_df.copy(), frames, reg_xys=REG_XYS, reg_depths=REG_DEPTHS
        )
    assert np.all(np.isfinite(r2[:5])), r2
    assert np.all(np.isnan(r2[5]))


def test_find_valid_frames_uses_positions_not_labels():
    trials_df = pd.DataFrame(
        {
            "imaging_harptime_stim_start": [0.0, 10.0, 20.0],
            "imaging_harptime_stim_stop": [5.0, 15.0, 25.0],
        },
        index=[7, 3, 12],  # e.g. after filtering trials
    )
    frame_times = np.array([-1, 1, 6, 11, 16, 21, 26], dtype=float)
    frames = stimulus_reconstruction.find_valid_frames(
        frame_times, trials_df, verbose=False
    )
    np.testing.assert_array_equal(frames, [1, 3, 5])


@pytest.mark.parametrize(
    "maker",
    [make_session, make_single_depth_session],
    ids=["multidepth", "single_depth"],
)
def test_fits_do_not_modify_imaging_df(maker):
    imaging_df, frames = maker()
    before = imaging_df.copy()
    coef, _, best_xy, best_depth = rf.fit_3d_rfs_hyperparam_tuning(
        imaging_df, frames, reg_xys=REG_XYS, reg_depths=REG_DEPTHS
    )
    rf.fit_3d_rfs_ipsi(imaging_df, frames, best_xy, best_depth)
    rf.fit_3d_rfs_grid_search(
        imaging_df, frames, reg_xys=REG_XYS, reg_depths=REG_DEPTHS
    )
    pd.testing.assert_frame_equal(imaging_df, before)


def test_trial_index_rf_use_frame_keeps_trial_ids():
    df = pd.DataFrame({"stim": [0, 1, 1, 1, 0, 1, 1, 0]})
    full = rf._trial_index(df)
    df["rf_use_frame"] = [True, True, False, True, True, False, True, True]
    masked = rf._trial_index(df)
    np.testing.assert_array_equal(
        masked, [np.nan, 1, np.nan, 1, np.nan, np.nan, 2, np.nan]
    )
    assert np.all(np.isnan(masked) | (masked == full))


def test_deprecated_flags_warn():
    imaging_df, frames = make_session()
    with pytest.warns(FutureWarning, match="tune_separately=False is deprecated"):
        rf.fit_3d_rfs_hyperparam_tuning(
            imaging_df,
            frames,
            reg_xys=REG_XYS,
            reg_depths=REG_DEPTHS,
            tune_separately=False,
        )
    with pytest.warns(FutureWarning, match="validation=True is deprecated"):
        rf.fit_3d_rfs_hyperparam_tuning(
            imaging_df,
            frames,
            reg_xys=REG_XYS,
            reg_depths=REG_DEPTHS,
            validation=True,
        )
    nrois = imaging_df["dffs"].iloc[0].shape[1]
    with pytest.warns(FutureWarning, match="validation=True is deprecated"):
        rf.fit_3d_rfs_ipsi(
            imaging_df,
            frames,
            [REG_XYS[0]] * nrois,
            [REG_DEPTHS[0]] * nrois,
            validation=True,
        )
