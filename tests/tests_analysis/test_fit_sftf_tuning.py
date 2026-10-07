"""Tests for `fit_gaussian_blob.fit_sftf_tuning` on synthetic grating responses."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from cottage_analysis.analysis import common_utils
from cottage_analysis.analysis.fit_gaussian_blob import (
    GratingParams,
    fit_sftf_tuning,
    grating_tuning,
)

SF = [0.01, 0.04, 0.16, 0.32]
TF = [0.5, 2.0, 8.0, 16.0]
ANGLES = np.arange(0, 360, 45)
N_RECORDINGS = 3
TUNED = GratingParams(
    log_amplitude=0.0,
    sf0=np.log(0.04),
    tf0=np.log(2.0),
    log_sigma_x2=0.0,
    log_sigma_y2=0.0,
    theta=0.3,
    offset=0.1,
    alpha0=np.pi / 2,
    log_kappa=0.5,
    dsi=0.8,
)


def make_trials(seed=0):
    """Full grid repeated N_RECORDINGS times (shuffled within each recording), with a
    tuned ROI (0) and a noise ROI (1)."""
    rng = np.random.default_rng(seed)
    grid = pd.DataFrame(
        [(sf, tf, a) for sf in SF for tf in TF for a in ANGLES],
        columns=["SpatialFrequency", "TemporalFrequency", "Angle"],
    )
    trials = pd.concat(
        [
            grid.sample(frac=1, random_state=i).assign(irecording=i)
            for i in range(N_RECORDINGS)
        ],
        ignore_index=True,
    )
    X = (
        np.log(trials.SpatialFrequency.to_numpy()),
        np.log(trials.TemporalFrequency.to_numpy()),
        np.deg2rad(trials.Angle.to_numpy()),
    )
    tuned = grating_tuning(X, *TUNED, min_sigma=0.25)
    trials[0] = tuned + rng.normal(0, 0.1, len(trials))
    trials[1] = rng.normal(0, 0.1, len(trials))
    return trials


def reference_fit(trials_df, niter, min_sigma):
    """The loop of fit_sftf_tuning before cross-validation was added."""
    df = trials_df.copy()
    df["log_SF"] = np.log(df["SpatialFrequency"])
    df["log_TF"] = np.log(df["TemporalFrequency"])
    df["Angle_rad"] = np.deg2rad(df["Angle"])
    X = df[["log_SF", "log_TF", "Angle_rad"]].to_numpy()
    lower = GratingParams(
        -np.inf,
        df.log_SF.min() - 1,
        df.log_TF.min() - 1,
        -np.inf,
        -np.inf,
        0,
        -np.inf,
        0,
        -np.inf,
        0,
    )
    upper = GratingParams(
        np.inf,
        df.log_SF.max() + 1,
        df.log_TF.max() + 1,
        np.inf,
        np.inf,
        0.5 * np.pi,
        np.inf,
        2 * np.pi,
        np.inf,
        1,
    )
    out = []
    for roi in [c for c in df.columns if type(c) == int]:

        def p0_func():
            return GratingParams(
                log_amplitude=np.random.normal(),
                sf0=df.groupby("log_SF")[roi].mean().idxmax(),
                tf0=df.groupby("log_TF")[roi].mean().idxmax(),
                log_sigma_x2=np.random.normal(),
                log_sigma_y2=np.random.normal(),
                theta=np.random.uniform(0, 0.5 * np.pi),
                offset=np.random.normal(),
                alpha0=df.groupby("Angle_rad")[roi].mean().idxmax(),
                log_kappa=np.random.normal(),
                dsi=np.random.uniform(0, 1),
            )

        popt, rsq = common_utils.iterate_fit(
            partial(grating_tuning, min_sigma=min_sigma),
            X.T,
            df[roi].to_numpy(),
            lower,
            upper,
            niter=niter,
            p0_func=p0_func,
        )
        out.append(dict(GratingParams(*popt)._asdict(), rsq=rsq))
    return pd.DataFrame(out)


def test_no_crossval_matches_reference():
    trials = make_trials()
    fit = fit_sftf_tuning(trials.copy(), niter=2, min_sigma=0.25)
    ref = reference_fit(trials, niter=2, min_sigma=0.25)
    assert list(fit.columns) == list(ref.columns)
    np.testing.assert_array_equal(fit.to_numpy(float), ref.to_numpy(float))


@pytest.mark.parametrize("fold_col, k_folds", [("irecording", 5), (None, 3)])
def test_crossval_separates_tuned_from_noise(fold_col, k_folds):
    trials = make_trials()
    fit = fit_sftf_tuning(
        trials, niter=2, min_sigma=0.25, k_folds=k_folds, fold_col=fold_col
    )
    n_folds = N_RECORDINGS if fold_col else k_folds
    assert fit.sftf_test_popts.iloc[0].shape == (n_folds, len(GratingParams._fields))
    tuned, noise = fit.iloc[0], fit.iloc[1]
    assert tuned.sftf_test_rsq > 0.5
    assert tuned.sftf_test_spearmanr_rval > 0 and tuned.sftf_test_spearmanr_pval < 1e-10
    assert noise.sftf_test_rsq < 0.05
    # held-out R² is at most the in-sample R² of the fit on all trials
    assert noise.sftf_test_rsq < noise.rsq


def test_parallel_matches_serial():
    trials = make_trials()
    kwargs = dict(niter=2, min_sigma=0.25, k_folds=2, fold_col="irecording")
    serial = fit_sftf_tuning(trials.copy(), n_jobs=1, **kwargs)
    parallel = fit_sftf_tuning(trials.copy(), n_jobs=2, **kwargs)
    cols = [c for c in serial.columns if c != "sftf_test_popts"]
    np.testing.assert_array_equal(
        serial[cols].to_numpy(float), parallel[cols].to_numpy(float)
    )
