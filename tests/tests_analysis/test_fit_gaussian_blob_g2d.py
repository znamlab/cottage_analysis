"""Standalone regression and comparison test suite for closed-loop `g2d` fitting.

Uses the self-contained 10-cell dataset in
`tests/tests_analysis/test_data/g2d_closedloop_10cells.npz` (extracted from
`PZAG16.3b_S20250224`). Requires no Flexilims connection, no external drives, and no
external packages beyond `cottage_analysis` and its core dependencies.

Provides:
- `load_g2d_test_dataset()`: reconstructs `(trials_df, ref_df, data_dict)` from the
  compressed `.npz` fixture.
- `compare_g2d_fit_to_reference(fit_df, ref_df, data_dict)`: computes a per-ROI
  comparison DataFrame of parameterization-invariant metrics (`delta_rsq`,
  `pred_corr`, `delta_angle_deg`, ` preferred_rs_ratio`, `preferred_of_ratio`,
  `delta_semimajor`, `delta_semiminor`, `delta_eccentricity`) so different versions
  of the `g2d` fitting procedure (e.g., sigma-theta vs. Cholesky precision on `dev`)
  can be compared directly.
"""

import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cottage_analysis.analysis import common_utils, fit_gaussian_blob

DATA_PATH = Path(__file__).resolve().parent / "test_data" / "g2d_closedloop_10cells.npz"


def _is_legacy_sigma_theta():
    """Return True if `Gaussian2DParams` uses `(log_sigma_x2, log_sigma_y2, theta)`."""
    return "log_sigma_x2" in fit_gaussian_blob.Gaussian2DParams._fields


def _convert_popt_for_current_branch(popt_legacy, min_sigma=0.25):
    """Convert a legacy 7-element `popt` to the current branch's parameterization."""
    if _is_legacy_sigma_theta():
        return np.asarray(popt_legacy, dtype=np.float64)
    if hasattr(fit_gaussian_blob, "convert_g2d_popt_old_to_cholesky"):
        popt_clamped = list(popt_legacy)
        # Below ~-18, exp(log_sigma) << min_sigma, and raw 1/exp(log_sigma) causes
        # float64 cancellation in c - l21**2 inside convert_g2d_popt_old_to_cholesky.
        popt_clamped[3] = max(float(popt_clamped[3]), -18.0)
        popt_clamped[4] = max(float(popt_clamped[4]), -18.0)
        return np.asarray(
            fit_gaussian_blob.convert_g2d_popt_old_to_cholesky(
                popt_clamped, min_sigma=min_sigma
            ),
            dtype=np.float64,
        )
    raise RuntimeError(
        "Unknown Gaussian2DParams parameterization and no converter found."
    )


def _clamp_floor_log_sigmas(popt, floor=-15.0):
    """Clamp log-sigmas below `floor` where exp(log_sigma) << min_sigma.

    Avoids spurious regression failures from unidentifiable optimizer noise on
    parameters hitting the min_sigma variance floor.
    """
    popt = np.asarray(popt, dtype=np.float64).copy()
    popt[..., 3] = np.maximum(popt[..., 3], floor)
    popt[..., 4] = np.maximum(popt[..., 4], floor)
    return popt


def _get_angle_deg(popt, min_sigma=0.25):
    """Call `get_gaussian_angle` compatibly across `reviews` (1 arg) and `dev` (2 args)."""
    sig = inspect.signature(fit_gaussian_blob.get_gaussian_angle)
    if "min_sigma" in sig.parameters:
        return fit_gaussian_blob.get_gaussian_angle(popt, min_sigma=min_sigma)
    return fit_gaussian_blob.get_gaussian_angle(popt)


def _wrap_angle_diff_deg(angle_a, angle_b):
    """Smallest absolute angular difference (in degrees) between two 180-deg-periodic axes."""
    diff = (np.asarray(angle_a) - np.asarray(angle_b) + 90.0) % 180.0 - 90.0
    return np.abs(diff)


def load_g2d_test_dataset(npz_path=DATA_PATH):
    """Load the 10-cell closed-loop test dataset and reconstruct `trials_df` and `ref_df`.

    Args:
        npz_path (Path or str, optional): Path to `g2d_closedloop_10cells.npz`.

    Returns:
        tuple: `(trials_df, ref_df, data)` where:
            - `trials_df` (pd.DataFrame): Trial-level dataframe ready to pass to
              `fit_gaussian_blob.fit_rs_of_tuning`.
            - `ref_df` (pd.DataFrame): Reference fit dataframe (10 rows, `roi` 0..9,
              with `original_roi` storing the session ROI index and all reference
              `closedloop_g2d` columns).
            - `data` (dict): Dictionary of raw numpy arrays and parsed `fit_params`.
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}

    data["fit_params"] = json.loads(str(data["fit_params_json"]))
    data["session_name"] = str(data["session_name"])
    data["project"] = str(data["project"])

    rs_thr = data["fit_params"]["rs_thr"]
    min_sigma = data["fit_params"]["min_sigma"]
    rs_flat = data["RS_stim_flat"]
    rs_eye_flat = data["RS_eye_stim_flat"]
    of_flat = data["OF_stim_flat"]
    dff_flat = data["dff_stim_flat"]

    running = (
        (rs_flat > rs_thr)
        & (rs_eye_flat > rs_thr)
        & (~np.isnan(of_flat))
        & (of_flat > 0)
    )
    data["rs_valid"] = np.log(rs_flat[running])
    data["rs_eye_valid"] = np.log(rs_eye_flat[running])
    data["of_valid"] = np.log(np.degrees(of_flat[running]))
    data["dff_valid"] = dff_flat[running, :]

    legacy_model_fn = getattr(
        fit_gaussian_blob, "_gaussian_2d_sigma_theta", fit_gaussian_blob.gaussian_2d
    )
    popt_k1 = data["rsof_popt_closedloop_g2d"]
    data["pred_dff_k1"] = np.column_stack(
        [
            legacy_model_fn(
                (data["rs_valid"], data["of_valid"]),
                *popt_k1[i],
                min_sigma=min_sigma,
            )
            for i in range(len(data["roi_ids"]))
        ]
    )

    splits = np.cumsum(data["trial_lengths"])[:-1]
    rs_stim = np.split(data["RS_stim_flat"], splits)
    rs_eye_stim = np.split(data["RS_eye_stim_flat"], splits)
    of_stim = np.split(data["OF_stim_flat"], splits)
    dff_stim = np.split(data["dff_stim_flat"], splits, axis=0)

    trials_df = pd.DataFrame(
        {
            "trial_no": data["trial_no"],
            "depth": data["depth"],
            "recording_name": data["recording_name"].astype(str),
            "closed_loop": data["closed_loop"],
            "RS_stim": rs_stim,
            "RS_eye_stim": rs_eye_stim,
            "OF_stim": of_stim,
            "dff_stim": dff_stim,
        }
    )

    n_rois = len(data["roi_ids"])
    ref_df = pd.DataFrame(
        {
            "roi": np.arange(n_rois, dtype=int),
            "original_roi": data["roi_ids"],
            "category": data["roi_categories"].astype(str),
            "is_depth_neuron": data["is_depth_neuron"],
            "rsof_popt_closedloop_g2d": list(data["rsof_popt_closedloop_g2d"]),
            "rsof_rsq_closedloop_g2d": data["rsof_rsq_closedloop_g2d"],
            "nemo_rsof_popt_closedloop_g2d": list(
                data["nemo_rsof_popt_closedloop_g2d"]
            ),
            "nemo_rsof_rsq_closedloop_g2d": data["nemo_rsof_rsq_closedloop_g2d"],
            "nemo_rsof_test_rsq_closedloop_g2d": data[
                "nemo_rsof_test_rsq_closedloop_g2d"
            ],
            "preferred_RS_closedloop_g2d": data["preferred_RS_closedloop_g2d"],
            "preferred_OF_closedloop_g2d": data["preferred_OF_closedloop_g2d"],
            "rsof_spearmanr_rval_closedloop_g2d": data[
                "rsof_spearmanr_rval_closedloop_g2d"
            ],
            "rsof_spearmanr_pval_closedloop_g2d": data[
                "rsof_spearmanr_pval_closedloop_g2d"
            ],
            "rsof_popt_closedloop_crossval_g2d": list(
                data["rsof_popt_closedloop_crossval_g2d"]
            ),
            "rsof_rsq_closedloop_crossval_g2d": data[
                "rsof_rsq_closedloop_crossval_g2d"
            ],
            "preferred_RS_closedloop_crossval_g2d": data[
                "preferred_RS_closedloop_crossval_g2d"
            ],
            "preferred_OF_closedloop_crossval_g2d": data[
                "preferred_OF_closedloop_crossval_g2d"
            ],
            "rsof_spearmanr_rval_closedloop_crossval_g2d": data[
                "rsof_spearmanr_rval_closedloop_crossval_g2d"
            ],
            "rsof_spearmanr_pval_closedloop_crossval_g2d": data[
                "rsof_spearmanr_pval_closedloop_crossval_g2d"
            ],
            "rsof_test_rsq_closedloop_g2d": data["rsof_test_rsq_closedloop_g2d"],
            "rsof_test_spearmanr_rval_closedloop_g2d": data[
                "rsof_test_spearmanr_rval_closedloop_g2d"
            ],
            "rsof_test_spearmanr_pval_closedloop_g2d": data[
                "rsof_test_spearmanr_pval_closedloop_g2d"
            ],
            "rsof_train_rsq_closedloop_g2d": [
                list(row) for row in data["rsof_train_rsq_closedloop_g2d"]
            ],
            "rsof_train_popt_closedloop_g2d": [
                [row_fold for row_fold in row]
                for row in data["rsof_train_popt_closedloop_g2d"]
            ],
            "derived_angle_deg": data["derived_angle_deg"],
            "derived_preferred_rs_cm": data["derived_preferred_rs_cm"],
            "derived_preferred_of_deg": data["derived_preferred_of_deg"],
            "derived_semimajor_length": data["derived_semimajor_length"],
            "derived_semiminor_length": data["derived_semiminor_length"],
            "derived_eccentricity_geometric": data["derived_eccentricity_geometric"],
            "derived_eccentricity_linear": data["derived_eccentricity_linear"],
        }
    )
    return trials_df, ref_df, data


def compare_g2d_fit_to_reference(fit_df, ref_df, data, sfx="closedloop_g2d"):
    """Compare a newly computed `g2d` fit DataFrame against the reference dataset.

    Evaluates parameterization-invariant metrics so different `g2d` fitting procedures
    (e.g., sigma-theta vs. Cholesky precision or different initializations) can be
    compared side-by-side.

    Args:
        fit_df (pd.DataFrame): Output of `fit_gaussian_blob.fit_rs_of_tuning`.
        ref_df (pd.DataFrame): Reference DataFrame from `load_g2d_test_dataset()`.
        data (dict): Raw data dictionary from `load_g2d_test_dataset()`.
        sfx (str, optional): Column suffix to compare. Defaults to `"closedloop_g2d"`.

    Returns:
        pd.DataFrame: Per-ROI comparison metrics.
    """
    min_sigma = data["fit_params"]["min_sigma"]
    rs_valid = data["rs_valid"]
    of_valid = data["of_valid"]
    pred_ref = data["pred_dff_k1"]

    rows = []
    for i in range(len(ref_df)):
        popt_new = np.asarray(fit_df.loc[i, f"rsof_popt_{sfx}"], dtype=float)
        rsq_new = float(fit_df.loc[i, f"rsof_rsq_{sfx}"])
        rsq_ref = float(ref_df.loc[i, f"rsof_rsq_{sfx}"])

        pred_new = fit_gaussian_blob.gaussian_2d(
            (rs_valid, of_valid), *popt_new, min_sigma=min_sigma
        )
        pred_corr = float(np.corrcoef(pred_ref[:, i], pred_new)[0, 1])
        pred_max_abs_diff = float(np.max(np.abs(pred_ref[:, i] - pred_new)))

        angle_new = float(_get_angle_deg(popt_new, min_sigma=min_sigma))
        angle_ref = float(ref_df.loc[i, "derived_angle_deg"])
        delta_angle = float(_wrap_angle_diff_deg(angle_new, angle_ref))

        pref_rs_new = float(fit_gaussian_blob.get_preferred_rs(popt_new))
        pref_rs_ref = float(ref_df.loc[i, "derived_preferred_rs_cm"])
        pref_of_new = float(fit_gaussian_blob.get_preferred_of(popt_new))
        pref_of_ref = float(ref_df.loc[i, "derived_preferred_of_deg"])

        semimajor_new = float(
            fit_gaussian_blob.get_semimajor_length(popt_new, min_sigma=min_sigma)
        )
        semimajor_ref = float(ref_df.loc[i, "derived_semimajor_length"])
        semiminor_new = float(
            fit_gaussian_blob.get_semiminor_length(popt_new, min_sigma=min_sigma)
        )
        semiminor_ref = float(ref_df.loc[i, "derived_semiminor_length"])

        ratio_new = semiminor_new / semimajor_new
        ecc_geom_new = float(np.sqrt(max(1.0 - ratio_new**2, 0.0)))
        ecc_geom_ref = float(ref_df.loc[i, "derived_eccentricity_geometric"])

        rows.append(
            {
                "roi": i,
                "original_roi": int(ref_df.loc[i, "original_roi"]),
                "category": ref_df.loc[i, "category"],
                "rsq_ref": rsq_ref,
                "rsq_new": rsq_new,
                "delta_rsq": rsq_new - rsq_ref,
                "pred_corr": pred_corr,
                "pred_max_abs_diff": pred_max_abs_diff,
                "angle_ref_deg": angle_ref,
                "angle_new_deg": angle_new,
                "delta_angle_deg": delta_angle,
                "pref_rs_ref_cm": pref_rs_ref,
                "pref_rs_new_cm": pref_rs_new,
                "log_pref_rs_diff": float(np.abs(np.log(pref_rs_new / pref_rs_ref))),
                "pref_of_ref_deg": pref_of_ref,
                "pref_of_new_deg": pref_of_new,
                "log_pref_of_diff": float(np.abs(np.log(pref_of_new / pref_of_ref))),
                "semimajor_ref": semimajor_ref,
                "semimajor_new": semimajor_new,
                "delta_semimajor": semimajor_new - semimajor_ref,
                "semiminor_ref": semiminor_ref,
                "semiminor_new": semiminor_new,
                "delta_semiminor": semiminor_new - semiminor_ref,
                "ecc_geom_ref": ecc_geom_ref,
                "ecc_geom_new": ecc_geom_new,
                "delta_ecc_geom": ecc_geom_new - ecc_geom_ref,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def g2d_dataset():
    """Module-scoped fixture loading the 10-cell closed-loop test dataset."""
    return load_g2d_test_dataset()


def test_dataset_integrity_and_forward_model(g2d_dataset):
    """Verify the stored 10-cell dataset structure, forward `gaussian_2d` predictions,
    and geometric property getters (including after Cholesky conversion on `dev`).
    """
    trials_df, ref_df, data = g2d_dataset
    assert len(ref_df) == 10
    assert len(trials_df) > 0
    assert (trials_df["closed_loop"] == 1).all()
    assert trials_df["dff_stim"].iloc[0].shape[1] == 10

    min_sigma = data["fit_params"]["min_sigma"]
    rs_valid = data["rs_valid"]
    of_valid = data["of_valid"]
    dff_valid = data["dff_valid"]
    legacy_model_fn = getattr(
        fit_gaussian_blob, "_gaussian_2d_sigma_theta", fit_gaussian_blob.gaussian_2d
    )

    for i in range(len(ref_df)):
        popt_legacy = ref_df.loc[i, "rsof_popt_closedloop_g2d"]
        pred_legacy = legacy_model_fn(
            (rs_valid, of_valid), *popt_legacy, min_sigma=min_sigma
        )
        np.testing.assert_allclose(
            pred_legacy, data["pred_dff_k1"][:, i], rtol=1e-12, atol=1e-12
        )

        popt_branch = _convert_popt_for_current_branch(popt_legacy, min_sigma=min_sigma)
        pred = fit_gaussian_blob.gaussian_2d(
            (rs_valid, of_valid), *popt_branch, min_sigma=min_sigma
        )
        np.testing.assert_allclose(
            pred, data["pred_dff_k1"][:, i], rtol=1e-4, atol=1e-4
        )

        rsq = common_utils.calculate_r_squared(dff_valid[:, i], pred)
        np.testing.assert_allclose(
            rsq, ref_df.loc[i, "rsof_rsq_closedloop_g2d"], rtol=1e-4, atol=1e-5
        )

        np.testing.assert_allclose(
            fit_gaussian_blob.get_preferred_rs(popt_branch),
            ref_df.loc[i, "derived_preferred_rs_cm"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            fit_gaussian_blob.get_preferred_of(popt_branch),
            ref_df.loc[i, "derived_preferred_of_deg"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            fit_gaussian_blob.get_semimajor_length(popt_branch, min_sigma=min_sigma),
            ref_df.loc[i, "derived_semimajor_length"],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            fit_gaussian_blob.get_semiminor_length(popt_branch, min_sigma=min_sigma),
            ref_df.loc[i, "derived_semiminor_length"],
            rtol=1e-4,
            atol=1e-4,
        )
        # Angle is well-defined whenever the Gaussian is not circular (ecc > 0.05)
        if ref_df.loc[i, "derived_eccentricity_geometric"] > 0.05:
            angle_deg = _get_angle_deg(popt_branch, min_sigma=min_sigma)
            assert (
                _wrap_angle_diff_deg(angle_deg, ref_df.loc[i, "derived_angle_deg"])
                < 1e-2
            )


def test_fit_rs_of_tuning_k1_closedloop(g2d_dataset):
    """Run `fit_rs_of_tuning` (k_folds=1, all closed-loop trials) on the 10-cell dataset
    and compare against the stored reference fits.
    """
    trials_df, ref_df, data = g2d_dataset
    fp = data["fit_params"]

    fit_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df,
        model="gaussian_2d",
        choose_trials=None,
        rs_thr=fp["rs_thr"],
        param_range=fp["param_range"],
        niter=fp["niter"],
        min_sigma=fp["min_sigma"],
        k_folds=1,
        random_state=fp["random_state"],
        run_closedloop_only=True,
    )

    comp = compare_g2d_fit_to_reference(fit_df, ref_df, data)

    if _is_legacy_sigma_theta():
        # Exact regression test for the legacy sigma-theta fitting procedure
        np.testing.assert_allclose(
            fit_df["rsof_rsq_closedloop_g2d"].to_numpy(dtype=float),
            ref_df["rsof_rsq_closedloop_g2d"].to_numpy(dtype=float),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            _clamp_floor_log_sigmas(
                np.stack(fit_df["rsof_popt_closedloop_g2d"].values)
            ),
            _clamp_floor_log_sigmas(
                np.stack(ref_df["rsof_popt_closedloop_g2d"].values)
            ),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            fit_df["preferred_RS_closedloop_g2d"].to_numpy(dtype=float),
            ref_df["preferred_RS_closedloop_g2d"].to_numpy(dtype=float),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            fit_df["preferred_OF_closedloop_g2d"].to_numpy(dtype=float),
            ref_df["preferred_OF_closedloop_g2d"].to_numpy(dtype=float),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            fit_df["rsof_spearmanr_rval_closedloop_g2d"].to_numpy(dtype=float),
            ref_df["rsof_spearmanr_rval_closedloop_g2d"].to_numpy(dtype=float),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        # Parameterization-invariant comparison for modified g2d procedures (e.g. dev)
        tuned = comp[comp["rsq_ref"] > 0.04]
        # R^2 on tuned cells should be at least as good (or within a small tolerance)
        assert (tuned["delta_rsq"] > -0.01).all(), (
            f"R^2 degraded on tuned cells:\n"
            f"{tuned[['original_roi', 'category', 'rsq_ref', 'rsq_new', 'delta_rsq']]}"
        )
        # Predicted tuning surfaces on tuned cells should be highly correlated
        assert (tuned["pred_corr"] > 0.98).all(), (
            f"Predicted surface correlation below 0.98:\n"
            f"{tuned[['original_roi', 'category', 'pred_corr']]}"
        )


def test_fit_rs_of_tuning_even_trials_closedloop(g2d_dataset):
    """Run `fit_rs_of_tuning` (`choose_trials="even"`, `k_folds=1`) on the 10 cells."""
    trials_df, ref_df, data = g2d_dataset
    fp = data["fit_params"]

    fit_even = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df,
        model="gaussian_2d",
        choose_trials="even",
        rs_thr=fp["rs_thr"],
        param_range=fp["param_range"],
        niter=fp["niter"],
        min_sigma=fp["min_sigma"],
        k_folds=1,
        random_state=fp["random_state"],
        run_closedloop_only=True,
    )

    if _is_legacy_sigma_theta():
        np.testing.assert_allclose(
            fit_even["rsof_rsq_closedloop_crossval_g2d"].to_numpy(dtype=float),
            ref_df["rsof_rsq_closedloop_crossval_g2d"].to_numpy(dtype=float),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            _clamp_floor_log_sigmas(
                np.stack(fit_even["rsof_popt_closedloop_crossval_g2d"].values)
            ),
            _clamp_floor_log_sigmas(
                np.stack(ref_df["rsof_popt_closedloop_crossval_g2d"].values)
            ),
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        delta_rsq = fit_even["rsof_rsq_closedloop_crossval_g2d"].to_numpy(
            dtype=float
        ) - ref_df["rsof_rsq_closedloop_crossval_g2d"].to_numpy(dtype=float)
        tuned_mask = ref_df["rsof_rsq_closedloop_g2d"].to_numpy(dtype=float) > 0.04
        assert (delta_rsq[tuned_mask] > -0.01).all()


def test_fit_rs_of_tuning_k5_closedloop(g2d_dataset):
    """Run 5-fold cross-validated `fit_rs_of_tuning` (`k_folds=5`) on the 10 cells."""
    trials_df, ref_df, data = g2d_dataset
    fp = data["fit_params"]

    fit_k5 = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df,
        model="gaussian_2d",
        choose_trials=None,
        rs_thr=fp["rs_thr"],
        param_range=fp["param_range"],
        niter=fp["niter"],
        min_sigma=fp["min_sigma"],
        k_folds=5,
        random_state=fp["random_state"],
        run_closedloop_only=True,
    )

    if _is_legacy_sigma_theta():
        np.testing.assert_allclose(
            fit_k5["rsof_test_rsq_closedloop_g2d"].to_numpy(dtype=float),
            ref_df["rsof_test_rsq_closedloop_g2d"].to_numpy(dtype=float),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(fit_k5["rsof_train_rsq_closedloop_g2d"].tolist(), dtype=float),
            np.array(ref_df["rsof_train_rsq_closedloop_g2d"].tolist(), dtype=float),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        delta_test_rsq = fit_k5["rsof_test_rsq_closedloop_g2d"].to_numpy(
            dtype=float
        ) - ref_df["rsof_test_rsq_closedloop_g2d"].to_numpy(dtype=float)
        tuned_mask = ref_df["rsof_rsq_closedloop_g2d"].to_numpy(dtype=float) > 0.04
        assert (delta_test_rsq[tuned_mask] > -0.02).all()


def run_comparison_report():
    """Run the current `g2d` fit on the 10-cell dataset and print a comparison table."""
    import time

    trials_df, ref_df, data = load_g2d_test_dataset()
    fp = data["fit_params"]
    param_type = (
        "legacy (sigma_x2, sigma_y2, theta)"
        if _is_legacy_sigma_theta()
        else "Cholesky (log_l11, l21, log_l22)"
    )
    print(
        f"Loaded 10-cell dataset from {data['session_name']} | "
        f"Active parameterization: {param_type}"
    )

    t0 = time.perf_counter()
    fit_k1 = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df,
        model="gaussian_2d",
        choose_trials=None,
        rs_thr=fp["rs_thr"],
        param_range=fp["param_range"],
        niter=fp["niter"],
        min_sigma=fp["min_sigma"],
        k_folds=1,
        random_state=fp["random_state"],
        run_closedloop_only=True,
    )
    elapsed_k1 = time.perf_counter() - t0

    comp = compare_g2d_fit_to_reference(fit_k1, ref_df, data)
    display_cols = [
        "original_roi",
        "category",
        "rsq_ref",
        "rsq_new",
        "delta_rsq",
        "pred_corr",
        "angle_ref_deg",
        "angle_new_deg",
        "delta_angle_deg",
        "ecc_geom_ref",
        "ecc_geom_new",
    ]
    print(f"\n--- k=1 Closed-Loop Fit Comparison (elapsed: {elapsed_k1:.2f} s) ---")
    print(comp[display_cols].to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    return comp


if __name__ == "__main__":
    run_comparison_report()
