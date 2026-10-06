"""Extract a standalone 10-cell closed-loop test dataset for `g2d` fitting.

This script loads one session (`PZAG16.3b_S20250224`), selects 10 stratified ROIs
covering diverse 2D Gaussian tuning regimes, verifies that `fit_rs_of_tuning`
reproduces the stored `neurons_df.pickle` reference fits, and saves a self-contained
compressed `.npz` archive to `tests/tests_analysis/test_data/g2d_closedloop_10cells.npz`.

Once generated, the `.npz` file has zero external dependencies (no Flexilims, no
external drives, and no `v1_depth_map` package required).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from cottage_analysis.analysis import fit_gaussian_blob
from cottage_analysis.pipelines import pipeline_utils

PROJECT = "colasa_3d-vision_revisions"
SESSION_NAME = "PZAG16.3b_S20250224"
PHOTODIODE_PROTOCOL = 5
FILTER_DATASETS = {"anatomical_only": 3, "ast_neuropil": False, "annotated": True}

FIT_PARAMS = {
    "rs_thr": 0.01,
    "param_range": {
        "rs_min": 0.005,
        "rs_max": 5.0,
        "of_min": 0.03,
        "of_max": 3000.0,
    },
    "niter": 10,
    "min_sigma": 0.25,
    "random_state": 42,
}

# 10 stratified ROIs from PZAG16.3b_S20250224:
# - 365: high-R2 diagonal ridge, hits rs_max bound
# - 367: high-R2 diagonal ridge, interior preferred RS/OF
# - 122: high-R2 diagonal ridge, interior preferred RS/OF
# - 113: compact elliptical blob, interior peak
# - 315: compact elliptical blob, negative angle, both axes > min_sigma floor
# - 54:  circular blob at min_sigma floor (ecc ~ 0)
# - 173: axis-aligned horizontal ridge (angle ~ 3 deg)
# - 30:  axis-aligned vertical ridge (angle ~ 83 deg)
# - 302: high-R2 diagonal ridge, hits of_max bound
# - 312: untuned / near-zero R2 cell
SELECTED_ROIS = [
    (365, "ridge_rs_bound"),
    (367, "ridge_interior"),
    (122, "ridge_interior"),
    (113, "compact_elliptical"),
    (315, "compact_elliptical_above_floor"),
    (54, "compact_circular_floor"),
    (173, "axis_aligned_rs"),
    (30, "axis_aligned_of"),
    (302, "ridge_of_bound"),
    (312, "untuned_low_rsq"),
]

OUTPUT_PATH = (
    Path(__file__).resolve().parent / "test_data" / "g2d_closedloop_10cells.npz"
)


def _extract_valid_running_frames(trials_df_cl, rs_thr=0.01):
    """Replicate the closed-loop frame masking and log conversion of `fit_rs_of_tuning`."""
    rs_list, rs_eye_list, of_list, dff_list = [], [], [], []
    for _, trial in trials_df_cl.iterrows():
        trial_rs = trial["RS_stim"]
        trial_rs_eye = trial["RS_eye_stim"]
        trial_of = trial["OF_stim"]
        trial_dff = trial["dff_stim"]
        running = (
            (trial_rs > rs_thr)
            & (trial_rs_eye > rs_thr)
            & (~np.isnan(trial_of))
            & (trial_of > 0)
        )
        if np.sum(running) == 0:
            continue
        rs_list.append(trial_rs[running])
        rs_eye_list.append(trial_rs_eye[running])
        of_list.append(trial_of[running])
        dff_list.append(trial_dff[running, :])

    rs = np.log(np.concatenate(rs_list))
    rs_eye = np.log(np.concatenate(rs_eye_list))
    of = np.log(np.degrees(np.concatenate(of_list)))
    dff = np.concatenate(dff_list, axis=0)
    return rs, rs_eye, of, dff


def main():
    roi_ids = np.array([r for r, _ in SELECTED_ROIS], dtype=np.int64)
    roi_categories = np.array([cat for _, cat in SELECTED_ROIS], dtype="U32")

    print(f"Loading session {SESSION_NAME} from project {PROJECT}...")
    _, neurons_df, _, trials_df_all = pipeline_utils.load_session(
        project=PROJECT,
        session_name=SESSION_NAME,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        filter_datasets=FILTER_DATASETS,
    )

    # Keep only non-multidepth closed-loop trials
    is_multidepth = trials_df_all["recording_name"].str.contains("multidepth")
    trials_df_cl = trials_df_all[
        (~is_multidepth) & (trials_df_all["closed_loop"] == 1)
    ].copy()
    trials_df_cl = trials_df_cl.reset_index(drop=True)

    # Slice dff_stim to the 10 selected ROIs
    trials_df_sub = pd.DataFrame(
        {
            "trial_no": trials_df_cl["trial_no"].astype(np.int64).values,
            "depth": trials_df_cl["depth"].astype(np.float64).values,
            "recording_name": trials_df_cl["recording_name"].astype(str).values,
            "closed_loop": trials_df_cl["closed_loop"].astype(np.int64).values,
            "RS_stim": [
                np.asarray(x, dtype=np.float64) for x in trials_df_cl["RS_stim"]
            ],
            "RS_eye_stim": [
                np.asarray(x, dtype=np.float64) for x in trials_df_cl["RS_eye_stim"]
            ],
            "OF_stim": [
                np.asarray(x, dtype=np.float64) for x in trials_df_cl["OF_stim"]
            ],
            "dff_stim": [
                np.asarray(x[:, roi_ids], dtype=np.float64)
                for x in trials_df_cl["dff_stim"]
            ],
        }
    )

    # Run the three closed-loop g2d fit variants from analysis_pipeline.py
    common_kwargs = dict(
        trials_df=trials_df_sub,
        model="gaussian_2d",
        rs_thr=FIT_PARAMS["rs_thr"],
        param_range=FIT_PARAMS["param_range"],
        niter=FIT_PARAMS["niter"],
        min_sigma=FIT_PARAMS["min_sigma"],
        random_state=FIT_PARAMS["random_state"],
        run_closedloop_only=True,
    )

    print("Running k=1 full closed-loop fit on 10 ROIs...")
    fit_k1 = fit_gaussian_blob.fit_rs_of_tuning(
        choose_trials=None, k_folds=1, **common_kwargs
    )

    print("Running k=1 even-trials fit on 10 ROIs...")
    fit_even = fit_gaussian_blob.fit_rs_of_tuning(
        choose_trials="even", k_folds=1, **common_kwargs
    )

    print("Running k=5 cross-validation fit on 10 ROIs...")
    fit_k5 = fit_gaussian_blob.fit_rs_of_tuning(
        choose_trials=None, k_folds=5, **common_kwargs
    )

    # Verify R2 match against saved NEMO neurons_df.pickle (allowing minor BLAS/arch diff)
    ref_sub = neurons_df.iloc[roi_ids].reset_index(drop=True)
    for col in [
        "rsof_rsq_closedloop_g2d",
        "rsof_spearmanr_rval_closedloop_g2d",
    ]:
        np.testing.assert_allclose(
            fit_k1[col].to_numpy(dtype=float),
            ref_sub[col].to_numpy(dtype=float),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Mismatch on {col}",
        )
    np.testing.assert_allclose(
        fit_even["rsof_rsq_closedloop_crossval_g2d"].to_numpy(dtype=float),
        ref_sub["rsof_rsq_closedloop_crossval_g2d"].to_numpy(dtype=float),
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        fit_k5["rsof_test_rsq_closedloop_g2d"].to_numpy(dtype=float),
        ref_sub["rsof_test_rsq_closedloop_g2d"].to_numpy(dtype=float),
        rtol=1e-3,
        atol=1e-4,
    )
    print("Verification passed: recomputed R2 matches saved NEMO neurons_df.pickle.")

    # Compute valid running frames & model predictions
    rs_valid, rs_eye_valid, of_valid, dff_valid = _extract_valid_running_frames(
        trials_df_sub, rs_thr=FIT_PARAMS["rs_thr"]
    )
    popt_k1 = np.stack(fit_k1["rsof_popt_closedloop_g2d"].values).astype(np.float64)
    popt_even = np.stack(fit_even["rsof_popt_closedloop_crossval_g2d"].values).astype(
        np.float64
    )
    train_popt_k5 = np.array(
        fit_k5["rsof_train_popt_closedloop_g2d"].tolist(), dtype=np.float64
    )
    train_rsq_k5 = np.array(
        fit_k5["rsof_train_rsq_closedloop_g2d"].tolist(), dtype=np.float64
    )
    train_rval_k5 = np.array(
        fit_k5["rsof_train_spearmanr_rval_closedloop_g2d"].tolist(), dtype=np.float64
    )
    train_pval_k5 = np.array(
        fit_k5["rsof_train_spearmanr_pval_closedloop_g2d"].tolist(), dtype=np.float64
    )

    pred_dff_k1 = np.column_stack(
        [
            fit_gaussian_blob.gaussian_2d(
                (rs_valid, of_valid), *popt_k1[i], min_sigma=FIT_PARAMS["min_sigma"]
            )
            for i in range(len(roi_ids))
        ]
    )

    # Derived parameterization-invariant geometric properties
    min_sigma = FIT_PARAMS["min_sigma"]
    derived_angle_deg = np.array(
        [fit_gaussian_blob.get_gaussian_angle(p) for p in popt_k1], dtype=np.float64
    )
    derived_preferred_rs_cm = np.array(
        [fit_gaussian_blob.get_preferred_rs(p) for p in popt_k1], dtype=np.float64
    )
    derived_preferred_of_deg = np.array(
        [fit_gaussian_blob.get_preferred_of(p) for p in popt_k1], dtype=np.float64
    )
    derived_semimajor = np.array(
        [
            fit_gaussian_blob.get_semimajor_length(p, min_sigma=min_sigma)
            for p in popt_k1
        ],
        dtype=np.float64,
    )
    derived_semiminor = np.array(
        [
            fit_gaussian_blob.get_semiminor_length(p, min_sigma=min_sigma)
            for p in popt_k1
        ],
        dtype=np.float64,
    )
    ratio = derived_semiminor / derived_semimajor
    derived_eccentricity_geometric = np.sqrt(np.maximum(1.0 - ratio**2, 0.0))
    derived_eccentricity_linear = 1.0 - ratio

    # Flatten ragged trial arrays for version-agnostic .npz storage
    trial_lengths = np.array([len(x) for x in trials_df_sub["RS_stim"]], dtype=np.int64)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        # Metadata
        session_name=np.array(SESSION_NAME),
        project=np.array(PROJECT),
        roi_ids=roi_ids,
        roi_categories=roi_categories,
        is_depth_neuron=ref_sub["is_depth_neuron"].to_numpy(dtype=bool),
        fit_params_json=np.array(json.dumps(FIT_PARAMS)),
        # Trial structure (for reconstructing trials_df)
        trial_lengths=trial_lengths,
        trial_no=trials_df_sub["trial_no"].to_numpy(dtype=np.int64),
        depth=trials_df_sub["depth"].to_numpy(dtype=np.float64),
        recording_name=trials_df_sub["recording_name"].to_numpy(dtype="U64"),
        closed_loop=trials_df_sub["closed_loop"].to_numpy(dtype=np.int64),
        RS_stim_flat=np.concatenate(trials_df_sub["RS_stim"].values).astype(np.float64),
        RS_eye_stim_flat=np.concatenate(trials_df_sub["RS_eye_stim"].values).astype(
            np.float64
        ),
        OF_stim_flat=np.concatenate(trials_df_sub["OF_stim"].values).astype(np.float64),
        dff_stim_flat=np.concatenate(trials_df_sub["dff_stim"].values, axis=0).astype(
            np.float64
        ),
        # Reference fit results: k=1 full closedloop
        rsof_popt_closedloop_g2d=popt_k1,
        rsof_rsq_closedloop_g2d=fit_k1["rsof_rsq_closedloop_g2d"].to_numpy(
            dtype=np.float64
        ),
        nemo_rsof_popt_closedloop_g2d=np.stack(
            ref_sub["rsof_popt_closedloop_g2d"].values
        ).astype(np.float64),
        nemo_rsof_rsq_closedloop_g2d=ref_sub["rsof_rsq_closedloop_g2d"].to_numpy(
            dtype=np.float64
        ),
        nemo_rsof_test_rsq_closedloop_g2d=ref_sub[
            "rsof_test_rsq_closedloop_g2d"
        ].to_numpy(dtype=np.float64),
        preferred_RS_closedloop_g2d=fit_k1["preferred_RS_closedloop_g2d"].to_numpy(
            dtype=np.float64
        ),
        preferred_OF_closedloop_g2d=fit_k1["preferred_OF_closedloop_g2d"].to_numpy(
            dtype=np.float64
        ),
        rsof_spearmanr_rval_closedloop_g2d=fit_k1[
            "rsof_spearmanr_rval_closedloop_g2d"
        ].to_numpy(dtype=np.float64),
        rsof_spearmanr_pval_closedloop_g2d=fit_k1[
            "rsof_spearmanr_pval_closedloop_g2d"
        ].to_numpy(dtype=np.float64),
        # Reference fit results: k=1 even trials (crossval)
        rsof_popt_closedloop_crossval_g2d=popt_even,
        rsof_rsq_closedloop_crossval_g2d=fit_even[
            "rsof_rsq_closedloop_crossval_g2d"
        ].to_numpy(dtype=np.float64),
        preferred_RS_closedloop_crossval_g2d=fit_even[
            "preferred_RS_closedloop_crossval_g2d"
        ].to_numpy(dtype=np.float64),
        preferred_OF_closedloop_crossval_g2d=fit_even[
            "preferred_OF_closedloop_crossval_g2d"
        ].to_numpy(dtype=np.float64),
        rsof_spearmanr_rval_closedloop_crossval_g2d=fit_even[
            "rsof_spearmanr_rval_closedloop_crossval_g2d"
        ].to_numpy(dtype=np.float64),
        rsof_spearmanr_pval_closedloop_crossval_g2d=fit_even[
            "rsof_spearmanr_pval_closedloop_crossval_g2d"
        ].to_numpy(dtype=np.float64),
        # Reference fit results: k=5 cross-validation
        rsof_test_rsq_closedloop_g2d=fit_k5["rsof_test_rsq_closedloop_g2d"].to_numpy(
            dtype=np.float64
        ),
        rsof_test_spearmanr_rval_closedloop_g2d=fit_k5[
            "rsof_test_spearmanr_rval_closedloop_g2d"
        ].to_numpy(dtype=np.float64),
        rsof_test_spearmanr_pval_closedloop_g2d=fit_k5[
            "rsof_test_spearmanr_pval_closedloop_g2d"
        ].to_numpy(dtype=np.float64),
        rsof_train_rsq_closedloop_g2d=train_rsq_k5,
        rsof_train_popt_closedloop_g2d=train_popt_k5,
        rsof_train_spearmanr_rval_closedloop_g2d=train_rval_k5,
        rsof_train_spearmanr_pval_closedloop_g2d=train_pval_k5,
        # Parameterization-invariant derived quantities
        derived_angle_deg=derived_angle_deg,
        derived_preferred_rs_cm=derived_preferred_rs_cm,
        derived_preferred_of_deg=derived_preferred_of_deg,
        derived_semimajor_length=derived_semimajor,
        derived_semiminor_length=derived_semiminor,
        derived_eccentricity_geometric=derived_eccentricity_geometric,
        derived_eccentricity_linear=derived_eccentricity_linear,
    )
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Saved test dataset to {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
