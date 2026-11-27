"""`merge_fit_dataframes` with and without fit files next to neurons_df.

The pipeline calls it even when no RS/OF fit was run (`run_rsof_fit=False`); it must then
leave neurons_df untouched instead of failing on an empty merge. Flexilims is mocked.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cottage_analysis.pipelines import pipeline_utils


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    neurons_df = pd.DataFrame({"roi": np.arange(4), "depth_col": [1.0, 2.0, 3.0, 4.0]})
    neurons_df.to_pickle(tmp_path / "neurons_df.pickle")
    ds = SimpleNamespace(path_full=tmp_path / "neurons_df.pickle")
    monkeypatch.setattr(
        pipeline_utils.flz, "get_flexilims_session", lambda *a, **k: None
    )
    monkeypatch.setattr(pipeline_utils, "create_neurons_ds", lambda **kwargs: ds)
    return tmp_path


def _merge(**kwargs):
    return pipeline_utils.merge_fit_dataframes(
        "project", "session", use_slurm=False, **kwargs
    )


def test_no_fit_files_leaves_neurons_df_unchanged(session_dir):
    before = (session_dir / "neurons_df.pickle").read_bytes()
    out = _merge(
        conflicts="overwrite", suffix="_treadmill", target_column_suffix="_treadmill"
    )
    pd.testing.assert_frame_equal(
        out, pd.read_pickle(session_dir / "neurons_df.pickle")
    )
    assert (session_dir / "neurons_df.pickle").read_bytes() == before


def test_fit_file_is_merged_with_suffix(session_dir):
    fit = pd.DataFrame({"roi": np.arange(4), "rsq": [0.1, 0.2, 0.3, 0.4]})
    fit.to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k1_treadmill.pickle")
    out = _merge(
        conflicts="overwrite", suffix="_treadmill", target_column_suffix="_treadmill"
    )
    np.testing.assert_array_equal(out["rsq_treadmill"], fit["rsq"])
    np.testing.assert_array_equal(out["depth_col"], [1.0, 2.0, 3.0, 4.0])
    saved = pd.read_pickle(session_dir / "neurons_df.pickle")
    assert "rsq_treadmill" in saved.columns


def test_openloop_k5_merge_ignores_legacy_and_recording_files(session_dir):
    """Targeted open-loop k5 merge ignores legacy k5openloop and per-recording files."""
    neurons = pd.DataFrame(
        {
            "roi": np.arange(4),
            "rsof_test_rsq_openloop_actual_g2d": [0.01, 0.02, 0.03, 0.04],
            "rsof_minSigma_openloop_actual_g2d": [0.25, 0.25, 0.25, 0.25],
        }
    )
    neurons.to_pickle(session_dir / "neurons_df.pickle")

    # Legacy file without underscore before 'openloop' and per-recording file
    pd.DataFrame(
        {
            "roi": np.arange(4),
            "rsof_test_rsq_openloop_actual_g2d": [-9.0, -9.0, -9.0, -9.0],
            "rsof_minSigma_openloop_actual_g2d": [0.99, 0.99, 0.99, 0.99],
        }
    ).to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k5openloop.pickle")
    pd.DataFrame(
        {
            "roi": np.arange(4),
            "rsof_test_rsq_openloop_actual_g2d": [-8.0, -8.0, -8.0, -8.0],
            "rsof_minSigma_openloop_actual_g2d": [0.88, 0.88, 0.88, 0.88],
        }
    ).to_pickle(
        session_dir
        / "fit_rs_of_tuning_gaussian_2d_k5_recording_R174125_openloop_1.pickle"
    )

    new_fit = pd.DataFrame(
        {
            "roi": np.arange(4),
            "rsof_test_rsq_openloop_actual_g2d": [0.1, 0.2, 0.3, 0.4],
            "rsof_minSigma_openloop_actual_g2d": [0.25, 0.25, 0.25, 0.25],
        }
    )
    new_fit.to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k5_openloop.pickle")

    out = _merge(
        conflicts="overwrite",
        prefix="fit_rs_of_tuning_gaussian_2d_k5",
        suffix="_openloop",
        exclude_keywords=["recording", "openclosed"],
        include_keywords=[],
        target_column_suffix="",
    )
    np.testing.assert_array_equal(
        out["rsof_test_rsq_openloop_actual_g2d"], [0.1, 0.2, 0.3, 0.4]
    )
    assert not [c for c in out.columns if c.endswith(("_x", "_y"))]


def test_primary_merge_ignores_no_acc_files(session_dir):
    """Primary merge excludes k1no_acc / k5no_acc files so g2d columns do not collide."""
    pd.DataFrame(
        {
            "roi": np.arange(4),
            "preferred_RS_openloop_actual_g2d": [0.1, 0.2, 0.3, 0.4],
            "rsof_rsq_openloop_actual_g2d": [0.5, 0.6, 0.7, 0.8],
        }
    ).to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k1.pickle")
    pd.DataFrame(
        {
            "roi": np.arange(4),
            "rsof_test_rsq_closedloop_g2d": [0.11, 0.22, 0.33, 0.44],
        }
    ).to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k5.pickle")

    # Pre-existing no_acc files from analysis_no_acceleration.py share column names
    pd.DataFrame(
        {
            "roi": np.arange(4),
            "preferred_RS_openloop_actual_g2d": [-9.0, -9.0, -9.0, -9.0],
            "rsof_rsq_openloop_actual_g2d": [-9.0, -9.0, -9.0, -9.0],
        }
    ).to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k1no_acc.pickle")
    pd.DataFrame(
        {
            "roi": np.arange(4),
            "rsof_test_rsq_closedloop_g2d": [-9.0, -9.0, -9.0, -9.0],
        }
    ).to_pickle(session_dir / "fit_rs_of_tuning_gaussian_2d_k5no_acc.pickle")

    out = _merge(
        conflicts="overwrite",
        prefix="fit_rs_of_tuning_",
        suffix="",
        exclude_keywords=["recording", "openclosed", "openloop", "no_acc"],
        include_keywords=[],
        target_column_suffix="",
    )
    np.testing.assert_array_equal(
        out["preferred_RS_openloop_actual_g2d"], [0.1, 0.2, 0.3, 0.4]
    )
    np.testing.assert_array_equal(
        out["rsof_test_rsq_closedloop_g2d"], [0.11, 0.22, 0.33, 0.44]
    )
    assert not [c for c in out.columns if c.endswith(("_x", "_y"))]
