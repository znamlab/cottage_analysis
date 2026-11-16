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
