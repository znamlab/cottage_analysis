"""Batched Slurm-array RS/OF fitting gives the same results as a single fit.

Runs the array tasks and the merge in-process (no Slurm) on the 10-cell dataset of
`tests/tests_analysis/test_fit_gaussian_blob_g2d.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cottage_analysis.analysis import fit_gaussian_blob
from cottage_analysis.pipelines import fit_rsof_array

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests_analysis"))
from test_fit_gaussian_blob_g2d import load_g2d_test_dataset  # noqa: E402

FITS = [("gaussian_2d", None, 1), ("gaussian_2d", "even", 1), ("gaussian_OF", None, 5)]


def _assert_same_fit(actual, expected):
    assert set(actual.columns) == set(expected.columns)
    assert len(actual) == len(expected)
    for col in expected.columns:
        for a, e in zip(actual[col], expected[col]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(e), err_msg=col)


@pytest.fixture(scope="module")
def fit_params():
    fp = load_g2d_test_dataset()[2]["fit_params"]
    return dict(
        rs_thr=fp["rs_thr"],
        param_range=fp["param_range"],
        niter=2,  # equivalence does not depend on niter; keep the test fast
        min_sigma=fp["min_sigma"],
        run_closedloop_only=True,
    )


def test_batched_fits_match_single_fit(tmp_path, fit_params):
    trials_df = load_g2d_test_dataset()[0]
    work_dir = tmp_path / "_rsof_fit_tmp"
    fits = [
        dict(model=m, choose_trials=c, k_folds=k, target=tmp_path / f"fit_{i}.pickle")
        for i, (m, c, k) in enumerate(FITS)
    ]
    n_tasks = fit_rsof_array.prepare_work_dir(
        work_dir, trials_df, fits, fit_params, roi_batch_size=4
    )
    assert n_tasks == len(FITS) * 3  # 10 ROIs -> batches of 4, 4, 2
    # tasks only read their own ROIs: dff is not in the shared trials_df
    assert "dff_stim" not in pd.read_pickle(work_dir / "trials_df.pkl").columns
    assert len(list((work_dir / "dff_batches").glob("*.pkl"))) == 3
    for task_id in range(n_tasks):
        fit_rsof_array.run_fit_task(work_dir, task_id)
    fit_rsof_array.merge_results(work_dir, cleanup=True)
    assert not work_dir.exists()

    for fit in fits:
        expected = fit_gaussian_blob.fit_rs_of_tuning(
            trials_df=trials_df,
            model=fit["model"],
            choose_trials=fit["choose_trials"],
            k_folds=fit["k_folds"],
            **fit_params,
        )
        _assert_same_fit(pd.read_pickle(fit["target"]), expected)


def test_merge_refuses_missing_batch(tmp_path, fit_params):
    trials_df = load_g2d_test_dataset()[0]
    work_dir = tmp_path / "_rsof_fit_tmp"
    fits = [
        dict(
            model="gaussian_OF",
            choose_trials=None,
            k_folds=1,
            target=tmp_path / "f.pickle",
        )
    ]
    n_tasks = fit_rsof_array.prepare_work_dir(
        work_dir, trials_df, fits, fit_params, roi_batch_size=5
    )
    fit_rsof_array.run_fit_task(work_dir, 0)  # task 1 never ran
    with pytest.raises(FileNotFoundError):
        fit_rsof_array.merge_results(work_dir)
    assert work_dir.exists() and not (tmp_path / "f.pickle").exists()
    assert n_tasks == 2
