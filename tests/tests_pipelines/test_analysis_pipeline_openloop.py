"""Offline tests for open-loop 5-fold RS/OF fitting in `analysis_pipeline.main`."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cottage_analysis.pipelines import analysis_pipeline, pipeline_utils


def _run_pipeline_with_loops(
    tmp_path,
    monkeypatch,
    closed_loop_values,
    *,
    run_depth_fit=False,
    run_rsof_fit=True,
):
    """Run `analysis_pipeline.main` with mocked I/O and record fit/merge calls."""
    neurons_path = tmp_path / "neurons_df.pickle"
    ds = SimpleNamespace(
        path_full=neurons_path,
        get_flexilims_entry=lambda: None,
        update_flexilims=lambda **k: None,
    )
    trials_df = pd.DataFrame(
        {
            "depth": [0.1] * len(closed_loop_values),
            "closed_loop": closed_loop_values,
            "dff_stim": [np.zeros((5, 3)) for _ in closed_loop_values],
        }
    )
    s2p_ds = SimpleNamespace(extra_attributes={"fs": 15.0})

    monkeypatch.setattr(
        analysis_pipeline.flz, "get_flexilims_session", lambda *a, **k: "flm"
    )
    monkeypatch.setattr(analysis_pipeline.flz, "update_entity", lambda *a, **k: None)
    monkeypatch.setattr(analysis_pipeline.flz, "get_datasets", lambda *a, **k: [s2p_ds])
    monkeypatch.setattr(pipeline_utils, "create_neurons_ds", lambda **k: ds)
    monkeypatch.setattr(
        analysis_pipeline.spheres,
        "sync_all_recordings",
        lambda **k: (pd.DataFrame(), trials_df),
    )
    if run_depth_fit:
        monkeypatch.setattr(
            analysis_pipeline.find_depth_neurons,
            "find_depth_neurons",
            lambda **k: (pd.DataFrame({"roi": np.arange(3)}), ds),
        )
        monkeypatch.setattr(
            analysis_pipeline.find_depth_neurons,
            "fit_preferred_depth",
            lambda **k: (k["neurons_df"], ds),
        )

    fit_calls = []
    merge_calls = []
    job_counter = {"n": 100}

    def fake_load_and_fit(*args, **kwargs):
        job_counter["n"] += 1
        fit_calls.append((args, kwargs))
        return job_counter["n"]

    def fake_merge(*args, **kwargs):
        job_counter["n"] += 1
        merge_calls.append((args, kwargs))
        return job_counter["n"]

    monkeypatch.setattr(pipeline_utils, "load_and_fit", fake_load_and_fit)
    monkeypatch.setattr(pipeline_utils, "merge_fit_dataframes", fake_merge)
    monkeypatch.setattr(pipeline_utils, "run_basic_plots", lambda *a, **k: None)

    analysis_pipeline.main(
        project="proj",
        session_name="MOUSE_S1",
        conflicts="overwrite",
        photodiode_protocol=5,
        use_slurm=True,
        run_depth_fit=run_depth_fit,
        run_rf=False,
        run_rsof_fit=run_rsof_fit,
        run_plot=False,
    )
    return fit_calls, merge_calls


def test_openloop_session_submits_5fold_openloop_g2d_and_targeted_merge(
    tmp_path, monkeypatch
):
    """Sessions with closed_loop==0 trials fit and merge k5 open-loop g2d."""
    monkeypatch.setenv("HOME", str(tmp_path))
    fit_calls, merge_calls = _run_pipeline_with_loops(tmp_path, monkeypatch, [1, 0])

    assert len(fit_calls) == 12
    _, ol_kwargs = fit_calls[-1]
    assert ol_kwargs["model"] == "gaussian_2d"
    assert ol_kwargs["choose_trials"] is None
    assert ol_kwargs["k_folds"] == 5
    assert ol_kwargs["run_openloop_only"] is True
    assert ol_kwargs["file_special_sfx"] == "_openloop"

    assert len(merge_calls) == 2
    _, m1_kwargs = merge_calls[0]
    _, m2_kwargs = merge_calls[1]
    assert {"openloop", "no_acc"}.issubset(m1_kwargs["exclude_keywords"])
    assert m2_kwargs["prefix"] == "fit_rs_of_tuning_gaussian_2d_k5"
    assert m2_kwargs["suffix"] == "_openloop"
    assert m2_kwargs["exclude_keywords"] == ["recording", "openclosed"]
    assert m2_kwargs["job_dependency"] == [113]


def test_depth_only_merge_also_excludes_no_acc_and_openloop(tmp_path, monkeypatch):
    """Depth-only merge path uses the same no_acc/openloop exclusions before open-loop merge."""
    monkeypatch.setenv("HOME", str(tmp_path))
    fit_calls, merge_calls = _run_pipeline_with_loops(
        tmp_path,
        monkeypatch,
        [1, 0],
        run_depth_fit=True,
        run_rsof_fit=False,
    )

    assert len(fit_calls) == 0
    assert len(merge_calls) == 2
    _, m1_kwargs = merge_calls[0]
    _, m2_kwargs = merge_calls[1]
    assert {"recording", "openclosed", "openloop", "no_acc"}.issubset(
        m1_kwargs["exclude_keywords"]
    )
    assert m2_kwargs["prefix"] == "fit_rs_of_tuning_gaussian_2d_k5"
    assert m2_kwargs["suffix"] == "_openloop"


def test_pure_closedloop_session_skips_openloop_fit_and_second_merge(
    tmp_path, monkeypatch
):
    """Pure closed-loop sessions keep the 11 standard fits and single merge."""
    monkeypatch.setenv("HOME", str(tmp_path))
    fit_calls, merge_calls = _run_pipeline_with_loops(tmp_path, monkeypatch, [1, 1])

    assert len(fit_calls) == 11
    assert all(not kw["run_openloop_only"] for _, kw in fit_calls)
    assert len(merge_calls) == 1
