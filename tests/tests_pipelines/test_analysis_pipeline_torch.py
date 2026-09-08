import types

import numpy as np
import pandas as pd
import pytest
import flexiznam as flz

from cottage_analysis.analysis import spheres, treadmill
from cottage_analysis.pipelines import analysis_pipeline_torch, pipeline_utils


N_ROIS = 4


def _make_trials_df(n_closedloop=2, n_openloop=1, depths=(10.0, 20.0)):
    rows = []
    for i in range(n_closedloop):
        rows.append({"closed_loop": 1, "depth": depths[i % len(depths)]})
    for i in range(n_openloop):
        rows.append({"closed_loop": 0, "depth": depths[i % len(depths)]})
    df = pd.DataFrame(rows)
    df["dff_stim"] = [np.zeros((5, N_ROIS)) for _ in range(len(df))]
    return df


@pytest.fixture
def patched_pipeline(monkeypatch, tmp_path):
    """Replace every external system main() touches with an in-memory fake,
    and return the lists each fake records its calls into.
    """
    calls = {"load_and_fit_torch": [], "update_entity": []}

    monkeypatch.setattr(flz, "get_flexilims_session", lambda project: object())
    fake_dataset = types.SimpleNamespace(extra_attributes={"fs": 30.0})
    monkeypatch.setattr(flz, "get_datasets", lambda **kwargs: [fake_dataset])

    def fake_update_entity(*args, **kwargs):
        calls["update_entity"].append(kwargs)

    monkeypatch.setattr(flz, "update_entity", fake_update_entity)

    fake_neurons_ds = types.SimpleNamespace(path_full=tmp_path / "neurons_df.pkl")
    monkeypatch.setattr(
        pipeline_utils, "create_neurons_ds", lambda **kwargs: fake_neurons_ds
    )

    def fake_load_and_fit_torch(*args, **kwargs):
        calls["load_and_fit_torch"].append(kwargs)
        return None

    monkeypatch.setattr(pipeline_utils, "load_and_fit_torch", fake_load_and_fit_torch)

    return calls


def test_main_uses_spheres_sync_for_default_protocol(patched_pipeline, monkeypatch):
    trials_df = _make_trials_df(n_closedloop=2, n_openloop=1)
    sync_calls = []

    def fake_sync(**kwargs):
        sync_calls.append(kwargs)
        return None, trials_df

    monkeypatch.setattr(spheres, "sync_all_recordings", fake_sync)
    monkeypatch.setattr(
        treadmill,
        "sync_all_recordings",
        lambda **kwargs: pytest.fail("treadmill sync should not run for this protocol"),
    )

    analysis_pipeline_torch.main(
        project="fake_project", session_name="fake_session", use_slurm=False
    )

    assert len(sync_calls) == 1
    update_kwargs = patched_pipeline["update_entity"][0]["attributes"]
    assert update_kwargs == {
        "closedloop_trials": 2,
        "openloop_trials": 1,
        "ndepths": 2,
    }

    fit_calls = patched_pipeline["load_and_fit_torch"]
    assert len(fit_calls) == 13  # len(to_do) in analysis_pipeline_torch.main
    for kwargs in fit_calls:
        assert kwargs["max_rs2motor_diff"] is None
        assert kwargs["file_special_sfx"] == ""
        expect_more_starts = kwargs["k_folds"] > 1 or kwargs["choose_trials"] == "even"
        assert kwargs["n_starts"] == (10 if expect_more_starts else 5)


def test_main_uses_treadmill_sync_and_sets_treadmill_overrides(
    patched_pipeline, monkeypatch
):
    trials_df = _make_trials_df(n_closedloop=3, n_openloop=0)
    sync_calls = []

    def fake_sync(**kwargs):
        sync_calls.append(kwargs)
        return None, trials_df

    monkeypatch.setattr(treadmill, "sync_all_recordings", fake_sync)
    monkeypatch.setattr(
        spheres,
        "sync_all_recordings",
        lambda **kwargs: pytest.fail("spheres sync should not run for this protocol"),
    )

    analysis_pipeline_torch.main(
        project="fake_project",
        session_name="fake_session",
        use_slurm=False,
        protocol_base="SpheresTubeMotor",
    )

    assert len(sync_calls) == 1
    update_kwargs = patched_pipeline["update_entity"][0]["attributes"]
    assert update_kwargs == {"treadmill_trials": 3}

    fit_calls = patched_pipeline["load_and_fit_torch"]
    assert len(fit_calls) == 13
    for kwargs in fit_calls:
        assert kwargs["max_rs2motor_diff"] == 0.3
        assert kwargs["file_special_sfx"] == "_treadmill"
