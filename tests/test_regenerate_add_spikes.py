from pathlib import Path
import numpy as np
import pandas as pd

from cottage_analysis.analysis.spheres import spheres


def test_regenerate_passes_add_spikes(monkeypatch):
    seen = []
    recs = pd.DataFrame({"name": ["S_R1_SpheresPermTubeReward_multidepth"]})
    monkeypatch.setattr(spheres.flz, "get_entity", lambda **kw: {"id": "s"})
    monkeypatch.setattr(spheres.flz, "get_entities", lambda **kw: recs)

    def fake(*args, add_spikes=False, **kwargs):
        seen.append(add_spikes)
        imaging_df = pd.DataFrame({"x": [0]})
        vs_df = pd.DataFrame({"OriginalSize": [0.087]})
        return vs_df, imaging_df, None, None, None, None

    monkeypatch.setattr(spheres, "_process_single_recording_for_session", fake)
    for flag in (False, True):
        spheres.regenerate_frames_all_recordings(
            "S",
            flexilims_session=object(),
            is_multidepth=True,
            do_regenerate_frames=False,
            add_spikes=flag,
        )
    assert seen == [False, True]


def _make_mock_imaging_df(n_frames=60, n_rois=3):
    """Build a minimal 2-trial synthetic imaging_df for testing generate_trials_df."""
    depth = np.full(n_frames, np.nan)
    # Trial 0: stim frames 5..14 (depth 0.2), blank frames 15..29
    depth[5:15] = 0.2
    depth[15:30] = -10.0
    # Trial 1: stim frames 30..39 (depth 0.5), blank frames 40..59
    depth[30:40] = 0.5
    depth[40:60] = -10.0

    # Unique values per frame and ROI so frame shifts can be checked exactly
    base = np.arange(n_frames, dtype=float)[:, None] * 100 + np.arange(n_rois)[None, :]
    dffs = np.split(base, n_frames, axis=0)
    spks = np.split(base + 0.5, n_frames, axis=0)

    return pd.DataFrame(
        {
            "imaging_volume": np.arange(n_frames, dtype=int),
            "imaging_harptime": np.arange(n_frames, dtype=float) / 15.0,
            "depth": depth,
            "RS": np.arange(n_frames, dtype=float) * 0.01,
            "RS_eye": np.arange(n_frames, dtype=float) * 0.01,
            "OF": np.arange(n_frames, dtype=float) * 0.1,
            "mouse_z_harp": np.arange(n_frames, dtype=float) * 0.001,
            "dffs": dffs,
            "spks": spks,
        }
    )


def test_generate_trials_df_lag(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(spheres.flz, "get_processed_path", lambda p: Path("/tmp"))
    rec = SimpleNamespace(
        name="S_R1_SpheresPermTubeReward",
        path="dummy",
        genealogy=["M", "S", "R1"],
    )

    df0 = spheres.generate_trials_df(
        rec, _make_mock_imaging_df(), add_spikes=True, lag=0
    )
    df4 = spheres.generate_trials_df(
        rec, _make_mock_imaging_df(), add_spikes=True, lag=4
    )
    df10 = spheres.generate_trials_df(
        rec, _make_mock_imaging_df(), add_spikes=True, lag=10
    )

    assert len(df0) == len(df4) == len(df10) == 2

    # Trial 0 stim is frames 5..14: well inside recording bounds, so RS_stim is identical
    # and dff_stim / spks_stim are shifted by +4 (or +10) frames (= +400 or +1000 in value)
    for lag, df_lag in [(4, df4), (10, df10)]:
        for t in range(2):
            np.testing.assert_array_equal(df_lag.RS_stim.iloc[t], df0.RS_stim.iloc[t])
            np.testing.assert_array_equal(df_lag.OF_stim.iloc[t], df0.OF_stim.iloc[t])
            np.testing.assert_allclose(
                df_lag.dff_stim.iloc[t], df0.dff_stim.iloc[t] + lag * 100
            )
            np.testing.assert_allclose(
                df_lag.spks_stim.iloc[t], df0.spks_stim.iloc[t] + lag * 100
            )
            # Every epoch has matching stimulus and neural response lengths
            for epoch in ("stim", "blank", "blank_pre"):
                assert (
                    len(df_lag[f"RS_{epoch}"].iloc[t])
                    == df_lag[f"dff_{epoch}"].iloc[t].shape[0]
                    == df_lag[f"spks_{epoch}"].iloc[t].shape[0]
                )

    # Last trial's blank stops at frame 59 at lag=0; at lag=10 it is clamped to 49 so
    # dff_blank reads frames 50..59 without going out of bounds
    assert df0.imaging_blank_stop.iloc[1] == 59
    assert df10.imaging_blank_stop.iloc[1] == 49
    assert df10.dff_blank.iloc[1].shape == (10, 3)
    np.testing.assert_allclose(df10.dff_blank.iloc[1][-1, 0], 59 * 100)
