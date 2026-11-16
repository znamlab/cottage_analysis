import numpy as np
import pandas as pd

from cottage_analysis.analysis.population_depth_decoder import (
    downsample_dff,
    find_stationary_samples,
)


def _trials(rs_pre, rs_stim, n_rois=3):
    trials_df = pd.DataFrame(
        {
            "depth": [0.1] * len(rs_stim),
            "RS_blank_pre": [np.asarray(x, float) for x in rs_pre],
            "RS_stim": [np.asarray(x, float) for x in rs_stim],
            "dff_stim": [np.zeros((len(x), n_rois)) for x in rs_stim],
        }
    )
    return downsample_dff(trials_df, frame_rate=10)


def test_stationary_uses_raw_speed_in_preceding_second():
    # 10 Hz, 0.5 s samples of 5 frames, 1 s window of 10 frames
    rs_stim = np.zeros(30)
    rs_stim[12] = 0.06  # one fast frame in sample 2
    trials_df = _trials([np.zeros(10)], [rs_stim])
    mask = find_stationary_samples(trials_df, frame_rate=10, downsample_window=0.5)
    # samples end on frames 5, 10, 15, 20, 25, 30; the fast frame 12 is in the
    # windows ending at 15 and 20
    np.testing.assert_array_equal(mask, [1, 1, 0, 0, 1, 1])


def test_stationary_window_extends_into_blank_and_needs_full_window():
    rs_stim = np.zeros(20)
    trials_df = _trials([[0.0] * 7 + [0.1] * 3, [0.0] * 2], [rs_stim, rs_stim])
    mask = find_stationary_samples(trials_df, frame_rate=10, downsample_window=0.5)
    # trial 1: running at the end of the blank excludes the first sample only
    # trial 2: 2 blank frames, the first sample has no full 1 s window
    np.testing.assert_array_equal(mask, [0, 1, 1, 1, 0, 1, 1, 1])


def test_stationary_threshold_is_strict_and_matches_sample_count():
    rs_stim = np.full(23, 0.05)  # 23 frames -> 4 samples, last 3 frames dropped
    trials_df = _trials([np.zeros(10)], [rs_stim])
    mask = find_stationary_samples(trials_df, frame_rate=10, downsample_window=0.5)
    assert len(mask) == trials_df["dff_stim_downsample"].iloc[0].shape[0] == 4
    assert not mask.any()


def test_downsample_dff_defaults_to_non_overlapping_window_only():
    dff = np.arange(10, dtype=float)[:, None]
    rs = np.arange(10, dtype=float)
    trials_df = pd.DataFrame({"depth": [0.1], "RS_stim": [rs], "dff_stim": [dff]})

    out_default = downsample_dff(trials_df.copy(), frame_rate=10, downsample_window=0.5)
    assert "dff_stim_rolling" not in out_default.columns
    assert "RS_stim_rolling" not in out_default.columns
    # Direct 5-frame non-overlapping means: [0..4]->2.0, [5..9]->7.0
    np.testing.assert_allclose(
        out_default["dff_stim_downsample"].iloc[0], [[2.0], [7.0]]
    )
    np.testing.assert_allclose(out_default["RS_stim_downsample"].iloc[0], [2.0, 7.0])


def test_downsample_dff_with_rolling_window():
    dff = np.arange(10, dtype=float)[:, None]
    rs = np.arange(10, dtype=float)
    trials_df = pd.DataFrame({"depth": [0.1], "RS_stim": [rs], "dff_stim": [dff]})

    out_roll = downsample_dff(
        trials_df.copy(), rolling_window=0.5, frame_rate=10, downsample_window=0.5
    )
    assert "dff_stim_rolling" in out_roll.columns
    assert "RS_stim_rolling" in out_roll.columns
    # Rolling mean (window=5, min_periods=1) on 0..9 gives [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7]
    # 5-frame non-overlapping means: mean([0, 0.5, 1, 1.5, 2])=1.0, mean([3, 4, 5, 6, 7])=5.0
    np.testing.assert_allclose(out_roll["dff_stim_downsample"].iloc[0], [[1.0], [5.0]])
    np.testing.assert_allclose(out_roll["RS_stim_downsample"].iloc[0], [1.0, 5.0])


def test_downsample_dff_no_downsample_frame_by_frame():
    dff = np.arange(10, dtype=float)[:, None]
    rs = np.zeros(10, dtype=float)
    rs[4] = 0.06
    trials_df = pd.DataFrame(
        {
            "depth": [0.1],
            "RS_blank_pre": [np.zeros(5)],
            "RS_stim": [rs],
            "dff_stim": [dff],
        }
    )

    out = downsample_dff(trials_df.copy(), frame_rate=10, downsample_window=None)
    assert out["dff_stim_downsample"].iloc[0].shape == (10, 1)
    np.testing.assert_allclose(out["dff_stim_downsample"].iloc[0], dff)
    np.testing.assert_allclose(out["RS_stim_downsample"].iloc[0], rs)
    assert len(out["depth_labels"].iloc[0]) == 10

    mask = find_stationary_samples(
        out, frame_rate=10, downsample_window=None, still_thr=0.05, still_time=0.5
    )
    # 5-frame still window: fast frame 4 excludes frames 4, 5, 6, 7, 8
    np.testing.assert_array_equal(mask, [1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
