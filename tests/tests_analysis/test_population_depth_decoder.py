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
