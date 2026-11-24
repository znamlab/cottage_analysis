import numpy as np
import pandas as pd

from cottage_analysis.analysis import gratings


def _make_img_df(n_volumes=60, n_rois=4, volumes_per_trial=10, seed=0):
    rng = np.random.default_rng(seed)
    dff = rng.normal(size=(n_volumes, n_rois))
    trial = np.arange(n_volumes) // volumes_per_trial
    img_df = pd.DataFrame(
        {
            "imaging_volume": np.arange(n_volumes),
            "SpatialFrequency": 0.02 * (1 + trial % 3),
            "TemporalFrequency": 1.0 + trial % 2,
            "Angle": 30.0 * trial,
        }
    )
    img_df["dffs"] = list(dff[:, None, :])
    img_df["spks"] = list(2 * dff[:, None, :])
    return img_df, dff


def test_stim_volumes_stop_before_next_trial():
    """`dff_stim` holds the volumes [stim_start, stim_end), as the trial mean does."""
    img_df, dff = _make_img_df()
    skip = 2
    trials_df, dff_mean = gratings.generate_trials_df(img_df, skip_first_n_volumes=skip)

    # onsets are found from a change of stimulus, so the first trial is not detected,
    # and the last one has no end
    assert len(trials_df) == 4
    for i, trial in trials_df.iterrows():
        start, end = (i + 1) * 10 + skip, (i + 2) * 10
        assert (trial.stim_start, trial.stim_end) == (start, end)
        np.testing.assert_array_equal(trial.dff_stim, dff[start:end])
        np.testing.assert_allclose(dff_mean[i], dff[start:end].mean(axis=0))
        np.testing.assert_allclose(
            trial[[0, 1, 2, 3]].to_numpy(float), trial.dff_stim.mean(axis=0)
        )


def test_spikes_use_the_same_window():
    img_df, dff = _make_img_df()
    trials_df, _ = gratings.generate_trials_df(
        img_df, skip_first_n_volumes=2, use_col="spks"
    )
    assert "dff_stim" not in trials_df.columns
    np.testing.assert_array_equal(trials_df.spks_stim.iloc[0], 2 * dff[12:20])
