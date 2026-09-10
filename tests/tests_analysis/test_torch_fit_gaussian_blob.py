import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cottage_analysis.analysis import torch_fit_gaussian_blob as tfgb

## Testing mathematical correctness of Gaussian functions
def test_gaussian_1d_peak_value_and_symmetry():
    # amplitude=2, x0=1.0, sigma^2 (+min_sigma)=1.0, offset=0.5
    params = torch.tensor([np.log(2.0), 1.0, np.log(0.75), 0.5])
    x = torch.tensor([0.0, 1.0, 2.0])
    y = tfgb.gaussian_1d(x, params, min_sigma=0.25)

    assert y.shape == (3,)
    # np.log(...) in `params` promotes it to float64, so compare loosely on dtype
    torch.testing.assert_close(y[1], torch.tensor(2.5), check_dtype=False)
    torch.testing.assert_close(y[0], y[2])  # symmetric around x0


def test_gaussian_1d_batches_over_rois():
    x = torch.linspace(-2.0, 2.0, 25)
    params = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    y = tfgb.gaussian_1d(x, params, min_sigma=0.25)
    assert y.shape == (25, 2)


def test_gaussian_2mult_matches_independent_reference_formula():
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    y_coord = torch.tensor([0.2, -0.3, 0.1, 1.0])
    log_amp, x0, y0, log_sx2, log_sy2, offset = 0.4, 0.1, -0.2, 0.3, -0.1, 0.05
    params = torch.tensor([log_amp, x0, y0, log_sx2, log_sy2, offset])

    result = tfgb.gaussian_2mult(x, y_coord, params, bounds=None, min_sigma=0.25)

    x_np, y_np = x.numpy(), y_coord.numpy()
    amp = np.exp(log_amp)
    sx2, sy2 = np.exp(log_sx2) + 0.25, np.exp(log_sy2) + 0.25
    expected = offset + amp * np.exp(-((x_np - x0) ** 2) / (2 * sx2)) * np.exp(
        -((y_np - y0) ** 2) / (2 * sy2)
    )
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_gaussian_additive_matches_sum_of_two_independent_gaussians():
    rs = torch.tensor([0.0, 1.0, 2.0])
    of = torch.tensor([0.0, 0.5, 1.5])
    log_amp_x, log_amp_y, x0, y0, log_sx2, log_sy2, offset = (
        0.2,
        -0.3,
        0.5,
        0.2,
        0.1,
        -0.1,
        0.05,
    )
    params = torch.tensor([log_amp_x, log_amp_y, x0, y0, log_sx2, log_sy2, offset])

    result = tfgb.gaussian_additive((rs, of), params, bounds=None, min_sigma=0.25)

    rs_np, of_np = rs.numpy(), of.numpy()
    amp_x, amp_y = np.exp(log_amp_x), np.exp(log_amp_y)
    sx2, sy2 = np.exp(log_sx2) + 0.25, np.exp(log_sy2) + 0.25
    expected = (
        offset
        + amp_x * np.exp(-((rs_np - x0) ** 2) / (2 * sx2))
        + amp_y * np.exp(-((of_np - y0) ** 2) / (2 * sy2))
    )
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


def test_gaussian_ratio_uses_log_difference_of_rs_and_of():
    # rs, of are already in log-space in the model's input convention, so the
    # "ratio" model should peak wherever rs - of == x0.
    rs = torch.tensor([1.0, 2.0])
    of = torch.tensor([1.0, 0.5])  # rs - of = [0.0, 1.5]
    params = torch.tensor([0.0, 1.5, 0.0, 0.0])  # x0 = 1.5 -> peaks on second sample
    y = tfgb.gaussian_ratio((rs, of), params, bounds=None, min_sigma=0.25)
    assert y[1] > y[0]


def test_calculate_chunk_size_respects_memory_budget_and_alignment(monkeypatch):
    # monkeypatch torch.cuda calls so it can be run on any machine without a real GPU
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 16 * 1024**3))

    chunk = tfgb._calculate_chunk_size(
        n_samples=1000, n_params=7, n_fits=500, max_chunk_size=8192
    )
    assert 0 < chunk <= 8192
    assert chunk % 32 == 0

    # a tighter memory budget should yield a smaller chunk size
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (256 * 1024**2, 256 * 1024**2))
    tight_chunk = tfgb._calculate_chunk_size(
        n_samples=1000, n_params=7, n_fits=500, max_chunk_size=8192
    )
    assert tight_chunk < chunk


## Test data munging
def _make_trial_row(rs, of, rs_eye, responses, depth):
    return pd.Series(
        {
            "RS_stim": np.asarray(rs),
            "OF_stim": np.asarray(of),
            "RS_eye_stim": np.asarray(rs_eye),
            "dff_stim": np.asarray(responses),
            "depth": depth,
        }
    )


def test_process_rs_of_for_fit_filters_and_transforms_running_frames():
    n_rois = 3
    responses = np.arange(4 * n_rois, dtype=float).reshape(4, n_rois)
    # index 0: below rs threshold; index 3: rs == 0 -> both dropped
    trial = _make_trial_row(
        rs=[0.005, 0.02, 0.03, 0.0],
        of=[0.1, 0.2, 0.3, 0.4],
        rs_eye=[0.005, 0.02, 0.03, 0.0],
        responses=responses,
        depth=10.0,
    )
    trials_df = pd.DataFrame([trial])

    rs, of, rs_eye, out_responses, depth = tfgb.process_rs_of_for_fit(
        trials_df, rs_threshold=0.01
    )

    np.testing.assert_allclose(rs, np.log([0.02, 0.03]))
    np.testing.assert_allclose(of, np.log(np.degrees([0.2, 0.3])))
    np.testing.assert_allclose(out_responses, responses[[1, 2], :])
    np.testing.assert_allclose(depth, [10.0, 10.0])


def test_process_rs_of_for_fit_trial_average_produces_one_row_per_trial():
    trial1 = _make_trial_row(
        rs=[0.02, 0.03, 0.04],
        of=[0.1, 0.2, 0.3],
        rs_eye=[0.02, 0.03, 0.04],
        responses=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        depth=5.0,
    )
    trial2 = _make_trial_row(
        rs=[0.05, 0.06],
        of=[0.4, 0.5],
        rs_eye=[0.05, 0.06],
        responses=np.array([[4.0, 40.0], [6.0, 60.0]]),
        depth=8.0,
    )
    trials_df = pd.DataFrame([trial1, trial2])

    rs, of, rs_eye, responses, depth = tfgb.process_rs_of_for_fit(
        trials_df, rs_threshold=0.01, trial_average=True
    )

    assert rs.shape == (2,)
    assert of.shape == (2,)
    assert rs_eye.shape == (2,)
    assert depth.shape == (2,)
    assert responses.shape == (2, 2)  # (n_trials, n_rois) -- stacked, not flattened

    np.testing.assert_allclose(rs, np.log([np.mean([0.02, 0.03, 0.04]), np.mean([0.05, 0.06])]))
    np.testing.assert_allclose(
        of, np.log(np.degrees([np.mean([0.1, 0.2, 0.3]), np.mean([0.4, 0.5])]))
    )
    np.testing.assert_allclose(responses, [[2.0, 20.0], [5.0, 50.0]])
    np.testing.assert_allclose(depth, [5.0, 8.0])


def test_process_rs_of_for_fit_returns_empty_arrays_when_no_frame_is_running():
    trial = _make_trial_row(
        rs=[0.001, 0.002],
        of=[0.1, 0.2],
        rs_eye=[0.001, 0.002],
        responses=np.ones((2, 4)),
        depth=1.0,
    )
    trials_df = pd.DataFrame([trial])

    with pytest.warns(UserWarning, match="No valid frames"):
        rs, of, rs_eye, responses, depth = tfgb.process_rs_of_for_fit(
            trials_df, rs_threshold=0.01
        )

    assert rs.size == 0
    assert responses.shape == (0, 4)  # n_rois preserved even though 0 samples remain


def test_validate_and_filter_fit_arrays_raises_on_shape_mismatch():
    rs = np.array([1.0, 2.0])
    of = np.array([1.0, 2.0, 3.0])  # wrong length
    responses = np.zeros((2, 2))
    depth = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="same number of samples"):
        tfgb._validate_and_filter_fit_arrays(rs, of, responses, depth, trial_average=False)


def test_validate_and_filter_fit_arrays_drops_non_finite_samples_unless_trial_averaged():
    rs = np.array([1.0, np.nan, 3.0])
    of = np.array([1.0, 2.0, np.inf])
    responses = np.zeros((3, 2))
    depth = np.array([0.0, 1.0, 2.0])

    rs_f, of_f, responses_f, depth_f = tfgb._validate_and_filter_fit_arrays(
        rs, of, responses, depth, trial_average=False
    )
    assert rs_f.shape == (1,)  # only sample 0 has finite rs and of

    # trial-averaged samples are trusted as-is and not filtered
    rs_f, of_f, responses_f, depth_f = tfgb._validate_and_filter_fit_arrays(
        rs, of, responses, depth, trial_average=True
    )
    assert rs_f.shape == (3,)


def test_r2_per_roi_matches_hand_computed_values():
    y_true = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y_pred = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])  # col 1 = mean(y_true)

    r2 = tfgb._r2_per_roi(y_true, y_pred)
    torch.testing.assert_close(r2[0], torch.tensor(1.0))
    torch.testing.assert_close(r2[1], torch.tensor(0.0))


def test_per_roi_spearman_recovers_perfect_and_anti_correlation():
    y_true = np.column_stack([[1, 2, 3, 4], [1, 2, 3, 4]])
    y_pred = np.column_stack([[1, 2, 3, 4], [4, 3, 2, 1]])

    rval, pval = tfgb._per_roi_spearman(y_true, y_pred)
    np.testing.assert_allclose(rval, [1.0, -1.0])


def test_calculate_param_range_pads_one_log_unit_beyond_data_range():
    rs = np.array([-2.0, -1.0, 0.0, 1.0])  # already log-RS
    of = np.array([0.5, 1.5, 2.5])  # already log-OF

    param_range = tfgb._calculate_param_range(rs, of)

    assert param_range.keys() == {"rs_min", "rs_max", "of_min", "of_max"}
    np.testing.assert_allclose(param_range["rs_min"], np.exp(-3.0))
    np.testing.assert_allclose(param_range["rs_max"], np.exp(2.0))
    np.testing.assert_allclose(param_range["of_min"], np.exp(-0.5))
    np.testing.assert_allclose(param_range["of_max"], np.exp(3.5))


def test_calculate_param_range_raises_on_empty_arrays():
    with pytest.raises(ValueError, match="zero-size array"):
        tfgb._calculate_param_range(np.array([]), np.array([]))


## Tests for configuring boundaries for fit
@pytest.fixture
def minimal_trials_df():
    return pd.DataFrame([{"dff_stim": np.zeros((2, 1))}])


class _StopEarly(Exception):
    """Sentinel raised by a fake format_model_bounds to short-circuit fit_rs_of_tuning
    right after it resolves param_range, before touching CUDA/device logic at all.
    """


def test_fit_rs_of_tuning_falls_back_to_default_param_range_when_none(
    monkeypatch, minimal_trials_df
):
    captured = {}

    def fake_format_model_bounds(model, **kwargs):
        captured.update(kwargs)
        raise _StopEarly()

    monkeypatch.setattr(tfgb.torch_utils, "format_model_bounds", fake_format_model_bounds)

    with pytest.raises(_StopEarly):
        tfgb.fit_rs_of_tuning(minimal_trials_df, param_range=None)

    assert captured == tfgb.DEFAULT_PARAM_RANGE


def test_fit_rs_of_tuning_passes_explicit_param_range_through_unchanged(
    monkeypatch, minimal_trials_df
):
    explicit_param_range = {
        "rs_min": 0.1,
        "rs_max": 10.0,
        "of_min": 1.0,
        "of_max": 100.0,
        "log_amplitude_max": 5.0,
    }
    captured = {}

    def fake_format_model_bounds(model, **kwargs):
        captured.update(kwargs)
        raise _StopEarly()

    monkeypatch.setattr(tfgb.torch_utils, "format_model_bounds", fake_format_model_bounds)

    with pytest.raises(_StopEarly):
        tfgb.fit_rs_of_tuning(minimal_trials_df, param_range=explicit_param_range)

    assert captured == explicit_param_range


## Tests for end-to-end fitting with GPU
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fit_rs_of_tuning requires a CUDA GPU")
def test_fit_rs_of_tuning_recovers_known_gaussian_rs_params():
    """End-to-end recovery test: simulate noiseless dff from a known gaussian_RS
    curve, run the real TRF fit, and check R^2 is high.

    NOTE: this test was written and reviewed without running it (no GPU
    available in the authoring environment) -- run it once on NEMO in the
    v1_depth_seq conda env and tighten/loosen the r2 threshold if needed.
    """
    n_rois = 3
    n_frames_per_trial = 50
    n_trials = 4
    log_rs = np.linspace(-3.0, 1.5, n_frames_per_trial * n_trials)
    rs_linear = np.exp(log_rs)

    # ground truth in the same parameterisation gaussian_1d expects
    true_params = torch.tensor([np.log(2.0), -0.5, 0.0, 0.1])
    dff_all = (
        tfgb.gaussian_1d(torch.tensor(log_rs), true_params, min_sigma=0.25)
        .numpy()[:, None]
        .repeat(n_rois, axis=1)
    )

    depths = np.repeat([10.0, 20.0], n_trials // 2 * n_frames_per_trial)
    rows = []
    for i in range(n_trials):
        sl = slice(i * n_frames_per_trial, (i + 1) * n_frames_per_trial)
        rows.append(
            {
                "RS_stim": rs_linear[sl],
                "RS_eye_stim": rs_linear[sl],
                "OF_stim": np.full(n_frames_per_trial, 1.0),
                "dff_stim": dff_all[sl, :],
                "depth": depths[sl][0],
                "closed_loop": 1,
            }
        )
    trials_df = pd.DataFrame(rows)

    result = tfgb.fit_rs_of_tuning(
        trials_df,
        model="gaussian_RS",
        use_col="dff_stim",
        n_starts=2,
        k_folds=1,
        run_closedloop_only=True,
        min_sigma=0.25,
    )

    r2 = result["rsof_rsq_closedloop_grs"].to_numpy()
    assert (r2 > 0.8).all()
