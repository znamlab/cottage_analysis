import numpy as np
import pandas as pd
import pytest

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


def _reference_tune_and_test(X_train, y_train, X_val, y_val, X_test, Cs):
    """Sequential implementation (linear kernel), kept as the reference."""
    from sklearn.metrics import accuracy_score
    from sklearn.svm import LinearSVC

    best_acc = 0
    best_params = {"C": Cs[0], "gamma": 1}
    for C in Cs:
        clf = LinearSVC(C=C, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        if acc > best_acc:
            best_acc = acc
            best_params = {"C": C}
    clf = LinearSVC(C=best_params["C"], random_state=42).fit(
        np.vstack([X_train, X_val]), np.hstack([y_train, y_val])
    )
    return best_params, clf.predict(X_test), clf.decision_function(X_test)


def _decoder_like_data(seed, n_rois=60, n_classes=5, n=(400, 120, 150), signal=0.3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, signal, (n_classes, n_rois))

    def draw(k):
        y = rng.integers(0, n_classes, k)
        return centers[y] + rng.normal(0, 1, (k, n_rois)), y

    return [*draw(n[0]), *draw(n[1]), draw(n[2])[0]]


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tune_and_test_svm_matches_sequential_reference(n_jobs, seed):
    from cottage_analysis.analysis.population_depth_decoder import (
        tune_and_test_svm_classifier,
    )

    X_train, y_train, X_val, y_val, X_test = _decoder_like_data(seed)
    Cs = np.logspace(-5, 3, 9)
    ref_params, ref_pred, ref_dec = _reference_tune_and_test(
        X_train, y_train, X_val, y_val, X_test, Cs
    )
    params, clf, y_pred = tune_and_test_svm_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        Cs=Cs,
        n_jobs=n_jobs,
    )
    assert params == ref_params
    np.testing.assert_array_equal(y_pred, ref_pred)
    np.testing.assert_array_equal(clf.decision_function(X_test), ref_dec)


def test_tune_and_test_svm_ties_and_largest_C():
    """Ties go to the smallest C, and the largest C (no speculative refit) works."""
    from cottage_analysis.analysis.population_depth_decoder import (
        tune_and_test_svm_classifier,
    )

    # separable data: every C reaches accuracy 1, so the first C must win
    X_train, y_train, X_val, y_val, X_test = _decoder_like_data(3, signal=5.0)
    Cs = [0.1, 1, 10]
    for n_jobs in (1, 3):
        params, _, y_pred = tune_and_test_svm_classifier(
            X_train, y_train, X_val, y_val, X_test, Cs=Cs, n_jobs=n_jobs
        )
        assert (
            params
            == _reference_tune_and_test(X_train, y_train, X_val, y_val, X_test, Cs)[0]
            == {"C": 0.1}
        )
    # single-candidate grid: the only C is the largest, so it is refitted afterwards
    params, _, y_pred = tune_and_test_svm_classifier(
        X_train, y_train, X_val, y_val, X_test, Cs=[10.0], n_jobs=2
    )
    ref = _reference_tune_and_test(X_train, y_train, X_val, y_val, X_test, [10.0])
    assert params == ref[0]
    np.testing.assert_array_equal(y_pred, ref[1])


def test_tune_and_test_svm_memmapped_inputs():
    """Arrays above joblib's 1 MB threshold reach the workers as read-only memmaps."""
    from cottage_analysis.analysis.population_depth_decoder import (
        tune_and_test_svm_classifier,
    )

    X_train, y_train, X_val, y_val, X_test = _decoder_like_data(
        4, n_rois=100, n=(2000, 400, 400)
    )
    assert X_train.nbytes > 1e6
    Cs = [0.001, 0.01, 0.1]
    ref = _reference_tune_and_test(X_train, y_train, X_val, y_val, X_test, Cs)
    params, clf, y_pred = tune_and_test_svm_classifier(
        X_train, y_train, X_val, y_val, X_test, Cs=Cs, n_jobs=3
    )
    assert params == ref[0]
    np.testing.assert_array_equal(y_pred, ref[1])
    np.testing.assert_array_equal(clf.decision_function(X_test), ref[2])


def test_linear_kernel_uses_linear_svc_and_other_kernels_use_svc():
    from sklearn.svm import SVC, LinearSVC

    from cottage_analysis.analysis.population_depth_decoder import (
        tune_and_test_svm_classifier,
    )

    X_train, y_train, X_val, y_val, X_test = _decoder_like_data(5)
    _, clf, _ = tune_and_test_svm_classifier(
        X_train, y_train, X_val, y_val, X_test, kernel="linear", Cs=[0.1]
    )
    assert isinstance(clf, LinearSVC)
    assert clf.loss == "squared_hinge" and clf.multi_class == "ovr"
    params, clf, _ = tune_and_test_svm_classifier(
        X_train, y_train, X_val, y_val, X_test, kernel="rbf", Cs=[1], gammas=[0.01]
    )
    assert isinstance(clf, SVC) and clf.kernel == "rbf"
    assert params == {"C": 1, "gamma": 0.01}


def test_calculate_acc_conmat_creates_and_preserves_decoder_dict(tmp_path):
    """calculate_acc_conmat initializes decoder_dict when absent and preserves
    earlier recording_type keys when present."""
    import pickle

    from cottage_analysis.analysis.population_depth_decoder import calculate_acc_conmat

    decoder_dict_path = tmp_path / "decoder_results.pickle"
    assert not decoder_dict_path.exists()

    def _run(recording_type, y_test, y_pred_folds):
        k_folds = len(y_pred_folds)
        test_idx = np.array_split(np.arange(len(y_test)), k_folds)
        for i, y_pred in enumerate(y_pred_folds):
            with open(
                tmp_path / f"decoder_outputs_{recording_type}_fold{i}.pickle", "wb"
            ) as f:
                pickle.dump({"best_params": {"C": 0.1 * (i + 1)}, "y_pred": y_pred}, f)
        decoder_inputs = {
            "trials_df": pd.DataFrame({"depth": [0.1, 0.2]}),
            "kernel": "linear",
            "y_test_all": y_test,
            "test_frame_indices": test_idx,
        }
        return calculate_acc_conmat(
            decoder_dict_path=decoder_dict_path,
            recording_type=recording_type,
            depth_list=[0.1, 0.2],
            decoder_inputs=decoder_inputs,
            k_folds=k_folds,
            use_slurm=False,
        )

    y_ol = np.array([0, 1, 0, 1])
    _run("openloop", y_ol, [np.array([0, 1]), np.array([0, 0])])
    assert decoder_dict_path.exists()

    y_cl = np.array([1, 0, 1, 0])
    out = _run("closedloop", y_cl, [np.array([1, 0]), np.array([1, 0])])

    with open(decoder_dict_path, "rb") as f:
        saved = pickle.load(f)
    assert saved["accuracy_openloop"] == pytest.approx(0.75)
    assert saved["accuracy_closedloop"] == pytest.approx(1.0)
    assert out["best_params_all_closedloop"] == {"C": [0.1, 0.2]}
    np.testing.assert_array_equal(saved["y_test_all_openloop"], y_ol)
    np.testing.assert_array_equal(saved["y_test_all_closedloop"], y_cl)


def test_depth_decoder_driver_does_not_write_decoder_dict(tmp_path, monkeypatch):
    """depth_decoder must not write decoder_results.pickle from the driver process,
    leaving all writes to the dependency-serialized calculate_acc_conmat job."""
    from types import SimpleNamespace

    from cottage_analysis.analysis import population_depth_decoder as pdd

    decoder_dict_path = tmp_path / "decoder_results.pickle"
    neurons_ds = SimpleNamespace(path_full=tmp_path / "neurons_df.pickle")
    trials_df = pd.DataFrame(
        {
            "closed_loop": [1, 1],
            "depth": [0.1, 0.2],
            "depth_col": [np.array([0, 0]), np.array([1, 1])],
        }
    )

    monkeypatch.setattr(
        pdd,
        "preprocess_data",
        lambda **kw: (
            trials_df,
            ([], [], [], [], [], []),
            (np.array([True]), np.array([0.1, 0.2])),
            ([np.array([0, 1]), np.array([2, 3])], "dff", "depth_col"),
        ),
    )

    def _fake_fold(*args, **kwargs):
        assert not decoder_dict_path.exists()
        return "fold_job"

    def _fake_acc(*args, **kwargs):
        assert not decoder_dict_path.exists()
        assert kwargs["decoder_dict_path"] == decoder_dict_path
        return "acc_job"

    monkeypatch.setattr(pdd, "fit_each_fold", _fake_fold)
    monkeypatch.setattr(pdd, "calculate_acc_conmat", _fake_acc)

    res = pdd.depth_decoder(
        trials_df=trials_df,
        flexilims_session=None,
        session_name="M_S",
        closed_loop=1,
        k_folds=2,
        use_slurm=False,
        neurons_ds=neurons_ds,
        decoder_dict_path=decoder_dict_path,
    )
    assert res == "acc_job"
    assert not decoder_dict_path.exists()
