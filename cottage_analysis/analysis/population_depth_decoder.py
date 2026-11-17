import functools

print = functools.partial(print, flush=True)

import os
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
from pathlib import Path
import pickle


from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.svm import SVC, LinearSVC
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix

import flexiznam as flz
from cottage_analysis.analysis import common_utils, find_depth_neurons
from cottage_analysis.io_module import suite2p as s2p_io

from znamutils.decorators import slurm_it

CONDA_ENV = "v1_depth_map"
LINEAR_SVC_RANDOM_STATE = 42


def calculate_error(conmat):
    """Mean depth-step error of a confusion matrix (see summary_analysis.depth_decoder_stats).

    Imported when called, as depth_decoder_stats imports the pipelines.
    """
    from cottage_analysis.summary_analysis.depth_decoder_stats import (
        calculate_error as _calculate_error,
    )

    return _calculate_error(conmat)


def _accuracy_or_nan(y_true, y_pred):
    """Accuracy, or NaN for a speed bin without frames (newer scikit-learn raises on empty input)."""
    if len(y_true) == 0:
        return np.nan
    return accuracy_score(y_true, y_pred)


def _confusion_matrix_or_empty(y_true, y_pred):
    """Confusion matrix as float, a 0 x 0 matrix for a speed bin without frames.

    The caller pads the missing depths with NaN, which gives a matrix of NaNs for an empty bin.
    """
    if len(y_true) == 0:
        return np.zeros((0, 0))
    return confusion_matrix(y_true, y_pred).astype("float")


def rolling_average(arr, window, axis=0):
    # calculate rolling average along an axis
    if arr.ndim == 1:
        return pd.Series(arr).rolling(window=window, min_periods=1).mean().values
    else:
        df = pd.DataFrame(arr)
        if axis == 1:
            df = df.T
        result = df.rolling(window=window, min_periods=1).mean()
        if axis == 1:
            result = result.T
        return result.values


def downsample(arr, factor, mode="average"):
    # downsample 1d or 2d array by factor along axis 0
    end = int(factor * int(arr.shape[0] / factor))

    if mode == "average":
        if arr.ndim == 1:
            arr_crop = arr[:end].reshape(-1, factor)
            arr_mean = np.mean(arr_crop, axis=1)
        elif arr.ndim == 2:
            arr_crop = arr[:end, :].reshape(-1, factor, arr.shape[1])
            arr_mean = np.mean(arr_crop, axis=1).reshape(-1, arr.shape[1])
        else:
            raise ValueError("Array must be 1D or 2D")
    elif mode == "skip":
        arr_mean = arr[::factor]
    return arr_mean


def downsample_dff(
    trials_df,
    rolling_window=None,
    frame_rate=15,
    downsample_window=0.5,
):
    if rolling_window is not None and rolling_window > 0:
        trials_df["dff_stim_rolling"] = trials_df["dff_stim"].apply(
            lambda x: rolling_average(
                x, window=round(rolling_window * frame_rate), axis=0
            )
        )
        trials_df["RS_stim_rolling"] = trials_df["RS_stim"].apply(
            lambda x: rolling_average(
                x, window=round(rolling_window * frame_rate), axis=0
            )
        )
        dff_col, rs_col = "dff_stim_rolling", "RS_stim_rolling"
    else:
        dff_col, rs_col = "dff_stim", "RS_stim"

    trials_df["dff_stim_downsample"] = trials_df[dff_col].apply(
        lambda x: downsample(
            x, factor=round(downsample_window * frame_rate), mode="average"
        )
    )
    trials_df["RS_stim_downsample"] = trials_df[rs_col].apply(
        lambda x: downsample(
            x, factor=round(downsample_window * frame_rate), mode="average"
        )
    )
    if any(trials_df.columns.str.contains("RS_volume")):
        # Take the maximum running speed in each imaging volume and then compute the rolling average across imaging volumes
        if rolling_window is not None and rolling_window > 0:
            trials_df["RS_volume_max_stim_rolling"] = trials_df["RS_volume_stim"].apply(
                lambda x: rolling_average(
                    np.nanmax(x, axis=1),
                    window=round(rolling_window * frame_rate),
                    axis=0,
                )
            )
            rs_volume_col = "RS_volume_max_stim_rolling"
        else:
            trials_df["RS_volume_max_stim"] = trials_df["RS_volume_stim"].apply(
                lambda x: np.nanmax(x, axis=1)
            )
            rs_volume_col = "RS_volume_max_stim"
        # Downsample the (rolling average of the) maximum running speed
        trials_df["RS_volume_max_stim_downsample"] = trials_df[rs_volume_col].apply(
            lambda x: downsample(
                x, factor=round(downsample_window * frame_rate), mode="average"
            )
        )
    depth_list = np.sort(trials_df.depth.unique())
    trials_df["depth_label"] = trials_df["depth"].apply(
        lambda x: np.where(depth_list == x)[0]
    )
    trials_df["depth_labels"] = trials_df.apply(
        lambda x: np.repeat(
            x["depth_label"], x["dff_stim_downsample"].shape[0], axis=0
        ),
        axis=1,
    )

    return trials_df


def find_stationary_samples(
    trials_df,
    frame_rate=15,
    downsample_window=0.5,
    still_thr=0.05,
    still_time=1,
):
    """Find stationary decoder samples: maximum running speed during the preceding
    `still_time` s is < `still_thr` m/s.

    Uses the raw per-frame running speed, not the smoothed and downsampled one. For
    volumetric imaging (`RS_volume_*` columns), the speed of a volume is the maximum over
    its frames. A sample of round(downsample_window * frame_rate) frames (as in
    downsample_dff; one frame if downsample_window is None) is stationary if the
    int(still_time * frame_rate) frames ending on its last frame are. The window extends
    into the preceding blank (`*_blank_pre`) for the first samples of a trial; samples
    without a full window are not stationary.

    Args:
        trials_df (pd.DataFrame): trials_df after downsample_dff.
        frame_rate (float): imaging frame (or volume) rate. Truncated to int as in
            preprocess_data.
        downsample_window (float): downsample window in s used in downsample_dff, or
            None if the samples were not downsampled.
        still_thr (float): running speed threshold in m/s.
        still_time (float): duration in s the mouse must have been stationary.

    Returns:
        np.ndarray: boolean mask in the sample order of np.hstack(dff_stim_downsample).
    """
    frame_rate = int(frame_rate)
    bin_frames = (
        1 if downsample_window is None else round(downsample_window * frame_rate)
    )
    still_frames = int(still_time * frame_rate)
    use_volume = "RS_volume_stim" in trials_df.columns

    def _speed(trial, epoch):
        col = f"RS_volume_{epoch}" if use_volume else f"RS_{epoch}"
        if col not in trial or len(trial[col]) == 0:
            return np.array([])
        if use_volume:
            return np.nanmax(np.asarray(trial[col], dtype=float), axis=1)
        return np.asarray(trial[col], dtype=float)

    masks = []
    for _, trial in trials_df.iterrows():
        rs_pre = _speed(trial, "blank_pre")
        rs = np.concatenate([rs_pre, _speed(trial, "stim")])
        n_samples = trial["dff_stim_downsample"].shape[0]
        sample_end = len(rs_pre) + (np.arange(n_samples) + 1) * bin_frames
        trial_mask = np.zeros(n_samples, dtype=bool)
        for i, end in enumerate(sample_end):
            window = rs[max(end - still_frames, 0) : end]
            trial_mask[i] = (
                len(window) == still_frames
                and np.isfinite(window).any()
                and np.nanmax(window) < still_thr
            )
        masks.append(trial_mask)
    return np.concatenate(masks)


def split_train_test_val(trials_df, k_folds=5, random_state=42, trial_average=False):
    # train test val split based on trials
    depth_list = np.sort(trials_df.depth.unique())
    if len(trials_df) % len(depth_list) != 0:
        trials_df = trials_df[: -(len(trials_df) % len(depth_list))]
    if trial_average:
        dff_col = "trial_mean_dff"
        depth_col = "depth_label"
    else:
        dff_col = "dff_stim_downsample"
        depth_col = "depth_labels"

    # get the indices of each number in dff_col
    # if trial_average is True, it's np.arange(nframes) for each row
    # if trial_average is False, it's the trial number for each row
    trials_df["frame_num"] = trials_df.apply(
        lambda x: x[dff_col].shape[0], axis=1
    ).astype("int")
    trials_df["frame_num_prev_trials"] = (
        trials_df["frame_num"].shift(1).fillna(0).cumsum().astype("int")
    )
    trials_df["frame_indices"] = trials_df.apply(
        lambda x: (np.arange(x["frame_num"]) + x["frame_num_prev_trials"]).astype(
            "int"
        ),
        axis=1,
    )

    # train test val split
    depth_label = np.hstack(trials_df["depth_label"].values)

    (
        dff_train_all,
        dff_val_all,
        dff_test_all,
        depth_train_all,
        depth_val_all,
        depth_test_all,
    ) = ([], [], [], [], [], [])
    if k_folds == 1:
        test_size = 0.2
        sss = StratifiedShuffleSplit(
            n_splits=k_folds, test_size=test_size, random_state=random_state
        )
    else:
        test_size = 1 / k_folds
        sss = StratifiedKFold(n_splits=k_folds, random_state=random_state, shuffle=True)

    # Train test split
    test_frame_indices = []
    test_index_all = []
    train_index_temp_all = []
    for fold, (train_index_temp, test_index) in enumerate(
        sss.split(np.arange(len(depth_label)), depth_label)
    ):
        depth_label_train = depth_label[train_index_temp]
        dff_test = np.vstack(trials_df.iloc[test_index][dff_col].values)
        depth_test = np.hstack(trials_df.iloc[test_index][depth_col].values)
        dff_test_all.append(dff_test)
        depth_test_all.append(depth_test)
        test_index_all.append(test_index)
        train_index_temp_all.append(train_index_temp)
        test_frame_indices.append(
            np.hstack(trials_df.iloc[test_index]["frame_indices"].values)
        )
        assert (
            len(test_frame_indices[fold]) == dff_test.shape[0]
        ), "ERROR: Test indices do not match."

    # Train val split
    train_index_all, val_index_all = [], []
    for fold, (train_index, val_index) in enumerate(
        sss.split(np.arange(len(depth_label_train)), depth_label_train)
    ):
        if train_index.max() == len(train_index_temp_all[fold]):
            train_index = train_index[:-1]
            print(
                f"Trial number is not the multiple of {int(len(depth_list)*k_folds)}. \n\
                  Train index max is equal to the length of the training set. Removing the last index."
            )
        if val_index.max() == len(train_index_temp_all[fold]):
            val_index = val_index[:-1]
            print(
                f"Trial number is not the multiple of {int(len(depth_list)*k_folds)}. \n\
                  Validation index max is equal to the length of the training set. Removing the last index."
            )
        train_index = train_index_temp_all[fold][train_index]
        val_index = train_index_temp_all[fold][val_index]
        dff_train = np.vstack(trials_df.iloc[train_index][dff_col].values)
        dff_val = np.vstack(trials_df.iloc[val_index][dff_col].values)
        depth_train = np.hstack(trials_df.iloc[train_index][depth_col].values)
        depth_val = np.hstack(trials_df.iloc[val_index][depth_col].values)
        dff_train_all.append(dff_train)
        dff_val_all.append(dff_val)
        depth_train_all.append(depth_train)
        depth_val_all.append(depth_val)
        train_index_all.append(train_index)
        val_index_all.append(val_index)

    # Check if the split has been done correctly (no repetition in train, val, test sets)
    for i in range(k_folds):
        all = np.sort(
            np.hstack([train_index_all[i], test_index_all[i], val_index_all[i]])
        )
        assert len(np.unique(all)) == len(
            trials_df
        ), "ERROR: There are repetitions in the train, val, test sets."

    return (
        trials_df,
        dff_train_all,
        dff_val_all,
        dff_test_all,
        depth_train_all,
        depth_val_all,
        depth_test_all,
        test_frame_indices,
        dff_col,
        depth_col,
    )


def subset_decoder_inputs(decoder_inputs, roi_vector, k_folds=5):
    """Select a subset of ROIs from the decoder inputs dictionary using the provided roi_vector."""
    subset = {}
    for key, value in decoder_inputs.items():
        if key == "trials_df":
            subset[key] = common_utils.choose_rois_subset(
                decoder_inputs[key], roi_vector=roi_vector
            )
        elif key in ("dff_test_all", "dff_val_all", "dff_train_all"):
            tmp = []
            for i in range(k_folds):
                tmp.append(value[i][:, roi_vector])
            subset[key] = tmp
        else:
            subset[key] = value

    return subset


def _svm_param_grid(kernel, Cs, gammas):
    """Hyperparameter combinations in the order they are evaluated."""
    if kernel == "linear":
        return [{"C": C} for C in Cs]
    return [{"C": C, "gamma": gamma} for C in Cs for gamma in gammas]


def _fit_svm(X, y, kernel, params):
    """Fit one SVM classifier.

    kernel "linear" uses LinearSVC (liblinear): squared hinge loss, one-vs-rest,
    default settings. It gives accuracies and confusion matrices close to
    SVC(kernel="linear") (libsvm: hinge loss, one-vs-one), in a fraction of the time.
    Other kernels use SVC.
    """
    if kernel == "linear":
        clf = LinearSVC(random_state=LINEAR_SVC_RANDOM_STATE, **params)
    else:
        clf = SVC(kernel=kernel, **params)
    clf.fit(X, y)
    return clf


def _grid_task(X_train, y_train, X_val, y_val, kernel, params):
    t0 = time.time()
    clf = _fit_svm(X_train, y_train, kernel, params)
    acc = accuracy_score(y_val, clf.predict(X_val))
    label = f"C{params['C']}" + (
        f" gamma{params['gamma']}" if "gamma" in params else ""
    )
    print(f"Accuracy for {label}: {acc} ({time.time() - t0:.0f} s)")
    return acc


def _refit_task(X_train_val, y_train_val, X_test, kernel, params):
    t0 = time.time()
    clf = _fit_svm(X_train_val, y_train_val, kernel, params)
    return clf, clf.predict(X_test), time.time() - t0


def _select_best_params(grid, accs, kernel, Cs, gammas):
    """Same rule as the original sequential loop: first best in grid order."""
    best_acc = 0
    best_params = {"C": Cs[0], "gamma": gammas[0]}
    for params, acc in zip(grid, accs):
        if acc > best_acc:
            best_acc = acc
            best_params = dict(params)
    if kernel == "linear":
        print(f"Best C: {best_params['C']}")
    else:
        print(f"Best C: {best_params['C']}, Best gamma: {best_params['gamma']}")
    return best_params


def svm_classifier_hyperparam_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    class_labels,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
    n_jobs=1,
):
    """Select SVM hyperparameters on the validation set.

    The grid is fitted in parallel when n_jobs > 1; selection is unchanged (first
    best accuracy in grid order), so results do not depend on n_jobs.
    """
    grid = _svm_param_grid(kernel, Cs, gammas)
    accs = Parallel(n_jobs=n_jobs)(
        delayed(_grid_task)(X_train, y_train, X_val, y_val, kernel, p) for p in grid
    )
    return _select_best_params(grid, accs, kernel, Cs, gammas)


def test_svm_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    class_labels,
    kernel="linear",
    best_params={"C": 1, "gamma": 1},
):
    # Train the final classifier on the best hyperparameters
    if kernel == "linear":
        params = {"C": best_params["C"]}
    else:
        params = {"C": best_params["C"], "gamma": best_params["gamma"]}
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])
    clf, y_pred, _ = _refit_task(X_train_val, y_train_val, X_test, kernel, params)
    return clf, y_pred


def tune_and_test_svm_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
    n_jobs=1,
):
    """Hyperparameter search and final train+val fit, in parallel when n_jobs > 1.

    Equivalent to svm_classifier_hyperparam_tuning followed by test_svm_classifier.
    With n_jobs > 1, the final train+val fits are also started for every candidate
    except the largest C (the slowest to fit with SVC, and rarely selected), alongside
    the grid, so the final model is usually ready when the grid finishes. Selection and
    fits are unchanged, so best_params and y_pred are identical to the sequential
    version.

    Returns:
        best_params (dict), clf (fitted LinearSVC or SVC), y_pred (np.ndarray)
    """
    if n_jobs == 1:
        best_params = svm_classifier_hyperparam_tuning(
            X_train, y_train, X_val, y_val, None, kernel, Cs, gammas, 1
        )
        clf, y_pred = test_svm_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            None,
            None,
            kernel,
            best_params,
        )
        return best_params, clf, y_pred

    grid = _svm_param_grid(kernel, Cs, gammas)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])
    max_C = max(Cs)
    tasks = [("grid", i) for i in range(len(grid))] + [
        ("refit", i) for i, p in enumerate(grid) if p["C"] != max_C
    ]
    # start the slowest fits (largest C) first
    tasks.sort(key=lambda t: -grid[t[1]]["C"])
    results = Parallel(n_jobs=n_jobs)(
        (
            delayed(_grid_task)(X_train, y_train, X_val, y_val, kernel, grid[i])
            if kind == "grid"
            else delayed(_refit_task)(X_train_val, y_train_val, X_test, kernel, grid[i])
        )
        for kind, i in tasks
    )
    out = dict(zip(tasks, results))
    accs = [out[("grid", i)] for i in range(len(grid))]
    best_params = _select_best_params(grid, accs, kernel, Cs, gammas)
    best_i = next(
        (
            i
            for i, p in enumerate(grid)
            if all(best_params.get(k) == v for k, v in p.items())
        ),
        None,
    )
    if best_i is not None and ("refit", best_i) in out:
        clf, y_pred, duration = out[("refit", best_i)]
        print(f"Final fit (started in parallel): {duration:.0f} s")
    else:
        t0 = time.time()
        clf, y_pred = test_svm_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            None,
            None,
            kernel,
            best_params,
        )
        print(f"Final fit: {time.time() - t0:.0f} s")
    return best_params, clf, y_pred


def preprocess_data(
    trials_df,
    flexilims_session,
    session_name,
    closed_loop=1,
    n_rois=None,
    trial_average=False,
    rolling_window=None,
    frame_rate=15,
    downsample_window=0.5,
    random_state=42,
    kernel="linear",
    k_folds=5,
    neurons_ds=None,
):
    # set test_size:
    if k_folds == 1:
        test_size = 0.2
    else:
        test_size = 1 / k_folds

    # add iscell
    suite2p_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        allow_multiple=False,
        return_dataseries=False,
    )
    iscell = s2p_io.load_is_cell(suite2p_ds.path_full)

    # process dff and trials_df
    if downsample_window is not None:
        trials_df = downsample_dff(
            trials_df,
            rolling_window=rolling_window,
            frame_rate=int(frame_rate),
            downsample_window=downsample_window,
        )
    else:
        trials_df["dff_stim_downsample"] = trials_df["dff_stim"].values
        trials_df["RS_stim_downsample"] = trials_df["RS_stim"].values
        if any(trials_df.columns.str.contains("RS_volume")):
            trials_df["RS_volume_max_stim_downsample"] = trials_df[
                "RS_volume_stim"
            ].apply(
                lambda x: np.nanmax(x, axis=1),
            )
        depth_list = np.sort(trials_df.depth.unique())
        trials_df["depth_label"] = trials_df["depth"].apply(
            lambda x: np.where(depth_list == x)[0]
        )
        trials_df["depth_labels"] = trials_df.apply(
            lambda x: np.repeat(
                x["depth_label"], x["dff_stim_downsample"].shape[0], axis=0
            ),
            axis=1,
        )
    trials_df = find_depth_neurons.trial_average_dff(
        trials_df,
        rs_thr_min=None,
        rs_thr_max=None,
        still_only=False,
        still_time=0,
        frame_rate=int(frame_rate),
        closed_loop=closed_loop,
    )
    depth_list = np.sort(trials_df.depth.unique())
    # subset trials_df to n neurons if n is not None using random state
    if n_rois is not None and neurons_ds is not None:
        os.makedirs(neurons_ds.path_full.parent / "decoder_subsets", exist_ok=True)
        if closed_loop:
            sfx = "closedloop"
        else:
            sfx = "openloop"
        trials_df, roi_vector = common_utils.choose_rois_subset(
            trials_df=trials_df, iscell=iscell, n_rois=n_rois, random_state=random_state
        )
        np.savez(
            file=neurons_ds.path_full.parent
            / "decoder_subsets"
            / f"decoder_subset_{sfx}_n{n_rois}_seed{random_state}.npz",
            random_state=random_state,
            roi_vector=roi_vector,
            allow_pickle=True,
        )

    # split train test val (test 0.2, val 0.2, train 0.6)
    (
        trials_df,
        dff_train_all,
        dff_val_all,
        dff_test_all,
        depth_train_all,
        depth_val_all,
        depth_test_all,
        test_frame_indices,
        dff_col,
        depth_col,
    ) = split_train_test_val(
        trials_df=trials_df,
        k_folds=k_folds,
        random_state=random_state,
        trial_average=trial_average,
    )

    return (
        trials_df,
        (
            dff_train_all,
            dff_val_all,
            dff_test_all,
            depth_train_all,
            depth_val_all,
            depth_test_all,
        ),
        (iscell, depth_list),
        (test_frame_indices, dff_col, depth_col),
    )


def _save_decoder_dict(decoder_dict, decoder_dict_path):
    """Write decoder_dict atomically so concurrent readers never see a partial file."""
    decoder_dict_path = Path(decoder_dict_path)
    tmp_path = decoder_dict_path.with_name(
        f".{decoder_dict_path.name}.{os.getpid()}.tmp"
    )
    with open(tmp_path, "wb") as f:
        pickle.dump(decoder_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, decoder_dict_path)


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={
        "mem": "32G",
        "time": "7-00:00:00",
        "partition": "ncpu",
        "cpus-per-task": 8,
    },
    print_job_id=True,
)
def fit_each_fold(
    i,
    decoder_inputs,
    recording_type,
    decoder_inputs_path=None,
    decoder_dict_path=None,
    n_rois=None,
    random_state=None,
    special_sfx="",
):
    print(f"Fitting fold {i+1}...")
    # load decoder inputs and results
    if len(decoder_inputs) == 0:
        assert (
            decoder_inputs_path is not None
        ), f"ERROR: decoder_inputs_path must be provided if decoder_dict is empty."
        if not os.path.exists(decoder_inputs_path):
            raise FileNotFoundError(
                f"Decoder inputs file not found: {decoder_inputs_path}"
            )
        with open(decoder_inputs_path, "rb") as f:
            decoder_inputs = pickle.load(f)
        # filter decoder inputs if using a subset of ROIs
        if n_rois is not None and random_state is not None:
            print(
                f"Filtering decoder inputs with ROI vector for n = {n_rois}, random_state = {random_state}"
            )
            roi_vector = np.load(
                file=Path(decoder_inputs_path).parent
                / "decoder_subsets"
                / f"decoder_subset_{recording_type}_n{n_rois}_seed{random_state}.npz",
            )["roi_vector"]
            decoder_inputs = subset_decoder_inputs(decoder_inputs, roi_vector, k_folds)

    (
        trials_df,
        dff_train_all,
        dff_val_all,
        dff_test_all,
        depth_train_all,
        depth_val_all,
        depth_test_all,
        iscell,
        depth_list,
        test_frame_indices,
        dff_col,
        depth_col,
        kernel,
        Cs,
        gammas,
        y_test_all,
    ) = (
        decoder_inputs["trials_df"],
        decoder_inputs["dff_train_all"],
        decoder_inputs["dff_val_all"],
        decoder_inputs["dff_test_all"],
        decoder_inputs["depth_train_all"],
        decoder_inputs["depth_val_all"],
        decoder_inputs["depth_test_all"],
        decoder_inputs["iscell"],
        decoder_inputs["depth_list"],
        decoder_inputs["test_frame_indices"],
        decoder_inputs["dff_col"],
        decoder_inputs["depth_col"],
        decoder_inputs["kernel"],
        decoder_inputs["Cs"],
        decoder_inputs["gammas"],
        decoder_inputs["y_test_all"],
    )

    # only select current fold and cells
    dff_train = dff_train_all[i]
    dff_val = dff_val_all[i]
    dff_test = dff_test_all[i]
    depth_train = depth_train_all[i]
    depth_val = depth_val_all[i]
    depth_test = depth_test_all[i]

    if n_rois is None:
        print("Using all cells for fitting...")
        dff_train = dff_train[:, iscell]
        dff_val = dff_val[:, iscell]
        dff_test = dff_test[:, iscell]

    # get best hyperparameters for this fold, then train on combined train and val
    # data and test on the current fold. Uses the CPUs allocated by slurm, if any.
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    best_params, clf, y_pred = tune_and_test_svm_classifier(
        X_train=dff_train,
        y_train=depth_train,
        X_val=dff_val,
        y_val=depth_val,
        X_test=dff_test,
        kernel=kernel,
        Cs=Cs,
        gammas=gammas,
        n_jobs=n_jobs,
    )

    decoder_outputs = {
        "y_pred": y_pred,
        "best_params": best_params,
    }
    if n_rois is None:
        decoder_outputs_str = f"decoder_outputs_{recording_type}{special_sfx}"
    elif n_rois is not None and random_state is not None:
        decoder_outputs_str = f"decoder_outputs_{recording_type}{special_sfx}_n{n_rois}_seed{random_state}"
    with open(
        Path(decoder_dict_path).parent / f"{decoder_outputs_str}_fold{i}.pickle",
        "wb",
    ) as f:
        pickle.dump(decoder_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return decoder_outputs


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={
        "mem": "32G",
        "time": "4:00:00",
        "partition": "ncpu",
    },
    print_job_id=True,
)
def calculate_acc_conmat(
    decoder_dict_path,
    recording_type,
    depth_list,
    decoder_inputs={},
    decoder_inputs_path=None,
    k_folds=5,
    n_rois=None,
    random_state=None,
    special_sfx="",
):
    print(f"Reading decoder inputs from {decoder_inputs_path}...")
    with open(decoder_dict_path, "rb") as f:
        decoder_dict = pickle.load(f)
    if decoder_inputs_path is not None:
        if os.path.exists(decoder_inputs_path):
            with open(decoder_inputs_path, "rb") as f:
                decoder_inputs = pickle.load(f)

    # getting ROI vector to filter raw decoder inputs if subset of ROIs used for training
    if n_rois is not None and random_state is not None:
        roi_vector = np.load(
            file=Path(decoder_inputs_path).parent
            / "decoder_subsets"
            / f"decoder_subset_{recording_type}_n{n_rois}_seed{random_state}.npz",
        )["roi_vector"]
        decoder_inputs = subset_decoder_inputs(decoder_inputs, roi_vector, k_folds)

    # concatenate results from all folds
    for i in range(k_folds):
        if n_rois is None:
            decoder_outputs_path = (
                Path(decoder_dict_path).parent
                / f"decoder_outputs_{recording_type}{special_sfx}_fold{i}.pickle"
            )
        elif n_rois is not None and random_state is not None:
            decoder_outputs_path = (
                Path(decoder_dict_path).parent
                / f"decoder_outputs_{recording_type}{special_sfx}_n{n_rois}_seed{random_state}_fold{i}.pickle"
            )
        print(f"Loading decoder outputs from {decoder_outputs_path}...")
        with open(decoder_outputs_path, "rb") as f:
            decoder_outputs = pickle.load(f)

        decoder_dict[f"best_params_all_{recording_type}"]["C"].append(
            decoder_outputs["best_params"]["C"]
        )
        if decoder_inputs["kernel"] != "linear":
            decoder_dict[f"best_params_all_{recording_type}"]["gamma"].append(
                decoder_outputs["best_params"]["gamma"]
            )
        decoder_dict[f"y_preds_all_{recording_type}"][
            decoder_inputs["test_frame_indices"][i]
        ] = decoder_outputs["y_pred"]

    y_test_all = decoder_dict[f"y_test_all_{recording_type}"]
    y_preds_all = decoder_dict[f"y_preds_all_{recording_type}"]
    # calculate acc & conmat
    acc = accuracy_score(y_test_all, y_preds_all)
    conmat = confusion_matrix(
        y_test_all, y_preds_all, labels=np.arange(len(depth_list))
    )
    decoder_dict[f"accuracy_{recording_type}"] = acc
    decoder_dict[f"conmat_{recording_type}"] = conmat
    decoder_dict[f"error_{recording_type}"] = calculate_error(conmat)
    print(f"Accuracy {recording_type}: {acc}")
    if n_rois is None:
        decoder_results_str = f"decoder_results{special_sfx}.pickle"
    elif n_rois is not None and random_state is not None:
        decoder_results_str = f"decoder_results_{recording_type}{special_sfx}_n{n_rois}_seed{random_state}.pickle"
    print(f"Writing decoder results to {decoder_results_str}")
    _save_decoder_dict(
        decoder_dict, Path(decoder_dict_path).parent / decoder_results_str
    )
    return decoder_dict


def depth_decoder(
    trials_df,
    flexilims_session,
    session_name,
    closed_loop=1,
    n_rois=None,
    trial_average=False,
    rolling_window=None,
    frame_rate=15,
    downsample_window=0.5,
    random_state=42,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
    k_folds=5,
    use_slurm=False,
    neurons_ds=None,
    decoder_dict_path=None,
    special_sfx="",
    job_dependency=None,
):
    if decoder_dict_path is None:
        assert (
            neurons_ds is not None
        ), "ERROR: neurons_ds or decoder_dict_path must be provided."
        decoder_dict_path = (
            neurons_ds.path_full.parent / f"decoder_results{special_sfx}.pickle"
        )
    # set slurm folder
    if use_slurm:
        slurm_folder = Path(os.path.expanduser(f"~/slurm_logs"))
        slurm_folder.mkdir(exist_ok=True)
        slurm_folder = Path(slurm_folder / f"{session_name}")
        slurm_folder.mkdir(exist_ok=True)
    else:
        slurm_folder = None

    # choose closedloop or openloop
    trials_df = trials_df[trials_df.closed_loop == closed_loop]

    # prepare data
    (
        trials_df,
        (
            dff_train_all,
            dff_val_all,
            dff_test_all,
            depth_train_all,
            depth_val_all,
            depth_test_all,
        ),
        (iscell, depth_list),
        (test_frame_indices, dff_col, depth_col),
    ) = preprocess_data(
        trials_df=trials_df,
        flexilims_session=flexilims_session,
        session_name=session_name,
        n_rois=n_rois,
        closed_loop=closed_loop,
        trial_average=trial_average,
        rolling_window=rolling_window,
        frame_rate=frame_rate,
        downsample_window=downsample_window,
        random_state=random_state,
        kernel=kernel,
        k_folds=k_folds,
        neurons_ds=neurons_ds,
    )

    # save decoder inputs if use_slurm
    y_test_all = np.hstack(trials_df[depth_col])
    y_preds_all = np.zeros_like(y_test_all)
    decoder_inputs = {
        "trials_df": trials_df,
        "dff_train_all": dff_train_all,
        "dff_val_all": dff_val_all,
        "dff_test_all": dff_test_all,
        "depth_train_all": depth_train_all,
        "depth_val_all": depth_val_all,
        "depth_test_all": depth_test_all,
        "iscell": iscell,
        "depth_list": depth_list,
        "test_frame_indices": test_frame_indices,
        "dff_col": dff_col,
        "depth_col": depth_col,
        "kernel": kernel,
        "Cs": Cs,
        "gammas": gammas,
        "y_test_all": y_test_all,
    }

    if kernel == "linear":
        best_params_all = {
            "C": [],
        }
    else:
        best_params_all = {
            "C": [],
            "gamma": [],
        }

    if closed_loop:
        recording_type = "closedloop"
    else:
        recording_type = "openloop"

    if use_slurm:
        assert (
            neurons_ds is not None
        ), "ERROR: neurons_ds must be provided when use_slurm is True."
        decoder_inputs_path = (
            neurons_ds.path_full.parent
            / f"decoder_inputs_{recording_type}{special_sfx}.pickle"
        )
        if n_rois is None:
            decoder_inputs_path = (
                neurons_ds.path_full.parent
                / f"decoder_inputs_{recording_type}{special_sfx}.pickle"
            )
            with open(decoder_inputs_path, "wb") as f:
                pickle.dump(decoder_inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        if n_rois is not None:
            # make sure the normal decoder inputs for the whole dataset exists and do not overwrite it with the subset data,
            # instead use the roi vector to subset trials_df
            assert os.path.exists(decoder_inputs_path)
        decoder_inputs = {}  # don't pass the dict on to the slurm job
    else:
        decoder_inputs_path = None

    # save temp decoder results. Load the existing results only now, after the
    # slow preprocessing, so we don't overwrite results that jobs from a previous
    # depth_decoder call (e.g. openloop) wrote in the meantime.
    if os.path.exists(decoder_dict_path):
        with open(decoder_dict_path, "rb") as f:
            decoder_dict = pickle.load(f)
    else:
        decoder_dict = {}
    decoder_dict[f"best_params_all_{recording_type}"] = best_params_all
    decoder_dict[f"y_test_all_{recording_type}"] = y_test_all
    decoder_dict[f"y_preds_all_{recording_type}"] = y_preds_all
    decoder_dict[f"trials_df_{recording_type}"] = trials_df
    if n_rois is None:
        decoder_results_path = (
            neurons_ds.path_full.parent / f"decoder_results{special_sfx}.pickle"
        )
    else:
        decoder_results_path = (
            neurons_ds.path_full.parent
            / "decoder_subsets"
            / f"decoder_results_{recording_type}_n{n_rois}_seed{random_state}.pickle"
        )
    _save_decoder_dict(decoder_dict, decoder_results_path)

    # fit each fold
    outputs = []
    for i in range(k_folds):
        out = fit_each_fold(
            i,
            decoder_inputs=decoder_inputs,
            decoder_inputs_path=decoder_inputs_path,
            decoder_dict_path=decoder_results_path,
            n_rois=n_rois,
            random_state=random_state,
            recording_type=recording_type,
            special_sfx=special_sfx,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"decoder_{recording_type}{special_sfx}_fold{i}"
            if n_rois is None
            else f"decoder_{recording_type}{special_sfx}_n{n_rois}_seed{random_state}_fold{i}",
        )
        outputs.append(out)

    # calculate the accuracy on all test data
    if use_slurm:
        acc_dependency = outputs + (list(job_dependency) if job_dependency else [])
    else:
        acc_dependency = None
    decoder_dict = calculate_acc_conmat(
        decoder_dict_path=decoder_results_path,
        recording_type=recording_type,
        depth_list=depth_list.tolist(),
        decoder_inputs=decoder_inputs,
        decoder_inputs_path=decoder_inputs_path,
        n_rois=n_rois,
        random_state=random_state,
        k_folds=k_folds,
        special_sfx=special_sfx,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"decoder_{recording_type}{special_sfx}_accuracy"
        if n_rois is None
        else f"decoder_{recording_type}{special_sfx}_n{n_rois}_seed{random_state}_accuracy",
        job_dependency=acc_dependency,
    )

    return decoder_dict


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={
        "mem": "32G",
        "time": "8:00:00",
        "partition": "ncpu",
    },
    print_job_id=True,
)
def find_acc_speed_bins(
    decoder_dict_path,
    recording_type,
    speed_bins,
    continuous_still=True,
    still_thr=0.05,
    still_time=1,
    frame_rate=15,
    add_errors=False,
    downsample_window=0.5,
):
    speed_bins = np.array(speed_bins)
    with open(decoder_dict_path, "rb") as f:
        decoder_dict = pickle.load(f)
    trials_df = decoder_dict[f"trials_df{recording_type}"]
    y_test = decoder_dict[f"y_test_all{recording_type}"]
    y_preds = decoder_dict[f"y_preds_all{recording_type}"]
    labels = np.unique(y_test)

    def fill_in_missing_depths_to_conmat(y_test, y_preds, trials_df):
        ndepths = len(trials_df.depth.unique())
        y_test_missing = list(set(np.arange(ndepths)) - set(np.unique(y_test)))
        y_preds_missing = list(set(np.arange(ndepths)) - set(np.unique(y_preds)))
        common_missing = list(set(y_test_missing) & set(y_preds_missing))
        return common_missing

    acc_speed_bins = []
    conmat_speed_bins = []
    # if calculating the error, make empty lists for the error of the decoder by speed bins, chance-level confusion matrices and their errors
    if add_errors:
        error_speed_bins = []
    rs_arr = np.hstack(trials_df["RS_stim_downsample"])
    if continuous_still:
        idx = find_stationary_samples(
            trials_df,
            frame_rate=frame_rate,
            downsample_window=downsample_window,
            still_thr=still_thr,
            still_time=still_time,
        )
        acc_speed_bins.append(_accuracy_or_nan(y_test[idx], y_preds[idx]))
        conmat = _confusion_matrix_or_empty(y_test[idx], y_preds[idx])
        missing_idx = fill_in_missing_depths_to_conmat(
            y_test[idx], y_preds[idx], trials_df
        )
        for miss in missing_idx:
            conmat = np.insert(conmat, miss, np.nan, axis=0)
            conmat = np.insert(conmat, miss, np.nan, axis=1)
        conmat_speed_bins.append(conmat)
        print(f"Still accuracy: {_accuracy_or_nan(y_test[idx], y_preds[idx])}")

        if add_errors:
            # calculating the error of the confusion matrix during stationary periods
            error_speed_bins.append(calculate_error(conmat))

    for i in range(len(speed_bins) - 1):
        idx = (rs_arr >= speed_bins[i]) & (rs_arr < speed_bins[i + 1])
        acc_speed_bins.append(_accuracy_or_nan(y_test[idx], y_preds[idx]))
        conmat = _confusion_matrix_or_empty(y_test[idx], y_preds[idx])
        missing_idx = fill_in_missing_depths_to_conmat(
            y_test[idx], y_preds[idx], trials_df
        )
        for miss in missing_idx:
            conmat = np.insert(conmat, miss, np.nan, axis=0)
            conmat = np.insert(conmat, miss, np.nan, axis=1)
        conmat_speed_bins.append(conmat)
        print(
            f"Speed bin {speed_bins[i]} - {speed_bins[i+1]} accuracy: {_accuracy_or_nan(y_test[idx], y_preds[idx])}"
        )
        if add_errors:
            # calculating the error of the confusion matrix for each speed bin
            error_speed_bins.append(calculate_error(conmat))

    for speed_bin in [
        speed_bins[-1]
    ]:  # speed that's larger than the max boundary of speed_bins
        idx = rs_arr >= speed_bin
        acc_speed_bins.append(_accuracy_or_nan(y_test[idx], y_preds[idx]))
        conmat = _confusion_matrix_or_empty(y_test[idx], y_preds[idx])
        missing_idx = fill_in_missing_depths_to_conmat(
            y_test[idx], y_preds[idx], trials_df
        )
        for miss in missing_idx:
            conmat = np.insert(conmat, miss, np.nan, axis=0)
            conmat = np.insert(conmat, miss, np.nan, axis=1)
        conmat_speed_bins.append(conmat)
        print(
            f"Speed bin > {speed_bin} accuracy: {_accuracy_or_nan(y_test[idx], y_preds[idx])}"
        )
        if add_errors:
            # calculating the error of the confusion matrix in the last speed bin
            error_speed_bins.append(calculate_error(conmat))
    decoder_dict[f"acc_speed_bins{recording_type}"] = acc_speed_bins
    decoder_dict[f"conmat_speed_bins{recording_type}"] = conmat_speed_bins
    decoder_dict[f"error_speed_bins{recording_type}"] = (
        error_speed_bins if add_errors else None
    )
    decoder_dict[f"error{recording_type}"] = (
        calculate_error(decoder_dict[f"conmat{recording_type}"]) if add_errors else None
    )
    _save_decoder_dict(decoder_dict, decoder_dict_path)
    return decoder_dict
