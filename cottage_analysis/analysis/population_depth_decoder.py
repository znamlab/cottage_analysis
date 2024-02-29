import functools

print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pickle
from tqdm import tqdm
import scipy
import itertools

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import flexiznam as flz
from cottage_analysis.analysis import spheres, common_utils, find_depth_neurons
from cottage_analysis.pipelines import pipeline_utils


def rolling_average_2d(arr, window, axis=0):
    # calculate rolling average along an axis for 2d array
    return (
        pd.DataFrame(arr).rolling(window=window, axis=axis, min_periods=1).mean().values
    )


def downsample_2d(arr, factor, mode="average"):
    # downsample 2d array by factor along axis 0
    end = int(factor * int(arr.shape[0] / factor))
    if mode == "average":
        arr_crop = arr[:end, :].reshape(-1, factor, arr.shape[1])
        arr_mean = np.mean(arr_crop, axis=1).reshape(-1, arr.shape[1])
    elif mode == "skip":
        arr_mean = arr[::factor, :]
    return arr_mean


def downsample_dff(
    trials_df,
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
):
    trials_df["dff_stim_rolling"] = trials_df["dff_stim"].apply(
        lambda x: rolling_average_2d(x, window=int(rolling_window * frame_rate), axis=0)
    )
    trials_df["dff_stim_downsample"] = trials_df["dff_stim_rolling"].apply(
        lambda x: downsample_2d(
            x, factor=int(downsample_window * frame_rate), mode="average"
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


def fit_svm_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    X_val,
    y_val,
    class_labels,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
):
    # tune hyperparameters:
    best_acc = 0
    best_params = {"C": Cs[0], "gamma": gammas[0]}
    if kernel == "linear":
        for C in Cs:
            print(f"Fitting C{C}...")
            clf = SVC(C=C, kernel=kernel)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"Accuracy for C{C}: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_params = {"C": C}
        print(f"Best C: {best_params['C']}")
    else:
        for C in Cs:
            for gamma in gammas:
                print(f"Fitting C{C}, gamma{gamma}...")
                clf = SVC(C=C, gamma=gamma, kernel=kernel)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                acc = accuracy_score(y_val, y_pred)

                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C, "gamma": gamma}
        print(f"Best C: {best_params['C']}, Best gamma: {best_params['gamma']}")

    # Train the final classifier on the best hyperparameters
    if kernel == "linear":
        clf = SVC(C=best_params["C"], kernel=kernel)
    else:
        clf = SVC(C=best_params["C"], gamma=best_params["gamma"], kernel=kernel)
    clf.fit(X_train, y_train)

    # Evaluate the accuracy of the classifier
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conmat = confusion_matrix(y_test, y_pred, labels=class_labels)

    return acc, conmat, best_params


def depth_decoder(
    trials_df,
    flexilims_session,
    session_name,
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
    test_size=0.2,
    random_state=42,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
):
    # add iscell
    suite2p_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        allow_multiple=False,
        return_dataseries=False,
    )
    iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[
        :, 0
    ]

    # process dff and trials_df
    trials_df = downsample_dff(
        trials_df,
        rolling_window=rolling_window,
        frame_rate=frame_rate,
        downsample_window=downsample_window,
    )

    # train test val split
    depth_list = np.sort(trials_df.depth.unique())
    depth_label = np.hstack(trials_df["depth_label"].values)
    stratified_kfold = StratifiedKFold(
        n_splits=int(1 / test_size), shuffle=True, random_state=random_state
    )
    # Train test split
    for fold, (train_index, test_index) in enumerate(
        stratified_kfold.split(np.arange(len(depth_label)), depth_label)
    ):
        depth_label_train = depth_label[train_index]
        dff_test = np.vstack(trials_df.iloc[test_index]["dff_stim_downsample"].values)
        depth_test = np.hstack(trials_df.iloc[test_index]["depth_labels"].values)
        break  # to only take the first fold

    # Train val split
    for fold, (train_index, val_index) in enumerate(
        stratified_kfold.split(np.arange(len(depth_label_train)), depth_label_train)
    ):
        dff_train = np.vstack(trials_df.iloc[train_index]["dff_stim_downsample"].values)
        dff_val = np.vstack(trials_df.iloc[val_index]["dff_stim_downsample"].values)
        depth_train = np.hstack(trials_df.iloc[train_index]["depth_labels"].values)
        depth_val = np.hstack(trials_df.iloc[val_index]["depth_labels"].values)
        break

    acc, conmat, best_params = fit_svm_classifier(
        X_train=dff_train,
        y_train=depth_train,
        X_test=dff_test,
        y_test=depth_test,
        X_val=dff_val,
        y_val=depth_val,
        class_labels=np.arange(len(depth_list)),
        kernel=kernel,
        Cs=Cs,
        gammas=gammas,
    )

    return acc, conmat, best_params
