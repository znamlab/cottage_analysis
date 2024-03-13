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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import flexiznam as flz
from cottage_analysis.analysis import spheres, common_utils, find_depth_neurons
from cottage_analysis.pipelines import pipeline_utils


def stratified_shuffle_split(X,y, test_size=0.2, random_state=42):
    # Ensure the data is in a numpy array for easier indexing
    X = np.array(X)
    y = np.array(y)
    
    # Determine unique classes and their distribution
    classes, y_indices = np.unique(y, return_inverse=True)
    class_counts = np.bincount(y_indices)
    
    # Calculate the number of samples per class in the test set
    test_counts = np.ceil(class_counts * test_size).astype(int)
    train_counts = class_counts - test_counts
    
    # Initialize lists to hold the indices for the split
    train_indices = []
    test_indices = []
    
    # For each class, shuffle and split the data
    for class_index, class_count in enumerate(class_counts):
        class_indices = np.where(y == classes[class_index])[0]
        np.random.shuffle(class_indices)
        
        class_test_indices = class_indices[:test_counts[class_index]]
        class_train_indices = class_indices[test_counts[class_index]:]
        
        train_indices.extend(class_train_indices)
        test_indices.extend(class_test_indices)

    
    # # Shuffle the final indices to mix classes
    # np.random.shuffle(train_indices)
    # np.random.shuffle(test_indices)
    
    return train_indices, test_indices


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


def split_train_test_val(trials_df, test_size=0.2, random_state=42, trial_average=False):
    depth_list = np.sort(trials_df.depth.unique())
    if len(trials_df)%len(depth_list) != 0:
        trials_df = trials_df[:-(len(trials_df)%len(depth_list))] 
    if trial_average:
        dff_col = "trial_mean_dff"
        depth_col = "depth_label"
    else:
        dff_col = "dff_stim_downsample"
        depth_col = "depth_labels"
    # train test val split
    depth_label = np.hstack(trials_df["depth_label"].values)

    # Train test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for fold, (train_index_temp, test_index) in enumerate(sss.split(np.arange(len(depth_label)), depth_label)):
        depth_label_train = depth_label[train_index_temp]
        dff_test = np.vstack(trials_df.iloc[test_index][dff_col].values)
        depth_test = np.hstack(trials_df.iloc[test_index][depth_col].values)

    # Train val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size/(1-test_size), random_state=random_state)
    for fold, (train_index, val_index) in enumerate(sss.split(np.arange(len(depth_label_train)), depth_label_train)):
        train_index = train_index_temp[train_index]
        val_index = train_index_temp[val_index]
        dff_train = np.vstack(trials_df.iloc[train_index][dff_col].values)
        dff_val = np.vstack(trials_df.iloc[val_index][dff_col].values)
        depth_train = np.hstack(trials_df.iloc[train_index][depth_col].values)
        depth_val = np.hstack(trials_df.iloc[val_index][depth_col].values)
        
    return dff_train, dff_val, dff_test, depth_train, depth_val, depth_test
    

def depth_decoder(
    trials_df,
    flexilims_session,
    session_name,
    closed_loop=1,
    trial_average=False,
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
    test_size=0.2,
    random_state=42,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
):
    # choose closedloop or openloop
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    
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
    ].astype('bool')

    # process dff and trials_df
    trials_df = downsample_dff(
        trials_df,
        rolling_window=rolling_window,
        frame_rate=frame_rate,
        downsample_window=downsample_window,
    )
    trials_df = find_depth_neurons.trial_average_dff(
        trials_df,
        rs_thr_min=None,
        rs_thr_max=None,
        still_only=False,
        still_time=0,
        frame_rate=15,
        closed_loop=closed_loop,
    )
    depth_list = np.sort(trials_df.depth.unique())

    # train test val split
    dff_train, dff_val, dff_test, depth_train, depth_val, depth_test = split_train_test_val(
        trials_df=trials_df, 
        test_size=0.2, 
        random_state=42, 
        trial_average=trial_average)
    
    # only select cells
    dff_train = dff_train[:, iscell]
    dff_val = dff_val[:, iscell]
    dff_test = dff_test[:, iscell]

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
