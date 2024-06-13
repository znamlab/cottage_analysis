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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import flexiznam as flz
from cottage_analysis.analysis import spheres, common_utils, find_depth_neurons
from cottage_analysis.pipelines import pipeline_utils


def rolling_average(arr, window, axis=0):
    # calculate rolling average along an axis
    if arr.ndim == 1:
        return (
            pd.Series(arr).rolling(window=window, min_periods=1).mean().values
        )
    else:
        return (
            pd.DataFrame(arr).rolling(window=window, axis=axis, min_periods=1).mean().values
        )


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
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
):
    trials_df["dff_stim_rolling"] = trials_df["dff_stim"].apply(
        lambda x: rolling_average(x, window=int(rolling_window * frame_rate), axis=0)
    )
    trials_df["RS_stim_rolling"] = trials_df["RS_stim"].apply(
        lambda x: rolling_average(x, window=int(rolling_window * frame_rate), axis=0)
    )
    trials_df["dff_stim_downsample"] = trials_df["dff_stim_rolling"].apply(
        lambda x: downsample(
            x, factor=int(downsample_window * frame_rate), mode="average"
        )
    )
    trials_df["RS_stim_downsample"] = trials_df["RS_stim_rolling"].apply(
        lambda x: downsample(
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


def split_train_test_val(trials_df, k_folds=5, random_state=42, trial_average=False):
    # train test val split based on trials
    depth_list = np.sort(trials_df.depth.unique())
    if len(trials_df)%len(depth_list) != 0:
        trials_df = trials_df[:-(len(trials_df)%len(depth_list))] 
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
    trials_df["frame_num_prev_trials"] = trials_df["frame_num"].shift(1).fillna(0).cumsum().astype("int")
    trials_df["frame_indices"] = trials_df.apply(
        lambda x: (np.arange(x["frame_num"])+x["frame_num_prev_trials"]).astype("int"), axis=1
    )
    
    # train test val split
    depth_label = np.hstack(trials_df["depth_label"].values)

    dff_train_all, dff_val_all, dff_test_all, depth_train_all, depth_val_all, depth_test_all = [], [], [], [], [], []
    if k_folds == 1:
        test_size=0.2
        sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=test_size, random_state=random_state)
    else:
        test_size = 1/k_folds
        sss = StratifiedKFold(n_splits=k_folds, random_state=random_state, shuffle=True)
        
    # Train test split
    test_frame_indices = []
    test_index_all = []
    train_index_temp_all = []
    for fold, (train_index_temp, test_index) in enumerate(sss.split(np.arange(len(depth_label)), depth_label)):
        depth_label_train = depth_label[train_index_temp]
        dff_test = np.vstack(trials_df.iloc[test_index][dff_col].values)
        depth_test = np.hstack(trials_df.iloc[test_index][depth_col].values)
        dff_test_all.append(dff_test)
        depth_test_all.append(depth_test) 
        test_index_all.append(test_index)
        train_index_temp_all.append(train_index_temp)
        test_frame_indices.append(np.hstack(trials_df.iloc[test_index]["frame_indices"].values))
        assert(len(test_frame_indices[fold]) == dff_test.shape[0]), "ERROR: Test indices do not match."
        
    # Train val split
    train_index_all, val_index_all = [], []
    for fold, (train_index, val_index) in enumerate(sss.split(np.arange(len(depth_label_train)), depth_label_train)):
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
        all = np.sort(np.hstack([train_index_all[i], test_index_all[i], val_index_all[i]]))
        assert(len(np.unique(all)) == len(trials_df)), "ERROR: There are repetitions in the train, val, test sets."
        
    return trials_df, dff_train_all, dff_val_all, dff_test_all, depth_train_all, depth_val_all, depth_test_all, test_frame_indices, dff_col, depth_col


def svm_classifier_hyperparam_tuning(
    X_train,
    y_train,
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
                clf = SVC(C=C, gamma=gamma, kernel=kernel)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                acc = accuracy_score(y_val, y_pred)

                if acc > best_acc:
                    best_acc = acc
                    best_params = {"C": C, "gamma": gamma}
        print(f"Best C: {best_params['C']}, Best gamma: {best_params['gamma']}")
        
    return best_params


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
        clf = SVC(C=best_params["C"], kernel=kernel)
    else:
        clf = SVC(C=best_params["C"], gamma=best_params["gamma"], kernel=kernel)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])
    clf.fit(X_train_val, y_train_val)

    # Test on the test set
    y_pred = clf.predict(X_test)

    return clf, y_pred


def preprocess_data(    
    trials_df,
    flexilims_session,
    session_name,
    closed_loop=1,
    trial_average=False,
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
    random_state=42,
    kernel="linear",
    k_folds=5,):
    # set test_size:
    if k_folds == 1:
        test_size = 0.2
    else:
        test_size = 1/k_folds
    
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
        frame_rate=int(frame_rate),
        downsample_window=downsample_window,
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
        
    # split train test val (test 0.2, val 0.2, train 0.6)
    trials_df, dff_train_all, dff_val_all, dff_test_all, depth_train_all, depth_val_all, depth_test_all, test_frame_indices, dff_col, depth_col = split_train_test_val(
        trials_df=trials_df, 
        k_folds=k_folds,
        random_state=random_state, 
        trial_average=trial_average)
    
    return (trials_df, (dff_train_all, dff_val_all, dff_test_all, depth_train_all, depth_val_all, depth_test_all), (iscell, depth_list), (test_frame_indices, dff_col, depth_col))


def depth_decoder(
    trials_df,
    flexilims_session,
    session_name,
    closed_loop=1,
    trial_average=False,
    rolling_window=0.5,
    frame_rate=15,
    downsample_window=0.5,
    random_state=42,
    kernel="linear",
    Cs=[0.1, 1, 10],
    gammas=[1, 0.1, 0.01],
    k_folds=5,
):
            
    # choose closedloop or openloop
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    
    # prepare data
    (
        trials_df,
        (dff_train_all, dff_val_all, dff_test_all, depth_train_all, depth_val_all, depth_test_all),
        (iscell, depth_list), 
        (test_frame_indices, dff_col, depth_col)
     ) = preprocess_data(
        trials_df=trials_df,
        flexilims_session=flexilims_session,
        session_name=session_name,
        closed_loop=closed_loop,
        trial_average=trial_average,
        rolling_window=rolling_window,
        frame_rate=frame_rate,
        downsample_window=downsample_window,
        random_state=random_state,
        kernel=kernel,
        k_folds=k_folds,
    )

    # loop through each fold
    y_test_all = np.hstack(trials_df[depth_col])
    y_preds_all = np.zeros_like(y_test_all)
    
    if kernel == "linear":
        best_params_all = {
            "C": [],
        }
    else:
        best_params_all = {
            "C": [],
            "gamma": [],
        }
    
    for i in range(k_folds):
        print(f"Fitting fold {i+1}...")
        # only select current fold and cells
        dff_train = dff_train_all[i][:, iscell]
        dff_val = dff_val_all[i][:, iscell]
        dff_test = dff_test_all[i][:, iscell]
        depth_train = depth_train_all[i]
        depth_val = depth_val_all[i]
        depth_test = depth_test_all[i]
        
        # get best hyperparameters for this fold
        best_params = svm_classifier_hyperparam_tuning(
            X_train=dff_train,
            y_train=depth_train,
            X_val=dff_val,
            y_val=depth_val,
            class_labels=np.arange(len(depth_list)),
            kernel=kernel,
            Cs=Cs,
            gammas=gammas,
        )
        best_params_all["C"].append(best_params["C"])
        if kernel != "linear":
            best_params_all["gamma"].append(best_params["gamma"])
        
        # train and test on the current fold (train on combined train and val data)
        clf, y_pred = test_svm_classifier(
            X_train=dff_train,
            y_train=depth_train,
            X_val=dff_val,
            y_val=depth_val,
            X_test=dff_test,
            y_test=depth_test,
            class_labels=np.arange(len(depth_list)),
            kernel=kernel,
            best_params=best_params,
        )
        y_preds_all[test_frame_indices[i]] = y_pred
    
    # calculate the accuracy on all test data   
    acc= accuracy_score(y_test_all, y_preds_all)
    conmat = confusion_matrix(y_test_all, y_preds_all, labels=np.arange(len(depth_list)))
        
    return acc, conmat, best_params_all, y_test_all, y_preds_all, trials_df


def find_acc_speed_bins(trials_df, speed_bins, y_test, y_preds, continuous_still=True, still_thr=0.05, still_time=1, frame_rate=15):
    acc_speed_bins = []
    conmat_speed_bins = []
    rs_arr = np.hstack(trials_df["RS_stim_downsample"])
    if continuous_still:
        idx =  common_utils.find_thresh_sequence(
                            array=rs_arr,
                            threshold_max=still_thr,
                            length=int(still_time * frame_rate),
                            shift=int(still_time * frame_rate),
                        )
        acc_speed_bins.append(accuracy_score(y_test[idx], y_preds[idx]))
        conmat_speed_bins.append(confusion_matrix(y_test[idx], y_preds[idx]))
        print(f"Still accuracy: {accuracy_score(y_test[idx], y_preds[idx])}")
    for i in range(len(speed_bins)-1):
        idx = (rs_arr >= speed_bins[i]) & (rs_arr < speed_bins[i+1])
        acc_speed_bins.append(accuracy_score(y_test[idx], y_preds[idx]))
        conmat_speed_bins.append(confusion_matrix(y_test[idx], y_preds[idx]))
        print(f"Speed bin {speed_bins[i]} - {speed_bins[i+1]} accuracy: 
              {accuracy_score(y_test[idx], y_preds[idx])}")
    return acc_speed_bins, conmat_speed_bins