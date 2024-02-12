import functools
print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # for pdfs
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
    return pd.DataFrame(arr).rolling(window=window, axis=axis, min_periods=1).mean().values

def downsample_2d(arr, factor, mode="average"):
    # downsample 2d array by factor along axis 0
    end = int(factor * int(arr.shape[0] / factor))
    if mode == "average":
        arr_crop = arr[:end, :].reshape(-1, factor, arr.shape[1])
        arr_mean = np.mean(arr_crop, axis=1).reshape(-1, arr.shape[1])
    elif mode == "skip":
        arr_mean = arr[::factor, :]
    return arr_mean

def fit_svm_classifier(X, y, class_labels, test_size=0.2, random_state=42, kernel="linear", Cs=[0.1, 1, 10], gammas=[1, 0.1, 0.01]):
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
        
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

            if acc > best_acc:
                best_acc = acc
                best_params = {"C": C}
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

    return acc, conmat


def depth_decoder(trials_df, rolling_window=0.5, frame_rate=15, downsample_window=0.5, test_size=0.2, random_state=42, kernel="linear", Cs=[0.1, 1, 10]):
    trials_df["dff_stim_rolling"] = trials_df["dff_stim"].apply(lambda x: rolling_average_2d(x, window=int(rolling_window*frame_rate), axis=0)) 
    trials_df["dff_stim_downsample"] = trials_df["dff_stim_rolling"].apply(lambda x: downsample_2d(x, factor=int(downsample_window*frame_rate), mode="average"))
    depth_list = np.sort(trials_df.depth.unique())
    trials_df["depth_label"] = trials_df["depth"].apply(lambda x: np.where(depth_list==x)[0])
    trials_df["depth_label"] = trials_df.apply(lambda x: np.repeat(x["depth_label"], x["dff_stim_downsample"].shape[0], axis=0), axis=1)

    downsampled_trace = np.vstack(trials_df["dff_stim_downsample"].values)
    depth_trace = np.hstack(trials_df["depth_label"].values)
    
    acc, conmat = fit_svm_classifier(X=downsampled_trace, y=depth_trace, class_labels=np.arange(len(depth_list)), test_size=test_size, random_state=random_state, kernel=kernel, Cs=Cs)
    
    return acc, conmat