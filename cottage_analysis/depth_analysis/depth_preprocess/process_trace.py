#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 14:08:41 2021

@author: hey2

Functions to process extracted fluorescence trace from suite2p
"""
import numpy as np
from sklearn import mixture


def calculate_dFF(f, mode="gmm", n_components=2, verbose=True):
    """
    Calculate dF/F from raw fluorescence trace.

    :param f: np.ndarray, shape nrois x time, raw fluorescence trace for all rois extracted from suite2p
    :param mode: str, 'average' or 'gmm', default 'gmm'
    :param n_components: int, number of components for GMM. default 2.
    :return: dffs: np.ndarray, shape nrois x time, dF/F for all rois extracted from suite2p
    """
    if mode == "average":
        f0 = np.average(f, axis=1).reshape(-1, 1)
    elif mode == "gmm":
        f0 = np.zeros(f.shape[0])
        for i in range(f.shape[0]):
            gmm = mixture.GaussianMixture(
                n_components=n_components, random_state=42
            ).fit(f[i].reshape(-1, 1))
            gmm_means = np.sort(gmm.means_[:, 0])
            f0[i] = gmm_means[0]
            if verbose:
                if i % 100 == 0:
                    print(i, flush=True)
        f0 = f0.reshape(-1, 1)
    dffs = (f - f0) / f0
    return dffs
