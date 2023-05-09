import functools

print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import scipy
from scipy.optimize import curve_fit

import flexiznam as flz

from cottage_analysis.depth_analysis.filepath import generate_filepaths
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.depth_analysis.analysis import common_utils


MIN_SIGMA=0.1
def twoD_Gaussian(xy_tuple, log_amplitude, xo, yo, log_sigma_x2, log_sigma_y2, theta, offset):
    (x,y) = xy_tuple
    sigma_x_sq = np.exp(log_sigma_x2) + MIN_SIGMA # 0.25
    sigma_y_sq = np.exp(log_sigma_y2) + MIN_SIGMA  # 0.25
    amplitude = np.exp(log_amplitude)
    a = (np.cos(theta)**2)/(2*sigma_x_sq) + (np.sin(theta)**2)/(2*sigma_y_sq)
    b = (np.sin(2*theta))/(4*sigma_x_sq) - (np.sin(2*theta))/(4*sigma_y_sq)
    c = (np.sin(theta)**2)/(2*sigma_x_sq) + (np.cos(theta)**2)/(2*sigma_y_sq)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            +c*((y-yo)**2)))
    return g


def fit_gaussian_blob(project, mouse, session, protocol="SpheresPermTubeReward", rs_thr=0.01, param_range={'rs_min':0.005, 'rs_max':5, 'of_min':0.03, 'of_max':3000}, batch_num=5):
    # Load files
    root = Path(flz.PARAMETERS["data_root"]["processed"])
    session_folder = root / project / mouse / session

    with open(session_folder / "plane0/trials_df.pickle", "rb") as handle:
        trials_df = pickle.load(handle)
    with open(session_folder / "plane0/neurons_df.pickle", "rb") as handle:
        neurons_df = pickle.load(handle)
    
    neurons_df['preferred_RS_closed_loop'] = np.nan #m/s
    neurons_df['preferred_OF_closed_loop'] = np.nan #rad/s
    neurons_df['gaussian_blob_popt_closed_loop'] = [[np.nan]] *len(neurons_df) # !! Calculated with RS in cm and OF in degrees/s
    neurons_df['gaussian_blob_rsq_closed_loop'] = np.nan
    
    neurons_df['preferred_RS_open_loop_actual'] = np.nan #m/s
    neurons_df['preferred_OF_open_loop_actual'] = np.nan #rad/s
    neurons_df['gaussian_blob_popt_open_loop_actual'] = [[np.nan]] *len(neurons_df) # !! Calculated with RS in cm and OF in degrees/s
    neurons_df['gaussian_blob_rsq_open_loop_actual'] = np.nan
    
    neurons_df['preferred_RS_open_loop_virtual'] = np.nan #m/s
    neurons_df['preferred_OF_open_loop_virtual'] = np.nan #rad/s
    neurons_df['gaussian_blob_popt_open_loop_virtual'] = [[np.nan]] *len(neurons_df) # !! Calculated with RS in cm and OF in degrees/s
    neurons_df['gaussian_blob_rsq_open_loop_virtual'] = np.nan
    
    # Determine whether this session has open loop or not
    if len(trials_df.closed_loop.unique())==2:
        protocols = [protocol, f"{protocol}Playback"]
    elif len(trials_df.closed_loop.unique())==1:
        protocols = [protocol]

    # Loop through all protocols
    for iprotocol in range(len(protocols)):
        print(
            f"---------Process protocol {iprotocol+1}/{len(protocols)}---------",
            flush=True,
        )
        if 'Playback' in protocols[iprotocol]:
            is_closedloop = 0
            protocol_sfx = 'open_loop'
        else:
            is_closedloop = 1
            protocol_sfx = 'closed_loop'
        trials_df_protocol = trials_df[trials_df.closed_loop==is_closedloop]
        
        # Concatenate arrays of RS/OF/dff from all trials together
        rs = np.concatenate(trials_df_protocol['RS_stim'].values)
        rs_eye = np.concatenate(trials_df_protocol['RS_eye_stim'].values)
        of = np.concatenate(trials_df_protocol['OF_stim'].values)
        dff = np.concatenate(trials_df_protocol['dff_stim'].values, axis=1)

        # Take out the values where running is below a certain threshold
        running = (rs>rs_thr) & (rs_eye>rs_thr) & (~np.isnan(of)) # !!! OF has a small number of frame = nan, investigate synchronisation.py
        rs = rs[running]
        rs_eye = rs_eye[running]
        of = of[running]
        dff = dff[:,running]

        # Fit data to 2D gaussian function
        if is_closedloop:
            Xs = [np.log(rs * 100)] # m-->cm
        else:
            Xs = [np.log(rs * 100), np.log(rs_eye * 100)] # m-->cm
        Y = np.log(np.degrees(of)) # rad-->deg
        rs_min = param_range['rs_min']*100 # m-->cm
        rs_max = param_range['rs_max']*100 # m-->cm
        of_min = param_range['of_min'] # degrees/s
        of_max = param_range['of_max'] # degrees/s
        
        for iX,X in enumerate(Xs):
            if is_closedloop:
                rs_type=''
            else:
                if iX==0:
                    rs_type='_actual'
                else:
                    rs_type='_virtual'
            print(
            f"Fitting {protocol_sfx}{rs_type} running...",
            flush=True,
            )
            for iroi in tqdm(range(dff.shape[0])):
                popt_arr = []
                rsq_arr = []
                np.random.seed(42)
                for ibatch in range(batch_num):
                    mu0 = 0
                    sigma0 = 1

                    p0 = np.concatenate((np.random.normal(mu0, sigma0, size=1),
                                            np.atleast_1d(
                                                np.min([np.abs(np.random.normal(mu0, sigma0, size=1)), np.log(rs_min)])),
                                            np.atleast_1d(
                                                [np.min([np.abs(np.random.normal(mu0, sigma0, size=1)) + np.log(of_min)])]),
                                            np.random.normal(mu0, sigma0, size=2),
                                            np.atleast_1d(
                                                np.max([np.abs(np.random.normal(mu0, sigma0, size=1)), np.radians(90)])),
                                            np.random.normal(mu0, sigma0, size=1)))
                    popt, pcov = curve_fit(twoD_Gaussian, (X,Y), dff[iroi, :], maxfev=100000,
                                            bounds=([-np.inf, np.log(rs_min), np.log(of_min), -np.inf, -np.inf, 0, -np.inf],
                                                    [np.inf, np.log(rs_max), np.log(of_max), np.inf, np.inf, np.radians(90),
                                                    np.inf])
                                            )

                    dff_fit = twoD_Gaussian(np.array([X,Y]), *popt)
                    r_sq = common_utils.calculate_R_squared(dff, dff_fit)
                    popt_arr.append(popt)
                    rsq_arr.append(r_sq)
                idx_best = np.argmax(np.array(rsq_arr))
                popt_best = popt_arr[idx_best]
                rsq_best = rsq_arr[idx_best]
                
                neurons_df.loc[iroi, f'preferred_RS_{protocol_sfx}{rs_type}'] = np.exp(popt_best[1])/100 #m
                neurons_df.loc[iroi, f'preferred_OF_{protocol_sfx}{rs_type}'] = np.radians(np.exp(popt_best[2])) #rad/s
                neurons_df[f'gaussian_blob_popt_{protocol_sfx}{rs_type}'].iloc[iroi] = popt_best # !! Calculated with RS in cm and OF in degrees/s
                neurons_df.loc[iroi, f'gaussian_blob_rsq_{protocol_sfx}{rs_type}'] = rsq_best
                
    with open(session_folder / "plane0/neurons_df.pickle", "wb") as handle:
        pickle.dump(neurons_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return neurons_df
    