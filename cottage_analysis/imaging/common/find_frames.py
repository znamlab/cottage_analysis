#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:36:04 2021

@author: hey2

Find frames for visual stimulation based on photodiode signal

"""

#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from scipy.signal import find_peaks



#%%
def find_VS_frames(photodiode_df, frame_rate=144, upper_thr=250, lower_thr=50, plot=False, plot_start=0, plot_range=2000, plot_dir=None):
    # Get elapsed time
    photodiode_df['ElapsedTime'] = None
    photodiode_df['ElapsedTime'] = photodiode_df.HarpTime - photodiode_df.HarpTime[0]
    photodiode_df['ElapsedTime'].iloc[0] = 0

    elapsed_time = np.array(photodiode_df.ElapsedTime).reshape(-1)
    photodiode = np.array(photodiode_df.Photodiode).reshape(-1)

    # Find peaks of photodiode
    distance = int(1 / frame_rate * 2 / 0.001)
    high_peaks, _ = find_peaks(photodiode, height=upper_thr, distance=distance)
    first_frame = high_peaks[0]
    low_peaks, _ = find_peaks(-photodiode, height=-lower_thr, distance=distance)
    low_peaks = low_peaks[low_peaks > first_frame]

    # Get rid of framedrops
    photodiode_df['FramePeak'] = None
    photodiode_df.FramePeak.iloc[high_peaks] = 1
    photodiode_df.FramePeak.iloc[low_peaks] = 0
    photodiode_df_simple = photodiode_df[photodiode_df.FramePeak.notnull()]
    photodiode_df_simple = photodiode_df_simple[photodiode_df_simple.FramePeak.diff() != 0]
    photodiode_df_simple['VisStim_Frame'] = np.arange(len(photodiode_df_simple))
    photodiode_df_simple = photodiode_df_simple.drop(columns=['FramePeak'])


    if plot:
        plt.figure()
        plt.plot(elapsed_time[plot_start:(plot_start + plot_range)], photodiode[plot_start:(plot_start + plot_range)])
        all_frame_idxs = photodiode_df_simple.index.values.reshape(-1)
        take_start = np.argmin(np.abs(all_frame_idxs - plot_start))
        take_stop = np.argmin(np.abs(all_frame_idxs - plot_start - plot_range))
        take_idxs = all_frame_idxs[take_start:take_stop]

        plot_peaks = np.intersect1d(all_frame_idxs, (np.arange(plot_start, (plot_start + plot_range), step=1)))
        # plt.figure()
        plt.plot(photodiode_df_simple.loc[take_idxs, 'ElapsedTime'], photodiode_df_simple.loc[take_idxs, 'Photodiode'])
        plt.plot(elapsed_time[plot_peaks], photodiode[plot_peaks], "x")
        plt.plot(elapsed_time[plot_start:(plot_start + plot_range)],
                 np.zeros_like(photodiode[plot_start:(plot_start + plot_range)]) + upper_thr, "--", color="gray")
        plt.plot(elapsed_time[plot_start:(plot_start + plot_range)],
                 np.zeros_like(photodiode[plot_start:(plot_start + plot_range)]) + lower_thr, "--", color="gray")
        plt.xlabel('Time(s)')
        plt.savefig(plot_dir+'Frame_finder_check.png')

    return photodiode_df_simple



def find_imaging_frames(harp_message, frame_number, exposure_time=0.015, register_address=32):
    frame_triggers = harp_message[harp_message.RegisterAddress == register_address]
    frame_triggers = frame_triggers.rename(columns={"Timestamp": "HarpTime"}, inplace=False)
    frame_triggers['HarpTime_diff'] = frame_triggers.HarpTime.diff()

    frame_triggers['Exposure'] = np.nan
    frame_triggers.Exposure.loc[
        frame_triggers[(np.abs(frame_triggers['HarpTime_diff'] - exposure_time) <= 0.0002)].index.values] = 1
    frame_triggers = frame_triggers[frame_triggers.Exposure == 1]
    frame_triggers['ImagingFrame'] = np.arange(len(frame_triggers))
    if len(frame_triggers[frame_triggers.Exposure == 1]) == frame_number:
        frame_triggers = frame_triggers
    elif ((len(frame_triggers[frame_triggers.Exposure == 1]) - frame_number) == 1):
        frame_triggers = frame_triggers[:-1]
        print('WARNING: SAVED VIDEO FRAME IS 1 FRAME LESS THAN FRAME TRIGGERS!!!')
    else:
        print('ERROR: FRAME NUMBER NOT CORRECT!!!')
    frame_triggers = frame_triggers.drop(columns=['HarpTime_diff', 'Exposure', 'RegisterAddress'])

    return frame_triggers

