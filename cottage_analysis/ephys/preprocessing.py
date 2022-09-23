"""
Preprocessing

Mostly renaming and synchronising
"""
import numpy as np
import pandas as pd

from cottage_analysis.imaging.common import find_frames


def preprocess_exp(data, plot_dir=None):
    """Run all mandatory preprocessing function

    Namely:
    - sync_harp2onix
    - find_frame_start_hf

    Args:
        data (dict): output of onix.load_onix_recording
        plot_dir (Path or str): path to save frame detection figure

    Returns:
        data (dict): updated data dict
    """
    data['harp_message'], data['harp2onix'] = sync_harp2onix(data['harp_message'])
    data['hf_frames'] = find_frame_start_hf(data['harp_message'], plot_dir=plot_dir)
    return data


def sync_harp2onix(harp_message):
    """Synchronise harp to onix using clock digital input

    Args:
        harp_message: output of load_harp

    Returns:
        harp_message: same as input, with two new fields: 'analog_time_onix' and
            'digital_time_onix'
        harp2onix: conversion function
    """

    # find when clock switches on
    onix_clock = np.diff(np.hstack([0, harp_message['onix_clock']])) == 1
    onix_clock_in_harp = harp_message['digital_time'][onix_clock]
    # assume a perfect 100Hz since so far it has been good
    real_time = np.arange(len(onix_clock_in_harp)) * 10e-3
    delta_clock = np.diff(onix_clock_in_harp)
    if np.nanmax(np.abs(delta_clock - 0.01)) * 1000 > 5:
        raise ValueError('Onix clock deviation from 100Hz!')

    # linear regression:
    t0 = onix_clock_in_harp[0]
    slope = np.nanmean(real_time[1:] / (onix_clock_in_harp[1:] - t0))

    def harp2onix(data):
        """Convert harp timestamp in onix time"""
        return (data - t0) * slope

    harp_message['analog_time_onix'] = harp2onix(harp_message['analog_time'])
    harp_message['digital_time_onix'] = harp2onix(harp_message['digital_time'])
    return harp_message, harp2onix


def find_frame_start_hf(harp_message, frame_rate=144, upper_thr=0.5, lower_thr=-0.5,
                        plot=True, plot_start=1000, plot_range=200,  plot_dir=None):
    """Wrapper around find_frames.find_VS_frames for ONIX

    Args:
        harp_message (dict): ouput of load_harp
        photodiode_df (pd.DataFrame): dataframe with `HarpTime` and `Photodiode` fields
        frame_rate (float): Expected frame rate. Peaks separated by less than half a
                            frame will be ignored
        upper_thr (float): Photodiode value above which the quad is considered white
        lower_thr (float): Photodiode value below which the quad is considered black
        photodiode_sampling (float): Sampling rate of the photodiode signal in Hz
        plot (bool): Should a summary figure be generated?
        plot_start (int): sample to start the plot
        plot_range (int): samples to plot after plot_start
        plot_dir (Path or str): directory to save the figure

    Returns:
        frame_starts (pd.DataFrame): a dataframe containing detected frame timing.
            It contains:
                - 'Photodiode': photodiode value at peak
                - 'onix_time': time of peak
                - 'ElapsedTime': time of peak since first peak
                - 'VisStim_Frame': peak index

    """
    zsc_hf_pd = harp_message['photodiode']
    zsc_hf_pd = (zsc_hf_pd - np.nanmean(zsc_hf_pd)) / np.nanstd(zsc_hf_pd)
    photodiode_df = pd.DataFrame(dict(Photodiode=zsc_hf_pd,
                                      HarpTime=harp_message['analog_time_onix']))
    frame_starts = find_frames.find_VS_frames(photodiode_df,
                                              frame_rate=frame_rate,
                                              upper_thr=upper_thr,
                                              lower_thr=lower_thr, plot=plot,
                                              plot_start=plot_start,
                                              plot_range=plot_range,
                                              plot_dir=plot_dir)
    frame_starts.rename(columns=dict(HarpTime='onix_time'), inplace=True)
    return frame_starts