#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:09:09 2021

@author: hey2

Load all the data needed for widefield & 2p imaging analysis 
Format the data in desired format
"""


#%% Import packages
import set_filepaths
import pandas as pd
import numpy as np
import os


#%% Get all file paths from set_filepaths.py
filepaths_dict = set_filepaths.get_filepaths()

#%% Load data files & Format data files

# Read timestamp file of wf videos
def get_wf_video_timestamps(filepaths_dict):
    '''
    Read timestamp file of widefield videos. 
    Format the data into a dataframe containing cols: PylonTimestamp, BonsaiTimestamp, PylonElapsedTime, FrameIndex

    Parameters
    ----------
    filepaths_dict : dict
        A dict containing all filepaths

    Returns
    -------
    wf_video_timestamps : dataframe
        A dataframe containing timestamps of wf videos. 
        cols: PylonTimestamp, BonsaiTimestamp, PylonElapsedTime, FrameIndex (actual frame number including the dropped frames)

    '''
    
    # Read csv file 
    filepath = filepaths_dict['wf_video_timestamps']
    assert os.path.isfile(filepath)
    wf_video_timestamps = pd.read_csv(filepath,sep=',')
    
    # Use Pylon timestamp instead of Bonsai one!! (Bonsai takes into account the time of processing!)
    wf_video_timestamps['PylonElapsedTime'] = (wf_video_timestamps['PylonTimestamp'] - wf_video_timestamps.loc[0,'PylonTimestamp'])/1e9  #in seconds
    
    # Find frame number (including the dropped frames)
    median_frame_time = np.nanmedian(wf_video_timestamps['PylonTimestamp'].diff()/1e9)
    wf_video_timestamps['FrameIndex'] = round(wf_video_timestamps['PylonElapsedTime']/median_frame_time)
    
    # Returns
    return wf_video_timestamps

