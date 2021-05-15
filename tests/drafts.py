#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:30:41 2021

@author: hey2
"""
#%% Clear all
run_clear = True

if run_clear:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    
    
#%% import packages
from cottage_analysis.io_module import harp
from cottage_analysis.io_module.video import io_func as video_io
from cottage_analysis.imaging.common import align_timestamps, io_formatting
import os

# =============================================================================
# fpath = '/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision/Data/PZAH4.1c/S20210406/ParamLog/R184923/PZAH4.1c_harpmessage_S20210406_R184923.bin'
# msg_df = harp.read_message(fpath, verbose=True, valid_addresses=32)
# 
# =============================================================================

import numpy as np
import pandas as pd



#%%
# Set-up
general_path_dict = {
    'root': '/Users/hey2/Desktop/cottage_analysis/tests/test_data',
    'mouse_name': 'test_mouse',
    'session_dir': 'Stest',
    'recording_dir': 'Rtest'
    }

filename_dict = {
    'wf_camera': 'wf_camera_data',
    'wf_camera_timestamps': 'wf_camera_timestamps',
    'wf_camera_metadata': 'wf_camera_metadata',
    'harp': 'harpmessage',
    'VS_frame_logger' : 'FrameLog',
    'VS_param_logger_Retinotopic' : 'Retinotopic_NewParams'
    }


# Get filepaths of all files
def get_filepaths(general_path_dict,filename_dict):
    '''
    Generate filepaths for all data needed to be loaded 

    Parameters
    ----------
    general_path_dict : dict
        A dict containing general path info 
    filename_dict : dict
        A dict containing specific filename info for each file

    Returns
    -------
    filepath_dict : dict
        A dict containing filepath for all files needed to be loaded 

    '''
    
    filepath_dict = {}
    
    
    root = general_path_dict['root']
    mouse_name = general_path_dict['mouse_name']
    session_dir = general_path_dict['session_dir']
    recording_dir = general_path_dict['recording_dir']
    
    # Path for widefield videos
    wf_camera_path = root + '/' + mouse_name + '/' + session_dir + '/' + recording_dir + '/'\
        + filename_dict['wf_camera'] + '.bin'
    #assert(os.path.isfile(wf_camera_path))
    
    # Path for widefield video recorded timestamps (Pylon & Bonsai)
    wf_camera_timestamps_path = root + '/' + mouse_name + '/' + session_dir + '/' + recording_dir + '/'\
        + filename_dict['wf_camera_timestamps'] + '.csv'
    assert(os.path.isfile(wf_camera_timestamps_path))
    
    # Path for widefield video metadata
    wf_camera_metadata = root + '/' + mouse_name + '/' + session_dir + '/' + recording_dir + '/'\
        + filename_dict['wf_camera_metadata'] + '.txt'
    assert(os.path.isfile(wf_camera_metadata))
    
    # Path for harp message
    harp_path = root + '/' + mouse_name + '/' + session_dir + '/' + 'ParamLog/' + recording_dir + '/' \
    + mouse_name + '_' + filename_dict['harp'] + '_' + session_dir + '_' + recording_dir + '.bin'
    assert(os.path.isfile(harp_path))
    
    # Path for Vis-stim frame logger 
    VS_frame_logger_path = root + '/' + mouse_name + '/' + session_dir + '/' + 'ParamLog/' + recording_dir + '/' \
    + mouse_name + '_' + filename_dict['VS_frame_logger'] + '_' + session_dir + '_' + recording_dir + '.csv'
    assert(os.path.isfile(VS_frame_logger_path))
    
    # Path for Vis-stim param logger Retino
    VS_param_logger_Retino_path = root + '/' + mouse_name + '/' + session_dir + '/' + 'ParamLog/' + recording_dir + '/' \
    + mouse_name + '_' + filename_dict['VS_param_logger_Retinotopic'] + '_' + session_dir + '_' + recording_dir + '.csv'
    assert(os.path.isfile(VS_param_logger_Retino_path))

    # Save all paths to dict
    filepath_dict = {
    'wf_camera': wf_camera_path,
    'wf_camera_timestamps': wf_camera_timestamps_path,
    'wf_camera_metadata': wf_camera_metadata,
    'harp': harp_path,
    'VS_frame_logger' : VS_frame_logger_path ,
    'VS_param_logger_Retinotopic' : VS_param_logger_Retino_path 
    }


    
    #Returns
    return filepath_dict



filepath_dict = get_filepaths(general_path_dict=general_path_dict,filename_dict=filename_dict)



#%% Test cottage_analysis.imaging.common : io_formatting
# Load camera timestamps / VS frame logger / VS param logger
wf_camera_timestamps = io_formatting.load_csv(filepath_dict['wf_camera_timestamps'])
VS_frame_logger = io_formatting.load_csv(filepath_dict['VS_frame_logger'])
VS_param_logger_Retinotopic = io_formatting.load_csv(filepath_dict['VS_param_logger_Retinotopic'])

# Format dataframes
wf_camera_timestamps = io_formatting.format_camera_timestamps(wf_camera_timestamps)
VS_frame_logger = io_formatting.format_VS_frame_logger(VS_frame_logger)
VS_param_logger_Retinotopic  = io_formatting.format_VS_param_logger(VS_param_logger=VS_param_logger_Retinotopic, \
                                                                    VS_frame_logger=VS_frame_logger,\
                                                                        which_protocol='Retinotopic')
    
assert(len(wf_camera_timestamps)==42340)
assert({'Frame','Timestamp_zeroed'}.issubset(wf_camera_timestamps.columns))

assert(len(VS_frame_logger)==108215)
assert({'Frame','HarpTime','Timestamp_zeroed'}.issubset(VS_frame_logger.columns))

assert(len(VS_param_logger_Retinotopic)==1566)
assert({'HarpTime','Timestamp_zeroed','Xdeg','Ydeg','Angle'}.issubset(VS_param_logger_Retinotopic.columns))

    
    
#%% Test align_timestamp
# Align timestamp
wf_VS_DF = align_timestamps.align_timestamps(df1=wf_camera_timestamps, df2=VS_param_logger_Retinotopic, align_basis='Timestamp_zeroed')

    

    
    
    
    
   
    
    