#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:49:40 2021

@author: hey2

Test cottage_analysis.imaging.common.io_formatting
"""

from cottage_analysis.imaging.common import imaging_loggers_formatting as logger_format
from cottage_analysis.imaging.common import align_timestamps
import os
import numpy as np
import pandas as pd


#%%
# Set-up
general_path_dict = {
    'root': 'tests/test_data',
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
wf_camera_timestamps = logger_format.load_csv(filepath_dict['wf_camera_timestamps'])
VS_frame_logger = logger_format.load_csv(filepath_dict['VS_frame_logger'])
VS_param_logger_Retinotopic = logger_format.load_csv(filepath_dict['VS_param_logger_Retinotopic'])

# Format dataframes
wf_camera_timestamps = logger_format.format_camera_timestamps(wf_camera_timestamps)
VS_frame_logger = logger_format.format_VS_frame_logger(VS_frame_logger)
VS_param_logger_Retinotopic  = logger_format.format_VS_param_logger(VS_param_logger=VS_param_logger_Retinotopic, \
                                                                    VS_frame_logger=VS_frame_logger,\
                                                                        which_protocol='Retinotopic')

    
# Format dataframes
def test_align_timestamps(df1=wf_camera_timestamps, df2=VS_param_logger_Retinotopic, align_basis='ElapsedTime'):
    DF = align_timestamps.align_timestamps(df1=df1, df2=df2, align_basis=align_basis)
    assert(len(DF)==len(wf_camera_timestamps))
    assert(np.isnan(DF.loc[0,'HarpTime']))
    assert(np.isnan(DF.loc[520,'Xdeg']))
    assert(DF.loc[521,'Xdeg'] == -105)
    assert(DF.loc[521,'Ydeg'] == 30)
    assert(DF.loc[521,'Angle'] == 45)
