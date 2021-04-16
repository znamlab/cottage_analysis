#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:18:27 2021

@author: hey2

Set file paths for all data needed to be loaded
"""

#%% Set-up

general_path_dict = {
    'root': '/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision',
    'mouse_name': 'PZAH4.1c',
    'session_dir': 'S20210406',
    'recording_dir': 'R184923'
    }

filenames_dict = {
    'wf_video': 'wf_camera_data',
    'wf_video_timestamps': 'wf_camera_timestamps',
    'wf_video_metadata': 'wf_camera_metadata',
    'harp': 'harpmessage',
    'VS_frame_logger' : 'FrameLog',
    'VS_param_logger_Retino' : 'Retinotopic_NewParams'
    }


# Just for testing purpose
def get_general_path_dict(general_path_dict):
    return general_path_dict

def get_filenames_dict(filenames_dict):
    return filenames_dict

    
        
#%% Get filepaths of all files
def get_filepaths(general_path_dict=general_path_dict,filenames_dict=filenames_dict):
    '''
    Generate filepaths for all data needed to be loaded 

    Parameters
    ----------
    general_path_dict : dict
        A dict containing general path info 
    filenames_dict : dict
        A dict containing specific filename info for each file

    Returns
    -------
    filepaths_dict : dict
        A dict containing filepath for all files needed to be loaded 

    '''
    
    filepaths_dict = {}
    
    
    root = general_path_dict['root']
    mouse_name = general_path_dict['mouse_name']
    session_dir = general_path_dict['session_dir']
    recording_dir = general_path_dict['recording_dir']
    
    # Path for widefield videos
    wf_video_path = root + '/' + mouse_name + '/' + session_dir + '/' + recording_dir + '/'\
        + filenames_dict['wf_video'] + '.bin'
    
    # Path for widefield video recorded timestamps (Pylon & Bonsai)
    wf_video_timestamps_path = root + '/' + mouse_name + '/' + session_dir + '/' + recording_dir + '/'\
        + filenames_dict['wf_video_timestamps'] + '.csv'
    
    # Path for widefield video metadata
    wf_video_metadata = root + '/' + mouse_name + '/' + session_dir + '/' + recording_dir + '/'\
        + filenames_dict['wf_video_metadata'] + '.txt'
        
    # Path for harp message
    harp_path = root + '/' + mouse_name + '/' + session_dir + '/' + 'ParamLog/' + recording_dir + '/' \
    + mouse_name + '_' + filenames_dict['harp'] + '_' + session_dir + '_' + recording_dir + '.bin'
    
     # Path for Vis-stim frame logger 
    VS_frame_logger_path = root + '/' + mouse_name + '/' + session_dir + '/' + 'ParamLog/' + recording_dir + '/' \
    + mouse_name + '_' + filenames_dict['VS_frame_logger'] + '_' + session_dir + '_' + recording_dir + '.csv'

    # Path for Vis-stim param logger Retino
    VS_param_logger_Retino_path = root + '/' + mouse_name + '/' + session_dir + '/' + 'ParamLog/' + recording_dir + '/' \
    + mouse_name + '_' + filenames_dict['VS_param_logger_Retino'] + '_' + session_dir + '_' + recording_dir + '.csv'


    # Save all paths to dict
    filepaths_dict = {
    'wf_video': wf_video_path,
    'wf_video_timestamps': wf_video_timestamps_path,
    'wf_video_metadata': wf_video_metadata,
    'harp': harp_path,
    'VS_frame_logger' : VS_frame_logger_path ,
    'VS_param_logger_Retino' : VS_param_logger_Retino_path 
    }


    
    #Returns
    return filepaths_dict

 