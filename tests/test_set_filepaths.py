#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:43:14 2021

@author: hey2

Test cottage_analysis/imaging/common/set_filepaths,py
"""
#%% Import packages
from cottage_analysis.imaging.common import set_filepaths
import os

#%% Set-up
test_general_path_dict = {
    'root': 'tests/test_data',
    'mouse_name': 'test_mouse',
    'session_dir': 'Stest',
    'recording_dir': 'Rtest'
    }

test_filenames_dict = {
    'wf_video': 'wf_camera_data',
    'wf_video_timestamps': 'wf_camera_timestamps',
    'wf_video_metadata': 'wf_camera_metadata',
    'harp': 'harpmessage',
    'VS_frame_logger' : 'FrameLog',
    'VS_param_logger_Retino' : 'Retinotopic_NewParams'
    }


#%% Test funcs
def test_get_filepaths(general_path_dict=test_general_path_dict,filenames_dict=test_filenames_dict):
    
    test_filepaths_dict = set_filepaths.get_filepaths(general_path_dict,filenames_dict)
    assert (os.path.isfile(test_filepaths_dict['VS_param_logger_Retino']))
    
    