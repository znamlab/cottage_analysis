#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:30:31 2021

@author: hey2

Load and format loggers for camera timestamps, vis-stim frames, parameters

"""


#%% Import packages
import os
import numpy as np
import pandas as pd



#%% Load loggers

def load_csv(filepath):
    '''
    Load csv files (camera timestamp files, frame logger, param logger)

    Parameters
    ----------
    filepath : string
        Full filepath

    Returns
    -------
    data_df : Dataframe

    '''
    assert os.path.isfile(filepath)
    
    data_df = pd.read_csv(filepath,sep=',')
    
    
    # Returns
    return data_df



#%% Format logger dataframes

def format_camera_timestamps(cam_timestamps):
    '''
    Format dataframe for camera timestamps
    df columns = 'Frame','Timestamp_zeroed' (only take the Pylon Timestamp, don't used the Bonsai timestamp)

    Parameters
    ----------
    cam_timestamps : Dataframe
        Loaded dataframe from camera_timestamp.csv

    Returns
    -------
    formatted_df : Dataframe
        Formatted dataframe for camera timestamps

    '''
    formatted_df = pd.DataFrame(columns=['Frame','Timestamp_zeroed'])  
    
    # Calculate elapsed time from the start of camera trigger
    # Use Pylon timestamp instead of Bonsai one!! (Bonsai takes into account the time of processing!)
    formatted_df['Timestamp_zeroed'] = (cam_timestamps['PylonTimestamp'] - cam_timestamps.loc[0,'PylonTimestamp'])/1e9
    
    # Find frame number (including the dropped frames)
    median_frame_time = np.nanmedian(cam_timestamps['PylonTimestamp'].diff()/1e9)
    formatted_df['Frame'] = round(formatted_df['Timestamp_zeroed']/median_frame_time)
    
    
    # Returns
    return formatted_df



def format_VS_frame_logger(VS_frame_logger):
    '''
    Format dataframe for VisStim frame_logger
    df columns = 'HarpTime','Timestamp_zeroed'
    
    Parameters
    ----------
    VS_frame_logger : Dataframe
        Loaded dataframe from VS frame_logger.csv

    Returns
    -------
    formatted_df : Dataframe
        Formatted dataframe for VS_frame_logger

    '''
    formatted_df = pd.DataFrame(columns=['Frame','HarpTime','Timestamp_zeroed'])  

    formatted_df['Frame'] =  VS_frame_logger['Frame']    
    formatted_df['HarpTime'] =  VS_frame_logger['HarpTime']   
    formatted_df['Timestamp_zeroed'] = VS_frame_logger['HarpTime'] - VS_frame_logger.loc[0,'HarpTime']
    
    
    # Returns
    return formatted_df
    


def format_VS_param_logger(VS_param_logger, VS_frame_logger, which_protocol):
    '''
    Format dataframe for VisStim param_logger
    df columns = 'HarpTime','Timestamp_zeroed' (calculated from the start of frame logger!!), Params...
    
    Parameters
    ----------
    VS_param_logger : Dataframe
        Loaded dataframe from VS param_logger.csv
    VS_frame_logger : Dataframe
        Loaded dataframe from VS frame_logger.csv
    which_protocol: string
        The Vis-Stim protocol used ('Retinotopic','Fourier')

    Returns
    -------
    formatted_df : Dataframe
        Formatted dataframe for VS_param_logger

    '''
    formatted_df = pd.DataFrame(columns=['HarpTime','Timestamp_zeroed'])  

    formatted_df['HarpTime'] =  VS_param_logger['HarpTime']   
    formatted_df['Timestamp_zeroed'] = VS_param_logger['HarpTime'] - VS_frame_logger.loc[0,'HarpTime']
    
    # Assign params according to stimulation protocol 
    if which_protocol == 'Retinotopic':
        formatted_df['Xdeg'] =  VS_param_logger['Xdeg']   
        formatted_df['Ydeg'] =  VS_param_logger['Ydeg']   
        formatted_df['Angle'] =  VS_param_logger['Angle']
        
    elif which_protocol == 'Fourier':
        formatted_df['BarID'] =  VS_param_logger['BarID']   
        formatted_df['LocationX'] =  VS_param_logger['LocationX']   
        formatted_df['LocationY'] =  VS_param_logger['LocationY'] 
        formatted_df['Angle'] =  VS_param_logger['Angle']  
    
    
    # Returns
    return formatted_df