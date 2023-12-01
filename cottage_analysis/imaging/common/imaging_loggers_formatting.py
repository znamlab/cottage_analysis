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
    """
    Load csv files (camera timestamp files, frame logger, param logger)

    Parameters
    ----------
    filepath : string
        Full filepath

    Returns
    -------
    data_df : Dataframe

    """
    assert os.path.isfile(filepath)

    data_df = pd.read_csv(filepath, sep=",")

    # Returns
    return data_df


#%% Format logger dataframes


def format_camera_timestamps(cam_timestamps):
    """
    Format dataframe for camera timestamps
    df columns = 'Frame','ElapsedTime' (only take the Pylon Timestamp, don't used the Bonsai timestamp)

    Parameters
    ----------
    cam_timestamps : Dataframe
        Loaded dataframe from camera_timestamp.csv

    Returns
    -------
    formatted_df : Dataframe
        Formatted dataframe for camera timestamps

    """
    formatted_df = pd.DataFrame(columns=["Frame", "ElapsedTime"])

    # Calculate elapsed time from the start of camera trigger
    # Use Pylon timestamp instead of Bonsai one!! (Bonsai takes into account the time of processing!)
    formatted_df["ElapsedTime"] = (
        cam_timestamps["PylonTimestamp"] - cam_timestamps.loc[0, "PylonTimestamp"]
    ) / 1e9

    # Find frame number (including the dropped frames)
    median_frame_time = np.nanmedian(cam_timestamps["PylonTimestamp"].diff() / 1e9)
    formatted_df["Frame"] = round(formatted_df["ElapsedTime"] / median_frame_time)

    # Returns
    return formatted_df


def format_VS_photodiode_logger(photodiode_file):
    """
    Format photodiode logger.

    :param str photodiode_filepath: Filepath for photodiode logger
    :return:
    pd.DataFrame formatted_df: formatted dataframe for photodiode logger
    """

    VS_photodiode_logger = pd.read_csv(photodiode_file, sep=",")
    formatted_df = pd.DataFrame(columns=["HarpTime", "Photodiode"])
    formatted_df["HarpTime"] = VS_photodiode_logger["HarpTime"]
    formatted_df["Photodiode"] = VS_photodiode_logger["Photodiode"]

    formatted_df["ElapsedTime"] = (
        VS_photodiode_logger["HarpTime"] - VS_photodiode_logger.loc[0, "HarpTime"]
    )

    # Returns
    return formatted_df


def format_VS_frame_logger(VS_frame_logger):
    """
    Format dataframe for VisStim frame_logger
    df columns = 'HarpTime','ElapsedTime'

    Parameters
    ----------
    VS_frame_logger : Dataframe
        Loaded dataframe from VS frame_logger.csv

    Returns
    -------
    formatted_df : Dataframe
        Formatted dataframe for VS_frame_logger

    """
    formatted_df = pd.DataFrame(columns=["Frame", "HarpTime", "ElapsedTime"])

    formatted_df["Frame"] = VS_frame_logger["Frame"]
    formatted_df["HarpTime"] = VS_frame_logger["HarpTime"]
    formatted_df["ElapsedTime"] = (
        VS_frame_logger["HarpTime"] - VS_frame_logger.loc[0, "HarpTime"]
    )

    # Returns
    return formatted_df


def format_VS_param_logger(VS_param_file, which_protocol):
    """
    Format visual stimulation parameter logger.

    :param str VS_param_file: Filepath for vis-stim param logger
    :param str which_protocol: 'SpherePermTubeReward', 'Retinotopy', 'Fourier', 'Episodic', 'SphereSparseNoise'
    :return:
    pd.DataFrame formatted_df: formatted dataframe for vis-stim parameter logger
    """
    VS_param_logger = pd.read_csv(VS_param_file, sep=",")
    formatted_df = pd.DataFrame(columns=["HarpTime"])

    formatted_df["HarpTime"] = VS_param_logger["HarpTime"]
    # formatted_df['ElapsedTime'] = VS_param_logger['HarpTime'] - VS_frame_logger.loc[0,'HarpTime']

    # Assign params according to stimulation protocol
    if which_protocol == "Retinotopy":
        formatted_df["Xdeg"] = VS_param_logger["Xdeg"]
        formatted_df["Ydeg"] = VS_param_logger["Ydeg"]
        formatted_df["Angle"] = VS_param_logger["Angle"]

    elif which_protocol == "Fourier":
        formatted_df["BarID"] = VS_param_logger["BarID"]
        formatted_df["LocationX"] = VS_param_logger["LocationX"]
        formatted_df["LocationY"] = VS_param_logger["LocationY"]
        formatted_df["Angle"] = VS_param_logger["Angle"]

    elif which_protocol == "Episodic":
        formatted_df["StimID"] = VS_param_logger["StimID"]
        formatted_df["Azimuth"] = VS_param_logger["Azimuth"]
        formatted_df["Elevation"] = VS_param_logger["Elevation"]
        formatted_df["Angle"] = VS_param_logger["Angle"]

    elif which_protocol == "SphereSparseNoise":
        formatted_df["SphereID"] = VS_param_logger["SphereID"]
        formatted_df["Depth"] = VS_param_logger["Depth"]
        formatted_df["Azimuth"] = VS_param_logger["Azimuth"]
        formatted_df["Elevation"] = VS_param_logger["Elevation"]

    elif which_protocol == "SpheresPermTubeReward":
        formatted_df["SphereID"] = VS_param_logger["SphereID"]
        formatted_df["Depth"] = VS_param_logger["Radius"]
        formatted_df["Theta"] = VS_param_logger["Theta"]
        formatted_df["Z0"] = VS_param_logger["Z0"]
        formatted_df["X"] = VS_param_logger["X"]
        formatted_df["Y"] = VS_param_logger["Y"]

    elif which_protocol == "SpheresPermTubeRewardPlayback":
        formatted_df["SphereID"] = VS_param_logger["SphereID"]
        if "Depth" in VS_param_logger.columns:
            formatted_df["Depth"] = VS_param_logger["Depth"]
        elif "Radius" in VS_param_logger.columns:
            formatted_df["Depth"] = VS_param_logger["Radius"]
        formatted_df["Theta"] = VS_param_logger["Theta"]
        formatted_df["Z0"] = VS_param_logger["Z0"]
        formatted_df["X"] = VS_param_logger["X"]
        formatted_df["Y"] = VS_param_logger["Y"]

    # Returns
    return formatted_df


def format_img_frame_logger(harpmessage_file, register_address=32):
    """
    Format imaging frame trigger logger (harpmessage).

    :param str harpmessage_file: filepath for harpmessage
    :param int register_address: at which address was the frame trigger registered. Default 32.
    :return: pd.DataFrame formatted_df
    """
    if "csv" in str(harpmessage_file):
        harp_message = pd.read_csv(
            harpmessage_file,
            sep=",",
            usecols=["RegisterAddress", "Timestamp", "DataElement0"],
        )
        harp_message = harp_message[harp_message.RegisterAddress == register_address]

        # Find 8 bit correspondance to different channels because the lick signal is also registered on the same channel
        bits = np.array(np.hstack(harp_message.DataElement0.values), dtype="uint8")
        bits = np.unpackbits(bits, bitorder="little")
        bits = bits.reshape((len(harp_message), 8))

        for iport in range(8):
            harp_message[f"Port{iport}"] = bits[:, iport]

        formatted_df = harp_message[harp_message.Port0.diff() != 0]
        formatted_df.rename(columns={"Timestamp": "HarpTime"}, inplace=True)

    elif "npz" in str(harpmessage_file):
        harp_message = np.load(harpmessage_file)
        formatted_df = pd.DataFrame(columns=["HarpTime", "FrameTriggers"])
        formatted_df["HarpTime"] = harp_message["digital_time"]
        formatted_df["FrameTriggers"] = harp_message["frame_triggers"]
        formatted_df["RegisterAddress"] = register_address
        formatted_df = formatted_df[formatted_df.FrameTriggers.diff() != 0]

    return formatted_df
