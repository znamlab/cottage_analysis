#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 18:36:04 2021

@author: hey2

Align timestamps for Vis-Stim & Recorded WF videos / 2p frames
Assign corresponding harp timestamps of df2 (usually logged stimuli parameters) to df1 (usually timestamps of recorded widefield/2p videos).


"""

# %% Import packages
import pandas as pd


# %%
def align_dataframes(df1, df2, align_basis, direction):
    """
    Align 2 dataframes

    Parameters
    ----------
    df1 : Dataframe
        Usually timestamps of widefield/2p videos, can be cropped before this step
    df2 : Dataframe
        Usually logged parameters of stimuli, can be cropped before this step
    align_basis : string
        The column as the basis for the 2 dataframes to be aligned
    direction: string, "backward"/"forward"/"nearest"
        Align direction. A “backward” search selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key.
        A “forward” search selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
        A “nearest” search selects the row in the right DataFrame whose ‘on’ key is closest in absolute distance to the left’s key.

    Returns
    -------
    DF: Dataframe
        Result df with every entry of df1 & their corresponding df2 parameters (with aligned timestamp)

    """

    DF = pd.merge_asof(
        df1, df2, on=align_basis, allow_exact_matches=False, direction=direction
    )
    assert len(DF) == len(df1)

    return DF


def align_timestamps(VS_frames, VS_params, imaging_frames):
    VS_DF = align_dataframes(
        df1=VS_frames, df2=VS_params, align_basis="HarpTime", direction="backward"
    )
    img_VS = align_dataframes(
        df1=imaging_frames, df2=VS_DF, align_basis="HarpTime", direction="backward"
    )

    return img_VS
