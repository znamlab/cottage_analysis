import numpy as np
import pandas as pd
import pickle
from cottage_analysis.analysis import spheres, find_depth_neurons
from cottage_analysis.analysis import common_utils


def sta(frames, dffs):
    """Calculate sta for all ROIs

    Args:
        frames (np.ndarray): 3d array containing reconstructed frames, (n_frames, n_elev, n_azim)
        dffs (np.2darray): 2d array containing dffs for all ROIs, (n_frames, n_rois), np.concatenate(imaging_df.dffs) or np.concatenate(imaging_df.spks)

    Returns:
        np.ndarray: stas for all ROIs, (n_rois, n_elev, n_azim)
    """
    # Select only the frames with stimuli
    stimulus_frames = np.sum(frames, axis=(1, 2)) > 0
    # calculate STA for all ROIs
    dffs = dffs - np.mean(dffs[stimulus_frames, :], axis=0)
    stas = dffs.T @ np.roll(np.reshape(frames, (frames.shape[0], -1)), 1, axis=0)
    stas = np.reshape(stas, (stas.shape[0], *frames.shape[1:]))
    sum_frames = np.sum(frames, axis=0)
    stas = stas / sum_frames
    return stas


def sta_by_depth(frames, dffs, imaging_df, delays=None, frame_rate=15, verbose=True):
    """Calculate sta for all ROIs by depth

    Args:
        frames (np.ndarray): 3d array containing reconstructed frames, (n_frames, n_elev, n_azim)
        dffs (np.2darray): 2d array containing dffs for all ROIs, (n_frames, n_rois), np.concatenate(imaging_df.dffs) or np.concatenate(imaging_df.spks)
        imaging_df (pd.DataFrame): imaging_df
        delays (list, optional): A list of delay values in s. Defaults to None.
        frame_rate (int, optional): frame rate of imaging. Defaults to 15.
        verbose (bool, optional): verbose or not. Defaults to True.

    Returns:
        (np.ndarray,int,int): (1) stas for all ROIs, (n_delays, n_depths, n_rois, n_elev, n_azim), (2) number of delays, (3) number of depths
    """
    # delay: current dff is in respond to previous frame. positive delay means shifting stimuli to the left or shifting dff to the right
    if delays is None:
        delays = [0]

    depth_list = np.sort(imaging_df.depth.unique())[
        np.sort(imaging_df.depth.unique()) > 0
    ]
    full_stas = np.zeros(
        (len(delays), dffs.shape[1], len(depth_list), frames.shape[1], frames.shape[2])
    )

    # loop over delays and depths
    for idelay, delay in enumerate(delays):
        if verbose:
            print(f"doing delay {delay*1000} ms")
        for idepth, depth in enumerate(depth_list):
            if verbose:
                print(f"... doing depth {depth*100} cm")
            depth_idx = (imaging_df.depth == depth).values
            frames_depth = frames[depth_idx, :, :]
            dffs_depth = dffs[depth_idx, :]

            # shift dffs
            shift = int(delay * frame_rate)
            dffs_depth = np.roll(dffs_depth, shift, axis=0)

            # crop frames_depth and dffs_depth to get rid of the first shift frames
            if delay >= 0:
                dffs_depth = dffs_depth[shift:, :]
                frames_depth = frames_depth[shift:, :, :]
            else:
                dffs_depth = dffs_depth[:shift, :]
                frames_depth = frames_depth[:shift, :, :]

            # calculate stas
            stas = sta(frames_depth, dffs_depth)
            full_stas[idelay, :, idepth, :, :] = stas
            ndelays = len(delays)
            ndepths = len(depth_list)

    return (full_stas, ndelays, ndepths)
