import numpy as np
import pandas as pd
import pickle
from cottage_analysis.analysis import spheres, find_depth_neurons
from cottage_analysis.analysis import common_utils


def sta(frames, dffs):
    '''Calculate sta for all ROIs

    Args:
        frames (np.ndarray): 3d array containing reconstructed frames, (n_frames, n_elev, n_azim)
        dffs (np.2darray): 2d array containing dffs for all ROIs, (n_frames, n_rois), np.concatenate(imaging_df.dffs) or np.concatenate(imaging_df.spks)

    Returns:
        np.ndarray: stas for all ROIs, (n_rois, n_elev, n_azim)
    '''
    # Select only the frames with stimuli
    stimulus_frames = np.sum(frames, axis=(1,2)) > 0
    # calculate STA for all ROIs
    dffs = dffs - np.mean(dffs[stimulus_frames, :], axis=0)
    stas = dffs.T @ np.roll(np.reshape(frames, (frames.shape[0], -1)), 1, axis=0)
    stas = np.reshape(stas, (stas.shape[0], *frames.shape[1:]))
    sum_frames = np.sum(frames, axis=0)
    stas = stas / sum_frames
    return stas


def sta_by_depth(frames, dffs, imaging_df, delays=None, frame_rate=15, verbose=True):
    '''Calculate sta for all ROIs by depth

    Args:
        frames (np.ndarray): 3d array containing reconstructed frames, (n_frames, n_elev, n_azim)
        dffs (np.2darray): 2d array containing dffs for all ROIs, (n_frames, n_rois), np.concatenate(imaging_df.dffs) or np.concatenate(imaging_df.spks)
        imaging_df (pd.DataFrame): imaging_df
        delays (list, optional): A list of delay values in s. Defaults to None.
        frame_rate (int, optional): frame rate of imaging. Defaults to 15.
        verbose (bool, optional): verbose or not. Defaults to True.

    Returns:
        (np.ndarray,int,int): (1) stas for all ROIs, (n_delays, n_depths, n_rois, n_elev, n_azim), (2) number of delays, (3) number of depths
    '''
    # delay: current dff is in respond to previous frame. positive delay means shifting stimuli to the left or shifting dff to the right
    if delays is None:
        delays = [0]
        
    depth_list = np.sort(imaging_df.depth.unique())[np.sort(imaging_df.depth.unique())>0]
    full_stas = np.zeros((len(delays), dffs.shape[1], len(depth_list), frames.shape[1], frames.shape[2]))

    # loop over delays and depths
    for idelay, delay in enumerate(delays):
        if verbose:
            print(f"doing delay {delay*1000} ms")
        for idepth, depth in enumerate(depth_list):
            if verbose:
                print(f"... doing depth {depth*100} cm")
            depth_idx = (imaging_df.depth == depth).values
            frames_depth = frames[depth_idx,:,:]
            dffs_depth = dffs[depth_idx,:]
            
            # shift dffs
            shift = int(delay*frame_rate)
            dffs_depth = np.roll(dffs_depth, shift, axis=0)
            
            # crop frames_depth and dffs_depth to get rid of the first shift frames
            if delay >= 0:
                dffs_depth = dffs_depth[shift:,:]
                frames_depth = frames_depth[shift:,:,:]
            else:
                dffs_depth = dffs_depth[:shift,:]
                frames_depth = frames_depth[:shift,:,:]
                
            # calculate stas
            stas = sta(frames_depth, dffs_depth)
            full_stas[idelay,:,idepth,:,:] = stas
            ndelays = len(delays)
            ndepths = len(depth_list)
            
    return (full_stas, ndelays, ndepths)

# def sta_by_depth(
#     trials_df,
#     reconstructed_frames,
#     frame_times,
#     frame_rate,
#     delays=None,
#     spk_per_frame=None,
#     verbose=True,
# ):
#     """Spike triggered average of reconstructed frames by depth

#     Delays, in second, are delay applied to the stimulus sequence. If delay is -100,
#     that means that spikes were triggered by stimulus 100ms before them.

#     Args:
#         trials_df (pd.DataFrame): stimulus structure with each row as a trial.
#         reconstructed_frames (np.array): n frames x n elev x n azim binary array of
#                                          stimuli
#         frame_times (np.array): time of each frame, same unit as corridor_df.start_time
#         frame_rate (float): frame rate to calculate delays. 144 for monitor frames, 15 for imaging frames.
#         delays (np.array): array of delays in seconds
#         spk_per_frame (np.array): spike for each frame, use to weight average. If None
#                                   will do simple average
#         verbose (bool): print progress

#     Returns:
#         sta (np.array): n depth x n delay x n elev x n azim weighted average
#         nspkes (np.array): n depth vector of number of spikes
#         depths (np.array): ordered depths corresponding to first sta dimension
#         delays (np.array): ordered delays corresponding to second sta dimension
#     """
#     if delays is None:
#         delays = [0]
#     if spk_per_frame is None:
#         spk_per_frame = np.ones(reconstructed_frames.shape[0])

#     depths = np.sort(trials_df.depth.unique())
#     full_sta = np.zeros((len(depths), len(delays), *reconstructed_frames.shape[1:]))
#     nspks = np.zeros(len(depths))

#     for idepth, depth in enumerate(depths):
#         if verbose:
#             print(f"... doing depth {depth*100} cm")
#         depth_df = trials_df[trials_df.depth == depth]
#         # find frames at this depth
#         # starts = depth_df.imaging_frame_stim_start.values
#         # ends = depth_df.imaging_frame_stim_stop.values
#         starts = frame_times.searchsorted(depth_df.imaging_harptime_stim_start)
#         ends = frame_times.searchsorted(depth_df.imaging_harptime_stim_stop)
#         ends = ends[: len(starts)]
#         frame_index = np.hstack(
#             [np.arange(s, e, dtype=int) for s, e in zip(starts, ends)]
#         )
#         # keep non-shifted spikes for all delay
#         # do it like that to look for valid frames only once
#         spk_per_frame_at_depth = spk_per_frame[frame_index]
#         nspks[idepth] = np.sum(spk_per_frame_at_depth)
#         valid_frames = spk_per_frame_at_depth != 0
#         for idelay, delay in enumerate(delays):
#             if verbose:
#                 print(f"... ... doing delay {delay * 1000} ms")
#             shift = int(delay * frame_rate)
#             # shift the stim
#             shifted_frames = np.clip(
#                 frame_index[valid_frames] + shift, 0, len(frame_times)
#             )
#             stims = reconstructed_frames[shifted_frames].reshape(
#                 len(shifted_frames), -1
#             )
#             sta = np.dot(stims.T, spk_per_frame_at_depth[valid_frames])
#             sta = sta.reshape(reconstructed_frames.shape[1:])
#             full_sta[idepth, idelay] = sta
#             # !! Needs to add normalized STA
#     return full_sta, nspks, depths, delays
