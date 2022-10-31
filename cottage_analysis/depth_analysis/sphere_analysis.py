import numpy as np


def sta_by_depth(corridor_df, reconstructed_frames, frame_times,
                 frame_rate=144, delays=None, spk_per_frame=None, verbose=True):
    """Spike triggered average of reconstructed frames by depth

    Args:
        corridor_df (pd.DataFrame): Stimulus structure, must have a 'depth',
        'start_time' and 'end_time' columns
        reconstructed_frames (np.array): n frames x n elev x n azim binary array of
                                         stimuli
        frame_times (np.array): time of each frame, same unit as corridor_df.start_time
        frame_rate (float): frame rate to calculate delays
        delays (np.array): array of delays in seconds
        spk_per_frame (np.array): spike for each frame, use to weight average. If None
                                  will do simple average
        verbose (bool): print progress

    Returns:
        sta (np.array): n depth x n delay x n elev x n azim weighted average
        nspkes (np.array): n depth vector of number of spikes
        depths (np.array): ordered depths corresponding to first sta dimension
        delays (np.array): ordered delays corresponding to second sta dimension
    """
    if delays is None:
        delays = [0]
    if spk_per_frame is None:
        spk_per_frame = np.ones(reconstructed_frames.shape[0])

    depths = np.sort(corridor_df.depth.unique())
    sta = np.zeros((len(depths), len(delays), *reconstructed_frames.shape[1:]))
    nspks = np.zeros(len(depths))

    for idepth, depth in enumerate(depths):
        if verbose:
            print('... doing depth %d cm' % depth)
        depth_df = corridor_df[corridor_df.depth == depth]
        # find frames at this depth
        starts = frame_times.searchsorted(depth_df.start_time)
        ends = frame_times.searchsorted(depth_df.end_time)
        frame_index = np.hstack([np.arange(s, e, dtype=int)
                                 for s, e in zip(starts, ends)])
        # keep non-shifted spikes for all delay
        # do it like that to look for valid frames only once
        spk_per_frame_at_depth = spk_per_frame[frame_index]
        nspks[idepth] = np.sum(spk_per_frame_at_depth)
        valid_frames = spk_per_frame_at_depth != 0
        for idelay, delay in enumerate(delays):
            if verbose:
                print('... ... doing delay %d ms' % (delay * 1000))
            shift = int(delay * frame_rate)
            # shift the stim
            # We want to trigger frame (n-shift) when there is a spike at frame n.
            shifted_frames = np.clip(frame_index[valid_frames] - shift, 0,
                                     len(frame_times))
            stims = reconstructed_frames[shifted_frames].reshape(len(shifted_frames), -1)
            sta = np.dot(stims.T, spk_per_frame_at_depth[valid_frames])
            sta = sta.reshape(reconstructed_frames.shape[1:])
            sta[idepth, idelay] = sta
    return sta, nspks, depths, delays
