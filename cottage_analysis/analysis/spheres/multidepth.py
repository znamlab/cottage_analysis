"""
Function used by multi-depth sphere protocol

This protocol is very similar to spheres, so it uses most functions from there and this
file contains only the few parts that are different
"""

import numpy as np


def find_trial_times(param_log, jitter=0.5, verbose=True):
    """Finds the onset and offset times of trials for multi-depth recordings.

    Args:
        param_log (pd.DataFrame): DataFrame containing the stimulus parameters,
            including 'logger_fname', 'HarpTime' and 'Radius'.
        jitter (float): Maximum acceptable delay between the onset of stimuli
            at different depths. Default to 0.5
        verbose (bool): Print info. Default to True

    Returns:
        np.ndarray: A 2xN array where the first row contains the onset times
            and the second row contains the corresponding offset times for each
            trial.

    Raises:
        ValueError: If the number of onsets and offsets do not match, or if
            multiple corridors start after the last offset.
        NotImplementedError: If the offsets and onsets are not matching.
    """
    # Find onset and offset of trials for each depth
    all_onsets = {}
    all_offsets = {}
    for log, df in param_log.groupby("logger_fname"):
        stim_onoff = (df.Radius > 0).astype(float).diff().values
        assert df.HarpTime.is_monotonic_increasing, "HarpTime is not sorted"
        # work on values to avoid indexing issues
        times = df.HarpTime.values

        # Find onset that are preceded by 2s of stim off (to avoid flicker)
        onsets = np.where(stim_onoff == 1)[0]
        onset_times = times[onsets]
        start_windows = times.searchsorted(onset_times - 2)
        clean_onsets = []
        for start, onset in zip(start_windows, onsets):
            chunk = np.abs(stim_onoff[start:onset])
            # in most cases there wasn't anything (no sphere log) in the 2s before and chunk
            # is empty
            if not chunk.size or chunk.max() == 0:
                clean_onsets.append(onset)
        clean_onset_times = times[clean_onsets]

        # Find offset that are followed by 2s of stim off (to avoid flicker)
        offsets = np.where(stim_onoff == -1)[0]
        offset_times = times[offsets]
        end_windows = times.searchsorted(offset_times + 2)
        clean_offsets = []
        for offset, end in zip(offsets, end_windows):
            chunk = np.abs(stim_onoff[offset + 1 : end])
            if not chunk.size or chunk.max() == 0:
                clean_offsets.append(offset)
        clean_offset_times = times[clean_offsets]

        all_onsets[int(df.Radius.max())] = np.array(clean_onset_times)
        all_offsets[int(df.Radius.max())] = np.array(clean_offset_times)

    # Keep only trials where all depths are in sync
    jitter = 0.5  # maximum acceptable delay between the onset of 1 stimuli
    n_depths = len(all_onsets.keys())
    print(f"Recording with {n_depths} depths")

    allon = np.sort(np.hstack(list(all_onsets.values())))
    on_delay = allon[None, :] - allon[:, None]
    n_on_in_sync = (np.abs(on_delay) < jitter).sum(axis=0)
    alloff = np.sort(np.hstack(list(all_offsets.values())))
    off_delay = alloff[None, :] - alloff[:, None]
    n_off_in_sync = (np.abs(off_delay) < jitter).sum(axis=0)

    # Keep only onsets where all trials started at the same time
    valid_onsets = allon[n_on_in_sync == n_depths]
    # And only the first of these n_depths almost synchroneous onsets
    first_of_bunch = np.hstack([1, np.diff(valid_onsets) > 1]).astype(bool)
    onset_times = valid_onsets[first_of_bunch]
    # Keep only offsets where all trials started at the same time
    valid_offsets = alloff[n_off_in_sync == n_depths]
    # And the last of these n_depths
    last_of_bunch = np.hstack([np.diff(valid_offsets) > 1, 1]).astype(bool)
    offset_times = valid_offsets[last_of_bunch]

    # Cut onset after the last offset (there shouldn't be more than one)
    too_late = onset_times > offset_times[-1]
    if np.sum(too_late) > 1:
        raise ValueError(f"{np.sum(too_late)} corridors start after the last offset.")
    onset_times = onset_times[~too_late]

    closest_offset = offset_times.searchsorted(onset_times)
    # If all went well we should have 1 for 1 matches and diff==1
    matching = np.diff(closest_offset)
    if np.any(matching == 0):
        # We have 2 onsets in a row. That means one offset was not quite in sync
        print(f"{np.sum(matching==0)} onsets with no offset")
        to_remove = np.where(matching == 0)[0] + 1
        onset_times = np.delete(onset_times, to_remove)
        # redo the matching
        closest_offset = offset_times.searchsorted(onset_times)

    if not np.all(np.diff(closest_offset) == 1):
        raise NotImplementedError("Offsets and onsets are not matching")
    trial_on_off = np.vstack([onset_times, offset_times[closest_offset]])
    param_log_index = param_log.HarpTime.searchsorted(trial_on_off)

    if verbose:
        n_on = len(valid_onsets)
        n_off = len(valid_offsets)
        txt = f"{n_on}/{len(allon)} valid onsets ({n_on/len(allon)*100:.2f}%), "
        txt += f"{n_off}/{len(alloff)} valid offsets ({n_off/len(alloff)*100:.2f}%)"
        print(txt)
        print(f"{len(trial_on_off[0])} valid trials left")
    return trial_on_off, param_log_index
