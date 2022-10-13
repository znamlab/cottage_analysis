"""
Module to find frames based on photodiode flicker
"""
import numpy as np
from scipy.signal import find_peaks


def detect_alternating_frames(photodiode, frame_rate=144, photodiode_sampling=1000,
                              num_steps=5):
    """Detect frames from photodiode signal

    Simple wrapper around `scipy.signal.find_peaks` to detect frames from photodiode
    signal. This expects an alternating signal between dark and bright patches.

    Args:
        photodiode (np.array): photodiode signal
        frame_rate (float): expected frame rate, peaks occuring faster than two frame
                            rate will be ignored
        photodiode_sampling (float): Sampling rate of the photodiode signal
        num_steps (int): The number of grey levels that the photodiode can take. This
                         will put a minimum threshold on peak prominence.

    Returns:
        frame_index (np.array): array of index where frames are detected
    """
    normed_pd = np.array(photodiode, dtype=float)
    normed_pd -= np.nanmin(normed_pd)
    normed_pd /= np.nanmax(normed_pd)
    dst = int(1 / (frame_rate * 2) * photodiode_sampling)
    ppks, _ = find_peaks(normed_pd, distance=dst, prominence=1 / num_steps / 5,
                         wlen=dst * 10)
    npks, _ = find_peaks(1-normed_pd, distance=dst, prominence=1 / num_steps / 2,
                         wlen=dst * 10)
    pks = np.sort(np.hstack([ppks, npks]))

    # When two local peaks in the same frame have exactly the same amplitude,
    # find_peaks doesn't know which one to pick and keeps both. Let's keep only the last
    plateau = np.where(np.diff(normed_pd[pks]) == 0)[0]
    valid = np.ones(len(pks), dtype=bool)
    valid[plateau] = False
    return np.array(pks[valid])
