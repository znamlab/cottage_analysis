"""
Module to find frames based on photodiode flicker
"""
import numpy as np
import scipy.signal as scsi
from cottage_analysis.utilities import continuous_data_analysis as cda


def detect_frame_onset(photodiode, frame_rate=144, photodiode_sampling=1000,
                       highcut=400, debug=False):
    """Detect frames from photodiode signal

    Simple wrapper around `scipy.signal.find_peaks` to detect frame borders from
    photodiode signal.

    Args:
        photodiode (np.array): photodiode signal
        frame_rate (float): expected frame rate, peaks occuring faster than two frame
                            rate will be ignored
        photodiode_sampling (float): Sampling rate of the photodiode signal
        highcut (float): If not None, use low pass filter cutting components
                         above `highcut` hertz
        debug (bool): False by default. If True, returns a dict with intermediary results

    Returns:
        border_index (np.array): index of frame borders
        peak_index (np.array): index of peak of each frame, len(border_index) - 1
        debug_dict (dict): Dictionary of intermediary element. Only if `debug` == True
    """
    if highcut is not None:
        fpd = cda.filter(photodiode, highcut=frame_rate * 3, sampling=photodiode_sampling,
                         design='bessel')
        absdiff = np.abs(np.diff(fpd))
    else:
        fpd = None
        absdiff = np.abs(np.diff(photodiode))
    absdiff -= np.quantile(absdiff, 0.01)
    absdiff /= np.quantile(absdiff, 0.99)
    dst = int(1 / (frame_rate * 2) * photodiode_sampling)
    frame_border, _ = scsi.find_peaks(absdiff, distance=dst, height=0.05)

    # When two local peaks in the same frame have exactly the same amplitude,
    # find_peaks doesn't know which one to pick and keeps both. Let's keep only the last
    plateau = np.where((np.diff(frame_border) <= dst) &
                       (np.diff(absdiff[frame_border]) == 0))[0]
    valid = np.ones(len(frame_border), dtype=bool)
    valid[plateau] = False
    borders = np.array(frame_border[valid])

    # Now find the peak of the frame, it is len(frame_border) - 1
    peaks = np.array([absdiff[b:e].argmin() + b
                      for b, e in zip(frame_border[:-1], frame_border[1:])], dtype=int)
    if debug:
        debug_dict = dict(plateau=plateau, all_pks=frame_border, distance=dst,
                          filtered_trace=fpd, diff_trace=absdiff)
        return borders, peaks, debug_dict
    return borders, peaks


def plot_frame_detection_report(photodiode, frame_rate=144, photodiode_sampling=1000,
                                highcut=400, plot_window=(-50, 50), num_examples=1,
                                border_index=None, peak_index=None, debug_dict=None):
    """Detect frames and generate a few debuging figures

    If border_index, peak_index or debug_dict is None, frame detection will be
    performed first, otherwise the results will be used directly.
    This will select `num_examples` frames randomly and plot `plot_window` samples
    around them. Another `num_examples` of frames with a frame drop will be selected
    and ploted the same way

    Args:
        photodiode (np.array): photodiode signal
        frame_rate (float): expected frame rate, peaks occuring faster than two frame
                            rate will be ignored
        photodiode_sampling (float): Sampling rate of the photodiode signal
        highcut (float): If not None, use low pass filter cutting components
                         above `highcut` hertz
        plot_window ([int, int]): limit of the window to plot around each example frame
        num_examples (int): number of figures randomly selected and with frame drop to
                            plot (len(fig) == num_examples * 2)
        border_index (np.array): index of frame borders
        peak_index (np.array): index of peak of each frame, len(border_index) - 1
        debug_dict (dict): Dictionary of intermediary element. Only if `debug` == True


    Returns:
        figs (list): a list of figure handles
    """
    if (border_index is None) or (debug_dict is None):
        border_index, debug_dict = detect_frame_onset(photodiode, frame_rate,
                                                      photodiode_sampling, highcut,
                                                      debug=True)
    rng = np.random.default_rng(42)
    skip = np.diff(border_index) > photodiode_sampling / frame_rate * 1.5
    example_frames = np.hstack([border_index[rng.integers(len(border_index),
                                                          size=num_examples)],
                                border_index[:-1][skip][rng.integers(np.sum(skip),
                                                                     size=num_examples)]])
    figs = []
    w = np.array(plot_window, dtype=int)
    i = np.arange(len(photodiode))
    vlines_kwargs = [dict(color='Grey', alpha=0.2, ls='--', lw=0.5),
                     dict(color='k', alpha=0.5, ls='-')]
    dot_kwargs = [dict(color='Grey', alpha=0.2, marker='.', s=5),
                  dict(color='darkred', alpha=0.5, marker='o',  s=50,
                       ec='None')]
    for f in example_frames:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(debug_dict['filtered_trace'][slice(*w+f)], label='filtered')
        ax.plot(photodiode[slice(*w + f)], label='raw')
        ax.legend(loc='upper right')
        for iw, which_pk in enumerate([debug_dict['all_pks'], border_index]):
            v = (which_pk > w[0] + f) & (which_pk < w[1] + f)
            for d in i[which_pk[v]] - (w[0]+f):
                ax.axvline(d, **vlines_kwargs[iw])
        v = peak_index[(peak_index > w[0] + f) & (peak_index < w[1] + f)]
        ax.scatter(i[v] - (w[0] + f), photodiode[v], **dot_kwargs[1])
        ax.set_title('Frame at sample %d' % f)
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(debug_dict['diff_trace'][slice(*w+f)])
        for iw, which_pk in enumerate([debug_dict['all_pks'], border_index]):
            v = which_pk[(which_pk > w[0] + f) & (which_pk < w[1] + f)]
            ax.scatter(i[v] - (w[0] + f), debug_dict['diff_trace'][v], **dot_kwargs[iw])
        ax.set_ylabel('Abs diff trace')
        ax.set_xlabel('Sample')
        figs.append(fig)
    return figs


def ideal_photodiode(time_base, switch_time, sequence, highcut=150):
    """Make an idealise photodiode trace from sequence

    The photodiode is imaging a quad that changes color every frame but the photodiode
    filter and the pixel response time make this alternation smooth and not step-wise.
    This function attempts to filter the raw sequence to re-create an ideal version of
    what the photodiode signal should look like


    Args:
        time_base (np.array): Time of the real photodiode signal, will be the shape of
                              the output
        switch_time (np.array): Time points at which the photodiode changes colour
        sequence (np.array): Values of the photodiode for each of the switch time
        highcut (float): Frequency for low pass filter

    Returns:
        perfect_sequence (np.array): Continuous version of photodiode_value (same size as
                                     time_base)
        fake_photodiode (np.array): Filtered version of perfect_sequence
    """
    sampling = 1/np.mean(np.diff(time_base))
    change_indices = time_base.searchsorted(switch_time)
    perfect_sequence = np.zeros_like(time_base)
    for i, v in enumerate(sequence[:-1]):
        perfect_sequence[change_indices[i]: change_indices[i+1]] = v

    freq = highcut / sampling
    sos = scsi.butter(N=1, Wn=freq, btype='lowpass', output='sos')
    fake_photodiode = scsi.sosfilt(sos, perfect_sequence)
    return perfect_sequence, fake_photodiode


def sync_by_correlation(frame_time, photodiode_time, photodiode_signal, switch_time,
                        sequence, num_frame_to_corr=10, maxlag=100, expected_lag=24,
                        return_chunk=False):
    """Find best shift to synchronise photodiode with ideal sequence

    This will cut a chunk of the photodiode signal centered on `frame_time` that is twice
    `num_frame_to_corr` long, find the corresponding chunk in `sequence` to generate
    and idealised photodiode and correlate the two signals to find the ideal lag

    Args:
        frame_time (float): The time of the (one) frame to synchronise
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        switch_time (np.array): Time of all changes of photodiode quad colour
        sequence (np.array): Value of the quad colour after each switch.
        num_frame_to_corr (int): number of frame before and after frame_time to keep
                                 for correlation
        maxlag (int): Maximum lag tested (in samples, centered on expected_lag).
        expected_lag (float): expected lag (in samples) to center search
        return_chunk (bool): If False, return only delay, otherwise also returns the
                             chunks that were used to correlate and the corr coef

    Returns:
        delay (int): best delay in samples
        max_corr (float): correlation at best delay
        chunk (dict): only if return_chunk, dict with
    """
    log_index = switch_time.searchsorted(frame_time)
    beg = max(0, log_index - num_frame_to_corr)
    end = min(len(sequence) - 1, log_index + num_frame_to_corr)

    chunk = (photodiode_time > switch_time[beg]) & (photodiode_time < switch_time[end])
    chunk_time = photodiode_time[chunk]
    chunk_signal = photodiode_signal[chunk]
    # for the ideal_pd we want it to be the same size as chunk_time + max_shift * 2,
    # but shifted by expected_lag
    borders = np.where(chunk)[0][[0, -1]] + np.array([-maxlag, maxlag + 1]) - expected_lag
    b, e = np.clip(borders, 0, len(photodiode_time))
    ideal_time = photodiode_time[b: e]
    seq_trace, ideal_pd = ideal_photodiode(time_base=ideal_time, switch_time=switch_time,
                                           sequence=sequence)
    corr = scsi.correlate(chunk_signal, ideal_pd, mode='valid')
    # note that scipy correlate is just the dot product, not the pearson
    lags = scsi.correlation_lags(len(chunk_signal), len(ideal_pd), mode='valid')
    lags += maxlag + expected_lag
    argmax = corr.argmax()
    delay = lags[argmax]
    if return_chunk:
        chunk_dict = dict(time_base=chunk_time,
                          ideal_time=ideal_time,
                          photodiode=chunk_signal,
                          ideal_pd=ideal_pd,
                          seq_trace=seq_trace,
                          correlation=corr,
                          lags=lags)
        return delay, corr[argmax], chunk_dict
    return delay, corr[argmax]
