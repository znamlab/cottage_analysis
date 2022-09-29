import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, bessel


def filter(data, sampling, lowcut=None, highcut=None, design='butter'):
    """Wrapper around butter and filtfilt to filter continuous data

    Args:
        data (np.array): signal to process
        sampling (float): sampling frequency, in Hz
        lowcut (float or None): cutoff frequency for highpass filter
        highcut (float or None): cutoff frequency for lowpass filter
        design (str): `butter` or `bessel`

    Returns:
        filtered_data (np.array
    """
    n = int(lowcut is not None) * 2 + int(highcut is not None)
    filter_type = [None, 'lowpass', 'highpass', 'bandpass'][n]
    if filter_type is None:
        warnings.warn('Both frequencies are None. Do nothing')
        return data
    if filter_type == 'lowpass':
        freq = highcut/sampling
    elif filter_type == 'highpass':
        freq = lowcut/sampling
    else:
        freq = (lowcut/sampling, highcut/sampling)
    if design.lower() == 'butter':
        filt_func = butter
    elif design.lower() == 'bessel':
        filt_func = bessel
    else:
        raise IOError('`design` must be `bessel` or `butter`')
    sos = filt_func(N=4, Wn=freq, btype=filter_type, output='sos')
    fdata = sosfiltfilt(sos, data)
    return fdata


def do_sta(analog_signal, event, window=[-.5, .5], verbose=True,
           s_rate=None, event_in_samples=False, t0=0, dtype=float,
           return_variance=False):
    """"Do a stimulus triggered average of analogSignal over the given time
    window

    Can also estimate the variance using Welford algorithm.
    [https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance]
    Using extract_sts + np.std might be better for things fitting in memory
    """

    times = np.array(event, dtype='float')

    if s_rate is not None:
        winind = np.array(np.asarray(window) * s_rate, dtype='int')
    else:
        assert all(np.array(window, dtype=int) == np.array(window))
        winind = np.array(window, dtype=int)

    mean = None
    m2 = None
    tot_n = float(len(times))
    n = 0
    for e in times:
        if not event_in_samples:
            if s_rate is None:
                raise IOError(
                    'Event is not in samples and I do not know the sampling rate')
            ind_ev = int((e - t0) * s_rate)
        else:
            ind_ev = int(e)
        if ind_ev + winind[1] > len(analog_signal):
            if verbose:
                print('skip last cause it does not fit')
            continue
        if ind_ev + winind[0] < 0:
            if verbose:
                print('skip first cause it does not fit')
            continue

        x = np.array(analog_signal[ind_ev + winind[0]: ind_ev + winind[1]],
                     dtype=dtype)  # current value
        if mean is None:  # initialise zero values
            mean = np.zeros_like(x)
            if return_variance:
                m2 = np.zeros_like(x)
        n += 1
        delta = x - mean
        mean = mean + delta / n
        if m2 is not None:
            m2 = m2 + delta * (x - mean)

    if return_variance:
        if mean is None:
            m2 = None
        elif n < 2:
            m2 *= 0
        else:
            m2 = m2 / (n - 1)
        return mean, m2
    return mean


def extract_sts(analog_signal, event, window=[-.5, .5], verbose=True,
                s_rate=None, event_in_samples=False, event_t0=0, dtype=float):
    """"Find stimulus triggered snippets of analogSignal over the given time
    window

    Returns the individual elements that can be averaged to generate the STA

    event_t0 is subtracted to every time. It needs to be like event, in sample or time
    """
    times = np.array(event, dtype='float')

    if s_rate is not None:
        winind = np.array(np.asarray(window) * s_rate, dtype='int')
    else:
        assert all(np.array(window, dtype=int) == np.array(window))
        winind = np.array(window, dtype=int)

    n = int(len(times))
    ndim = analog_signal.ndim
    if ndim == 2:
        out = np.zeros((n, np.diff(winind)[0], analog_signal.shape[1]), dtype=dtype)
    elif ndim == 1:
        out = np.zeros((n, np.diff(winind)[0]), dtype=dtype)
    else:
        raise IOError('Analog signal must have 1 or 2 dim')
    for ind, e in enumerate(times):
        if not event_in_samples:
            if s_rate is None:
                raise IOError(
                    'Event is not in samples and I do not know the sampling rate')
            ind_ev = int((e - event_t0) * s_rate)
        else:
            ind_ev = int(e - event_t0)
        if ind_ev + winind[1] > len(analog_signal):
            if verbose:
                print('skip last cause it does not fit')
            continue
        elif ind_ev + winind[0] < 0:
            if verbose:
                print('skip first cause it does not fit')
            continue
        out[ind] = np.asarray(analog_signal[ind_ev + winind[0]: ind_ev + winind[1]],
                              dtype=dtype)
    return out
