import numpy as np


def searchclosest(ts0, ts1):
    """Find the closest time of ts1 for each element of ts0

    From: https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    Args:
        ts0 (np.array): array in which we need to find the values, needs to be sorted
        ts1 (np.array): values to find in ts0, needs to be sorted

    Returns:
        closest (np.array): array of indices of closest values, same shape as ts1
    """
    idx = ts0.searchsorted(ts1)
    # fix border issues (ensure idx and idx-1 are in [0, len(ts0)-1])
    idx = np.clip(idx, 1, len(ts0) - 1)
    # We have ts1[idx[i] - 1] < ts0[i] <= ts1[idx[i]]
    left = ts0[idx - 1]
    right = ts0[idx]
    idx -= ts1 - left < right - ts1
    return idx


def cc_func(ts0, ts1, trange, absolute_time=False, keep_zero=True, check=False):
    """Compute crosscorrelogram between two time series

    This is what ephys people call crosscorrelogram. More precisely it is just events of
    ts1 in a window around each event of ts0

    Args:
        ts0 (np.array): first time series
        ts1 (np.array): second time series
        trange (float, float): window to extract the crosscorrelogram
        absolute_time (bool): If False (default) return the time relative to ts0 event,
                              otherwise the absolute time
        keep_zero (bool): Keep exact match (useful to remove for autocorrelograms)
        check (bool): check if series are sorted

    Returns:
        cc (list of np.array): A list of len(ts0) arrays containing times of ts1
                               falling in `trange` around each ts0 event
    """
    trange = np.asarray(trange)
    ts0 = np.asarray(ts0)
    ts1 = np.asarray(ts1)
    assert len(trange) == 2
    if check:
        assert all(np.sort(ts1) == ts1)

    limits = np.vstack([ts0 + t for t in trange])
    lim_ind = ts1.searchsorted(limits)
    cc = [ts1[b:e] for b, e in lim_ind.T]
    if not absolute_time:
        cc = [c - s for c, s in zip(cc, ts0)]
    if not keep_zero:
        cc = [c[c != 0] for c in cc]
    return cc


def gaussian_density(data, sd, start=None, end=None, dstep=None, verbose=True):
    """Takes a sequence of spike times and produces a non-normalised density
    estimate by summing Normals defined by sd at each spike time. The range of
    the output is guessed from the extent of the data (which need not be
    ordered), the resolution is automagically determined from sd; we currently
    used sd*0.05 A 2d np.array is returned with the time scale and
    non-normalised 'density' as first and second rows."""

    # Note: once I've understood convolutions and Fourier transforms, they
    # probably represent the quick way of doing this.
    # note: try to fft this

    # Resolution as fraction of sd

    data = np.array(data)
    dmax = np.max(data) + sd * 4.0 if end is None else float(end)
    dmin = np.min(data) - sd * 4.0 if start is None else float(start)

    res = 0.05
    if dstep is None:
        dstep = sd * res
    else:
        if dstep > sd * res:
            if verbose:
                print("Warning dstep big relative to sd")
    time = np.arange(dmin, dmax, dstep)

    norm = (
        1
        / np.sqrt(2 * np.pi * sd**2)
        * np.exp(-((time - time[int(time.size / 2)]) ** 2) / (2 * sd**2))
    )
    kernel = np.vstack((time, norm))

    time, dens = kernel_density(
        data, kernel, dmin=dmin, dmax=dmax, dstep=dstep, verbose=verbose
    )
    # for t in data:
    #      dens[t_to_i(t-sd*3.)+r] += norm
    return np.vstack((time, dens))


def half_exp_density(data, sd):
    """Takes a sequence of spike times and produces a normalised density
    estimate by summing Half-exponential (asymetric) defined by sd at each spike
    time. The range of the output is guessed from the extent of the data (which
    need not be ordered), the resolution is automagically determined from sd; we
    currently used sd/10. A 2d np.array is returned with the time scale and
    non-normalised 'density' as first and second rows."""

    # Resolution as fraction of sd
    res = 0.1
    data = np.array(data)
    dmax = float(np.max(data) + sd * 4.0)
    dmin = float(np.min(data) - sd * 4.0)
    dstep = sd * res
    time = np.arange(start=dmin, stop=dmax, step=dstep)
    if time.size % 2 != 0:
        time = time[:-1]
    r = np.arange(0, len(time), dtype=int)
    hal = r.size // 2
    exp = np.zeros(r.size, dtype="float")
    exp[hal:] = (
        2
        / np.sqrt(2 * sd**2)
        * np.exp(-np.sqrt(2) * np.abs(time[hal:] - time[hal]) / sd)
    )
    exp /= exp.sum()
    time, dens = kernel_density(data, np.vstack((time, exp)))
    return np.vstack((time, dens))


def kernel_density(data, kernel, dmin=None, dmax=None, dstep=None, verbose=True):
    """Kernel density estimation

    Given a 2-D kernel (one line for the time, one for the values) and a
    list of time (data), compute the kde (just do the convolution basically)

    if dmin and/or dmax not None, use it as the minimum/maximum time value for
       the output
    else the output has the minimum size to fit all data points plus a kernel
       half-width


    return time, kde two 1-D arrays
    """

    if dstep is None:
        dstep = kernel[0][1] - kernel[0][0]
    length = kernel[0][-1] - kernel[0][0]
    if dmin is None:
        dmin = data.min() - length / 2
    if dmax is None:
        dmax = data.max() + length / 2

    time = np.arange(dmin, dmax, dstep)

    # add one dstep to have the smallest time bigger than dmin to dmax
    out = np.zeros(time.size, dtype=int)
    ignored = 0
    for d in data:
        if dmin < d < dmax:
            out[int((d - dmin) / dstep)] += 1
        else:
            ignored += 1
    bigkernel = np.zeros(out.size, dtype="float")
    beg = int(bigkernel.size / 2 - kernel.shape[1] / 2)
    end = int(bigkernel.size / 2 + int(kernel.shape[1] / 2.0 + 0.5))
    bigkernel[beg:end] = kernel[1][:]
    conv = np.convolve(bigkernel, out, mode="same")
    if ignored > 0 and verbose:
        print("%i data points out of [dmin, dmax] interval were ignored" % ignored)
    return time, conv
