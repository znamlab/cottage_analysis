"""
Simulated neuron responsed from 2D gaussian fits
"""

import numpy as np
from cottage_analysis.analysis import fit_gaussian_blob as fit_gb

VALID_KERNEL_NORMALIZATIONS = ("max", "area")


def _normalize_kernel(kernel, normalization):
    """Rescale a kernel in place according to `normalization`.

    Args:
        kernel (np.ndarray): kernel to rescale.
        normalization (str): "max" scales the kernel so its peak is 1 (matches the
            usual convention for an impulse-response-style calcium kernel). "area"
            scales it to unit gain (kernel sums to 1), so a sustained/constant input
            reaches the same steady-state magnitude at the output as at the input.

    Returns:
        np.ndarray: the rescaled kernel.
    """
    if normalization == "max":
        return kernel / np.max(kernel)
    elif normalization == "area":
        return kernel / kernel.sum()
    raise ValueError(
        f"Unknown normalization {normalization!r}; expected one of "
        f"{VALID_KERNEL_NORMALIZATIONS}."
    )


def make_exponential_kernel(tau, frame_rate, normalization="max"):
    """
    Create an exponential decay kernel.

    Args:
        tau (float): Decay time constant in seconds.
        frame_rate (float): Sampling rate in Hz.
        normalization (str, optional): "max" to normalize the kernel peak to 1, or
            "area" to normalize its sum (unit gain) to 1. Defaults to "max".

    Returns:
        np.ndarray: Normalized exponential decay kernel.
    """
    kernel_duration = 5 * tau
    time_kernel = np.arange(0, kernel_duration, 1 / frame_rate)
    kernel = np.exp(-time_kernel / tau)
    return _normalize_kernel(kernel, normalization)


def make_biexponential_kernel(tau_decay, tau_rise, frame_rate, normalization="max"):
    """
    Create a biexponential kernel.

    Args:
        tau_decay (float): Decay time constant in seconds.
        tau_rise (float): Rise time constant in seconds.
        frame_rate (float): Sampling rate in Hz.
        normalization (str, optional): "max" to normalize the kernel peak to 1, or
            "area" to normalize its sum (unit gain) to 1. Defaults to "max".

    Returns:
        np.ndarray: Normalized exponential decay kernel.
    """
    assert tau_decay > tau_rise, "tau_decay must be greater than tau_rise"
    kernel_duration = 5 * max(tau_decay, tau_rise)
    time_kernel = np.arange(0, kernel_duration, 1 / frame_rate)
    kernel = np.exp(-time_kernel / tau_decay) - np.exp(-time_kernel / tau_rise)
    return _normalize_kernel(kernel, normalization)


def simulate_calcium_responses(
    imaging_df,
    popt_list,
    tau_decay=0.8,
    tau_rise=0.15,
    frame_rate=30.0,
    min_sigma=0.25,
    make_circular=True,
    kernel_normalization="max",
):
    """
    Simulate calcium responses continuously based on 2D Gaussian fit parameters.

    Args:
        imaging_df (pd.DataFrame): DataFrame containing continuous continuous data
            with RS and OF for the whole recording.
        popt_list (list): List of 2D Gaussian fit parameters (arrays), one per ROI.
        tau_decay (float): Decay time constant in seconds. Default 0.8
        tau_rise (float): Rise time constant in seconds. Default 0.15
        frame_rate (float, optional): Sampling rate in Hz for the exponential kernel. Defaults to 30.0.
        min_sigma (float, optional): Minimum sigma for the 2D Gaussian. Defaults to 0.25.
        make_circular (bool, optional): If True, make the Gaussian circular by setting
            the major axis to the minor axis length.
        kernel_normalization (str, optional): "max" to normalize the calcium kernel's
            peak to 1, or "area" to normalize its sum (unit gain) to 1. Defaults to
            "max".

    Returns:
        np.ndarray: Simulated continuous responses for all ROIs, shape (time, n_rois).
    """
    if tau_rise is None:
        kernel = make_exponential_kernel(
            tau_decay, frame_rate, normalization=kernel_normalization
        )
    else:
        kernel = make_biexponential_kernel(
            tau_decay, tau_rise, frame_rate, normalization=kernel_normalization
        )

    # The entire recording's stimulus vectors
    # Note: outside of trials, RS or depth might be NaN or different
    # Make sure we handle potential NaNs by converting them to appropriate defaults or zero
    rs = imaging_df.RS.values.copy()
    of = imaging_df.OF.values.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.log(rs)
        of = np.log(np.rad2deg(of))

    # Identify frames with no movement or invalid data
    bad_indices = np.isnan(rs) | np.isnan(of) | np.isinf(rs) | np.isinf(of)

    # Replace invalid/zero values with a dummy low log-value (e.g., -10)
    # This ensures gaussian_2d returns the baseline 'offset' for these frames
    rs[bad_indices] = -10
    of[bad_indices] = -10

    n_frames = len(imaging_df)
    n_rois = len(popt_list)
    fake_dff_continuous = np.zeros((n_frames, n_rois)) + np.nan

    # ROIs Loop
    for iroi, popt in enumerate(popt_list):
        if popt is None or np.isnan(popt).any():
            continue

        if make_circular:
            # Make a copy of popt but circular, by reducing the major axis to the minor
            popt_model = popt.copy()
            popt_model[3] = popt_model[4] = min(popt[3:5])
        else:
            popt_model = popt

        fake_data = fit_gb.gaussian_2d((rs, of), *popt_model, min_sigma=min_sigma)

        # Convolve with an exponential decay to mimic calcium dynamics
        fake_data_slow = np.convolve(fake_data, kernel, mode="full")[:n_frames]
        fake_dff_continuous[~bad_indices, iroi] = fake_data_slow[~bad_indices]

    return fake_dff_continuous
