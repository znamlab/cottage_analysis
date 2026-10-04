"""
Simulated neuron responsed from 2D gaussian fits
"""

import numpy as np
from cottage_analysis.analysis import fit_gaussian_blob as fit_gb


def make_exponential_kernel(tau, frame_rate):
    """
    Create an exponential decay kernel.

    Args:
        tau (float): Decay time constant in seconds.
        frame_rate (float): Sampling rate in Hz.

    Returns:
        np.ndarray: Normalized exponential decay kernel.
    """
    kernel_duration = 5 * tau
    time_kernel = np.arange(0, kernel_duration, 1 / frame_rate)
    kernel = np.exp(-time_kernel / tau)
    kernel /= kernel.sum()
    return kernel


def make_biexponential_kernel(tau_decay, tau_rise, frame_rate):
    """
    Create a biexponential kernel.

    Args:
        tau_decay (float): Decay time constant in seconds.
        tau_rise (float): Rise time constant in seconds.
        frame_rate (float): Sampling rate in Hz.

    Returns:
        np.ndarray: Normalized exponential decay kernel.
    """
    assert tau_decay > tau_rise, "tau_decay must be greater than tau_rise"
    kernel_duration = 5 * max(tau_decay, tau_rise)
    time_kernel = np.arange(0, kernel_duration, 1 / frame_rate)
    kernel = np.exp(-time_kernel / tau_decay) - np.exp(-time_kernel / tau_rise)
    kernel /= kernel.sum()
    return kernel


def simulate_calcium_responses(
    imaging_df,
    popt_list,
    tau_decay=0.8,
    tau_rise=0.15,
    frame_rate=30.0,
    min_sigma=0.25,
    make_circular=True,
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

    Returns:
        np.ndarray: Simulated continuous responses for all ROIs, shape (time, n_rois).
    """
    if tau_rise is None:
        kernel = make_exponential_kernel(tau_decay, frame_rate)
    else:
        kernel = make_biexponential_kernel(tau_decay, tau_rise, frame_rate)

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
