"""
Quality control plot for sync
"""
import numpy as np
from matplotlib import pyplot as plt


def plot_onix_harp_clock_sync(
    oni_clock_di,
    oni_clock_times,
    clock_in_harp_di,
    harp_di_times,
    harp2onix,
    onix_sampling=250e6,
):
    """Plot clock sync

    Args:
        oni_clock_di (pd.Series): Digital input state on the breakout board
        oni_clock_times (pd.Series): Time corresponding to the digital states
        clock_in_harp_di (pd.Series): State of di recording onix clock in harp
        harp_di_times (pd.Series): Harp time corresponding to the harp digital states
        harp2onix (function): conversion function
        onix_sampling (float):  sampling rate of onix in Hz

    Returns:
        fig (plt.Figure): output figure

    """
    clock_onset = np.diff(np.hstack([0, oni_clock_di])) == 1
    clock_time = oni_clock_times[clock_onset] / onix_sampling
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.hist(np.diff(clock_time) * 1e6 - 10e3)
    ax.set_title("Onix clock deviation from 100 Hz")
    ax.set_yscale("log")
    ax.set_xlabel(r"Clock tick error ($\mu s$)")
    ax.set_ylabel("# of ticks")
    ax = fig.add_subplot(2, 2, 2)
    h_clock_onset = np.diff(np.hstack([0, clock_in_harp_di])) == 1
    oni_clock_in_harp = harp_di_times[h_clock_onset]
    ax.hist(np.diff(oni_clock_in_harp) * 1e6 - 10e3)
    ax.set_title("Harp clock deviation from 100 Hz")
    ax.set_yscale("log")
    ax.set_xlabel(r"Clock tick error ($\mu s$)")
    ax.set_ylabel("# of ticks")
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(clock_time.values[: len(oni_clock_in_harp)], oni_clock_in_harp, "o")
    ax.set_xlabel("ONI time (s)")
    ax.set_ylabel("Harp time (s)")
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(
        clock_time.values[: len(oni_clock_in_harp)],
        1000
        * (harp2onix(oni_clock_in_harp) - clock_time.values[: len(oni_clock_in_harp)]),
        "o",
    )
    ax.set_xlabel("ONI time (s)")
    ax.set_ylabel("Translated Harp - ONI time (ms)")
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    return fig
