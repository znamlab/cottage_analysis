import numpy as np
from matplotlib import pyplot as plt


def plot_onix_harp_clock_sync(oni_clock_di, oni_clock_times, oni_clock_in_harp, harp2onix,
                              onix_sampling=250e6):
    clock_onset = np.diff(np.hstack([0, oni_clock_di])) == 1
    clock_time = oni_clock_times[clock_onset] / onix_sampling
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.hist(np.diff(clock_time) * 1e6 - 10e3)
    ax.set_title('Onix clock deviation from 100 Hz')
    ax.set_yscale('log')
    ax.set_xlabel(r'Clock tick error ($\mu s$)')
    ax.set_ylabel('# of ticks')
    ax = fig.add_subplot(2, 2, 2)
    ax.hist(np.diff(oni_clock_in_harp) * 1e6 - 10e3)
    ax.set_title('Harp clock deviation from 100 Hz')
    ax.set_yscale('log')
    ax.set_xlabel(r'Clock tick error ($\mu s$)')
    ax.set_ylabel('# of ticks')
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(clock_time.values[:len(oni_clock_in_harp)], oni_clock_in_harp, 'o')
    ax.set_xlabel('ONI time (s)')
    ax.set_ylabel('Harp time (s)')
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(clock_time.values[:len(oni_clock_in_harp)],
            1000 * (harp2onix(oni_clock_in_harp) -
                    clock_time.values[:len(oni_clock_in_harp)]), 'o')
    ax.set_xlabel('ONI time (s)')
    ax.set_ylabel('Translated Harp - ONI time (ms)')
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    return fig