"""
Function used by multi-depth sphere protocol

This protocol is very similar to spheres, so it uses most functions from there and this
file contains only the few parts that are different
"""
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np


def find_trial_times(param_log, jitter=2, verbose=True, debug=False):
    """Finds the onset and offset times of trials for multi-depth recordings.

    Args:
        param_log (pd.DataFrame): DataFrame containing the stimulus parameters,
            including 'logger_fname', 'HarpTime' and 'Radius'.
        jitter (float): Maximum acceptable delay between the onset of stimuli
            at different depths. Default to 1
        verbose (bool): Print info. Default to True
        debug (bool): Return debug info

    Returns:
        tuple:
            - trial_on_off (np.ndarray): A 2xN array where the first row contains the
              onset times and the second row contains the corresponding offset times
              for each trial.
            - param_log_index (np.ndarray): A 2xN array of indices into `param_log`
              corresponding to `trial_on_off`.
            - If `debug` is True, a tuple with additional debugging information is
              returned: `(all_onsets, all_offsets, valid_onsets, closest_offset)`.

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
        # The logger starts at the first presentation so t0 is always onset
        clean_onset_times = np.hstack([times[0], times[clean_onsets]])

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
    # Find depth with the smallest number of onsets
    min_on = np.inf
    depth_with_least_onsets = None
    for k, v in all_onsets.items():
        if len(v) < min_on:
            depth_with_least_onsets = k
            min_on = len(v)
    # For each of the putative onset times, check that all depth have an onset
    onset_times = []
    skipped = 0
    t0 = param_log.HarpTime.min()
    for i_onset, ref_onset in enumerate(all_onsets[depth_with_least_onsets]):
        # find closest time in each element of all_onsets
        batch = np.zeros(len(all_onsets)) + np.inf
        for i_depth, onsets in enumerate(all_onsets.values()):
            closest = onsets[np.argmin(np.abs(onsets - ref_onset))]
            batch[i_depth] = closest
        if np.abs(batch - ref_onset).max() > jitter:
            skipped += 1
            continue
        # keep the first
        onset_times.append(np.min(batch))
    print(f"Skipped {skipped} onsets with bad jitter")
    onset_times = np.array(onset_times)
    # Same for offsets
    min_off = np.inf
    depth_with_least_offsets = None
    for k, v in all_offsets.items():
        if len(v) < min_off:
            depth_with_least_offsets = k
            min_off = len(v)

    # For each of the putative offset times, check that all depth have an offset
    offset_times = []
    skipped = 0
    for i_offset, ref_offset in enumerate(all_offsets[depth_with_least_offsets]):
        # find closest time in each element of all_offsets
        batch = np.zeros(len(all_offsets)) + np.inf
        for i_depth, offsets in enumerate(all_offsets.values()):
            closest = offsets[np.argmin(np.abs(offsets - ref_offset))]
            batch[i_depth] = closest
        if np.abs(batch - ref_offset).max() > jitter:
            skipped += 1
            continue
        # keep the last
        offset_times.append(np.max(batch))
    print(f"Skipped {skipped} offsets with bad jitter")
    offset_times = np.array(offset_times)

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
        n_on = len(onset_times)
        n_off = len(offset_times)
        txt = f"{n_on}/{min_on} valid onsets ({n_on/min_on*100:.2f}%), "
        txt += f"{n_off}/{min_off} valid offsets ({n_off/min_off*100:.2f}%)"
        print(txt)
        print(f"{len(trial_on_off[0])} valid trials left")
    if debug:
        return trial_on_off, param_log_index, all_onsets, all_offsets, closest_offset
    return trial_on_off, param_log_index


def trial_diagnostic(param_log, trial_on_off, all_offsets, all_onsets):
    loggers = list(natsorted(param_log["logger_fname"].unique()))

    fig, axes = plt.subplots(len(loggers), 2)
    fig.set_size_inches((15, 6))
    t0 = param_log.HarpTime.min()
    tmax = param_log.HarpTime.max()
    for iax, log in enumerate(loggers):
        depth = int(log.split("_")[-1].split(".")[0][:-2])
        for ax in [axes[iax, 0], axes[iax, 1]]:
            ax.vlines(trial_on_off[0] - t0, -0.5, 1.5, color="g", alpha=0.5, ls="--")

            df = param_log[param_log.logger_fname == log]
            dfv = df[df.Radius > 0]
            dfo = df[df.Radius < 0]
            ax.scatter(
                dfv.HarpTime - t0, np.zeros(dfv.shape[0]) - 0.2, marker=".", color="k"
            )
            ax.scatter(
                dfo.HarpTime - t0, np.zeros(dfo.shape[0]) - 0.2, marker=".", color="b"
            )
            ax.scatter(
                all_onsets[depth] - t0,
                np.zeros_like(all_onsets[depth]),
                marker=".",
                color="g",
            )
            ax.scatter(
                all_offsets[depth] - t0,
                np.zeros_like(all_offsets[depth]),
                marker=".",
                color="r",
            )
            ax.set_yticks([])
            ax.set_ylim(-0.5, 1.5)
        axes[iax, 0].set_ylabel(depth)
        axes[iax, 1].set_xlim((tmax - t0) / 2 + np.array([-60 * 2, 60 * 2]))
    for i in range(trial_on_off.shape[1]):
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.plot(
                [trial_on_off[0, i] - t0, trial_on_off[1, i] - t0], [1, 1], color="k"
            )
            ax.scatter(
                trial_on_off[0] - t0,
                np.ones_like(trial_on_off[0]),
                marker="v",
                color="g",
            )
            ax.scatter(
                trial_on_off[1] - t0,
                np.ones_like(trial_on_off[1]),
                marker="v",
                color="r",
            )

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    return fig
