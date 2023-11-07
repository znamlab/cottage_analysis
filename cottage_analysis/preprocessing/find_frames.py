"""
Module to find frames based on photodiode flicker
"""
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.signal as scsi
import matplotlib.pyplot as plt
from tqdm import tqdm
from cottage_analysis.utilities import continuous_data_analysis as cda
from znamutils.decorators import slurm_it


def sync_by_frame_alternating(
    photodiode,
    analog_time,
    frame_rate=144,
    photodiode_sampling=1000,
    plot=False,
    plot_start=10000,
    plot_range=1000,
    plot_dir=None,
):
    """Find frame refresh based on photodiode signal

    Signal is expected to alternate between high and low at each frame (flickering quad
    between white and black)

    Args:
        photodiode (np.ndarray): photodiode series extracted from harp
        analog_time (np.ndarray): analog time series extracted from harp
        frame_rate (float): Expected frame rate. Peaks separated by less than half a
                            frame will be ignored. Default to 144.
        photodiode_sampling (float): Sampling rate of the photodiode signal in Hz. Default to 1000.
        plot (bool): Should a summary figure be generated? Default to False.
        plot_start (int): sample to start the plot. Default to 10000.
        plot_range (int): samples to plot after plot_start. Default to 1000.
        plot_dir (Path or str): directory to save the figure. Default to None.

    Returns:
        frames_df (pd.DataFrame): a dataframe containing detected frame timing.
            It contains:
                - 'photodiode': photodiode value at peak
                - 'closest_frame': peak frame index.
                - 'peak_time': harptime for the photodiode to peak at each frame.
    """

    # Photodiode value above which the quad is considered white
    upper_thr = np.percentile(photodiode, 80)
    # Photodiode value above which the quad is considered black
    lower_thr = np.percentile(photodiode, 20)
    photodiode_df = pd.DataFrame({"photodiode": photodiode, "analog_time": analog_time})
    # Find peaks of photodiode
    distance = int(1 / frame_rate * 2 * photodiode_sampling)
    high_peaks, _ = scsi.find_peaks(photodiode, height=upper_thr, distance=distance)
    first_frame = high_peaks[0]
    low_peaks, _ = scsi.find_peaks(-photodiode, height=-lower_thr, distance=distance)
    low_peaks = low_peaks[low_peaks > first_frame]

    # Get rid of framedrops
    photodiode_df["FramePeak"] = None
    photodiode_df.FramePeak.iloc[high_peaks] = 1
    photodiode_df.FramePeak.iloc[low_peaks] = 0
    frames_df = photodiode_df[photodiode_df.FramePeak.notnull()]
    frames_df = frames_df[frames_df.FramePeak.diff() != 0]
    frames_df["closest_frame"] = np.arange(len(frames_df))
    frames_df = frames_df.drop(columns=["FramePeak"])
    frames_df = frames_df.rename(columns={"analog_time": "peak_time"})

    if plot:
        plt.figure()
        plt.plot(
            analog_time[plot_start : (plot_start + plot_range)],
            photodiode[plot_start : (plot_start + plot_range)],
        )
        all_frame_idxs = frames_df.index.values.reshape(-1)
        take_start = np.argmin(np.abs(all_frame_idxs - plot_start))
        take_stop = np.argmin(np.abs(all_frame_idxs - plot_start - plot_range))
        take_idxs = all_frame_idxs[take_start:take_stop]

        plot_peaks = np.intersect1d(
            all_frame_idxs, (np.arange(plot_start, (plot_start + plot_range), step=1))
        )
        plt.figure()
        plt.plot(
            frames_df.loc[take_idxs, "peak_time"],
            frames_df.loc[take_idxs, "photodiode"],
        )
        plt.plot(analog_time[plot_peaks], photodiode[plot_peaks], "x")
        plt.plot(
            analog_time[plot_start : (plot_start + plot_range)],
            np.zeros_like(photodiode[plot_start : (plot_start + plot_range)])
            + upper_thr,
            "--",
            color="gray",
        )
        plt.plot(
            analog_time[plot_start : (plot_start + plot_range)],
            np.zeros_like(photodiode[plot_start : (plot_start + plot_range)])
            + lower_thr,
            "--",
            color="gray",
        )
        plt.xlabel("Time(s)")
    if plot_dir is not None:
        plt.savefig(Path(plot_dir) / "Frame_finder_check.png")

    return frames_df


@slurm_it(conda_env="cottage_analysis")
def sync_by_correlation(
    frame_log,
    photodiode_time,
    photodiode_signal,
    time_column="HarpTime",
    sequence_column="PhotoQuadColor",
    frame_detection_height=0.1,
    num_frame_to_corr=5,
    maxlag=50e-3,
    expected_lag=15e-3,
    frame_rate=144,
    correlation_threshold=0.9,
    relative_corr_thres=0.03,
    minimum_lag=5e-3,
    do_plot=False,
    verbose=True,
    debug=False,
    save_folder=None,
    ignore_errors=False,
    last_frame_delay=100,
):
    """Find best shift to synchronise photodiode with ideal sequence

    This will generate an idealised photodiode trace based on the `switch_time` and the
    switch `sequence`. This trace will be synced by crosscorrelation

    Args:
        frame_log (pd.Dataframe): The logger from bonsai with RenderFrame values
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        time_column (str): Name of the column in `frame_log` containing time. Must
                           match photodiode_time (Default to HarpTime)
        sequence_column (str): Name of the column in `frame_log` containing sequence
                               information (Default to 'PhotoQuadColor')
        frame_detection_height (float): Height of the peak of diff of filtered
            photodiode signal to use for detection, relative to max diff peak. Default
            to 0.1
        num_frame_to_corr (int): number of frame before and after frame_time to keep
                                 for correlation
        maxlag (float): Maximum lag tested (in s, centered on expected_lag).
        expected_lag (float): expected lag (in s) to center search
        frame_rate (float): Frame rate in Hz
        correlation_threshold (float): threshold on the pearson correlation. Anything
                                       below is considered a failure to fit
        relative_corr_thres (float): When multiple correlations are above threshold,
                                     will consider only among those that less than
                                     `relative_corr_thres` away from max corr
        minimum_lag (float): Minimum possible lag. Anything below is considered a
                             failure to fit
        do_plot (bool): If True generate some quality measure plots during run and
                        return the figure handles
        verbose (bool): Print progress and general info.
        debug (bool): False by default. If True, returns a dict with intermediary results
        save_folder (str): If not None, and plot is True save figures results in this
            folder
        ignore_errors (bool): If True, will skip quality checks and try to force through
        last_frame_delay (float): If ignore_errors is True, frame detected more than
            this many seconds after the last rendered frame will be ignored

    Returns:
        frames_df (pd.DataFrame): dataframe with a line per detected frame
        extra_out (dict): A dictionary containing info to log as well as figures if
            `do_plot` is True, and debug information if debug is True.
    """

    pd_sampling = 1 / np.mean(np.diff(photodiode_time))

    # Normalise photodiode signal
    normed_pd = np.array(photodiode_signal, dtype=float)
    normed_pd -= np.quantile(normed_pd, 0.01)
    normed_pd /= np.quantile(normed_pd, 0.99)

    # First step: Frame detection
    frames_df, db_dict, figs = create_frame_df(
        frame_log=frame_log,
        photodiode_time=photodiode_time,
        photodiode_signal=normed_pd,
        time_column=time_column,
        frame_rate=frame_rate,
        height=frame_detection_height,
        do_plot=do_plot,
        verbose=verbose,
        debug=debug,
        save_folder=save_folder,
        ignore_errors=ignore_errors,
        last_frame_delay=last_frame_delay,
    )
    ndetected = len(frames_df)
    npresented = len(frame_log)
    if npresented < ndetected:
        msg = (
            f"Detected more frames ({ndetected}) than presented ({npresented})"
            "\n Check create_frame_df parameters"
        )
    elif npresented > ndetected * 2:
        msg = (
            f"Dropped more than half of the frames ({npresented - ndetected} dropped)"
            "\n Check create_frame_df parameters"
        )
    else:
        msg = None
    if msg is not None:
        if ignore_errors:
            warnings.warn(msg)
        else:
            raise ValueError(msg)

    if db_dict is not None:
        db_dict["normed_pd"] = normed_pd

    if figs is not None:
        fig_dict = dict(frame_dection=figs)
    else:
        fig_dict = dict()

    # Second step: cross correlation
    frames_df, db_di = run_cross_correlation(
        frames_df,
        frame_log,
        photodiode_time,
        normed_pd,
        time_column,
        sequence_column,
        num_frame_to_corr,
        maxlag,
        expected_lag,
        frame_rate,
        verbose,
        debug or do_plot,
        pd_sampling,
    )
    if db_di is not None:
        db_dict.update(db_di)

    # Now attempt the matching of the correlated frames to the logger (i.e. find if
    # `bef`, `center` and `aft` agree or if I can identify the best)
    frames_df = _match_fit_to_logger(
        frames_df,
        correlation_threshold=correlation_threshold,
        relative_corr_thres=relative_corr_thres,
        minimum_lag=minimum_lag,
        verbose=verbose,
    )
    extra_out = {}
    if do_plot:
        extra_out["figures"] = fig_dict
    if debug:
        extra_out["debug_info"] = db_dict

    # Then interpolate the missing frames
    interpolate_sync(frames_df, verbose=verbose)
    # and remove the last double detected frames
    frames_df = _remove_double_frames(frames_df, verbose=True)

    if do_plot and (save_folder is not None):
        if verbose:
            print("Plotting diagnostic figures")
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        plot_crosscorr_matrix(ax, db_dict["cc_dict"], db_dict["lags_sample"], frames_df)
        ax = fig.add_subplot(2, 1, 2)
        plot_crosscorr_matrix(ax, db_dict["cc_dict"], db_dict["lags_sample"], frames_df)
        xl = ax.get_xlim()
        mid = (xl[0] + xl[1]) / 2
        ax.set_xlim(mid - 100, mid + 100)
        fig.savefig(save_folder / "crosscorr_matrix.png")
        extra_out["figures"]["crosscorr_matrix"] = fig

        # plot a few example sync that succeeded and failed
        rng = np.random.default_rng(42)
        good = frames_df[frames_df.sync_reason == "consensus of 3"]
        good = good.sample(min(2, len(good)), random_state=rng).index
        ampl = frames_df[frames_df.sync_reason == "closest to photodiode"]
        ampl = ampl.sample(min(2, len(ampl)), random_state=rng).index
        bad = (
            frames_df[frames_df.closest_frame.isna()].sample(4, random_state=rng).index
        )
        toplot = list(good) + list(bad) + list(bad + 1) + list(bad - 1) + list(ampl)
        for frame in toplot:
            fig = plot_one_frame_check(
                frame,
                frames_df,
                frame_log,
                real_time=photodiode_time,
                normed_pd=normed_pd,
                ideal_time=db_dict["ideal_time"],
                ideal_pd=db_dict["ideal_photodiode_trace"],
                ideal_seqi=db_dict["ideal_seqi_trace"],
                num_frame_to_corr=None,
            )
            fig.suptitle(
                f"Frame {frame} matching frame log "
                f"{frames_df.loc[frame, 'closest_frame']}\n"
                f"Is interpolated: {not frames_df.loc[frame, 'interpolation_seeds']}"
            )
            fig.savefig(save_folder / f"frame_{frame}_check.png")
    if not debug:
        clean_df(frames_df)
    return frames_df, extra_out


def create_frame_df(
    frame_log,
    photodiode_time,
    photodiode_signal,
    time_column="HarpTime",
    frame_rate=144,
    height=0.1,
    do_plot=False,
    verbose=True,
    debug=False,
    save_folder=None,
    ignore_errors=False,
    last_frame_delay=100,
):
    """Create a dataframe with the frame information

    Args:
        frame_log (float): The logger from bonsai with RenderFrame values
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        time_column (str): Name of the column in `frame_log` containing time. Must
                           match photodiode_time (Default to HarpTime)
        frame_rate (float): Frame rate in Hz
        height (float): Height of the peak of diff of filtered photodiode signal to use
            for detection, relative to max diff peak. Default to 0.1
        do_plot (bool): If True generate some quality measure plots during run and
                        return the figure handles
        verbose (bool): Print progress and general info.
        debug (bool): False by default. If True, returns a dict with intermediary results
        save_folder (str): If not None, and plot is True save figures results in this
            folder
        ignore_errors (bool): If True, will skip quality checks and try to force through
        last_frame_delay (float): Frame detected more than this many seconds after the
            last rendered frame will be ignored (Default to 100)

    Returns:
        frames_df (pd.DataFrame): dataframe with a line per detected frame
        db_dict (dict): dict with intermediary results. None if debug is False
        figs (list): list of figure handles. None if do_plot is False
    """
    pd_sampling = 1 / np.mean(np.diff(photodiode_time))
    out = detect_frame_onset(
        photodiode=photodiode_signal,
        frame_rate=frame_rate,
        photodiode_sampling=pd_sampling,
        highcut=frame_rate * 3,
        debug=debug or do_plot,
        height=height,
    )
    if debug or do_plot:
        frame_borders, peak_index, out_dict = out
    else:
        out_dict = {}
        frame_borders, peak_index = out
    # cut frame detected before the recording started
    t0 = frame_log[time_column].iloc[0]
    to_cut = photodiode_time[frame_borders].searchsorted(t0)
    frame_borders = frame_borders[to_cut:]
    peak_index = peak_index[to_cut:]

    # Format the results in a nicer dataframe
    frame_skip = np.diff(frame_borders) > pd_sampling / frame_rate * 1.5
    frames_df = pd.DataFrame(
        dict(
            onset_sample=frame_borders[:-1],
            offset_sample=frame_borders[1:],
            peak_sample=peak_index,
            include_skip=frame_skip,
        )
    )
    frames_df["onset_time"] = photodiode_time[frames_df.onset_sample]
    frames_df["offset_time"] = photodiode_time[frames_df.offset_sample]
    frames_df["peak_time"] = photodiode_time[frames_df.peak_sample]

    # check if frames are detected after the presentation is over
    after_last = frames_df["onset_time"] >= frame_log[time_column].iloc[-1]
    if after_last.any():
        delay = frames_df["onset_time"].iloc[-1] - frame_log[time_column].iloc[-1]
        print(f"{after_last.sum()} frames detected after the last render time.")
        if last_frame_delay is not None:
            too_late = (
                frames_df.onset_time
                >= frame_log[time_column].iloc[-1] + last_frame_delay
            )
            print(
                f"Removing {too_late.sum()} frames detected more than "
                f"{last_frame_delay} s after the last render"
            )
            frames_df = frames_df[~too_late].copy()
            # recomputing delay
            delay = frames_df["onset_time"].iloc[-1] - frame_log[time_column].iloc[-1]
        if delay > 1:
            msg = (
                "Delay between last render and last frame is too large. "
                f"{delay:.2f}.  Something is wrong"
            )
            if not ignore_errors:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

    out_dict["detected_frames"] = len(frames_df)
    out_dict["logged_frames"] = len(frame_log)
    out_dict["percentage_drop"] = 100 * (1 - len(frames_df) / len(frame_log))
    out_dict["dropped_frames"] = len(frame_log) - len(frames_df)
    if verbose:
        print(
            f"Found {out_dict['detected_frames']} frames out of "
            f"{out_dict['logged_frames']} render"
            f" ({out_dict['percentage_drop']:.2f}%,"
            f" {out_dict['dropped_frames']} dropped)"
        )
    figs = None
    if do_plot:
        plot_window = np.array([-7.5, 7.5]) / frame_rate * pd_sampling
        figs = plot_frame_detection_report(
            border_index=frame_borders,
            peak_index=peak_index,
            debug_dict=out_dict,
            num_examples=1,
            plot_window=plot_window,
            photodiode=photodiode_signal,
            frame_rate=frame_rate,
            photodiode_sampling=pd_sampling,
            highcut=frame_rate * 3,
        )
        if save_folder is not None:
            for ifig, fig in enumerate(figs):
                fig.savefig(Path(save_folder) / f"frame_detection_fig{ifig}.png")

    return frames_df, out_dict, figs


def detect_frame_onset(
    photodiode,
    frame_rate=144.0,
    photodiode_sampling=1000.0,
    highcut=400.0,
    height=0.1,
    debug=False,
):
    """Detect frames from photodiode signal

    Simple wrapper around `scipy.signal.find_peaks` to detect frame borders from
    photodiode signal.

    Args:
        photodiode (np.array): photodiode signal
        frame_rate (float, optional): expected frame rate, peaks occuring faster than
            1.5 frame rate will be ignored
        photodiode_sampling (float, optional): Sampling rate of the photodiode signal
        highcut (float, optional): If not None, use low pass filter cutting components
            above `highcut` hertz
        height (float, optional): Minimum height of a peak. Default to 0.1
        debug (bool, optional): False by default. If True, returns a dict with
            intermediary results

    Returns:
        border_index (np.array): index of frame borders
        peak_index (np.array): index of peak of each frame, len(border_index) - 1
        debug_dict (dict): Dictionary of intermediary element. Only if `debug` == True
    """
    if highcut is not None:
        fpd = cda.filter(
            photodiode,
            highcut=frame_rate * 3,
            sampling=photodiode_sampling,
            design="bessel",
        )
        absdiff = np.abs(np.diff(fpd))
    else:
        fpd = None
        absdiff = np.abs(np.diff(photodiode))
    absdiff -= np.quantile(absdiff, 0.01)
    absdiff /= np.quantile(absdiff, 0.99)
    dst = int(1 / (frame_rate * 1.5) * photodiode_sampling)
    frame_border, _ = scsi.find_peaks(absdiff, distance=dst, height=height)

    # When two local peaks in the same frame have exactly the same amplitude,
    # find_peaks doesn't know which one to pick and keeps both. Let's keep only the last
    plateau = np.where(
        (np.diff(frame_border) <= dst) & (np.diff(absdiff[frame_border]) == 0)
    )[0]
    valid = np.ones(len(frame_border), dtype=bool)
    valid[plateau] = False
    borders = np.array(frame_border[valid])

    # Now find the peak of the frame, it is len(frame_border) - 1
    peaks = np.array(
        [
            absdiff[b:e].argmin() + b
            for b, e in zip(frame_border[:-1], frame_border[1:])
        ],
        dtype=int,
    )
    if debug:
        debug_dict = dict(
            plateau=plateau,
            all_pks=frame_border,
            distance=dst,
            filtered_trace=fpd,
            diff_trace=absdiff,
        )
        return borders, peaks, debug_dict
    return borders, peaks


def plot_frame_detection_report(
    photodiode,
    frame_rate=144,
    photodiode_sampling=1000,
    highcut=400,
    plot_window=(-50, 50),
    height=0.1,
    num_examples=1,
    border_index=None,
    peak_index=None,
    debug_dict=None,
    example_frames=None,
):
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
        height (float): Minimum height of a peak. Default to 0.1
        num_examples (int): number of figures randomly selected and with frame drop to
                            plot (len(fig) == num_examples * 2)
        border_index (np.array): index of frame borders
        peak_index (np.array): index of peak of each frame, len(border_index) - 1
        debug_dict (dict): Dictionary of intermediary element. Only if `debug` == True
        example_frames (np.array): If not None, use this array of frame index instead of
            randomly selecting them


    Returns:
        figs (list): a list of figure handles
    """
    if (border_index is None) or (peak_index is None) or (debug_dict is None):
        border_index, peak_index, debug_dict = detect_frame_onset(
            photodiode, frame_rate, photodiode_sampling, highcut, debug=True
        )
    if example_frames is None:
        rng = np.random.default_rng(42)
        skip = np.diff(border_index) > photodiode_sampling / frame_rate * 1.5
        example_frames = np.hstack(
            [
                border_index[rng.integers(len(border_index), size=num_examples)],
                border_index[:-1][skip][rng.integers(np.sum(skip), size=num_examples)],
            ]
        )
    figs = []
    w = np.array(plot_window, dtype=int)
    i = np.arange(len(photodiode))
    vlines_kwargs = [
        dict(color="Grey", alpha=0.2, ls="--", lw=0.5),
        dict(color="k", alpha=0.5, ls="-"),
    ]
    dot_kwargs = [
        dict(color="Grey", alpha=0.2, marker=".", s=5),
        dict(color="darkred", alpha=0.5, marker="o", s=50, ec="None"),
    ]
    for f in example_frames:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(photodiode[slice(*w + f)], label="raw")
        ax.plot(debug_dict["filtered_trace"][slice(*w + f)], label="filtered")
        ax.legend(loc="upper right")
        for iw, which_pk in enumerate([debug_dict["all_pks"], border_index]):
            v = (which_pk > w[0] + f) & (which_pk < w[1] + f)
            for d in i[which_pk[v]] - (w[0] + f):
                ax.axvline(d, **vlines_kwargs[iw])
        v = peak_index[(peak_index > w[0] + f) & (peak_index < w[1] + f)]
        ax.scatter(i[v] - (w[0] + f), photodiode[v], **dot_kwargs[1])
        ax.set_title("Frame at sample %d" % f)
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(debug_dict["diff_trace"][slice(*w + f)])
        ax.axhline(height, color="k", ls="--")
        for iw, which_pk in enumerate([debug_dict["all_pks"], border_index]):
            v = which_pk[(which_pk > w[0] + f) & (which_pk < w[1] + f)]
            ax.scatter(i[v] - (w[0] + f), debug_dict["diff_trace"][v], **dot_kwargs[iw])
        ax.set_ylabel("Abs diff trace")
        ax.set_xlabel("Sample")
        figs.append(fig)
    return figs


def run_cross_correlation(
    frames_df,
    frame_log,
    photodiode_time,
    photodiode_signal,
    time_column,
    sequence_column,
    num_frame_to_corr,
    maxlag,
    expected_lag,
    frame_rate,
    verbose,
    debug,
    pd_sampling=None,
):
    """Run cross correlation between photodiode signal and frame log

    Will test correlations with lags in [expected_lag-maxlag, expected_lag+maxlag]

    Args:
        frames_df (pd.Dataframe): Dataframe containing frame information as created by
            `create_frame_df`
        frame_log (pd.Dataframe): The logger from bonsai with RenderFrame values
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        time_column (str): Name of the column in `frame_log` containing time. Must
                           match photodiode_time (Default to HarpTime)
        sequence_column (str): Name of the column in `frame_log` containing sequence
                               information (Default to 'PhotoQuadColor')
        num_frame_to_corr (int): number of frame before and after frame_time to keep
                                 for correlation
        maxlag (float): Maximum lag tested (in s, centered on expected_lag).
        expected_lag (float): expected lag (in s) to center search
        frame_rate (float): Frame rate in Hz
        verbose (bool): Print progress and general info.
        debug (bool): False by default. If True, returns a dict with intermediary results
        pd_sampling (float): Sampling rate of the photodiode signal. If None, will be
            computed from photodiode_time

    Returns:
        frames_df (pd.DataFrame): dataframe with a line per detected frame

    """
    if pd_sampling is None:
        pd_sampling = 1 / np.mean(np.diff(photodiode_time))
    out_dict = {}
    frame_onsets = frames_df["onset_sample"].values
    # make lags into samples
    maxlag_samples = int(np.round(maxlag * pd_sampling))
    expected_lag_samples = int(np.round(expected_lag * pd_sampling))

    # make an idealised photodiode signal
    ideal_time, ideal_seqi_trace, ideal_pd = ideal_photodiode(
        frame_log,
        sampling_rate=pd_sampling,
        sequence_column="PhotoQuadColor",
        time_column="HarpTime",
        pad_frames=(maxlag + num_frame_to_corr) * 2,
        highcut=150,
    )
    if debug:
        out_dict["ideal_photodiode_trace"] = ideal_pd
        out_dict["ideal_time"] = ideal_time
        out_dict["ideal_seqi_trace"] = ideal_seqi_trace

    # find the closest switch time for each frame according to computer time
    real_switch_times = frame_log[time_column].values
    closest_switch = np.clip(
        real_switch_times.searchsorted(photodiode_time[frame_onsets]),
        0,
        len(real_switch_times) - 1,
    )
    frames_df["closest_frame_log_index"] = closest_switch
    # and the corresponding ideal photodiode sample
    ideal_onset = frame_log["ideal_switch_samples"].iloc[closest_switch].values

    # run the cross correlation
    out = _crosscorr_befcentaft(
        frame_onsets,
        photodiode_time=photodiode_time,
        photodiode_signal=photodiode_signal,
        ideal_onset_samples=ideal_onset,
        ideal_photodiode_trace=ideal_pd,
        ideal_frame_index=ideal_seqi_trace,
        expected_lag=expected_lag_samples,
        maxlag=maxlag_samples,
        num_frame_to_corr=num_frame_to_corr,
        frame_rate=frame_rate,
        verbose=verbose,
        debug=debug,
    )

    if debug:
        cc_dict, id_dict, lags, db, residuals = out
        out_dict.update(db)
        out_dict["lags_sample"] = lags
        out_dict["cc_dict"] = cc_dict
        out_dict["ideal_pd"] = ideal_pd
        out_dict["ideal_onset"] = ideal_onset

    else:
        cc_dict, id_dict, lags, residuals = out

    # add lag and corresponding frame index to the dataframe
    if verbose:
        print("Adding cross correlation results to dataframe")

    for iw, which in enumerate(["bef", "center", "aft"]):
        cc = cc_dict[which]
        frames_df[f"residuals_{which}"] = residuals[iw]
        frames_df[f"lag_{which}"] = lags[cc.argmax(axis=1)] / pd_sampling
        frames_df[f"peak_corr_{which}"] = cc.max(axis=1)
        # to find the match between photiodiode and frame log, we want to look at what is
        # the value of the lag-shifed sequence index in during the real frame.
        # The sync is made the closest computer frame log time
        time_of_match = (
            frame_log["ideal_switch_times"]
            .iloc[frames_df.closest_frame_log_index]
            .values
        )
        # we remove the lag to frames_df instead of adding it to frame_log
        time_of_match -= frames_df["lag_%s" % which].values
        # This gives us the time relative to the onset of the real photodiode signal
        # However, if there is a frame drop, onset is uselss when matching `aft`
        # Furthermore the detected frame is almost centered on the offset of the frame
        # as the signal is filtered strongly. So we have to look a bit before the detect
        # frame to find the actual sequence value that produced the signal.
        # So we match:
        # * onset + 0.2 frame for bef
        # * peak - 0.3 frame for center
        # * offset - 0.7 frame for aft
        matching_type = ["onset", "peak", "offset"][iw]
        time_to_onset = (
            frames_df[f"{matching_type}_time"].values - frames_df["onset_time"].values
        )
        time_of_match += time_to_onset
        time_of_match += [0.2, -0.3, -0.7][iw] / frame_rate
        frames_df[f"ideal_time_of_match_{which}"] = time_of_match
        index_of_match = ideal_time.searchsorted(time_of_match)
        too_late = index_of_match == len(ideal_time)
        if any(too_late):
            naft = too_late.sum()
            last = time_of_match.max() - ideal_time.max()
            warnings.warn(
                f"{naft} frames detected after the end of the ideal photodiode trace "
                f"(worse match {last:.2f} s after)"
            )
            index_of_match[too_late] = len(ideal_time) - 1
        cl = ideal_seqi_trace[index_of_match]
        frames_df[f"closest_frame_{which}"] = cl
        frames_df["quadcolor_%s" % which] = frame_log.iloc[cl][sequence_column].values

    # also add photodiode value at peak
    frames_df["photodiode"] = photodiode_signal[
        frames_df["peak_sample"].values.astype(int)
    ]
    # use this measure to find frame "jumps", where there is no alternation
    diff_sign = np.sign(np.diff(frames_df.photodiode))
    jumps = diff_sign[1:] == diff_sign[:-1]
    frames_df["is_jump"] = np.hstack([0, jumps, 0])
    return frames_df, out_dict


def ideal_photodiode(
    frame_log,
    sampling_rate,
    sequence_column="PhotoQuadColor",
    time_column="HarpTime",
    pad_frames=10,
    highcut=150,
):
    """Make an idealise photodiode trace from sequence

    The photodiode is imaging a quad that changes color every frame but the photodiode
    filter and the pixel response time make this alternation smooth and not step-wise.
    This function attempts to filter the raw sequence to re-create an ideal version of
    what the photodiode signal should look like.

    The output will have the first frame of the sequence at sample 0 and nth frame at
    sample `n / frame_rate * sampling_rate`. This will not correspond to real photodiode
    signal since it does not include any frame drop

    This will also had 2 columns in frame_log: "ideal_switch_times" and
    "ideal_switch_samples"

    Args:
        frame_log (pd.DataFrame): DataFrame with the frame log
        sampling_rate (float): Sampling rate of the photodiode signal
        sequence_column (str, optional): Name of the column in frame_log that contains
            the sequence. Defaults to "PhotoQuadColor".
        time_column (str, optional): Name of the column in frame_log that contains the
            computer time of the frame. Defaults to "HarpTime".
        pad_frames (int, optional): Number of frames to pad at the beginning and end of
            the trace. Defaults to 10.
        highcut (float, optional): Frequency for low pass filter. Defaults to 150.

    Returns:
        ideal_time (np.array): Time base for the ideal photodiode
        ideal_frame_index (np.array): Continuous index of the sequence index
        fake_photodiode (np.array): Filtered version of perfect_sequence
    """

    sequence = frame_log[sequence_column].values
    computer_switch_time = frame_log[time_column].values
    actual_rate = 1 / np.median(np.diff(computer_switch_time))
    ideal_switch_times = np.arange(len(sequence) + 1) / actual_rate
    frame_log["ideal_switch_times"] = ideal_switch_times[:-1]
    samples_one_frame = int(sampling_rate / actual_rate)

    # add padding at the beginning and end
    padding_samples = int(pad_frames * samples_one_frame)

    switch_samples = (ideal_switch_times * sampling_rate + padding_samples).astype(int)
    frame_log["ideal_switch_samples"] = switch_samples[:-1]
    ideal_time = np.arange(-padding_samples, switch_samples[-1] + padding_samples + 1)
    ideal_time = ideal_time.astype(float) / sampling_rate

    perfect_sequence = np.zeros_like(ideal_time)
    ideal_frame_index = np.zeros_like(ideal_time, dtype=int) - 1
    for i, v in enumerate(sequence):
        perfect_sequence[switch_samples[i] : switch_samples[i + 1]] = v
        ideal_frame_index[switch_samples[i] : switch_samples[i + 1]] = i

    freq = highcut / sampling_rate
    sos = scsi.butter(N=1, Wn=freq, btype="lowpass", output="sos")
    fake_photodiode = scsi.sosfilt(sos, perfect_sequence)
    return ideal_time, ideal_frame_index, fake_photodiode


def _crosscorr_befcentaft(
    frame_onsets,
    photodiode_time,
    photodiode_signal,
    ideal_onset_samples,
    ideal_photodiode_trace,
    ideal_frame_index,
    expected_lag,
    maxlag,
    num_frame_to_corr,
    frame_rate,
    verbose=True,
    debug=False,
    crosscorr_normalisation="pearson",
):
    """Run three crosscorrelations before, centered on and after each frame time

    Inner function of `sync_by_correlation`.

    Args:
        frame_onsets (np.array): sample of photodiode signal at which each frame starts
        photodiode_time (np.array): Time of photodiode signal. Expected to be regularly
                                    sampled
        photodiode_signal (np.array): Photodiode signal, same size as photodiode time
        switch_time (np.array): Time of all changes of photodiode quad colour
        ideal_onset_samples (np.array): Esitmated onset sample of each frame of frames_df
            in the ideal photodiode, assuming 0 lag.
        ideal_photodiode_trace (np.array): Continuous version of photodiode_value (same
            sampling as actual photodiode)
        ideal_frame_index (np.array): Continuous index of the sequence index
        expected_lag (int): expected lag (in samples) to center search
        maxlag (int): Maximum lag tested (in samples, centered on expected_lag).
        num_frame_to_corr (int): number of frame around frame_time to keep for correlation
        frame_rate (float): Frame rate in Hz
        verbose (bool): Print time taken. Default True
        debug (bool): Return intermediary results Default False
        crosscorr_normalisation (str): Method to use for cross correlation. One of
            `dot`, `pearson` or `difference`. Default `pearson`

    Returns:
        cc_mat (np.array): a (3 x len(frame_onsets) x len(lags)) array of correlation
                           coefficients
        lags (np.array): lag in samples
        db_dict (dict): only if debug=True. Dictionnary with intermediary results
    """
    pd_sampling = 1 / np.mean(np.diff(photodiode_time))

    # define the 3 correlation windows, bef, center and aft
    window = [
        np.array([-1, 1]) * maxlag
        + np.array(w * num_frame_to_corr / frame_rate * pd_sampling, dtype="int")
        for w in [np.array([-1, 0]), np.array([-0.5, 0.5]), np.array([0, 1])]
    ]
    # for bef window, we add 1.5 frame to have half of the current frame included
    window[0] += int(1.5 / frame_rate * pd_sampling)
    # for center window, we shift by 0.5 frame to center
    window[1] += int(0.5 / frame_rate * pd_sampling)

    if verbose:
        start = time.time()
        print("Starting crosscorrelation", flush=True)
    cc_mat = np.zeros((len(window), len(frame_onsets), maxlag * 2)) + np.nan
    eq_ind = np.zeros((len(window), len(frame_onsets), maxlag * 2), dtype="int") - 1
    residuals = np.zeros((len(window), len(frame_onsets))) + np.nan
    for iframe, foi in tqdm(enumerate(frame_onsets), total=len(frame_onsets)):
        for iw, win in enumerate(window):
            if (win[0] + foi) < 0:
                if verbose:
                    print(
                        "Frame %d at sample %d is too close from start of recording"
                        % (iframe, foi)
                    )
                continue
            elif (win[1] + foi) > (len(photodiode_signal) - expected_lag):
                if verbose:
                    print(
                        "Frame %d at sample %d is too close from end of recording"
                        % (iframe, foi)
                    )
                continue
            elif (win[0] + ideal_onset_samples[iframe] - expected_lag) < 0:
                if verbose:
                    print(
                        "Frame %d at sample %d is too close from start of ideal pd"
                        % (iframe, foi)
                    )
                continue
            elif (win[1] + ideal_onset_samples[iframe] - expected_lag) > len(
                ideal_photodiode_trace
            ):
                if verbose:
                    print(
                        "Frame %d at sample %d is too close from end of ideal pd"
                        % (iframe, foi)
                    )
                continue
            # ideal_pd is drifting, so we need to look for the closest computer time
            id_t = ideal_frame_index[
                slice(*win + ideal_onset_samples[iframe] - expected_lag)
            ]
            # we want the middle "maxlag * 2" samples, which is where correlation can
            # be done
            eq_ind[iw, iframe] = id_t[
                int(len(id_t) / 2 - maxlag) : int(len(id_t) / 2 + maxlag)
            ]

            corr, lags = cda.crosscorrelation(
                photodiode_signal[slice(*win + foi)],
                ideal_photodiode_trace[
                    slice(*win + ideal_onset_samples[iframe] - expected_lag)
                ],
                maxlag=maxlag,
                expected_lag=0,
                normalisation=crosscorr_normalisation,
            )
            if crosscorr_normalisation.lower() == "difference":
                corr *= -1
            cc_mat[iw, iframe] = corr
            lag = lags[corr.argmax()]
            residuals[iw, iframe] = np.nanmean(
                np.abs(
                    photodiode_signal[slice(*win + foi)]
                    - ideal_photodiode_trace[
                        slice(*win + ideal_onset_samples[iframe] - expected_lag - lag)
                    ]
                )
            )
    lags += expected_lag
    if verbose:
        end = time.time()
        print("done (%d s)" % (end - start), flush=True)
    cc_dict = {l: cc_mat[i] for i, l in enumerate(["bef", "center", "aft"])}
    id_dict = {l: eq_ind[i] for i, l in enumerate(["bef", "center", "aft"])}
    if debug:
        db_dict = dict(window=window)
        return cc_dict, id_dict, lags, db_dict, residuals
    return cc_dict, id_dict, lags, residuals


def _match_fit_to_logger(
    frames_df,
    correlation_threshold=0.8,
    relative_corr_thres=0.03,
    minimum_lag=5e-3,
    verbose=True,
):
    """Remove bad fit and pick the best of remaining

    Inner function of sync_by_correlation

    frames_df has bef, center and aft crosscorrelation, find which one are reasonable
    and pick the best among those

    More precisely: correlation coefficient will be filtered by `correlation_threshold`
    and `minimum_lag`. If all remaining correlation after filtering match the same
    frame, this frame is selected.
    If multiple frames remain and one correlation coefficient is `relative_corr_thres`
    above the others, this frame is selected.
    If correlation coefficient are within `relative_corr_thres` of each other, the
    frame the will get a photodiode signal closest to the ideal one is selected.
    If no correlation coefficient is left after filtering, keep lag and frame to NaN

    Args:
        frames_df (pd.DateFrame): Dataframe containing crosscorrelation information
        frame_log (pd.DateFrame): Dataframe from bonsai logger
        correlation_threshold (float): threshold on the pearson correlation. Anything
                                       below is considered a failure to fit
        relative_corr_thres (float): When multiple correlations are above threshold,
                                     will consider only among those that less than
                                     `relative_corr_thres` away from max corr
        minimum_lag (float): Minimum possible lag. Anything below is considered a
                             failure to fit
        clean_df (bool): Should the output contain all columns) (if clean_df=False,
                         default) or only the one selected and not the `bef`, `center`
                         and `aft` version
        verbose (bool): Print progress?

    Returns:
        df (pd.DataFrame): A dataframe with one of 3 crosscorr selected.
    """
    if verbose:
        start = time.time()
        print("Matching fit to logger", flush=True)

    # Initialize the dataframe columns
    frames_df["closest_frame"] = np.nan
    frames_df["lag"] = np.nan
    frames_df["residuals"] = np.nan
    frames_df["sync_reason"] = "not done"
    frames_df["crosscorr_picked"] = "not done"

    labels = ["bef", "center", "aft"]
    # Exclude correlation that are too low
    peak_correlations = frames_df.loc[:, ["peak_corr_%s" % l for l in labels]].values
    did_not_fit = peak_correlations < correlation_threshold

    # and impossible lags
    impossible_lag = (
        frames_df.loc[:, ["lag_%s" % l for l in labels]].values < minimum_lag
    )
    bad = did_not_fit | impossible_lag

    # find frames for which all good correlation agree
    frames = frames_df.loc[:, ["closest_frame_%s" % l for l in labels]].values.astype(
        float
    )
    frames[bad] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean_frame = np.nanmean(frames, axis=1)
    good = np.nansum(np.abs(frames - mean_frame[:, np.newaxis]), axis=1) == 0
    # remove the all bad lines (3 nans, wich return 0 when nansummed)
    all_bad = np.all(bad, axis=1)
    good[all_bad] = False

    # for the good I can take whichever is valid
    first_valid = np.nanargmax(frames[good], axis=1)
    lab = [labels[i] for i in first_valid]
    goodi = np.where(good)[0]
    frames_df.loc[good, "crosscorr_picked"] = lab
    frames_df.loc[good, "closest_frame"] = [
        frames_df.loc[g, f"closest_frame_{l}"] for g, l in zip(goodi, lab)
    ]
    frames_df.loc[good, "lag"] = [
        frames_df.loc[g, f"lag_{l}"] for g, l in zip(goodi, lab)
    ]
    frames_df.loc[good, "residuals"] = [
        frames_df.loc[g, f"residuals_{l}"] for g, l in zip(goodi, lab)
    ]
    frames_df.loc[good, "sync_reason"] = [
        f"consensus of {n}" for n in np.sum(~bad[good], axis=1)
    ]
    if verbose:
        print(
            "Sync'ed %d frames easily. That's %d%% of the recording."
            % (np.sum(good), np.sum(good) / len(good) * 100),
            flush=True,
        )

    # for the bad, It's hopeless
    frames_df.loc[all_bad, "sync_reason"] = "not synced"
    frames_df.loc[all_bad, "crosscorr_picked"] = "none"
    if verbose:
        print(
            f"{np.sum(all_bad)} frames cannot be sync'ed. "
            + f"That's {np.sum(all_bad)/len(all_bad)*100:.1f}% of the recording.",
            flush=True,
        )

    remaining = ~good & ~all_bad
    best_corr = np.max(peak_correlations, axis=1)
    valid_corr = peak_correlations - best_corr[:, np.newaxis] > -relative_corr_thres
    single_best = np.sum(valid_corr, axis=1) == 1
    use_best = single_best & remaining

    ind_best = np.argmax(peak_correlations[use_best], axis=1)
    lab = [labels[i] for i in ind_best]
    ubi = np.where(use_best)[0]
    frames_df.loc[use_best, "crosscorr_picked"] = lab
    frames_df.loc[use_best, "closest_frame"] = [
        frames_df.loc[g, f"closest_frame_{l}"] for g, l in zip(ubi, lab)
    ]
    frames_df.loc[use_best, "residuals"] = [
        frames_df.loc[g, f"residuals_{l}"] for g, l in zip(ubi, lab)
    ]
    frames_df.loc[use_best, "lag"] = [
        frames_df.loc[g, f"lag_{l}"] for g, l in zip(ubi, lab)
    ]
    frames_df.loc[use_best, "sync_reason"] = "relative peak corr"
    if verbose:
        print(
            f"Sync'ed {np.sum(use_best)} frames based on relative peak corr coeff. "
            + f"That's {np.sum(use_best) / len(use_best) * 100:.1f}% of the recording.",
            flush=True,
        )

    remaining = ~good & ~all_bad & ~use_best

    sequence_value = frames_df.loc[:, ["quadcolor_%s" % l for l in labels]].values
    dst2photodiode = np.abs(sequence_value - frames_df.photodiode.values[:, np.newaxis])
    # put the distance of invalid correlation to high value
    dst2photodiode[bad | ~valid_corr] = 2
    closest = np.argmin(dst2photodiode, axis=1)

    lab = [labels[i] for i in closest[remaining]]
    ri = np.where(remaining)[0]
    frames_df.loc[remaining, "crosscorr_picked"] = lab
    frames_df.loc[remaining, "closest_frame"] = [
        frames_df.loc[g, f"closest_frame_{l}"] for g, l in zip(ri, lab)
    ]
    frames_df.loc[remaining, "lag"] = [
        frames_df.loc[g, f"lag_{l}"] for g, l in zip(ri, lab)
    ]
    frames_df.loc[remaining, "residuals"] = [
        frames_df.loc[g, f"residuals_{l}"] for g, l in zip(ri, lab)
    ]
    frames_df.loc[remaining, "sync_reason"] = "closest to photodiode"

    if verbose:
        print(
            f"Sync'ed {np.sum(remaining)} frames based on photodiode value. "
            + f"That's {np.sum(remaining) / len(remaining) * 100:.1f}% of the recording.",
            flush=True,
        )

    if verbose:
        end = time.time()
        print("done (%d s)" % (end - start), flush=True)
    return frames_df


def interpolate_sync(frames_df, residuals_threshold=0.08, verbose=True):
    """Interpolate closest_frame and lag for frames that could not be sync'ed

    Args:
        frames_df (pd.DataFrame): the dataframe with the initial match
        verbose (bool, optional): print progress. Defaults to True.
    """
    # move initial value to other column
    frames_df["closest_frame_initital_guess"] = frames_df.closest_frame
    frames_df["lag_initial_guess"] = frames_df.lag

    # remove double detected frames to avoid diff==0 later
    _remove_double_frames(frames_df, verbose=True)

    consensus_frames = frames_df.sync_reason == "consensus of 3"
    ok_res = frames_df.residuals < residuals_threshold
    non_nan = frames_df.closest_frame.notna()
    good_frames = consensus_frames & ok_res & non_nan
    if verbose:
        print(f"Found {consensus_frames.sum()} consensus frames")
        print(f"Found {ok_res.sum()} frames with residuals < {residuals_threshold}")
        print(f"Found {non_nan.sum()} frames with non-nan closest_frame")
        print(f"{np.sum(good_frames)}/{len(good_frames)} frames left for interpolation")

    # interpolate closest_frame
    closest_frame = frames_df.closest_frame.values
    out_of_order = 1
    while np.sum(out_of_order):
        goodi = np.where(good_frames)[0]
        out_of_order = np.diff(closest_frame[goodi]) <= 0
        good_frames[goodi[1:][out_of_order]] = False
        good_frames[goodi[:-1][out_of_order]] = False
        if verbose:
            print(f"Found {np.sum(out_of_order)} out of order frames")
            print(f"{np.sum(good_frames)}/{len(good_frames)} frames left")

    assert all(
        np.diff(closest_frame[good_frames]) > 0
    ), "closest_frame should be sorted"

    frames_df["interpolation_seeds"] = False
    frames_df.loc[good_frames, "interpolation_seeds"] = True

    closest_frame[~good_frames] = np.interp(
        frames_df.loc[~good_frames, "onset_time"],
        frames_df.loc[good_frames, "onset_time"],
        frames_df.loc[good_frames, "closest_frame"],
        left=np.nan,
        right=np.nan,
    )
    closest_frame = np.round(closest_frame)
    frames_df.loc[:, "closest_frame"] = closest_frame
    frames_df.loc[~good_frames, "lag"] = np.nan
    if verbose:
        out_of_order = np.diff(closest_frame) < 0
        double_match = np.diff(closest_frame) == 0
        print(
            f"Found {np.sum(out_of_order)} out of order frames and "
            f"{np.sum(double_match)} frames matching the same index "
            "after interpolation"
        )


def _remove_double_frames(frames_df, verbose=True):
    """Clean-up the frame matching order, after the initial match

    The initial match is done frame by frame in _match_fit_to_logger, but this does not
    work for all frames. In particular, sometimes 2 frames in a row are matched to the
    same logger frame. This function tries to fix this.

    Args:
        frames_df (pd.DataFrame): the dataframe with the initial match
        verbose (bool, optional): print progress. Defaults to True.

    Returns:
        pd.DataFrame: the cleaned-up dataframe
    """

    # Find frames that are matched to the same logger frame
    no_increase = frames_df.iloc[:-1][
        np.diff(frames_df.closest_frame.values) == 0
    ].index

    # look if the frame at n-1 matches closest_frame-1 and n+2 matches closest_frame+1
    bef = np.clip(no_increase - 1, 0, len(frames_df) - 1)
    aft = np.clip(no_increase + 2, 0, len(frames_df) - 1)
    frame_bef = frames_df.loc[bef, "closest_frame"].values
    frame_aft = frames_df.loc[aft, "closest_frame"].values
    n_skiped = frame_aft - frame_bef
    # now three options:
    # Option1: we have 1, 2, 2, 4, then 3 is missing and n_skiped == 3. We can fix that.
    dfi = no_increase[n_skiped == 3]
    frames_df.loc[dfi, "closest_frame"] = frame_bef[n_skiped == 3] + 1
    frames_df.loc[dfi + 1, "closest_frame"] = frame_bef[n_skiped == 3] + 2
    if verbose and np.sum(n_skiped == 3):
        print(
            f"{np.sum(n_skiped == 3)} frames of by one and will be fixed "
            + f"{np.sum(n_skiped == 3) / len(n_skiped) * 100:.1f}%",
            flush=True,
        )

    # Option2: we have 1, 2, 2, X with X larger than 4, (or x<3, 5, 5, 6) We can't fix
    # that, put both frames to nan
    frames_df.loc[no_increase[n_skiped > 3], "closest_frame"] = np.nan
    frames_df.loc[no_increase[n_skiped > 3] + 1, "closest_frame"] = np.nan

    # Option3: we have 1, 2, 2, 3, then 2 has been detected twice. n_skiped == 2, we
    # merge the two frames
    dfi = no_increase[n_skiped == 2]
    for w in ["time", "sample"]:
        frames_df.loc[dfi, f"offset_{w}"] = frames_df.loc[dfi + 1, f"offset_{w}"].values
    frames_df.drop(dfi + 1, inplace=True)
    frames_df.reset_index(drop=True, inplace=True)
    if verbose and np.sum(n_skiped == 2):
        print(
            f"{np.sum(n_skiped == 2)} frames are double and will be removed. "
            + f"That's {np.sum(n_skiped == 2) / len(n_skiped) * 100:.1f}% of the "
            + "cases where the closest frame is the same.",
            flush=True,
        )
    return frames_df


def clean_df(frames_df):
    """Remove columns that are not needed anymore

    Args:
        frames_df (pd.DataFrame): the dataframe to clean-up

    """
    columns = [
        c
        for c in frames_df.columns
        if ("_bef" in c) or ("_aft" in c) or ("_center" in c)
    ]
    frames_df.drop(columns, axis=1, inplace=True)


def plot_one_frame_check(
    frame,
    frames_df,
    frame_log,
    real_time,
    normed_pd,
    ideal_time,
    ideal_pd,
    ideal_seqi,
    num_frame_to_corr=None,
):
    """Plot the photodiode signand the frame detection for one frame.

    Args:
        frame (int): the frame to plot
        frames_df (pd.DataFrame): the dataframe with the frame detection
        frame_log (pd.DataFrame): the dataframe with the frame log
        real_time (np.array): the time of the photodiode signal
        normed_pd (np.array): the photodiode signal
        ideal_time (np.array): the idealised time of the photodiode signal
        ideal_pd (np.array): the idealised photodiode signal
        ideal_seqi (np.array): the idealised sequence index
        num_frame_to_corr (int, optional): the number of frame to use for the
            correlation. If provide will indicate area of correlation. Defaults to None.

    Returns:
        None
    """
    if num_frame_to_corr is not None:
        frame_rate = 1 / np.median(np.diff(frame_log["HarpTime"].values))
        window = [
            np.array(w * num_frame_to_corr / frame_rate)
            for w in [np.array([-1, 0]), np.array([-0.5, 0.5]), np.array([0, 1])]
        ]
        # for bef window, we add 1 frame to have the current frame included
        window[0] += int(1.5 / frame_rate)
        # for center window, we shift by 0.5 frame to center
        window[1] += int(0.5 / frame_rate)

    fig = plt.figure(figsize=(12, 8))
    for iw, which in enumerate(["bef", "center", "aft"]):
        ax = fig.add_subplot(3, 1, iw + 1)

        fd = frames_df.loc[frame]
        fl = frame_log.iloc[int(fd["closest_frame_log_index"])]

        # plot real photodiode signal
        real_t0 = fd.onset_time
        w = np.array([-100, 100]) + fd["onset_sample"]
        ax.plot(
            real_time[slice(*w)] - real_t0,
            normed_pd[slice(*w)],
            label="photodiode",
            color="C0",
        )
        if num_frame_to_corr is not None:
            t = real_time[slice(*w)] - real_t0
            ok = (t > window[iw][0]) & (t < window[iw][1])
            ax.plot(
                t[ok], normed_pd[slice(*w)][ok], label="__nolegend__", lw=3, color="C0"
            )

        ax.axvspan(
            fd.onset_time - real_t0,
            fd.offset_time - real_t0,
            alpha=0.2,
            ymax=0.5,
            color="k",
        )
        ax.text(
            fd.onset_time
            - real_t0
            + (fd.offset_time - real_t0 - fd.onset_time + real_t0) / 2,
            0.3,
            f"{frame}, matching {fd[f'closest_frame_{which}']}",
            ha="center",
            va="center",
        )

        # plot idealised photodiode signal
        w = np.array([-100, 100]) + int(fl["ideal_switch_samples"])
        ideal_t0 = fl["ideal_switch_times"]
        ax.plot(
            ideal_time[slice(*w)] - ideal_t0 + fd[f"lag_{which}"],
            ideal_pd[slice(*w)],
            label="ideal",
            color="C1",
        )

        seq = ideal_seqi[slice(*w)]
        t = ideal_time[slice(*w)] - ideal_t0 + fd[f"lag_{which}"]
        switch = np.where(np.diff(seq) != 0)[0]
        for i, s in enumerate(switch[:-1]):
            b = t[s]
            e = t[switch[i + 1]]
            if b < -70e-3 or e > 70e-3:
                continue

            index = seq[s + 2]
            ax.axvspan(b, e, alpha=0.2, ymin=0.5, color=f"C{i%2}")
            ax.text(
                b + (e - b) / 2, 0.9, f"{index}", ha="center", va="center", rotation=90
            )

        ax.axvline(
            fd[f"ideal_time_of_match_{which}"] - ideal_t0 + fd[f"lag_{which}"],
            color="C1",
        )
        pc = fd[f"peak_corr_{which}"]
        r = fd[f"residuals_{which}"]
        if "crosscorr_picked" not in fd:
            ax.set_ylabel(f"{which} (c={pc:.2f}, r={r:.2f})", color="k")
        elif fd.crosscorr_picked == which:
            ax.set_ylabel(
                f"{which} (c={pc:.2f}, r={r:.2f}\n{fd.sync_reason})", color="r"
            )
        else:
            ax.set_ylabel(f"{which} (c={pc:.2f}, r={r:.2f})", color="k")
        if iw < 2:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-70e-3, 70e-3)
    fig.subplots_adjust(hspace=0)

    return fig


def plot_crosscorr_matrix(ax, cc_dict, lags, frames_df):
    """Plot the cross correlation matrix.

    It is the amplitude of the correlation for each frame/lag pair, selecting the
    picked `bef`, `center` or `aft` window for each frame.

    Args:
        ax (plt.Axes): the axes to plot on
        cc_dict (dict): the cross correlation dictionary
        lags (np.array): the lags used for the cross correlation
        frames_df (pd.DataFrame): the dataframe with the frame detection

    Returns:
        None
    """
    cc = np.zeros(cc_dict["center"].shape) + 0
    for i, picked in frames_df.crosscorr_picked.items():
        if picked == "none":
            continue
        cc[i] = cc_dict[picked][i]

    ax.imshow(
        cc.T,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=[0, len(cc), lags[0], lags[-1]],
        aspect="auto",
        origin="lower",
    )
    nans = np.where((frames_df.crosscorr_picked == "none").values)[0]
    ax.scatter(
        nans,
        np.ones(nans.shape),
        color="k",
        marker="|",
        alpha=0.1,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Lag")


def remove_frames_in_wrong_order(monitor_frames_df):
    monitor_frames_df = monitor_frames_df.copy()
    # Remove monitor frames with wrong order of frame indices
    print(f"Removing frames in wrong order of frame indices.")
    removed_frames = True
    while removed_frames:
        old_length = len(monitor_frames_df)
        diffs = monitor_frames_df.closest_frame.diff()
        diffs.iloc[0] = 1
        # check whether the negative diff precedes or follows a large positive diff
        # if it precedes, remove the frame with the negative diff
        # if it follows, remove the frame with the large positive diff
        diffs_before = diffs.shift(1, fill_value=1).values
        diffs_after = diffs.shift(-1, fill_value=1).values
        remove_before = np.roll(
            np.logical_and(diffs.values < 0, diffs_before > diffs_after), -1
        )
        remove_negative_diff = np.logical_and(
            diffs.values < 0, diffs_before <= diffs_after
        )
        monitor_frames_df = monitor_frames_df[~(remove_before | remove_negative_diff)]
        # Then remove the duplicates
        duplicates = monitor_frames_df.closest_frame.diff() == 0
        monitor_frames_df = monitor_frames_df[~duplicates]
        new_length = len(monitor_frames_df)
        print(f"Removed {old_length - new_length} frames including:")
        print(f"{np.sum(remove_before | remove_negative_diff)} negative diffs.")
        print(f"{np.sum(duplicates)} duplicates.")
        if new_length == old_length:
            removed_frames = False
    return monitor_frames_df
