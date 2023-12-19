"""
Find frames for visual stimulation based on photodiode signal
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfiltfilt
from functools import partial

print = partial(print, flush=True)


def find_pulses(data, sampling, lowcut=1, highcut=400):
    """Find square like pulses in continuous signal

    Useful for instance to find frame pulses in the photodiode
    Args:
        data (np.array): signal to process
        sampling (float): sampling frequency, in Hz
        lowcut (float or None): cutoff frequency for highpass filter
        highcut (float or None): cutoff frequency for lowpass filter

    Returns:

    """
    n = int(lowcut is not None) * 2 + int(highcut is not None)
    filter_type = [None, "lowpass", "highpass", "bandpass"][n]
    if filter_type is not None:
        freq = [
            None,
            highcut / sampling,
            lowcut / sampling,
            (lowcut / sampling, highcut / sampling),
        ][n]
        sos = butter(N=4, Wn=freq, btype=filter_type, output="sos")
        fdata = sosfiltfilt(sos, data)
    raise NotImplementedError


# %%
def find_VS_frames(
    photodiode_df,
    frame_rate=144,
    upper_thr=200,
    lower_thr=50,
    photodiode_sampling=1000,
    plot=False,
    plot_start=0,
    plot_range=2000,
    plot_dir=None,
):
    """Find frame refresh based on photodiode signal

    Signal is expected to alternate between high and low at each frame (flickering quad
    between white and black)

    Args:
        photodiode_df (pd.DataFrame): dataframe with `HarpTime` and `Photodiode` fields
        frame_rate (float): Expected frame rate. Peaks separated by less than half a
                            frame will be ignored
        upper_thr (float): Photodiode value above which the quad is considered white
        lower_thr (float): Photodiode value below which the quad is considered black
        photodiode_sampling (float): Sampling rate of the photodiode signal in Hz
        plot (bool): Should a summary figure be generated?
        plot_start (int): sample to start the plot
        plot_range (int): samples to plot after plot_start
        plot_dir (Path or str): directory to save the figure

    Returns:
        detected_frame (pd.DataFrame): a dataframe containing detected frame timing.
            It contains:
                - 'Photodiode': photodiode value at peak
                - 'HarpTime': time of peak
                - 'ElapsedTime': time of peak since first peak
                - 'VisStim_Frame': peak index
    """

    # Get elapsed time
    photodiode_df["ElapsedTime"] = None
    photodiode_df["ElapsedTime"] = photodiode_df.HarpTime - photodiode_df.HarpTime[0]
    photodiode_df["ElapsedTime"].iloc[0] = 0

    elapsed_time = np.array(photodiode_df.ElapsedTime).reshape(-1)
    photodiode = np.array(photodiode_df.Photodiode).reshape(-1)

    # Find peaks of photodiode
    distance = int(1 / frame_rate * 2 * photodiode_sampling)
    high_peaks, _ = find_peaks(photodiode, height=upper_thr, distance=distance)
    first_frame = high_peaks[0]
    low_peaks, _ = find_peaks(-photodiode, height=-lower_thr, distance=distance)
    low_peaks = low_peaks[low_peaks > first_frame]

    # Get rid of framedrops
    photodiode_df["FramePeak"] = None
    photodiode_df.FramePeak.iloc[high_peaks] = 1
    photodiode_df.FramePeak.iloc[low_peaks] = 0
    detect_frames = photodiode_df[photodiode_df.FramePeak.notnull()]
    detect_frames = detect_frames[detect_frames.FramePeak.diff() != 0]
    detect_frames["VisStim_Frame"] = np.arange(len(detect_frames))
    detect_frames = detect_frames.drop(columns=["FramePeak"])

    if plot:
        plt.figure()
        plt.plot(
            elapsed_time[plot_start : (plot_start + plot_range)],
            photodiode[plot_start : (plot_start + plot_range)],
        )
        all_frame_idxs = detect_frames.index.values.reshape(-1)
        take_start = np.argmin(np.abs(all_frame_idxs - plot_start))
        take_stop = np.argmin(np.abs(all_frame_idxs - plot_start - plot_range))
        take_idxs = all_frame_idxs[take_start:take_stop]

        plot_peaks = np.intersect1d(
            all_frame_idxs, (np.arange(plot_start, (plot_start + plot_range), step=1))
        )
        # plt.figure()
        plt.plot(
            detect_frames.loc[take_idxs, "ElapsedTime"],
            detect_frames.loc[take_idxs, "Photodiode"],
        )
        plt.plot(elapsed_time[plot_peaks], photodiode[plot_peaks], "x")
        plt.plot(
            elapsed_time[plot_start : (plot_start + plot_range)],
            np.zeros_like(photodiode[plot_start : (plot_start + plot_range)])
            + upper_thr,
            "--",
            color="gray",
        )
        plt.plot(
            elapsed_time[plot_start : (plot_start + plot_range)],
            np.zeros_like(photodiode[plot_start : (plot_start + plot_range)])
            + lower_thr,
            "--",
            color="gray",
        )
        plt.xlabel("Time(s)")
    if plot_dir is not None:
        plt.savefig(Path(plot_dir) / "Frame_finder_check.png")

    return detect_frames


def find_imaging_frames(
    harp_message,
    frame_number,
    frame_period=0.015,
    register_address=32,
    frame_period_tolerance=0.0002,
):
    """Find imaging triggers and the corresponding harptime from formatted harpmessage.

    Note that the resulting harp times now correspond to the start of each frame, at least for
    scanimage recordings.

    Args:
        harp_message (pd.DataFrame): Dataframe of formatted harpmessage.
        frame_number (int): Correct frame number extracted from suite2o
        frame_period (float, optional): Duration of a frame in s. Defaults to 0.015.
        register_address (int, optional): Register channel in harpmessage for imaging triggers. Defaults to 32.
        frame_period_tolerance (float, optional): Error tolerance for frame period. Defaults to 0.0002. For widefield: 0.0002, for 2p: 0.001

    Returns:
        frame_triggers (pd.DataFrame): DataFrame containing harptime for each imaging frame trigger.

    """
    # TODO: This version always rejects the last imaging frame. check
    frame_triggers = harp_message[harp_message.RegisterAddress == register_address]
    frame_triggers = frame_triggers[
        frame_triggers.FrameTriggers == 1
    ]  # only keep frame onset
    frame_triggers = frame_triggers.rename(
        columns={"Timestamp": "HarpTime"}, inplace=False
    )

    # shift diff by -1 to get the start of the frame
    frame_triggers["HarpTime_diff"] = frame_triggers.HarpTime.diff().shift(-1)

    frame_triggers["FramePeriod"] = np.nan
    frame_triggers.loc[
        np.abs(frame_triggers["HarpTime_diff"] - frame_period)
        <= frame_period_tolerance,
        "FramePeriod",
    ] = 1
    print(
        f"{np.sum(frame_triggers.FramePeriod!=1)} frames are not {frame_period:.4f} s"
    )
    frame_triggers = frame_triggers[frame_triggers.FramePeriod == 1]
    n_frame_triggers = len(frame_triggers)
    frame_triggers["ImagingFrame"] = np.arange(len(frame_triggers))
    print(f"ImagingFrames in video: {frame_number}")
    print(f"ImagingFrame triggers: {n_frame_triggers}")
    if n_frame_triggers == frame_number:
        frame_triggers = frame_triggers
    elif (n_frame_triggers - frame_number) == 1:
        frame_triggers = frame_triggers[:-1]
        print("WARNING: SAVED VIDEO FRAMES ARE 1 FRAME LESS THAN FRAME TRIGGERS!!!")
    elif (len(frame_triggers[frame_triggers.FramePeriod == 1]) - frame_number) == 2:
        frame_triggers = frame_triggers[:-2]
        print("WARNING: SAVED VIDEO FRAMES ARE 2 FRAMES LESS THAN FRAME TRIGGERS!!!")
    else:
        print(
            "FRAME NUMBER NOT CORRECT likely due to incomplete imaging volume at the end of the stack or bonsai crash."
        )
        frame_triggers = frame_triggers[:frame_number]
    frame_triggers = frame_triggers.drop(
        columns=["HarpTime_diff", "FramePeriod", "RegisterAddress"]
    )

    return frame_triggers
