"""
Created on Sat Jun 12 14:08:41 2021

@author: hey2

Functions to sort running speed, optic flow speed and dFF into desired array form or to process these values
"""
import numpy as np
from cottage_analysis.stimulus_structure.sphere_structure import (
    find_min_trial_num_all_depths,
    find_frame_num_per_trial,
)


def create_trace_arr_per_roi(
    which_roi,
    dffs,
    depth_list,
    stim_dict,
    mode,
    protocol="fix_length",
    isStim=True,
    blank_period=0,
    frame_rate=30,
):
    """
    Create an array to store the traces for each roi.

    :param which_roi: np.ndarray, array of ROI no. which are identified as neurons
    :param dffs: np.ndarray, array of dFF for all ROIs. nROI * timeseries.
    :param depth_list: list, list of depth values in meters.
    :param stim_dict: dict, stim_dict containing vis-stim frames for each trial for each depth
    :param mode: str, 'sort_by_depth' or 'all_trials' (sort_by_depth: depth * trial * frame_num; all_trials: 1 * all_trials * frame_num)
    :param protocol: str, 'fix_time' or 'fix_length' (fix_time: take min frame num for all trials; fix_length: take max frame num for all trials)
    :param isStim: bool, True: frames with vis-stim. False: frames with blank screen.
    :param blank_period: float, blank period in seconds.
    :param frame_rate: int, frame rate for imaging.
    :return: trace_arr: np.ndarray. Array form see mode and protocol.
    """

    if protocol == "fix_time":
        trial_num = find_min_trial_num_all_depths(stim_dict, depth_list, isStim=isStim)
        frame_num_pertrial = find_frame_num_per_trial(
            depth_list, stim_dict, mode="min", isStim=isStim
        )
        frame_num_pertrial = frame_num_pertrial + blank_period * frame_rate * 2
        dff = dffs[which_roi, :]
        if isStim:
            if mode == "sort_by_depth":
                trace_arr = np.zeros((len(depth_list), trial_num, frame_num_pertrial))
                for idepth in range(0, len(depth_list)):
                    depth = depth_list[idepth]
                    for itrial in range(0, trial_num):
                        frame_start = (
                            stim_dict["stim" + str(depth)]["start"][itrial]
                            - blank_period * frame_rate
                        )
                        trace_arr[idepth, itrial, :] = dff[
                            frame_start : (frame_start + frame_num_pertrial)
                        ]
            elif mode == "all_trials":
                trace_arr = np.zeros(
                    (1, len(stim_dict["stim_all"]["start"]), frame_num_pertrial)
                )
                for itrial in range(0, len(stim_dict["stim_all"]["start"])):
                    frame_start = (
                        stim_dict["stim_all"]["start"][itrial]
                        - blank_period * frame_rate
                    )
                    trace_arr[0, itrial, :] = dff[
                        frame_start : (frame_start + frame_num_pertrial)
                    ]

        else:
            trace_arr = np.zeros((1, trial_num, frame_num_pertrial))
            for itrial in range(0, trial_num):
                frame_start = stim_dict["blank"]["start"][itrial]
                trace_arr[0, itrial, :] = dff[
                    frame_start : (frame_start + frame_num_pertrial)
                ]

        return trace_arr

    elif protocol == "fix_length":
        trial_num = find_min_trial_num_all_depths(stim_dict, depth_list, isStim=isStim)
        frame_num_pertrial_max = find_frame_num_per_trial(
            depth_list, stim_dict, mode="max", isStim=isStim
        )
        dff = dffs[which_roi, :]

        if isStim:
            if mode == "sort_by_depth":
                # trace_arr: each entry: ---blank--- ---stim 000000--- ---blank---
                trace_arr = np.zeros(
                    (
                        len(depth_list),
                        trial_num,
                        (frame_num_pertrial_max + blank_period * frame_rate * 2),
                    )
                )
                trace_arr[:] = np.nan
                trace_list = [
                    [[[] for i in range(3)] for j in range(trial_num)]
                    for k in range(len(depth_list))
                ]
                for idepth in range(0, len(depth_list)):
                    depth = depth_list[idepth]
                    for itrial in range(0, trial_num):
                        frame_start = stim_dict["stim" + str(depth)]["start"][itrial]
                        frame_num_this_trial = (
                            stim_dict["stim" + str(depth)]["stop"][itrial]
                            - stim_dict["stim" + str(depth)]["start"][itrial]
                            + 1
                        )
                        trace_arr[
                            idepth,
                            itrial,
                            : int(frame_num_this_trial + blank_period * frame_rate),
                        ] = dff[
                            (frame_start - blank_period * frame_rate) : (
                                frame_start + frame_num_this_trial
                            )
                        ]
                        trace_arr[
                            idepth,
                            itrial,
                            frame_num_pertrial_max + blank_period * frame_rate :,
                        ] = dff[
                            (frame_start + frame_num_this_trial) : (
                                frame_start
                                + frame_num_this_trial
                                + blank_period * frame_rate
                            )
                        ]

                        trace_list[idepth][itrial][0] = dff[
                            (frame_start - blank_period * frame_rate) : frame_start
                        ].tolist()
                        trace_list[idepth][itrial][1] = dff[
                            frame_start : (frame_start + frame_num_this_trial)
                        ].tolist()
                        trace_list[idepth][itrial][2] = dff[
                            (frame_start + frame_num_this_trial) : (
                                frame_start
                                + frame_num_this_trial
                                + blank_period * frame_rate
                            )
                        ].tolist()

            elif mode == "all_trials":
                trace_arr = np.zeros(
                    (
                        1,
                        trial_num * len(depth_list),
                        (frame_num_pertrial_max + blank_period * frame_rate * 2),
                    )
                )
                trace_arr[:] = np.nan
                trace_list = [
                    [[[] for i in range(3)] for j in range(trial_num * len(depth_list))]
                    for k in range(1)
                ]
                for itrial in range(trial_num * len(depth_list)):
                    frame_start = stim_dict["stim_all"]["start"][itrial]
                    frame_num_this_trial = (
                        stim_dict["stim_all"]["stop"][itrial]
                        - stim_dict["stim_all"]["start"][itrial]
                        + 1
                    )
                    trace_arr[
                        0,
                        itrial,
                        : int(frame_num_this_trial + blank_period * frame_rate),
                    ] = dff[
                        (frame_start - blank_period * frame_rate) : (
                            frame_start + frame_num_this_trial
                        )
                    ]
                    trace_arr[
                        0, itrial, frame_num_pertrial_max + blank_period * frame_rate :
                    ] = dff[
                        (frame_start + frame_num_this_trial) : (
                            frame_start
                            + frame_num_this_trial
                            + blank_period * frame_rate
                        )
                    ]

                    trace_list[0][itrial][0] = dff[
                        (frame_start - blank_period * frame_rate) : frame_start
                    ].tolist()
                    trace_list[0][itrial][1] = dff[
                        frame_start : (frame_start + frame_num_this_trial)
                    ].tolist()
                    trace_list[0][itrial][2] = dff[
                        (frame_start + frame_num_this_trial) : (
                            frame_start
                            + frame_num_this_trial
                            + blank_period * frame_rate
                        )
                    ].tolist()

        else:
            trace_arr = np.zeros((1, trial_num, frame_num_pertrial_max))
            for itrial in range(0, trial_num):
                frame_start = stim_dict["blank"]["start"][itrial]
                trace_arr[0, itrial, :] = dff[
                    frame_start : (frame_start + frame_num_pertrial_max)
                ]
                trace_list = trace_arr.tolist()

        return trace_arr, trace_list


def create_speed_arr(
    speeds,
    depth_list,
    stim_dict,
    mode,
    protocol="fix_length",
    isStim=True,
    blank_period=0,
    frame_rate=30,
):
    """
    Create an array and/or list to store the speed (RS or OF).

    :param speeds: np.ndarray, 1 x time. Array of RS/OF speed of all imaging frames.
    :param depth_list: list, list of depth values in meters.
    :param stim_dict: dict, stim_dict containing vis-stim frames for each trial for each depth
    :param mode: str, 'sort_by_depth' or 'all_trials'. (sort_by_depth: depth * trial * frame_num; all_trials: 1 * all_trials * frame_num).
    :param protocol: str, 'fix_time' or 'fix_length'. (fix_time: take min frame num for all trials; fix_length: take max frame num for all trials)
    :param isStim: bool, True: frames with vis-stim. False: frames with blank screen.
    :param blank_period: float, blank period in seconds.
    :param frame_rate: int, frame rate for imaging.
    :return: speed_arr (mode='fix_time') or speed arr & speed_list (mode='fix_length'): np.ndarray and/or list. Array form see mode and protocol.
    """

    if protocol == "fix_time":
        trial_num = find_min_trial_num_all_depths(stim_dict, depth_list, isStim=isStim)
        frame_num_pertrial = find_frame_num_per_trial(
            depth_list, stim_dict, mode="min", isStim=isStim
        )
        frame_num_pertrial = frame_num_pertrial + blank_period * frame_rate * 2

        if isStim:
            if mode == "sort_by_depth":
                speed_arr = np.zeros((len(depth_list), trial_num, frame_num_pertrial))
                for idepth in range(0, len(depth_list)):
                    depth = depth_list[idepth]
                    for itrial in range(0, trial_num):
                        frame_start = (
                            stim_dict["stim" + str(depth)]["start"][itrial]
                            - blank_period * frame_rate
                        )
                        speed_arr[idepth, itrial, :] = speeds[
                            frame_start : (frame_start + frame_num_pertrial)
                        ]
            elif mode == "all_trials":
                speed_arr = np.zeros(
                    (1, len(stim_dict["stim_all"]["start"]), frame_num_pertrial)
                )
                for itrial in range(0, len(stim_dict["stim_all"]["start"])):
                    frame_start = (
                        stim_dict["stim_all"]["start"][itrial]
                        - blank_period * frame_rate
                    )
                    speed_arr[0, itrial, :] = speeds[
                        frame_start : (frame_start + frame_num_pertrial)
                    ]

        else:
            speed_arr = np.zeros((1, trial_num, frame_num_pertrial))
            for itrial in range(0, trial_num):
                frame_start = stim_dict["blank"]["start"][itrial]
                speed_arr[0, itrial, :] = speeds[
                    frame_start : (frame_start + frame_num_pertrial)
                ]

        return speed_arr

    elif protocol == "fix_length":
        trial_num = find_min_trial_num_all_depths(stim_dict, depth_list, isStim=isStim)
        frame_num_pertrial_max = find_frame_num_per_trial(
            depth_list, stim_dict, mode="max", isStim=isStim
        )

        if isStim:
            if mode == "sort_by_depth":
                # trace_arr: each entry: ---blank--- ---stim 000000--- ---blank---
                speed_arr = np.zeros(
                    (
                        len(depth_list),
                        trial_num,
                        (frame_num_pertrial_max + blank_period * frame_rate * 2),
                    )
                )
                speed_arr[:] = np.nan
                speed_list = [
                    [[[] for i in range(3)] for j in range(trial_num)]
                    for k in range(len(depth_list))
                ]
                for idepth in range(0, len(depth_list)):
                    depth = depth_list[idepth]
                    for itrial in range(0, trial_num):
                        frame_start = stim_dict["stim" + str(depth)]["start"][itrial]
                        frame_num_this_trial = (
                            stim_dict["stim" + str(depth)]["stop"][itrial]
                            - stim_dict["stim" + str(depth)]["start"][itrial]
                            + 1
                        )
                        speed_arr[
                            idepth,
                            itrial,
                            : int(frame_num_this_trial + blank_period * frame_rate),
                        ] = speeds[
                            (frame_start - blank_period * frame_rate) : (
                                frame_start + frame_num_this_trial
                            )
                        ]
                        speed_arr[
                            idepth,
                            itrial,
                            frame_num_pertrial_max + blank_period * frame_rate :,
                        ] = speeds[
                            (frame_start + frame_num_this_trial) : (
                                frame_start
                                + frame_num_this_trial
                                + blank_period * frame_rate
                            )
                        ]

                        speed_list[idepth][itrial][0] = speeds[
                            (frame_start - blank_period * frame_rate) : frame_start
                        ].tolist()
                        speed_list[idepth][itrial][1] = speeds[
                            frame_start : (frame_start + frame_num_this_trial)
                        ].tolist()
                        speed_list[idepth][itrial][2] = speeds[
                            (frame_start + frame_num_this_trial) : (
                                frame_start
                                + frame_num_this_trial
                                + blank_period * frame_rate
                            )
                        ].tolist()

            elif mode == "all_trials":
                speed_arr = np.zeros(
                    (
                        1,
                        trial_num * len(depth_list),
                        (frame_num_pertrial_max + blank_period * frame_rate * 2),
                    )
                )
                speed_arr[:] = np.nan
                speed_list = [
                    [[[] for i in range(3)] for j in range(trial_num * len(depth_list))]
                    for k in range(1)
                ]
                for itrial in range(trial_num * len(depth_list)):
                    frame_start = stim_dict["stim_all"]["start"][itrial]
                    frame_num_this_trial = (
                        stim_dict["stim_all"]["stop"][itrial]
                        - stim_dict["stim_all"]["start"][itrial]
                        + 1
                    )
                    speed_arr[
                        0,
                        itrial,
                        : int(frame_num_this_trial + blank_period * frame_rate),
                    ] = speeds[
                        (frame_start - blank_period * frame_rate) : (
                            frame_start + frame_num_this_trial
                        )
                    ]
                    speed_arr[
                        0, itrial, frame_num_pertrial_max + blank_period * frame_rate :
                    ] = speeds[
                        (frame_start + frame_num_this_trial) : (
                            frame_start
                            + frame_num_this_trial
                            + blank_period * frame_rate
                        )
                    ]

                    speed_list[0][itrial][0] = speeds[
                        (frame_start - blank_period * frame_rate) : frame_start
                    ].tolist()
                    speed_list[0][itrial][1] = speeds[
                        frame_start : (frame_start + frame_num_this_trial)
                    ].tolist()
                    speed_list[0][itrial][2] = speeds[
                        (frame_start + frame_num_this_trial) : (
                            frame_start
                            + frame_num_this_trial
                            + blank_period * frame_rate
                        )
                    ].tolist()

        else:
            speed_arr = np.zeros((1, trial_num, frame_num_pertrial_max))
            for itrial in range(0, trial_num):
                frame_start = stim_dict["blank"]["start"][itrial]
                speed_arr[0, itrial, :] = speeds[
                    frame_start : (frame_start + frame_num_pertrial_max)
                ]
                speed_list = speed_arr.tolist()

        return speed_arr, speed_list


def thr(arr, thr):
    """
    High-pass threshold an array and convert any values below the threshold to the threshold.
    """
    #     arr = np.abs(arr)
    arr = arr.copy()
    arr[arr < thr] = thr

    return arr


def calculate_OF(rs, img_VS, mode):
    """
    Calculate optic flow speeds based on running speed. In meters.

    :param rs: np.ndarray. 1 x time. Array of running speed for all imaging frames.
    :param img_VS: pd.DataFrame. Dataframe containing aligned parameters for each imaging frame.
    :param mode: str, 'no_RF' or 'RF'. 'no_RF': not taking into account the sphere position when calculating optic flow speed.
    :return: optics: np.ndarray. 1 X time. Array of optic flow speed for all imaging frames.
    """
    if mode == "no_RF":
        all_depths = img_VS.Depth.replace(-99.99, np.nan)
        optics = rs / (all_depths)
        optics = np.array(optics)

    return optics


def get_trace_arrs(
    roi,
    dffs,
    depth_list,
    stim_dict,
    mode="sort_by_depth",
    protocol="fix_length",
    blank_period=5,
    frame_rate=15,
):
    # Trace array of dFF
    trace_arr, _ = create_trace_arr_per_roi(
        roi,
        dffs,
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=blank_period,
        frame_rate=frame_rate,
    )
    trace_arr_mean = np.nanmean(trace_arr, axis=1)
    trace_arr_noblank, _ = create_trace_arr_per_roi(
        roi,
        dffs,
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
    trace_arr_noblank_mean = np.nanmean(trace_arr_noblank, axis=1)
    trace_arr_blank, _ = create_trace_arr_per_roi(
        roi,
        dffs,
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        isStim=False,
        blank_period=0,
        frame_rate=frame_rate,
    )
    return trace_arr_noblank, trace_arr_blank
