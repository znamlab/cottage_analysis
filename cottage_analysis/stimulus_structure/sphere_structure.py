import pandas as pd
import numpy as np


# Create stim_dict
def create_stim_dict(depth_list, img_VS, choose_trials=None):
    # Simplify img_VS
    img_VS["Stim"] = np.nan
    img_VS.loc[img_VS.Depth.notnull(), "Stim"] = 1
    img_VS.loc[img_VS.Depth < 0, "Stim"] = 0
    img_VS.loc[((img_VS[img_VS.Depth < 0]).index.values - 1), "Stim"] = 0

    img_VS_simple = img_VS[(img_VS["Stim"].diff() != 0) & (img_VS["Stim"].notnull())]
    img_VS_simple.Depth = np.round(img_VS_simple.Depth, 2)

    # Find stim frames in order of depth
    stim_dict = {}
    for istim in depth_list:
        stim_dict["stim" + str(istim)] = {}
        stim_dict["stim" + str(istim)]["start"] = img_VS_simple[
            (img_VS_simple["Depth"] == istim) & (img_VS_simple["Stim"] == 1)
        ].index.values
        stim_dict["stim" + str(istim)]["stop"] = img_VS_simple[
            (img_VS_simple["Depth"] == istim) & (img_VS_simple["Stim"] == 0)
        ].index.values
        if choose_trials != None:
            stim_dict["stim" + str(istim)]["start"] = stim_dict["stim" + str(istim)][
                "start"
            ][:choose_trials]
            stim_dict["stim" + str(istim)]["stop"] = stim_dict["stim" + str(istim)][
                "stop"
            ][:choose_trials]

    blank_points = img_VS_simple.index.values[1:-1]
    stim_dict["blank"] = {}
    stim_dict["blank"]["start"] = blank_points[0::2] + 1
    stim_dict["blank"]["stop"] = blank_points[1::2] - 1
    if choose_trials != None:
        stim_dict["blank"]["start"] = stim_dict["blank"]["start"][
            : choose_trials * len(depth_list)
        ]
        stim_dict["blank"]["stop"] = stim_dict["blank"]["stop"][
            : choose_trials * len(depth_list)
        ]

    # Find stim frames in order
    stim_dict["stim_all"] = {}
    stim_dict["stim_all"]["start"] = []
    stim_dict["stim_all"]["stop"] = []
    for istim in depth_list:
        stim_dict["stim_all"]["start"].append(
            list(stim_dict["stim" + str(istim)]["start"])
        )
        stim_dict["stim_all"]["stop"].append(
            list(stim_dict["stim" + str(istim)]["stop"])
        )
    stim_dict["stim_all"]["start"] = np.sort(
        np.array([j for i in stim_dict["stim_all"]["start"] for j in i])
    )
    stim_dict["stim_all"]["stop"] = np.sort(
        np.array([j for i in stim_dict["stim_all"]["stop"] for j in i])
    )
    stim_dict["stim_all"]["depth"] = np.array(img_VS_simple[::2].Depth)
    if choose_trials != None:
        stim_dict["stim_all"]["depth"] = stim_dict["stim_all"]["depth"][
            : choose_trials * len(depth_list)
        ]

    return stim_dict


def find_min_trial_num_all_depths(stim_dict, depth_list, isStim=True):
    """
    Find mininum number of trials across all depths
    """
    if isStim:
        trial_nums = []
        for idepth in range(len(depth_list)):
            depth = depth_list[idepth]
            frame_dict = stim_dict["stim" + str(depth)]
            trial_nums.append(len(frame_dict["start"]))
        trial_nums = np.array(trial_nums)
        #         if not np.all(trial_nums==trial_nums[0]):
        #             print('Trials nums are not the same. Take the min trial num: '+str(np.min(trial_nums)))
        trial_num = np.min(trial_nums)
    else:
        trial_num = len(stim_dict["blank"]["stop"])
    return trial_num


def find_frame_num_per_trial(depth_list, stim_dict, mode="max", isStim=True):
    """
    Find number of frames for each trial.

    :param list depth_list: list of depth values in meters
    :param dict stim_dict: stim_dict containing vis-stim frames for each trial for each depth
    :param str mode: 'min' or 'max'. 'min': minimum frame number across all trials; 'max': maximum frame number across all trials.
    :param bool isStim: True: frames with vis-stim. False: frames with blank screen.
    :return:
    """
    if isStim:
        trial_num = find_min_trial_num_all_depths(stim_dict, depth_list)
        frame_num_arr = np.zeros((len(depth_list), trial_num))
        for idepth in range(len(depth_list)):
            for itrial in range(trial_num):
                frame_start = stim_dict["stim" + str(depth_list[idepth])]["start"][
                    itrial
                ]
                frame_stop = stim_dict["stim" + str(depth_list[idepth])]["stop"][itrial]
                frame_num = frame_stop - frame_start + 1
                frame_num_arr[idepth, itrial] = frame_num
        if mode == "min":
            frame_num_result = int(np.min(frame_num_arr))
        elif mode == "max":
            frame_num_result = int(np.max(frame_num_arr))

    # Blank period
    else:
        trial_num = find_min_trial_num_all_depths(stim_dict, depth_list, isStim=False)
        frame_num_arr = np.zeros(trial_num)
        for itrial in range(trial_num):
            frame_start = stim_dict["blank"]["start"][itrial]
            frame_stop = stim_dict["blank"]["stop"][itrial]
            frame_num = frame_stop - frame_start + 1
            frame_num_arr[itrial] = frame_num
        frame_num_result = int(
            np.min(frame_num_arr)
        )  # always find the min frame num for blank period

    return frame_num_result
