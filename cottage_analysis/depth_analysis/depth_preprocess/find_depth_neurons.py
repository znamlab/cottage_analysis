"""
main script for basic preprocessing of 2p depth experimental data

For now, use env: 2p_analysis_clone
"""

# IMPORTS
from typing import Any

import os
import sys
import defopt
import pickle
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from suite2p.extraction import masks
import matplotlib
from matplotlib import cm

import cottage_analysis as cott
from cottage_analysis.depth_analysis.filepath import generate_filepaths
from cottage_analysis.imaging.common import find_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers
from cottage_analysis.stimulus_structure import sphere_structure as vis_stim_structure
from cottage_analysis.depth_analysis.plotting.plotting_utils import *
from cottage_analysis.depth_analysis.depth_preprocess.process_params import *
from cottage_analysis.depth_analysis.depth_preprocess.process_trace import *


def main(project, mouse, session, protocol):
    """
    :param str project: project name (determines the root directory for raw data)
    :param str mouse: mouse name
    :param str session: session name,Sdate
    :param str protocol: stimulation protocol name
    :return: None
    """

    # ----- SETUPS -----
    rawdata_root = "/camp/lab/znamenskiyp/data/instruments/raw_data/projects/"
    root = "/camp/lab/znamenskiyp/home/shared/projects/"
    # depth_list = [0.06, 0.19, 0.6, 1.9, 6]
    choose_trials = 13
    frame_rate = 15
    speed_thr_cal = (
        0.2  # m/s, threshold for running speed when calculating depth neurons
    )
    calculate_dFF = False
    if "Playback" in protocol:
        folder_no = 1
    else:
        folder_no = 0

    # ----- STEP1: Generate file path -----
    print("---START STEP 1---", "\n", "Getting data filepaths...", flush=True)
    (
        rawdata_folder,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    ) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        rawdata_root=rawdata_root,
        root=root,
    )
    if not os.path.exists(analysis_folder + "plane0/"):
        os.makedirs(analysis_folder + "plane0/")

    if "Playback" in protocol:
        (
            rawdata_folder_closeloop,
            protocol_folder_closeloop,
            analysis_folder_closeloop,
            suite2p_folder_closeloop,
            trace_folder_closeloop,
        ) = generate_filepaths.generate_file_folders(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol.replace("Playback", ""),
            rawdata_root=rawdata_root,
            root=root,
        )
        assert os.path.exists(analysis_folder_closeloop + "plane0/")
    print("---STEP 1 FINISHED.---", "\n", flush=True)

    # -----STEP2: Calculate dF/F from the raw fluorescence trace and Save -----
    print("---START STEP 2---", "\n", "Calculating dF/F for all ROIs...", flush=True)
    # Load suite2p files
    stat = np.load(suite2p_folder + "stat.npy", allow_pickle=True)
    ops = np.load(suite2p_folder + "ops.npy", allow_pickle=True)
    ops = ops.item()
    stats = np.load(suite2p_folder + "stat.npy", allow_pickle=True)
    iscell = np.load(suite2p_folder + "iscell.npy", allow_pickle=True)[:, 0]
    # F = np.load(trace_folder + 'F.npy', allow_pickle=True)
    Fast = np.load(trace_folder + "Fast.npy", allow_pickle=True)
    # Fneu = np.load(trace_folder + 'Fneu.npy', allow_pickle=True)
    # spks = np.load(trace_folder + 'spks.npy', allow_pickle=True)

    # The ROI no. for all cells (excluding non-cells)
    which_rois = (np.arange(Fast.shape[0]))[iscell.astype("bool")]

    # Calculate dF/F
    if calculate_dFF:
        dffs_ast = process_trace.calculate_dFF(
            f=Fast, mode="gmm", n_components=2, verbose=True
        )
        np.save(trace_folder + "dffs_ast.npy", dffs_ast)
        print("dFFs saved.", flush=True)
        visualize_dffs = False
        if visualize_dffs:
            plt.figure(figsize=(10, 5))
            plt.plot(dffs_ast[0])
            plt.savefig(trace_folder + "dffs_ast_example.png")
    print("---STEP 2 FINISHED.---", "\n", flush=True)

    # -----STEP3: Align timestamps of visual stimulation loggers and save as img_VS.pickle file-----
    # This is trial-based analysis and has not included screen frame no.
    print("---START STEP 3---", "\n", "Align timestamps for all loggers...", flush=True)
    # photodiode_file = generate_filepaths.generate_logger_path(project=project,
    #                                                           mouse=mouse,
    #                                                           session=session,
    #                                                           protocol=protocol,
    #                                                           rawdata_root=rawdata_root,
    #                                                           root=root,
    #                                                           logger_name='PhotodiodeLog')

    # VS_photodiode_logger = format_loggers.format_VS_photodiode_logger(photodiode_file)
    # # plt.plot(VS_photodiode_logger['Photodiode'][100000:110000])
    # if not os.path.exists(trace_folder+'checker_plots/'):
    #     os.makedirs(trace_folder+'checker_plots/')
    # VS_photodiode_logger = find_frames.find_VS_frames(photodiode_df=VS_photodiode_logger, frame_rate=144, upper_thr=70,
    #                                                   lower_thr=30, plot=True, plot_start=100000, plot_range=2000,
    #                                                   plot_dir=trace_folder+'checker_plots/')
    # print('Photodiode signal extracted.', flush=True)

    # Load vis-stim param logger
    VS_param_file = generate_filepaths.generate_logger_path(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        rawdata_root=rawdata_root,
        root=root,
        logger_name="NewParams",
    )
    VS_param_logger = format_loggers.format_VS_param_logger(
        VS_param_file, which_protocol=protocol
    )

    # Load frame trigger logger and find frame triggers
    harpmessage_file = generate_filepaths.generate_logger_path(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        rawdata_root=rawdata_root,
        root=root,
        logger_name="harp_message",
    )
    img_frame_logger = format_loggers.format_img_frame_logger(
        harpmessage_file, register_address=32
    )
    frame_number = ops["frames_per_folder"][folder_no]
    img_frame_logger = find_frames.find_imaging_frames(
        harp_message=img_frame_logger,
        frame_number=frame_number,
        exposure_time=0.0324 * 2,
        register_address=32,
        exposure_time_tolerance=0.001,
    )
    print("Frame trigger extracted.", flush=True)

    # Align vis-stim param timestamps
    # img_VS = align_timestamps.align_timestamps(VS_frames=VS_photodiode_logger, VS_params=VS_param_logger,
    #                                            imaging_frames=img_frame_logger)
    img_VS = pd.merge_asof(
        img_frame_logger,
        VS_param_logger,
        on="HarpTime",
        allow_exact_matches=True,
        direction="backward",
    )

    # Add running logger to the aligned dataframe (ALL DISTANCE IN METERS)
    rotary_encoder_file = generate_filepaths.generate_logger_path(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        rawdata_root=rawdata_root,
        root=root,
        logger_name="RotaryEncoder",
    )
    mousez_logger = pd.read_csv(
        rotary_encoder_file, sep=",", usecols=["HarpTime", "EyeZ", "MouseZ"]
    )
    img_VS = pd.merge_asof(
        img_VS,
        mousez_logger,
        on="HarpTime",
        allow_exact_matches=True,
        direction="backward",
    )

    img_VS.EyeZ = img_VS.EyeZ / 100  # Convert cm to m
    img_VS.MouseZ = img_VS.MouseZ / 100  # Convert cm to m
    img_VS.Depth = img_VS.Depth / 100  # Convert cm to m
    img_VS.Z0 = img_VS.Z0 / 100  # Convert cm to m

    depth_list = img_VS["Depth"].unique()
    depth_list = np.round(depth_list, 2)
    depth_list = depth_list[~np.isnan(depth_list)].tolist()
    depth_list.remove(-99.99)
    depth_list.sort()

    # print(img_VS[:20], flush=True)
    # Save img_VS
    with open(protocol_folder + "img_VS.pickle", "wb") as handle:
        pickle.dump(img_VS, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Timestamps aligned and saved.", flush=True)
    print("---STEP 3 FINISHED.---", "\n", flush=True)

    # -----STEP4: Get the visual stimulation structure and Save (find the imaging frames for visual stimulation)-----
    print("---START STEP 4---", "\n", "Get vis-stim structure...", flush=True)
    with open(protocol_folder + "img_VS.pickle", "rb") as handle:
        img_VS = pickle.load(handle)
    stim_dict = vis_stim_structure.create_stim_dict(
        depth_list=depth_list, img_VS=img_VS, choose_trials=choose_trials
    )
    with open(protocol_folder + "stim_dict.pickle", "wb") as handle:
        pickle.dump(stim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("---STEP 4 FINISHED.---", "\n", flush=True)

    # -----STEP5: Find depth neurons, save and Plot depth neuron spatial distribution-----
    # Depth neurons are calculated only based on the time where running speed exceeds a certain threshold (default 0.2 m/s)
    print("---START STEP 5---", "\n", "Find depth neurons...", flush=True)
    # Create an array for running speed (ndepth x ntrials x time)
    speeds = img_VS.MouseZ.diff() / img_VS.HarpTime.diff()  # m/s
    speeds[0] = 0
    # speed_thr = 0.01
    # speeds_thred = thr(speeds, speed_thr)
    speed_arr, _ = create_speed_arr(
        speeds=speeds,
        depth_list=depth_list,
        stim_dict=stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )

    # For each CELL (not all ROIs), create an array for dFF trace.
    # Average across time for each trial, and then do an anova for all depths with all trial average as data points.
    anova_ps = []
    dffs = np.load(trace_folder + "dffs_ast.npy")
    for roi in which_rois:
        trace_arr, _ = create_trace_arr_per_roi(
            which_roi=roi,
            dffs=dffs,
            depth_list=depth_list,
            stim_dict=stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        (ndepths, ntrials, nframes) = trace_arr.shape
        trace_arr[speed_arr < speed_thr_cal] = np.nan
        trace_arr_mean_eachtrial = np.nanmean(trace_arr, axis=2)
        _, p = scipy.stats.f_oneway(
            trace_arr_mean_eachtrial[0],
            trace_arr_mean_eachtrial[1],
            trace_arr_mean_eachtrial[2],
            trace_arr_mean_eachtrial[3],
            trace_arr_mean_eachtrial[4],
        )
        anova_ps.append(p)
    anova_ps = np.array(anova_ps)

    # # Gaussian fit for depth selectivity
    # from scipy.optimize import curve_fit
    # def gaussian_func(x, a, x0, sigma):
    #     return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    depth_neurons = which_rois[
        np.where(anova_ps < 0.05)[0]
    ]  # ROI no. for depth neurons
    if not "Playback" in protocol:
        np.save(analysis_folder + "plane0/anova_p.npy", anova_ps)
        np.save(analysis_folder + "plane0/depth_neurons.npy", depth_neurons)
    elif "Playback" in protocol:
        depth_neurons_openloop = depth_neurons
        np.save(analysis_folder + "plane0/anova_p.npy", anova_ps)
        np.save(analysis_folder + "plane0/depth_neurons.npy", depth_neurons_openloop)
        depth_neurons = np.load(analysis_folder_closeloop + "plane0/depth_neurons.npy")
        print(
            "This is a playback session. \n Depth neurons from this session is saved, "
            "but depth neurons from the closeloop session is used for later analysis."
        )

    max_depths = (
        np.ones(len(depth_neurons)) * 9999
    )  # index no. from depth_list indicating the max depth of each depth neuron
    max_depths_values = (
        np.ones(len(depth_neurons)) * 9999
    )  # depth value = the max depth of each depth neuron

    for iroi in range(len(depth_neurons)):
        roi = depth_neurons[iroi]
        trace_arr, _ = create_trace_arr_per_roi(
            roi,
            dffs,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=15,
        )
        trace_arr_mean_eachtrial = np.nanmean(trace_arr, axis=2)
        trace_arr_depth_mean = np.nanmean(trace_arr_mean_eachtrial, axis=1)
        max_depth = np.where(trace_arr_depth_mean == np.max(trace_arr_depth_mean))[0]
        max_depths[iroi] = max_depth
        max_depths_values[iroi] = depth_list[max_depth[0]]

    max_depths[max_depths == 9999] = np.nan
    np.save(analysis_folder + "plane0/max_depths_index.npy", max_depths)
    np.save(analysis_folder + "plane0/max_depths_values.npy", max_depths_values)
    print("Depth neurons and max depths saved.", flush=True)

    # PLOT DEPTH NEURONS
    stats_arr = stats_to_array(stats, ops["Ly"], ops["Lx"])
    cells_mask = stats_to_array(stats, Ly=ops["Ly"], Lx=ops["Lx"], label_id=True)
    cells_mask[cells_mask != 0] = 1
    cells_mask[cells_mask == 0] = np.nan

    # PLOT 1: based on depth with max response
    plot_rows = 2
    plot_cols = 2
    plt.figure(figsize=(plot_cols * 10, plot_rows * 10))

    # Plot 1 (0,0): Plot spatial distribution of depth neurons overlayed with mean image of FOV
    # Depth neurons coded in discrete colours
    plt.subplot(plot_rows, plot_cols, 1)
    cmap = cm.cool.reversed()
    line_colors = []
    norm = matplotlib.colors.Normalize(
        vmin=np.log(min(depth_list)), vmax=np.log(max(depth_list))
    )
    for depth in depth_list:
        rgba_color = cmap(norm(np.log(depth)), bytes=True)
        rgba_color = tuple(it / 255 for it in rgba_color)
        line_colors.append(rgba_color)

    im = np.zeros((ops["Ly"], ops["Lx"], 3))

    # careful imshow color in BGR not RGB, but colormap seems to swap it already
    for i in range(len(depth_list)):
        for n in depth_neurons[max_depths == i]:
            ypix = stat[n]["ypix"][~stat[n]["overlap"]]
            xpix = stat[n]["xpix"][~stat[n]["overlap"]]
            im[ypix, xpix, :] = np.asarray(
                [line_colors[i][0], line_colors[i][1], line_colors[i][2]]
            )

    non_depth_neurons = [x for x in which_rois if x not in depth_neurons]
    for n in non_depth_neurons:
        ypix = stat[n]["ypix"][~stat[n]["overlap"]]
        xpix = stat[n]["xpix"][~stat[n]["overlap"]]
        if len(xpix) > 0 and len(ypix) > 0:
            im[ypix, xpix, :] = np.tile(
                (stat[n]["lam"][~stat[n]["overlap"]])
                / np.max(stat[n]["lam"][~stat[n]["overlap"]]),
                (3, 1),
            ).T

    plt.imshow(im)
    plt.axis("off")

    # plt.imshow(ops['meanImg'], cmap='gray', vmax=1000)
    # colors = ['red', 'orange', 'forestgreen', 'skyblue', 'purple']
    # ax = plt.gca()
    # for iroi in range(len(depth_neurons)):
    #     roi = depth_neurons[iroi]
    #     x_center, y_center, radius = find_roi_center(cells_mask, roi)
    #     radius = 8
    #     circ = Circle((x_center, y_center), radius, color=colors[int(max_depths[iroi])])
    #     ax.add_patch(circ)
    # plt.legend([(str(int(i * 100)) + 'cm') for i in depth_list], bbox_to_anchor=(1, 1), fontsize=15)
    # leg = ax.get_legend()
    # for i in range(len(depth_list)):
    #     leg.legendHandles[i].set_color(colors[i])
    # plt.axis('off');

    # Plot 2 (0,1): Plot spatial distribution of depth neurons overlayed with mean image of FOV
    # Histogram of p values from ANOVA test for depth neurons
    plt.subplot(plot_rows, plot_cols, 2)
    _, _, patches = plt.hist(anova_ps, bins=50)
    # plt.title('P value of ANOVA test for comparing dF/F trial means', fontsize=20)
    plt.title("P value of ANOVA test", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.xlabel("p value", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.vlines(
        0.05, 0, np.max(patches.datavalues), linestyle="dotted", color="r", linewidth=2
    )
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plot 3 (1,0): Bar plot of percentage of neurons preferring each depth
    plt.subplot(plot_rows, plot_cols, 3)
    # plt.subplot2grid([plot_rows, plot_cols], [0,2])
    max_depths_count = []
    for i in range(len(depth_list)):
        max_depths_count.append(np.sum(max_depths == i))
    max_depths_count_cumsum = np.cumsum(np.array(max_depths_count))
    max_depths_count.append(len(which_rois) - np.sum(max_depths_count))
    max_depths_count = np.array(max_depths_count)
    for i in range(len(depth_list)):
        if i == 0:
            plt.bar(
                x="compare mean",
                height=max_depths_count[i] / len(which_rois) * 100,
                width=0.5,
                color=line_colors[i],
                align="center",
            )
        else:
            plt.bar(
                x="compare mean",
                height=max_depths_count[i] / len(which_rois) * 100,
                width=0.5,
                bottom=max_depths_count_cumsum[i - 1] / len(which_rois) * 100,
                color=line_colors[i],
                align="center",
            )
    plt.bar(
        x="compare mean",
        height=max_depths_count[-1] / len(which_rois) * 100,
        width=0.5,
        bottom=max_depths_count_cumsum[-1] / len(which_rois) * 100,
        color="gray",
        align="center",
    )
    plt.xlim([-1, 1])

    plt.ylabel("Percentage of total neurons (%)", fontsize=20)
    plt.legend(loc="upper right", fontsize=15, bbox_to_anchor=(1.4, 1))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=1)

    # Plot 4 (1,1): Bar plot of max preferred depth
    plt.subplot(plot_rows, plot_cols, 4)
    # plt.subplot2grid([plot_rows, plot_cols], [0,3])
    max_depths_count = []
    for i in range(len(depth_list)):
        max_depths_count.append(np.sum(max_depths == i))
    max_depths_count.append(len(which_rois) - np.sum(np.array(max_depths_count)))
    plt.bar(
        x=np.arange(len(depth_list) + 1),
        height=np.array(max_depths_count) / len(which_rois) * 100,
    )
    xticks = [i for i in (np.array(depth_list) * 100).astype("int")]
    xticks.append("NaN")
    plt.xticks(np.arange(len(depth_list) + 1), xticks, fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylabel("Percentage of total neurons (%)", fontsize=20)
    plt.xlabel("Preferred depth (cm)", fontsize=20)

    plt.tight_layout(pad=1)

    save_prefix = "/plane0/plots/depth_selectivity/"
    if not os.path.exists(analysis_folder + save_prefix):
        os.makedirs(analysis_folder + save_prefix)
    plt.savefig(analysis_folder + save_prefix + "depth_selectivity_anova.pdf")
    print("Plot 1 of depth neuron distribution saved.", flush=True)

    print("---STEP 5 FINISHED.---", "\n", flush=True)

    print("---PREPROCESS FINISHED.---", "\n", flush=True)


if __name__ == "__main__":
    defopt.run(main)