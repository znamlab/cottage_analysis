import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mutual_info_score
from typing import Sequence, Dict, Any
import scipy
from cottage_analysis.depth_analysis.plotting import plotting_utils
from cottage_analysis.depth_analysis.depth_preprocess.process_params import *


def plot_depth_neuron_distribution(
    project,
    mouse,
    session,
    neurons_df,
    trials_df,
    protocol="SpheresPermTubeReward",
    depth_min=0.02,
    depth_max=20,
    bin_number=50,
    mode="discrete",
):
    """
    Plot distribution of neurons' depth preferences in one session.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name
        neurons_df (pd.DataFrame): Dataframe containing analyzed info of all rois
        trials_df (pd.DataFrame): Dataframe containing info of all trials
        protocol (str, optional): protocol name. Defaults to 'SpheresPermTubeReward'.
        depth_min (float, optional): minimum fitted depth. Defaults to 0.02.
        depth_max (int, optional): maximum fitted depth. Defaults to 20.
        bin_number (int, optional): number of bins for continuous distribution. Defaults to 50.
        mode (str, optional): 'discrete' for bar graph, 'continuous' for histogram. Defaults to 'discrete'.
    """
    # Reload iscell file and filter out non-neuron rois
    iscell = common_utils.load_is_cell_file(project, mouse, session, protocol)
    neurons_df.iscell = iscell
    neurons_df = neurons_df[neurons_df.iscell == 1]

    if mode == "continuous":
        all_preferred_depths = (
            neurons_df[neurons_df.is_depth_neuron == 1]
        ).preferred_depth_closed_loop
        bins = np.geomspace(depth_min, depth_max, num=bin_number)
        plt.hist(all_preferred_depths, bins=bins)
        plt.xscale("log")
        plt.xlabel("Preferred depth (m)")
        plt.ylabel("Frequency")

    elif mode == "discrete":
        depth_list = find_depth_neurons.find_depth_list(trials_df)
        groups = depth_list.copy()
        groups.append("not-tuned")
        depth_perc = []
        for depth in depth_list:
            depth_perc.append(
                np.mean(
                    (neurons_df.best_depth == depth) & (neurons_df.is_depth_neuron == 1)
                )
            )
        not_tuned_perc = np.mean(neurons_df.is_depth_neuron == 0)
        depth_perc.append(not_tuned_perc)
        plt.bar(np.arange(len(groups)), depth_perc)
        plt.xticks(np.arange(len(groups)), groups)
        plt.xlabel("Preferred depth (m)")
        plt.ylabel("Proportion of neurons")
    plt.title("Depth preference")


def get_depth_color(depth, depth_list, cmap=cm.cool.reversed()):
    """
    Calculate the color for a certain depth out of a depth list

    Args:
        depth (float): preferred depth of a certain neuron.
        depth_list (float): list of all depths.
        cmap (colormap, optional): colormap used. Defaults to cm.cool.reversed().

    Returns:
        rgba_color: tuple of 3 with RGB color values.
    """
    norm = mpl.colors.Normalize(
        vmin=np.log(min(depth_list)), vmax=np.log(max(depth_list))
    )
    rgba_color = cmap(norm(np.log(depth)), bytes=True)
    rgba_color = tuple(it / 255 for it in rgba_color)

    return rgba_color


def plot_spatial_distribution(
    neurons_df, trials_df, ops, stat, iscell, cmap=cm.cool.reversed()
):
    """
    Plot spatial distribution of depth preference of a session.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        ops (np.ndarray): suite2p ops.
        stat (np.ndarray): suite2p stat.
        iscell (bool): suite2p iscell file (needs to reload before the plotting)
        cmap (matplotlib object, optional): Matplotlib colormao. Defaults to cm.cool.reversed().
    """
    # Reload iscell file and filter out non-neuron rois
    neurons_df.iscell = iscell

    # set cmap
    line_colors = []
    depth_list = find_depth_neurons.find_depth_list(trials_df)
    norm = mpl.colors.Normalize(
        vmin=np.log(min(depth_list)), vmax=np.log(max(depth_list))
    )
    for depth in depth_list:
        rgba_color = cmap(norm(np.log(depth)), bytes=True)
        rgba_color = tuple(it / 255 for it in rgba_color)
        line_colors.append(rgba_color)

    #  Create a background using mean_img
    background_color = np.array([0.133, 0.545, 0.133])
    im = np.swapaxes(
        np.swapaxes(np.tile(ops["meanImg"], (3, 1, 1)), 0, 2), 0, 1
    ) / np.max(ops["meanImg"])
    im = np.multiply(im, background_color.reshape(1, -1)) * 3

    #  Assign color to pixels of neuronal mask
    # careful imshow color in BGR not RGB, but colormap seems to swap it already
    for n in (
        neurons_df[(neurons_df.iscell == 1) & (neurons_df.is_depth_neuron == 1)]
    ).roi:
        ypix = stat[n]["ypix"][~stat[n]["overlap"]]
        xpix = stat[n]["xpix"][~stat[n]["overlap"]]
        if len(xpix) > 0 and len(ypix) > 0:
            lam_mat = np.tile(
                (stat[n]["lam"][~stat[n]["overlap"]])
                / np.max(stat[n]["lam"][~stat[n]["overlap"]]),
                (3, 1),
            ).T
            rgba_color = get_depth_color(
                depth=neurons_df.loc[n, "preferred_depth_closed_loop"],
                depth_list=depth_list,
                cmap=cmap,
            )
            im[ypix, xpix, :] = (
                (np.asarray(rgba_color)[:-1].reshape(-1, 1))
                @ (lam_mat[:, 0].reshape(1, -1))
            ).T

    non_depth_neurons = (
        neurons_df[(neurons_df.iscell == 1) & (neurons_df.is_depth_neuron != 1)]
    ).roi
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


def plot_depth_tuning_curve(
    neurons_df,
    trials_df,
    roi,
    rs_thr=0.2,
    plot_fit=True,
    linewidth=3,
    linecolor="k",
    fit_linecolor="r",
):
    """
    Plot depth tuning curve for one neuron.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        rs_thr (float, optional): Threshold to cut off non-running frames. Defaults to 0.2. (m/s)
        plot_fit (bool, optional): Whether to plot fitted tuning curve or not. Defaults to True.
        linewidth (int, optional): linewidth. Defaults to 3.
        linecolor (str, optional): linecolor of true data. Defaults to "k".
        fit_linecolor (str, optional): linecolor of fitted curve. Defaults to "r".
    """

    # Load average activity and confidence interval for this roi
    depth_list = np.array(find_depth_neurons.find_depth_list(trials_df))
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
        trials_df, rs_thr=rs_thr
    )[:, :, roi]
    CI_low, CI_high = common_utils.get_confidence_interval(mean_dff_arr)
    mean_arr = np.mean(mean_dff_arr, axis=1)

    # Load gaussian fit params for this roi
    if plot_fit:
        min_sigma = 0.5
        [a, x0, log_sigma, b] = neurons_df.loc[roi, "gaussian_depth_tuning_popt"]
        x = np.geomspace(depth_list[0], depth_list[-1], num=100)
        gaussian_arr = find_depth_neurons.gaussian_func(
            np.log(x), a, x0, log_sigma, b, min_sigma
        )

    # Plotting
    plt.plot(np.log(depth_list), mean_arr, color=linecolor)
    plt.fill_between(
        np.log(depth_list),
        CI_low,
        CI_high,
        color=linecolor,
        alpha=0.3,
        edgecolor=None,
        rasterized=False,
    )
    if plot_fit:
        plt.plot(np.log(x), gaussian_arr, color=fit_linecolor)
    plt.xticks(np.log(depth_list), depth_list)
    plt.xlabel("Preferred depth (cm)")
    plt.ylabel("\u0394F/F")

    plotting_utils.despine()


# -------OLD----------------


# --- Raster plot for different depths (running speed or dFF) --- #
def plot_raster_all_depths(
    values,
    dffs,
    depth_list,
    img_VS,
    stim_dict,
    distance_bins,
    plot_rows,
    plot_cols,
    which_row,
    which_col,
    heatmap_cmap,
    fontsize_dict,
    is_trace=True,
    roi=0,
    title="",
    frame_rate=15,
    distance_max=6,
    vmax=None,
    landscape=False,
):
    if is_trace:
        values_arr, _ = create_trace_arr_per_roi(
            roi,
            dffs,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        unit_scale = 1
    else:
        values_arr, _ = create_speed_arr(
            values,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        unit_scale = 100
    distance_arr, _ = create_speed_arr(
        img_VS["EyeZ"],
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
    for idepth in range(distance_arr.shape[0]):
        for itrial in range(distance_arr.shape[1]):
            distance_arr[idepth, itrial, :] = (
                distance_arr[idepth, itrial, :] - distance_arr[idepth, itrial, 0]
            )

    binned_stats = get_binned_arr(
        xarr=distance_arr,
        yarr=values_arr,
        bin_number=distance_bins,
        bin_edge_min=0,
        bin_edge_max=6,
    )
    if vmax == None:
        vmax = np.nanmax(binned_stats["binned_yrr"]) * unit_scale
    else:
        vmax = vmax
    for idepth in range(0, len(depth_list)):
        depth = depth_list[idepth]
        if landscape:
            plt.subplot2grid([plot_rows, plot_cols], [which_row, which_col + idepth])
        else:
            plt.subplot2grid([plot_rows, plot_cols], [which_row + idepth, which_col])
        if idepth == 0:
            title_on = True
        else:
            title_on = False
        # title = f'ROI {roi} {title}, \n'
        title = f"{title}"
        # if idepth!=len(depth_list)-1:
        #     colorbar_on = False
        # else:
        #     colorbar_on = True
        colorbar_on = False
        plot_raster(
            arr=np.array(binned_stats["binned_yrr"][idepth]) * unit_scale,
            vmin=0,
            vmax=vmax,
            cmap=heatmap_cmap,
            title=title,
            title_on=title_on,
            suffix="Depth: " + str(int(depth_list[idepth] * 100)) + " cm",
            fontsize_dict=fontsize_dict,
            frame_rate=frame_rate,
            extent=[
                0,
                distance_max * 100,
                binned_stats["binned_yrr"][idepth].shape[0],
                1,
            ],
            set_nan_cmap=False,
            colorbar_on=colorbar_on,
        )
        if idepth == 0:
            plt.ylabel("Trial no.", fontsize=fontsize_dict["ylabel"])
            plt.xticks(fontsize=fontsize_dict["xticks"], rotation=45)
            plt.yticks(fontsize=fontsize_dict["yticks"])
            plt.xlabel("Corridor \n position (cm)", fontsize=fontsize_dict["xlabel"])
        else:
            plt.xticks(fontsize=fontsize_dict["xticks"], rotation=45)
            plt.yticks([])
            plt.xlabel("Corridor \n position (cm)", fontsize=fontsize_dict["xlabel"])

    return binned_stats


# --- PSTH --- #
def plot_PSTH(
    values,
    dffs,
    depth_list,
    img_VS,
    stim_dict,
    distance_bins,
    line_colors,
    heatmap_cmap,
    fontsize_dict,
    is_trace=True,
    roi=0,
    ylim=None,
    title="",
    frame_rate=15,
    distance_max=6,
):
    if is_trace:
        values_arr, _ = create_trace_arr_per_roi(
            roi,
            dffs,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        unit_scale = 1
    else:
        values_arr, _ = create_speed_arr(
            values,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        unit_scale = 100  # scale depth unit from m to cm
    distance_arr, _ = create_speed_arr(
        img_VS["EyeZ"],
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
    for idepth in range(distance_arr.shape[0]):
        for itrial in range(distance_arr.shape[1]):
            distance_arr[idepth, itrial, :] = (
                distance_arr[idepth, itrial, :] - distance_arr[idepth, itrial, 0]
            )

    binned_stats = get_binned_arr(
        xarr=distance_arr,
        yarr=values_arr,
        bin_number=distance_bins,
        bin_edge_min=0,
        bin_edge_max=6,
    )

    for idepth, linecolor in zip(range(len(depth_list)), line_colors):
        CI_low, CI_high = get_confidence_interval(
            binned_stats["binned_yrr"][idepth] * unit_scale,
            mean_arr=np.nanmean(
                binned_stats["binned_yrr"][idepth] * unit_scale, axis=0
            ),
        )
        plot_line_with_error(
            xarr=np.linspace(0, distance_max * unit_scale, distance_bins),
            arr=np.nanmean(binned_stats["binned_yrr"][idepth] * unit_scale, axis=0),
            CI_low=CI_low,
            CI_high=CI_high,
            linecolor=linecolor,
            label=str(int(depth_list[idepth] * 100)) + " cm",
            fontsize=fontsize_dict["title"],
            linewidth=3,
        )
        plt.legend(fontsize=fontsize_dict["legend"], framealpha=0.3)
        plt.title("Running Speed (cm/s)", fontsize=fontsize_dict["title"])
    plt.xlabel("Distance (cm)", fontsize=fontsize_dict["xlabel"])
    plt.ylabel("Running speed (cm/s)", fontsize=fontsize_dict["ylabel"])
    plt.xticks(fontsize=fontsize_dict["xticks"])
    plt.yticks(fontsize=fontsize_dict["yticks"])
    xlim = plt.gca().get_xlim()
    if ylim == None:
        ylim = [-0.2, plt.gca().get_ylim()[1]]
        plt.ylim(ylim)
    else:
        ylim = ylim
        plt.ylim(ylim)
    despine()

    return binned_stats, xlim, ylim


# --- Depth tuning curve --- #
MIN_SIGMA = 0.5


def gaussian_func(x, a, x0, log_sigma, b):
    a = a
    sigma = np.exp(log_sigma) + MIN_SIGMA
    return (a * np.exp(-((x - x0) ** 2)) / (2 * sigma**2)) + b


# --- RS/OF - trace heatmap --- #
def get_RS_OF_heatmap_matrix(
    speeds,
    optics,
    roi,
    dffs,
    depth_list,
    img_VS,
    stim_dict,
    log_range,
    playback=False,
    speed_thr=0.01,
    of_thr=0.03,
    frame_rate=15,
):
    extended_matrix = np.zeros((log_range["rs_bin_num"], log_range["of_bin_num"]))

    # calculate all RS/OF arrays
    speed_arr_noblank, _ = create_speed_arr(
        speeds,
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
    frame_num_pertrial_max_playback = speed_arr_noblank.shape[2]
    total_trials = speed_arr_noblank.shape[1]

    of_arr_noblank, _ = create_speed_arr(
        optics,
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
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

    if playback:
        # When RS = 0:
        speeds_playback_unthred = (
            img_VS.MouseZ.diff() / img_VS.HarpTime.diff()
        )  # with no playback. EyeZ and MouseZ should be the same.
        speeds_playback_unthred[0] = 0
        speeds_playback_unthred[speeds_playback_unthred < 0] = 0
        speed_arr_playback_unthred, _ = create_speed_arr(
            speeds_playback_unthred,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        speeds_eye_playback_unthred = (
            img_VS.EyeZ.diff() / img_VS.HarpTime.diff()
        )  # EyeZ is how the perspective of animal moves
        speeds_eye_playback_unthred[0] = 0
        speeds_playback_unthred[speeds_playback_unthred < 0] = 0
        optics_playback_unthred = calculate_OF(
            rs=speeds_eye_playback_unthred, img_VS=img_VS, mode="no_RF"
        )
        of_arr_playback_unthred, _ = create_speed_arr(
            optics_playback_unthred,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )

        zero_idx_rs = np.where(speed_arr_playback_unthred <= speed_thr)
        speeds_playback_zeros_rs = np.array(
            [
                speed_arr_playback_unthred[
                    zero_idx_rs[0][i], zero_idx_rs[1][i], zero_idx_rs[2][i]
                ]
                for i in range(len(zero_idx_rs[0]))
            ]
        )
        of_playback_zeros_rs = np.array(
            [
                of_arr_playback_unthred[
                    zero_idx_rs[0][i], zero_idx_rs[1][i], zero_idx_rs[2][i]
                ]
                for i in range(len(zero_idx_rs[0]))
            ]
        )
        trace_playback_zeros_rs = np.array(
            [
                trace_arr_noblank[
                    zero_idx_rs[0][i], zero_idx_rs[1][i], zero_idx_rs[2][i]
                ]
                for i in range(len(zero_idx_rs[0]))
            ]
        )

        # When OF = 0:
        zero_idx_of = np.where(of_arr_playback_unthred <= of_thr)
        speeds_playback_zeros_of = np.array(
            [
                speed_arr_playback_unthred[
                    zero_idx_of[0][i], zero_idx_of[1][i], zero_idx_of[2][i]
                ]
                for i in range(len(zero_idx_of[0]))
            ]
        )
        of_playback_zeros_of = np.array(
            [
                of_arr_playback_unthred[
                    zero_idx_of[0][i], zero_idx_of[1][i], zero_idx_of[2][i]
                ]
                for i in range(len(zero_idx_of[0]))
            ]
        )
        trace_playback_zeros_of = np.array(
            [
                trace_arr_noblank[
                    zero_idx_of[0][i], zero_idx_of[1][i], zero_idx_of[2][i]
                ]
                for i in range(len(zero_idx_of[0]))
            ]
        )

    # Matrix binned by RS & OF
    binned_stats = get_binned_stats_2d(
        xarr1=thr(speed_arr_noblank * 100, speed_thr * 100),
        xarr2=np.degrees(of_arr_noblank),
        yarr=trace_arr_noblank,
        bin_edges=[
            np.logspace(
                log_range["rs_bin_log_min"],
                log_range["rs_bin_log_max"],
                num=log_range["rs_bin_num"],
                base=log_range["log_base"],
            ),
            np.logspace(
                log_range["of_bin_log_min"],
                log_range["of_bin_log_max"],
                num=log_range["of_bin_num"],
                base=log_range["log_base"],
            ),
        ],
        log=True,
        log_base=log_range["log_base"],
    )

    vmin_heatmap_closeloop = np.nanmin(binned_stats["bin_means"])
    vmax_heatmap_closeloop = np.nanmax(binned_stats["bin_means"])

    extended_matrix[1:, 1:] = binned_stats["bin_means"]

    if playback:
        binned_stats_zeros_rs = get_binned_stats_2d(
            xarr1=speeds_playback_zeros_rs * 100,
            xarr2=np.degrees(of_playback_zeros_rs),
            yarr=trace_playback_zeros_rs,
            bin_edges=[
                np.array([0, 1]),
                np.logspace(
                    log_range["of_bin_log_min"],
                    log_range["of_bin_log_max"],
                    num=log_range["of_bin_num"],
                    base=log_range["log_base"],
                ),
            ],
            log=True,
            log_base=log_range["log_base"],
        )

        binned_stats_zeros_of = get_binned_stats_2d(
            xarr1=speeds_playback_zeros_of * 100,
            xarr2=np.degrees(of_playback_zeros_of),
            yarr=trace_playback_zeros_of,
            bin_edges=[
                np.logspace(
                    log_range["rs_bin_log_min"],
                    log_range["rs_bin_log_max"],
                    num=log_range["rs_bin_num"],
                    base=log_range["log_base"],
                ),
                np.array([0, 1]),
            ],
            log=True,
            log_base=log_range["log_base"],
        )

        extended_matrix[0, 1:] = binned_stats_zeros_rs["bin_means"].flatten()
        extended_matrix[1:, 0] = binned_stats_zeros_of["bin_means"].flatten()

    return extended_matrix


def set_RS_OF_heatmap_axis_ticks(log_range, fontsize_dict, playback=False, log=True):
    bin_numbers = [log_range["rs_bin_num"] - 1, log_range["of_bin_num"] - 1]
    bin_edges1 = np.logspace(
        log_range["rs_bin_log_min"],
        log_range["rs_bin_log_max"],
        num=log_range["rs_bin_num"],
        base=log_range["log_base"],
    )
    bin_edges2 = np.logspace(
        log_range["of_bin_log_min"],
        log_range["of_bin_log_max"],
        num=log_range["of_bin_num"],
        base=log_range["log_base"],
    )
    if playback:
        bin_numbers = [log_range["rs_bin_num"], log_range["of_bin_num"]]
        bin_edges1 = np.insert(bin_edges1, 0, 0)
        bin_edges2 = np.insert(bin_edges2, 0, 0)
    bin_edges1 = bin_edges1.tolist()
    bin_edges2 = bin_edges2.tolist()
    ctr = 0
    for it in bin_edges1:
        if (it >= 1) or (it == 0):
            bin_edges1[ctr] = int(np.round(it))
        else:
            bin_edges1[ctr] = np.round(it, 2)
        ctr += 1
    ctr = 0
    for it in bin_edges2:
        if it >= 1:
            bin_edges2[ctr] = int(np.round(it))
        else:
            bin_edges2[ctr] = np.round(it, 2)
        ctr += 1
    # if log == False:
    #     _, _ = plt.xticks(np.arange(bin_numbers[0]), bin_centers1, rotation=60, ha='center',
    #                       fontsize=fontsize_dict['xticks'])
    #     _, _ = plt.yticks(np.arange(bin_numbers[1]), bin_centers2, fontsize=fontsize_dict['yticks'])
    else:
        ticks_select1 = (np.arange(-1, bin_numbers[0] * 2, 1) / 2)[0::2]
        ticks_select2 = (np.arange(-1, bin_numbers[1] * 2, 1) / 2)[0::2]
        _, _ = plt.xticks(
            ticks_select1,
            bin_edges1,
            rotation=60,
            ha="center",
            fontsize=fontsize_dict["xticks"],
        )
        _, _ = plt.yticks(ticks_select2, bin_edges2, fontsize=fontsize_dict["yticks"])


def plot_RS_OF_heatmap_extended(
    matrix,
    log_range,
    playback,
    fontsize_dict,
    log=True,
    xlabel="Running Speed (cm/s)",
    ylabel="Optic Flow (degree/s)",
    vmin=None,
    vmax=None,
):
    if not playback:
        matrix = matrix[1:, 1:]
    plt.imshow(matrix.T, cmap="Reds", origin="lower", vmin=vmin, vmax=vmax)
    set_RS_OF_heatmap_axis_ticks(
        log_range=log_range, fontsize_dict=fontsize_dict, playback=playback, log=log
    )
    plt.colorbar()
    plt.xlabel(xlabel, fontsize=fontsize_dict["xlabel"])
    plt.ylabel(xlabel, fontsize=fontsize_dict["ylabel"])
