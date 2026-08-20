import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import scipy
import seaborn as sns
from functools import partial

import flexiznam as flz
from cottage_analysis.analysis import (
    find_depth_neurons,
    fit_gaussian_blob,
    common_utils,
)
from cottage_analysis.plotting import plotting_utils


def calculate_speed_tuning(speed_arr, dff_arr, bins, smoothing_sd=1, ci_range=0.95):
    # calculate speed tuning
    bin_means, _, _ = scipy.stats.binned_statistic(
        x=speed_arr,
        values=dff_arr,
        statistic="mean",
        bins=bins,
    )
    bin_counts, _, _ = scipy.stats.binned_statistic(
        x=speed_arr,
        values=dff_arr,
        statistic="count",
        bins=bins,
    )
    ci = np.zeros((len(bin_means), 2)) * np.nan
    for ibin in range(len(bin_means)):
        idx = (speed_arr > bins[ibin]) & (speed_arr < bins[ibin + 1])
        if np.sum(idx) > 0:
            ci_low, ci_high = common_utils.get_bootstrap_ci(
                dff_arr[idx], n_bootstraps=1000, sig_level=1 - ci_range
            )
            ci[ibin, 0] = ci_low[0]
            ci[ibin, 1] = ci_high[0]
    smoothed_tuning = plotting_utils.get_tuning_function(
        bin_means, bin_counts, smoothing_sd=smoothing_sd
    )
    return bin_means, smoothed_tuning, ci


def plot_speed_tuning(
    trials_df,
    roi,
    is_closed_loop,
    nbins=20,
    which_speed="RS",
    speed_min=0.01,
    speed_max=1.5,
    speed_thr=0.01,
    of_min=1e-2,
    of_max=1e4,
    smoothing_sd=1,
    markersize=5,
    linewidth=1,
    markeredgecolor="w",
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    legend_on=False,
    ci_range=0.95,
    ylim=None,
    ax=None,
):
    """Plot a neuron's speed tuning to either running speed or optic flow speed.

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        nbins (int, optional): number of bins to bin the tuning curve. Defaults to 20.
        which_speed (str, optional): 'RS': running speed; 'OF': optic flow speed.
            Defaults to 'RS'.
        speed_min (float, optional): min RS speed for the bins (m/s). Defaults to 0.01.
        speed_max (float, optional): max RS speed for the bins (m/s). Defaults to 1.5.
        speed_thr (float, optional): threshold RS for logging (m/s). Defaults to 0.01.
        of_min (float, optional): min OF speed for the bins (deg/s). Defaults to 1e-2.
        of_max (float, optional): max OF speed for the bins (deg/s). Defaults to 1e4.
        smoothing_sd (int, optional): standard deviation of the gaussian kernel for
            smoothing. Defaults to 1.
        markersize (int, optional): size of the markers. Defaults to 5.
        linewidth (int, optional): width of the line. Defaults to 1.
        markeredgecolor (str, optional): color of the marker edge. Defaults to 'w'.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and
            tick. Defaults to {"title": 20, "label": 15, "tick": 15, "legend": 5}.
        legend_on (bool, optional): whether to show the legend. Defaults to False.
        ci_range (float, optional): confidence interval range. Defaults to 0.95.
        ylim (list, optional): y-axis limits. Defaults to None.
        ax (matplotlib.axes.Axes, optional): axes to plot on. Defaults to None.

    """
    if ax is None:
        ax = plt.gca()
    if np.all([np.all(np.isnan(v[:, roi])) for v in trials_df.dff_stim.values]):
        print("All NaN dff. Not plotting")
        return
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")

    if which_speed == "RS":
        depth_list.append("blank")
        bins = (
            np.linspace(start=speed_min, stop=speed_max, num=nbins + 1, endpoint=True)
            * 100
        )
    tuning = np.zeros(((len(depth_list)), nbins))
    smoothed_tuning = np.zeros(((len(depth_list)), nbins))
    ci = np.zeros(((len(depth_list)), nbins, 2))
    bin_centers = np.zeros(((len(depth_list)), nbins))

    # Find all speed and dff of this ROI for a specific depth
    for idepth, depth in enumerate(depth_list):
        if depth == "blank":
            all_speed = trials_df[f"{which_speed}_blank"].values
            all_dff = trials_df["dff_blank"].values
        else:
            all_speed = grouped_trials.get_group(depth)[f"{which_speed}_stim"].values
            all_dff = grouped_trials.get_group(depth)["dff_stim"].values
        dff_arr = np.array([j for i in all_dff for j in i[:, roi]])
        speed_arr = np.array([j for i in all_speed for j in i])
        if which_speed == "OF":
            speed_arr = np.degrees(speed_arr)  # rad --> degrees
        else:
            speed_arr = speed_arr * 100  # m/s --> cm/s
        dff_arr = dff_arr[speed_arr > speed_thr]
        speed_arr = speed_arr[speed_arr > speed_thr]
        if which_speed == "OF":
            if (of_min is None) or (of_max is None):
                bins = np.geomspace(
                    start=np.nanmin(speed_arr),
                    stop=np.nanmax(speed_arr),
                    num=nbins + 1,
                    endpoint=True,
                )
            else:
                bins = np.geomspace(
                    start=of_min,
                    stop=of_max,
                    num=nbins + 1,
                    endpoint=True,
                )
        bin_centers[idepth] = (bins[:-1] + bins[1:]) / 2
        tuning[idepth], smoothed_tuning[idepth], ci[idepth] = calculate_speed_tuning(
            speed_arr,
            dff_arr,
            bins,
            smoothing_sd=smoothing_sd,
            ci_range=ci_range,
        )
    # Plotting
    for idepth, depth in enumerate(depth_list):
        if depth == "blank":
            linecolor = "gray"
            label = "blank"
        else:
            linecolor = plotting_utils.get_color(
                value=depth,
                value_min=np.min(depth_list[:-1]),
                value_max=np.max(depth_list[:-1]),
                cmap=cm.cool.reversed(),
                log=True,
            )
            label = f"{int(depth_list[idepth] * 100)} cm"
        ax.plot(
            bin_centers[idepth, :],
            smoothed_tuning[idepth, :],
            color=linecolor,
            label=label,
            linewidth=linewidth,
        )
        ax.errorbar(
            x=bin_centers[idepth, :],
            y=tuning[idepth, :],
            yerr=np.abs(ci[idepth, :].T - tuning[idepth, :]),
            fmt="o",
            color=linecolor,
            ls="none",
            markersize=markersize,
            linewidth=linewidth,
            markeredgewidth=0.3,
            markeredgecolor=markeredgecolor,
        )
        if which_speed == "OF":
            ax.set_xscale("log")
    # Plot tuning to gray period
    if which_speed == "RS":
        ax.set_xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
    else:
        ax.set_xlabel(
            "Optic flow speed (degrees/s)",
            fontsize=fontsize_dict["label"],
        )
    ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"], labelpad=-5)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])

    if legend_on:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.95, 1.15),
            fontsize=fontsize_dict["legend"],
            frameon=False,
            handlelength=1,
            labelspacing=0.35,
        )
    if which_speed == "RS":
        ax.set_xticks(np.linspace(speed_min, speed_max, 4) * 100)
    elif which_speed == "OF":
        ax.set_xticks(np.geomspace(1e-2, 1e4, 3))
    if ylim is None:
        ax.set_ylim(
            [np.min([np.nanmin(ci[:, :, 0]), 0]), np.round(np.nanmax(ci[:, :, 1]), 1)]
        )
        ax.set_yticks([0, np.round(np.nanmax(ci[:, :, 1]), 1)])
        ax.set_yticklabels([0, np.round(np.nanmax(ci[:, :, 1]), 1)])
    else:
        ax.set_ylim(ylim)
        ax.set_yticks([0, ylim[1]])
        ax.set_yticklabels([0, ylim[1]])
    sns.despine(ax=ax, offset=3, trim=True)


def get_RS_OF_heatmap_axis_ticks(log_range, fontsize_dict, playback=False, log=True):
    """Generate tick positions and labels for running speed and optic flow axes.

    Args:
        log_range (dict): configuration dict specifying bin count and limits.
        fontsize_dict (dict): dictionary with text styling parameters.
        playback (bool, optional): whether this is for playback visualization. Defaults to False.
        log (bool, optional): whether to compute logarithmic bins. Defaults to True.

    Returns:
        tuple: (ticks_select1, ticks_select2, bin_edges1, bin_edges2) containing
            arrays for ticking axis selectively and exact bin edges.
    """
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
    # bin_edges1 = bin_edges1 / 100
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
    else:
        log_base = log_range.get("log_base", 10)
        # Compute true logarithmic mid-points and boundaries instead of pixel
        ticks_select1 = []
        for edge in bin_edges1:
            if edge <= 0:
                ticks_select1.append(
                    log_range["rs_bin_log_min"] - 1
                )  # Fallback roughly
            else:
                ticks_select1.append(np.log(edge) / np.log(log_base))

        ticks_select2 = []
        for edge in bin_edges2:
            if edge <= 0:
                ticks_select2.append(log_range["of_bin_log_min"] - 1)
            else:
                ticks_select2.append(np.log(edge) / np.log(log_base))

        # We return the raw log values as tick selectors
        ticks_select1 = np.array(ticks_select1)
        ticks_select2 = np.array(ticks_select2)

    return ticks_select1, ticks_select2, bin_edges1, bin_edges2


def add_rsof_colorbar(fig, ax, im, cbar_width, vmin, vmax, fontsize_dict):
    """Add a colorbar for the RS-OF heatmap.

    Args:
        fig (matplotlib.figure.Figure): figure object.
        ax (matplotlib.axes.Axes): axes of the heatmap.
        im (matplotlib.image.AxesImage): image returned by imshow.
        cbar_width (float): width of the colorbar inside the figure.
        vmin (float): minimum value for the colorbar ticks.
        vmax (float): maximum value for the colorbar ticks.
        fontsize_dict (dict): dictionary specifying font sizes.
    """
    plot_x, plot_y, plot_width, plot_height = (
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width,
        ax.get_position().height,
    )
    ax2 = fig.add_axes(
        [plot_x + plot_width * 1.1, plot_y, plot_width * 0.05, plot_height / 2]
    )
    cbar = fig.colorbar(im, cax=ax2, label="\u0394F/F")
    ax2.tick_params(labelsize=fontsize_dict.get("legend", 10), length=2, pad=2)
    ax2.set_ylabel(
        "\u0394F/F", rotation=270, fontsize=fontsize_dict.get("legend", 10), labelpad=4
    )
    cbar.set_ticks([vmin, vmax])


def set_rsof_ticks(ax, log_range, tick_dict, fontsize_dict):
    """Configure axis ticks for running speed and optic flow heatmap.

    Args:
        ax (matplotlib.axes.Axes): axes to configure ticks for.
        log_range (dict): configuration dict specifying bin count and limits.
            Expected keys include "log_base" (base for logarithmic scaling, typically 2
            or 10), "rs_bin_log_min" and "rs_bin_log_max" (min and max limits for the
            running speed in log space), "rs_bin_num" (number of running speed bins),
            and similar keys for optic flow: "of_bin_log_min", "of_bin_log_max",
            and "of_bin_num".
        tick_dict (dict or None): custom tick mapping containing predefined
            tick locators and formatting values. It must contain the following keys:
            "rs_tick_select" (data coordinates for RS ticks, normally log-scaled values)
            "rs_tick_values" (labels/values to display for RS ticks, e.g. raw cm/s),
            "of_tick_select" (data coordinates for OF ticks, normally log-scaled values)
            "of_tick_values" (labels/values to display for OF ticks, e.g. raw deg/s).
        fontsize_dict (dict): dictionary with text styling parameters.
    """
    if tick_dict is None:
        (
            ticks_select1,
            ticks_select2,
            bin_edges1,
            bin_edges2,
        ) = get_RS_OF_heatmap_axis_ticks(
            log_range=log_range,
            fontsize_dict=fontsize_dict,
        )
        ax.set_xticks(ticks_select1[0::2])
        ax.set_xticklabels(
            bin_edges1[0::2],
            fontsize=fontsize_dict.get("tick", 10),
        )

        ax.set_yticks(ticks_select2[1::2])
        ax.set_yticklabels(
            bin_edges2[1::2],
            fontsize=fontsize_dict.get("tick", 10),
        )
    else:
        ax.set_xticks(tick_dict["rs_tick_select"])
        ax.set_xticklabels(
            tick_dict["rs_tick_values"],
            fontsize=fontsize_dict.get("tick", 10),
        )
        ax.set_yticks(tick_dict["of_tick_select"])
        ax.set_yticklabels(
            tick_dict["of_tick_values"],
            fontsize=fontsize_dict.get("tick", 10),
        )


def plot_RS_OF_matrix(
    trials_df,
    roi,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
    is_closed_loop=1,
    vmin=None,
    vmax=None,
    xlabel="Running speed (cm/s)",
    ylabel="Optic flow speed (°/s)",
    title="",
    cbar_width=0.01,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    ax=None,
    max_acc_ratio=None,
    max_abs_rs2motor_diff_ratio=0.3,
    of_bins=None,
    rs_bins=None,
    tick_dict=None,
    use_full_range=False,
    return_matrix=False,
):
    """Plot the heatmap of the tuning matrix of a neuron.

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        log_range (dict, optional): dictionary of the log range for the heatmap.
            Defaults to {"rs_bin_log_min": 0, "rs_bin_log_max": 2.5, "rs_bin_num": 6,
            "of_bin_log_min": -1.5, "of_bin_log_max": 3.5, "of_bin_num": 11, "log_base":
            10}.
        is_closed_loop (int, optional): 1 for closed loop, 0 for open loop. Defaults to
            1.
        vmin (float, optional): min value of the heatmap. Defaults to None.
        vmax (float, optional): max value of the heatmap. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Running speed (cm/s)".
        ylabel (str, optional): y-axis label. Defaults to "Optic flow speed (°/s)".
        title (str, optional): title of the plot. Defaults to "".
        cbar_width (float, optional): width of the colorbar. Defaults to 0.01.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label, tick
            and legend. Defaults to {"title": 20, "label": 15, "tick": 15, "legend": 5}.
        ax (matplotlib.axes.Axes, optional): axes to plot on. Defaults to None.
        max_acc_ratio (float, optional): max acceleration ratio. Defaults to None.
        max_abs_rs2motor_diff_ratio (float, optional): max absolute running speed to
            motor speed difference ratio. Defaults to 0.3.
        of_bins (np.ndarray, optional): optical flow bins. Defaults to None.
        rs_bins (np.ndarray, optional): running speed bins. Defaults to None.
        tick_dict (dict, optional): custom tick dictionary to use instead of
            automatically generated ticks. Defaults to None.
        use_full_range (bool, optional): whether to use the entire dimension range for
            visualization. Defaults to False.
        return_matrix (bool, optional): whether to return the raw 2D binned matrix
            alongside the color limits. Defaults to False.

    Returns:
        float: min value of the heatmap.
        float: max value of the heatmap.
    """

    if ax is None:
        ax = plt.gca()

    log_base = log_range.get("log_base", 10)

    # Derive extent directly from the bins
    if rs_bins is not None and of_bins is not None:
        # We skip index 0 since we drop the first bin for plotting bin_means[1:, 1:]
        rs_lower = (
            np.log(rs_bins[1]) / np.log(log_base)
            if rs_bins[1] > 0
            else log_range.get("rs_bin_log_min", 0)
        )
        rs_upper = (
            np.log(rs_bins[-1]) / np.log(log_base)
            if rs_bins[-1] > 0
            else log_range.get("rs_bin_log_max", 2.5)
        )
        of_lower = (
            np.log(of_bins[1]) / np.log(log_base)
            if of_bins[1] > 0
            else log_range.get("of_bin_log_min", -1.5)
        )
        of_upper = (
            np.log(of_bins[-1]) / np.log(log_base)
            if of_bins[-1] > 0
            else log_range.get("of_bin_log_max", 3.5)
        )
        extent = [rs_lower, rs_upper, of_lower, of_upper]
    else:
        # Fallback to logical default log_bounds
        extent = [
            log_range.get("rs_bin_log_min", 0),
            log_range.get("rs_bin_log_max", 2.5),
            log_range.get("of_bin_log_min", -1.5),
            log_range.get("of_bin_log_max", 3.5),
        ]
    plt.sca(ax)
    fig = ax.get_figure()
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
    if rs_bins is None:
        rs_bins = (
            np.logspace(
                log_range["rs_bin_log_min"],
                log_range["rs_bin_log_max"],
                num=log_range["rs_bin_num"],
                base=log_range["log_base"],
            )
            # / 100
        )
        rs_bins = np.insert(rs_bins, 0, 0)
    if of_bins is None:
        of_bins = np.logspace(
            log_range["of_bin_log_min"],
            log_range["of_bin_log_max"],
            num=log_range["of_bin_num"],
            base=log_range["log_base"],
        )
        of_bins = np.insert(of_bins, 0, 0)

    rs_arr = np.array([j for i in trials_df.RS_stim.values for j in i]) * 100
    of_arr = np.degrees([j for i in trials_df.OF_stim.values for j in i])
    acc_max_ratio = np.array(
        [j for i in trials_df.acceleration_ratio_max_stim.values for j in i]
    )
    dff_arr = np.vstack(trials_df.dff_stim.values)[:, roi]

    if max_acc_ratio is not None:
        idx = acc_max_ratio < max_acc_ratio
        rs_arr = rs_arr[idx]
        of_arr = of_arr[idx]
        dff_arr = dff_arr[idx]

    if (
        max_abs_rs2motor_diff_ratio is not None
    ) and "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns:
        rs2motor_diff_ratio = np.array(
            [j for i in trials_df.max_abs_rs2motor_diff_ratio_stim.values for j in i]
        )
        idx = rs2motor_diff_ratio < max_abs_rs2motor_diff_ratio
        rs_arr = rs_arr[idx]
        of_arr = of_arr[idx]
        dff_arr = dff_arr[idx]
        valid = ~(np.isnan(dff_arr) | np.isinf(dff_arr))
        rs_arr = rs_arr[valid]
        of_arr = of_arr[valid]
        dff_arr = dff_arr[valid]

    bin_means, rs_edges, of_egdes, _ = scipy.stats.binned_statistic_2d(
        x=rs_arr, y=of_arr, values=dff_arr, statistic="mean", bins=[rs_bins, of_bins]
    )

    if vmin is None:
        if use_full_range:
            vmin = np.nanmin(bin_means[1:-1, 1:-1].flatten())
        else:
            vmin = np.nanmax([0, np.nanmin(bin_means[1:-1, 1:-1].flatten())])
    if vmax is None:
        if use_full_range:
            vmax = np.nanmax(bin_means[1:-1, 1:-1].flatten())
        else:
            vmax = np.nanmax([0, np.nanmax(bin_means[1:-1, 1:-1].flatten())])

    cmap = matplotlib.cm.Reds.copy()
    cmap.set_bad(color="lightgrey")

    im = ax.imshow(
        bin_means[1:, 1:].T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_title(title, fontsize=fontsize_dict["title"])
    plot_x, plot_y, plot_width, plot_height = (
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width,
        ax.get_position().height,
    )

    set_rsof_ticks(ax, log_range, tick_dict, fontsize_dict)

    if is_closed_loop:
        ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"], labelpad=0)
        ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=0)
    else:
        (
            ticks_select1,
            ticks_select2,
            bin_edges1,
            bin_edges2,
        ) = get_RS_OF_heatmap_axis_ticks(
            log_range=log_range,
            fontsize_dict=fontsize_dict,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        rect = ax.get_position()
        ax_left = fig.add_axes(
            [
                rect.x0 - rect.width / (log_range["rs_bin_num"] - 1) * 1.5,
                rect.y0,
                rect.width / (log_range["rs_bin_num"] - 1),
                rect.height,
            ]
        )
        ax_left.imshow(
            bin_means[0, 1:].reshape(1, -1).T,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.yticks(
            ticks_select2[1::2], bin_edges2[1::2], fontsize=fontsize_dict["tick"]
        )
        plt.xticks([])

        ax_down = fig.add_axes(
            [
                rect.x0,
                rect.y0 - rect.height / (log_range["of_bin_num"] - 1) * 1.5,
                rect.width,
                rect.height / (log_range["of_bin_num"] - 1),
            ]
        )
        ax_down.imshow(
            bin_means[1:, 0].reshape(-1, 1).T,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.xticks(
            ticks_select1[0::2], bin_edges1[0::2], fontsize=fontsize_dict["tick"]
        )
        plt.yticks([])
        ax_corner = fig.add_axes(
            [
                rect.x0 - rect.width / (log_range["rs_bin_num"] - 1) * 1.5,
                rect.y0 - rect.height / (log_range["of_bin_num"] - 1) * 1.5,
                rect.width / (log_range["rs_bin_num"] - 1),
                rect.height / (log_range["of_bin_num"] - 1),
            ]
        )
        ax_corner.imshow(
            bin_means[0, 0].reshape(1, 1),
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.yticks(ax_corner.get_yticks()[1::2], ["< 0.03"])
        plt.xticks(ax_corner.get_xticks()[1::2], ["< 1"])

        ax_down.set_xlabel(xlabel, fontsize=fontsize_dict["label"], labelpad=0)
        ax_left.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=0)
        ax_left.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax_down.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax_corner.tick_params(
            axis="both", which="major", labelsize=fontsize_dict["tick"]
        )
    if cbar_width is not None:
        add_rsof_colorbar(fig, ax, im, cbar_width, vmin, vmax, fontsize_dict)

    if return_matrix:
        return vmin, vmax, bin_means[1:, 1:].T
    return vmin, vmax


def plot_RS_OF_fit(
    neurons_df,
    roi,
    model="g2d",
    model_label="",
    min_sigma=0.25,
    vmin=0,
    vmax=None,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
    cbar_width=0.01,
    xlabel="Running speed (cm/s)",
    ylabel="Optical flow speed (°/s)",
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 10},
    ax=None,
    sfx="",
    label_r2=True,
    of_bins=None,
    rs_bins=None,
    tick_dict=None,
    mask=None,
):
    """Plot the fitted tuning of a neuron.

    Args:
        neurons_df (pd.DataFrame): DataFrame containing the fit parameters and R-squared
            values for neurons.
        roi (int): Index of the ROI (neuron) to plot.
        model (str, optional): The model used for fitting (e.g., "g2d", "gadd", "gof",
            "grs", "gratio"). Defaults to "g2d".
        model_label (str, optional): Title label for the model plot. Defaults to "".
        min_sigma (float, optional): Minimum standard deviation constraint for the
            Gaussian fit. Defaults to 0.25.
        vmin (float, optional): Minimum value for the heat map color mapping. Defaults
            to 0.
        vmax (float, optional): Maximum value for the heat map color mapping. Defaults
            to None.
        log_range (dict, optional): Dictionary defining the logarithmic range and bin
            numbers for running speed and optic flow.
        cbar_width (float, optional): Width of the colorbar. Defaults to 0.01.
        xlabel (str, optional): Label for the x-axis. Defaults to "Running speed (cm/s)"
        ylabel (str, optional): Label for the y-axis. Defaults to "Optical flow speed
            (°/s)".
        fontsize_dict (dict, optional): Dictionary specifying font sizes for title,
            label, tick, and legend.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on. Defaults to
            None.
        sfx (str, optional): Suffix to append to the column names when extracting fit
            parameters. Defaults to "".
        label_r2 (bool, optional): Whether to display the R-squared value of the fit on
            the plot. Defaults to True.
        of_bins (numpy.ndarray, optional): Array of optic flow bin edges in degrees/s.
            Defaults to None.
        rs_bins (numpy.ndarray, optional): Array of running speed bin edges in cm/s.
            Defaults to None.
        tick_dict (dict, optional): Dictionary containing custom tick locations and
            labels for both axes. Defaults to None.
        mask (numpy.ndarray, optional): Boolean array of True/False values to grey out
            specific bins of the fit, matching the shape of the extent. Defaults to None.

    Returns:
        tuple[float, float]: A tuple containing the minimum and maximum values of the
            predicted responses (vmin, vmax).
    """

    if ax is None:
        ax = plt.gca()

    log_base = log_range.get("log_base", 10)

    if rs_bins is not None:
        rs_min_log = np.log(rs_bins[rs_bins > 0].min()) / np.log(log_base)
        rs_max_log = np.log(rs_bins.max()) / np.log(log_base)
    else:
        rs_min_log = log_range["rs_bin_log_min"]
        rs_max_log = log_range["rs_bin_log_max"]

    if of_bins is not None:
        of_min_log = np.log(of_bins[of_bins > 0].min()) / np.log(log_base)
        of_max_log = np.log(of_bins.max()) / np.log(log_base)
    else:
        of_min_log = log_range["of_bin_log_min"]
        of_max_log = log_range["of_bin_log_max"]

    rs = np.logspace(rs_min_log, rs_max_log, 100, base=log_base) / 100  # cm/s --> m/s
    of = np.logspace(of_min_log, of_max_log, 100, base=log_base)  # deg/s

    rs_grid, of_grid = np.meshgrid(np.log(rs), np.log(of))
    if model == "gof":
        params = of_grid
    elif model == "grs":
        params = rs_grid
    elif model == "gratio":
        params = rs_grid - of_grid
    else:
        params = (rs_grid, of_grid)
    funcs = {
        "g2d": fit_gaussian_blob.gaussian_2d,
        "gadd": fit_gaussian_blob.gaussian_additive,
        "gof": fit_gaussian_blob.gaussian_1d,
        "gratio": fit_gaussian_blob.gaussian_1d,
        "grs": fit_gaussian_blob.gaussian_1d,
    }
    if "roi" in neurons_df.columns and (neurons_df.roi == roi).any():
        popt = neurons_df.loc[
            neurons_df.roi == roi, f"rsof_popt_closedloop_{model}{sfx}"
        ].iloc[0]
    else:
        print(f"ROI {roi} not found in neurons_df, using iloc!!!!!!")
        popt = neurons_df[f"rsof_popt_closedloop_{model}{sfx}"].iloc[roi]

    if np.all(np.isnan(popt)):
        print("All NaN roi, not plotting. ")
        return
    resp_pred = funcs[model](
        params,
        *popt,
        min_sigma=min_sigma,
    ).reshape((len(of), len(rs)))

    if vmin is None:
        vmin = np.nanmin(resp_pred)
    if vmax is None:
        vmax = np.nanmax(resp_pred)

    extent = [
        rs_min_log,
        rs_max_log,
        of_min_log,
        of_max_log,
    ]
    im = ax.imshow(
        resp_pred,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )

    if mask is not None:
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
        mask_rgba[mask] = [0.5, 0.5, 0.5, 1.0]
        ax.imshow(mask_rgba, origin="lower", extent=extent, aspect="equal")

    if (rs_bins is None) and (of_bins is None):
        # standard log scale ticks
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["1", "10", "100"], fontsize=fontsize_dict["tick"])
        ax.set_yticks([-1, 0, 1, 2, 3])
        ax.set_yticklabels(
            ["0.1", "1", "10", "100", "1000"], fontsize=fontsize_dict["tick"]
        )
    else:
        # Use custom bins/ticks mechanism
        set_rsof_ticks(ax, log_range, tick_dict, fontsize_dict)

    if cbar_width is not None:
        fig = ax.get_figure()
        add_rsof_colorbar(fig, ax, im, cbar_width, vmin, vmax, fontsize_dict)

    plt.sca(ax)
    plt.title(
        model_label,
        fontdict={"fontsize": fontsize_dict["label"]},
    )
    if label_r2:
        plt.text(
            x=log_range["rs_bin_log_min"] + 0.2,
            y=log_range["of_bin_log_max"] - 0.7,
            s=f"$R^2$ = {neurons_df[f'rsof_test_rsq_closedloop_{model}{sfx}'].iloc[roi]:.2f}",
            fontsize=fontsize_dict["tick"],
        )
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"], labelpad=0)
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=0)
    return resp_pred.min(), resp_pred.max()


def plot_r2_comparison(
    fig,
    neurons_df,
    models,
    labels,
    ci=None,
    plot_type="violin",
    markersize=10,
    alpha=0.3,
    color="k",
    plot=True,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    reference_model="g2d",
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    neurons_df = neurons_df[neurons_df["iscell"] == 1].copy()
    if plot_type == "violin":
        results = pd.DataFrame(columns=["model", "rsq"])
        ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
        for i, col in enumerate(models):
            neurons_df[col][neurons_df[col] < -1] = 0
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {"model": labels[i], "rsq": neurons_df[col]},
                    ),
                ],
                ignore_index=True,
            )
        sns.violinplot(data=results, x="model", y="rsq", ax=ax)
        ax.set_ylabel("R-squared", fontsize=fontsize_dict["label"])
        ax.set_xlabel("Model", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        plotting_utils.despine()

        print(
            f"{labels[0]} vs {labels[1]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[0]], results['rsq'][results['model'] == labels[1]])}"
        )
        print(
            f"{labels[0]} vs {labels[2]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[0]], results['rsq'][results['model'] == labels[2]])}"
        )
        print(
            f"{labels[1]} vs {labels[2]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[1]], results['rsq'][results['model'] == labels[2]])}"
        )

    elif plot_type == "bar":
        model_cols = [f"rsof_test_rsq_closedloop_{model}" for model in models]
        # Find the best model for each neuron
        neurons_df["best_model"] = neurons_df[model_cols].idxmax(axis=1)

        # Calculate percentage of neurons that have the best model
        neuron_sum = (
            neurons_df.groupby("session")[["roi"]].agg(["count"]).values.flatten()
        )
        props = []
        # calculate the proportion of neurons that have the best model for each session
        for i, model in enumerate(model_cols):
            prop = (
                neurons_df.groupby("session")[["best_model", "roi"]]
                .apply(lambda x: x[x["best_model"] == model][["roi"]].agg(["count"]))
                .values.flatten()
            ) / neuron_sum
            props.append(prop)
            # Plot bar plot
        if plot:
            ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
            for i, model in enumerate(model_cols):
                sns.stripplot(
                    x=np.ones(len(props[i])) * i,
                    y=props[i],
                    size=markersize,
                    alpha=alpha,
                    jitter=0.3,
                    edgecolor="white",
                    color=color,
                )
                plt.plot(
                    [i - 0.4, i + 0.4],
                    [np.median(props[i]), np.median(props[i])],
                    linewidth=3,
                    color="k",
                )
                if ci is not None:
                    plt.fill_between(
                        [i - 0.4, i + 0.4],
                        [ci[i][0], ci[i][0]],
                        [ci[i][1], ci[i][1]],
                        color=color,
                        alpha=0.7,
                        edgecolor="none",
                    )
        if plot:
            sns.despine(offset=5, ax=plt.gca())
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(labels, fontsize=fontsize_dict["label"], rotation=90)
            ax.set_ylabel(
                "Proportion of neurons with best model fit",
                fontsize=fontsize_dict["label"],
            )
            ax.set_ylim([0, 1])
            ax.tick_params(axis="y", which="major", labelsize=fontsize_dict["tick"])
            reference_id = models.index(reference_model)
            for i, model in enumerate(model_cols):
                if i != reference_id:
                    wilcoxon_result = scipy.stats.wilcoxon(
                        props[reference_id], props[i]
                    )
                    print(f"{labels[reference_id]} vs {labels[i]}: {wilcoxon_result}")
        return props


def plot_r2_cdfs(
    neurons_df,
    models,
    model_labels,
    xlim=(10**-4, 1),
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    def cdf(values):
        x = np.sort(values)
        y = np.linspace(0, 1, len(x) + 1)
        return x, y[1:]

    neurons_df_sig = neurons_df[
        (neurons_df["iscell"] == 1)
        & (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.001)
        & (neurons_df["preferred_depth_amplitude"] > 0.5)
    ]
    for model, label in zip(models, model_labels):
        plt.plot(*cdf(neurons_df_sig[f"rsof_test_rsq_closedloop_{model}"]), label=label)
    plt.xscale("log")
    plt.legend(frameon=False, fontsize=fontsize_dict["label"])
    plt.xlim(xlim)
    plt.ylim([0, 1])
    plt.gca().tick_params(axis="both", labelsize=fontsize_dict["tick"])
    plt.xlabel("$R^2$", fontsize=fontsize_dict["label"])
    plt.ylabel("Cumulative proportion of neurons", fontsize=fontsize_dict["label"])
    sns.despine(offset=5, ax=plt.gca())


def plot_r2_violin(
    neurons_df,
    models,
    model_labels,
    ylim=(10**-4, 1),
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    cols = [f"rsof_test_rsq_closedloop_{model}" for model in models]
    df = neurons_df[cols].melt(var_name="model", value_name="r2")
    df["model"] = df["model"].apply(lambda x: model_labels[cols.index(x)])
    df["r2"][df["r2"] < ylim[0]] = ylim[0]
    df["r2"][df["r2"] > ylim[1]] = ylim[1]
    sns.violinplot(
        data=df,
        y="r2",
        x="model",
        log_scale=True,
        hue="model",
        cut=0,
        inner="quartile",
        legend=False,
        fill=False,
        palette="Set1",
    )
    plt.ylim(ylim)
    plt.gca().tick_params(axis="y", labelsize=fontsize_dict["tick"])
    plt.gca().tick_params(axis="x", labelsize=fontsize_dict["label"], rotation=90)
    plt.xlabel("")
    # change the first xtick label
    ytick_labels = plt.gca().get_yticklabels()
    ytick_labels[1].set_text(f"\u2264 {ytick_labels[1].get_text()}")
    plt.gca().set_yticklabels(ytick_labels)
    plt.ylabel("$R^2$", fontsize=fontsize_dict["label"])
    sns.despine(offset=5, ax=plt.gca())


def plot_scatter(
    fig,
    neurons_df,
    xcol,
    ycol,
    xlabel="Running speed (cm/s)",
    ylabel="Preferred depth (cm)",
    s=10,
    alpha=0.2,
    c="g",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    aspect_equal=False,
    plot_diagonal=False,
    diagonal_color="k",
    diagonal_linewidth=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    log_scale=True,
    edgecolors="none",
):
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    X = neurons_df[xcol].values
    y = neurons_df[ycol].values
    ax.scatter(X, y, s=s, alpha=alpha, c=c, edgecolors=edgecolors, linewidths=0.5)
    if plot_diagonal:
        diag = [
            np.max((plt.xlim()[0], plt.ylim()[0])),
            np.min((plt.xlim()[1], plt.ylim()[1])),
        ]
        ax.plot(
            diag,
            diag,
            c=diagonal_color,
            linestyle="dotted",
            linewidth=diagonal_linewidth,
        )
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"], labelpad=1)
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=1)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    if aspect_equal:
        ax.set_aspect("equal")
    plotting_utils.despine()
    r, p = scipy.stats.spearmanr(X, y)
    print(f"Correlation between {xcol} and {ycol}: R = {r}, p = {p}")


def plot_2d_hist(
    fig,
    neurons_df,
    xcol,
    ycol,
    xlabel="Running speed (cm/s)",
    ylabel="Preferred depth (cm)",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    xlim=None,
    ylim=None,
    aspect_equal=True,
    plot_diagonal=False,
    diagonal_linewidth=1,
    diagonal_color="k",
    contour_color="k",
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    log_scale=True,
    color="k",
    linewidth=1,
    plot_scatter=True,
    s=3,
    alpha=0.5,
    edgecolors="none",
    rasterized=False,
):
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    X = neurons_df[xcol].values
    y = neurons_df[ycol].values
    # plota 2d histogram on log scale
    sns.kdeplot(
        x=X,
        y=y,
        color=contour_color,
        log_scale=log_scale,
        linewidths=linewidth,
        cut=0,
        levels=5,
    )
    if plot_scatter:
        ax.scatter(
            X,
            y,
            s=s,
            alpha=alpha,
            c=color,
            edgecolors=edgecolors,
            linewidths=0.5,
            rasterized=rasterized,
        )
    if plot_diagonal:
        diag = [
            np.max((plt.xlim()[0], plt.ylim()[0])),
            np.min((plt.xlim()[1], plt.ylim()[1])),
        ]
        ax.plot(
            diag,
            diag,
            c=diagonal_color,
            linestyle="dotted",
            linewidth=diagonal_linewidth,
        )
    if xlim is None:
        xlim = [np.nanmin(X) * 0.9, np.nanmax(X) / 0.9]
    if ylim is None:
        ylim = [np.nanmin(y) * 0.9, np.nanmax(y) / 0.9]
    plt.xlim(xlim)
    plt.ylim(ylim)

    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"], labelpad=1)
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=1)
    # from matplotlib.ticker import LogLocator
    # from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    # ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    # ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    # ax.yaxis.set_minor_locator(MultipleLocator(2))
    # ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    if aspect_equal:
        ax.set_aspect("equal")
    plotting_utils.despine()
    r, p = scipy.stats.spearmanr(X, y)
    print(f"Correlation between {xcol} and {ycol}: R = {r}, p = {p}")
    return r, p


def plot_speed_colored_by_depth(
    fig,
    neurons_df,
    xcol,
    ycol,
    zcol,
    xlabel="Running speed (cm/s)",
    ylabel="Optic flow speed (°/s)",
    zlabel="Preferred depth (cm)",
    s=10,
    alpha=0.2,
    cmap="cool_r",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    edgecolors="none",
    depths=np.geomspace(5, 640, 8),
):
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    depth_range = [np.min(depths), np.max(depths)]
    norm = matplotlib.colors.LogNorm(depth_range[0], depth_range[1])
    sns.scatterplot(
        neurons_df,
        x=xcol,
        y=ycol,
        hue=neurons_df[zcol],
        hue_norm=norm,
        palette="cool_r",
        s=s,
        alpha=alpha,
        ax=ax,
        edgecolor=edgecolors,
        linewidth=0.2,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sns.despine(ax=plt.gca())
    ax.set_aspect("equal", "box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.get_legend().remove()
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    cbar = plt.colorbar(sm, shrink=0.8, ax=ax, ticks=depth_range)
    cbar.ax.set_ylabel(
        zlabel, rotation=270, fontsize=fontsize_dict["legend"], labelpad=10
    )
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"])
    # set colorbar ticks to be at the center of the color range
    cbar.ax.set_yticklabels(["< 5", "> 640"])
    # move colorbar down to align with bottom of ax
    ax_pos = ax.get_position()
    cbar_pos = cbar.ax.get_position()
    cbar_pos.y0 = ax_pos.y0
    cbar_pos.y1 = ax_pos.y0 + ax_pos.height * 0.3
    cbar.ax.set_position(cbar_pos)
    cbar.ax.minorticks_off()

    ax_inset = fig.add_axes(
        [
            ax_pos.x0 + ax_pos.width * 1.2,
            ax_pos.y0 + ax_pos.height * 0.5,
            ax_pos.width * 0.5,
            ax_pos.height * 0.5,
        ]
    )
    for depth in depths:
        ax_inset.plot(
            xrange,
            np.rad2deg(xrange / depth),
            c=sm.to_rgba(depth),
            linewidth=0.5,
        )
    sns.despine(ax=ax)
    sns.despine(ax=ax_inset)
    ax_inset.set_aspect("equal", "box")
    ax_inset.set_xscale("log")
    ax_inset.set_yscale("log")
    # same x and y ticks as ax without labels
    from matplotlib.ticker import LogLocator

    ax_inset.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax_inset.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax_inset.minorticks_off()
    ax_inset.tick_params(labelbottom=False, labelleft=False)
    ax_inset.set_xlim(xrange)
    ax_inset.set_ylim(yrange)
    ax_inset.set_xlabel("Running speed", fontsize=fontsize_dict["tick"], labelpad=5)
    ax_inset.set_ylabel("Optic flow speed", fontsize=fontsize_dict["tick"], labelpad=5)


def add_trial_colorbox(
    ax,
    trial_starts,
    trial_lengths,
    depths,
    depth_list,
    param_trace,
    fs,
    ylim,
    cmap=cm.cool.reversed(),
    alpha=0.3,
):
    for i, trial_start in enumerate(trial_starts):
        color = plotting_utils.get_color(
            value=depths[i],
            value_min=np.min(depth_list),
            value_max=np.max(depth_list),
            cmap=cmap,
            log=True,
        )
        if ylim is None:
            rect = patches.Rectangle(
                (trial_starts[i] / fs, np.nanmin(param_trace)),
                trial_lengths[i] / fs,
                (np.nanmax(param_trace) - np.nanmin(param_trace)) * 1.1,
                linewidth=0,
                edgecolor="none",
                facecolor=color,
                alpha=alpha,
            )
        else:
            rect = patches.Rectangle(
                (trial_starts[i] / fs, ylim[0]),
                trial_lengths[i] / fs,
                ylim[1] - ylim[0],
                linewidth=0,
                edgecolor="none",
                facecolor=color,
                alpha=alpha,
            )
        ax.add_patch(rect)


def plot_speed_trace(
    trials_df,
    trial_list,
    param,
    fs,
    ax,
    ylabel,
    plot=True,
    xlim=(0, 100),
    ylim=None,
    linecolor="k",
    linewidth=1,
    plot_trial_number=False,
    colorbox_alpha=0.4,
    OF_to_degree=True,
    fontsize_dict={"title": 15, "label": 10, "ticks": 10},
):
    if "RS" in param:
        if f"{param}_merged" not in trials_df.columns:
            trials_df[f"{param}_merged"] = trials_df.apply(
                lambda x: np.concatenate([x[f"{param}_stim"], x[f"{param}_blank"]]),
                axis=1,
            )
        param_trace = np.concatenate(
            [row[f"{param}_merged"] for _, row in trials_df.iloc[trial_list].iterrows()]
        )
        if trial_list[0] == 0:
            blank_start = trials_df.iloc[trial_list[0]][f"{param[:2]}_blank_pre"][
                -int(fs * 10) :
            ]
            param_trace = np.concatenate([blank_start, param_trace])
        else:
            blank_start = trials_df.iloc[trial_list[0]][f"{param[:2]}_blank_pre"]
            param_trace = np.concatenate([blank_start, param_trace])
        param_trace = param_trace * 100
    elif "OF" in param:
        if f"{param}_merged" not in trials_df.columns:
            # Use RS_blank to pad NaN as OF might not be defined in blanks
            trials_df[f"{param}_merged"] = trials_df.apply(
                lambda x: np.concatenate(
                    [x[f"{param}_stim"], np.full(len(x["RS_blank"]), np.nan)]
                ),
                axis=1,
            )
        param_trace = np.concatenate(
            [row[f"{param}_merged"] for _, row in trials_df.iloc[trial_list].iterrows()]
        )
        if OF_to_degree:  # to convert OF from rads/s to degrees/s
            param_trace = np.degrees(param_trace)
        if trial_list[0] == 0:
            blank_start = trials_df.iloc[trial_list[0]]["RS_blank_pre"][-int(fs * 10) :]
            param_trace = np.concatenate([np.full(int(fs * 10), np.nan), param_trace])
        else:
            blank_start = trials_df.iloc[trial_list[0]]["RS_blank_pre"]
            param_trace = np.concatenate(
                [np.full(len(blank_start), np.nan), param_trace]
            )
        param_trace[param_trace < 1e-2] = 1e-2
        param_trace[0] = 1e-2

    trial_starts = np.cumsum(
        [
            len(row[f"{param}_merged"])
            for _, row in trials_df.iloc[trial_list].iterrows()
        ]
    )
    if "processed" not in param:
        trial_lengths = [
            len(row[f"{param}_stim"])
            for _, row in trials_df.iloc[trial_list].iterrows()
        ]
    else:
        trial_lengths = [
            len(row[f"{param}"]) for _, row in trials_df.iloc[trial_list].iterrows()
        ]
    trial_starts = np.concatenate([[0], trial_starts[:-1]]) + len(blank_start)

    depths = trials_df.iloc[trial_list]["depth"].values
    depth_list = np.sort(trials_df.depth.unique())

    if ylim is None:
        ylim = [np.nanmin(param_trace), np.nanmax(param_trace)]
    # plot param
    if plot:
        ax.plot(
            np.linspace(0, len(param_trace) / fs, len(param_trace)),
            param_trace,
            c=linecolor,
            linewidth=linewidth,
        )
        if "RS" in param:
            ax.set_ylim(ylim)
            ax.set_yticks([0, ylim[1] // 10 * 10])
        if "OF" in param:
            ax.set_yscale("log")
            if ylim is None:
                ax.set_ylim(1e-2, np.nanmax(param_trace) * 2)
                ax.set_yticks([1e-2, 1e2])
            else:
                ax.set_ylim(ylim)
                ax.set_yticks([ylim[0], ylim[1]])
        ax.set_ylabel(ylabel, rotation=90, labelpad=15, fontsize=fontsize_dict["label"])
        # ax.set_xlim(0, len(param_trace) / fs)
        ax.set_xlim(xlim)
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])

        # plot trials
        add_trial_colorbox(
            ax=ax,
            trial_starts=trial_starts,
            trial_lengths=trial_lengths,
            depths=depths,
            depth_list=depth_list,
            param_trace=param_trace,
            fs=fs,
            ylim=ylim,
            alpha=colorbox_alpha,
        )

        # plot trial number at xticks instead of time
        if plot_trial_number:
            ax.set_xticks((np.array(trial_starts) + np.array(trial_lengths) / 2) / fs)
            ax.set_xticklabels(np.arange(len(trial_list)))
            ax.set_xlabel("Trial number", fontsize=fontsize_dict["label"])

        # remove upper and right frame of the plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ylim = ax.get_ylim()
    return ylim


def plot_speed_trace_closed_open_loop(
    flexilims_session,
    session_name,
    trials_df,
    trial_list,
    positions,
    linewidth=1,
    fontsize_dict={"title": 8, "label": 8, "ticks": 6},
):
    suite2p_datasets = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        flexilims_session=flexilims_session,
        return_dataseries=False,
        filter_datasets={"anatomical_only": 3},
    )
    fs = suite2p_datasets[0].extra_attributes["fs"]

    fig = plt.gcf()
    xlims = []
    ylims = []
    axes = [fig.add_axes(position) for position in positions]
    for closed_loop, title in zip([1, 0], ["Closed loop", "Open loop"]):
        for param, ylabel in zip(["RS", "OF"], ["RS (cm/s)", "OF (°/s)"]):
            ax = axes[0]
            ylim = plot_speed_trace(
                trials_df=trials_df[trials_df.closed_loop == closed_loop],
                trial_list=trial_list,
                param=param,
                fs=fs,
                ax=ax,
                ylabel=ylabel,
                linecolor="k",
                linewidth=linewidth,
                plot=False,
                fontsize_dict=fontsize_dict,
            )
            ylims.append(ylim)

    i = 0
    for closed_loop, title in zip([1, 0], ["Closed loop", "Open loop"]):
        for param, ylabel in zip(["RS", "OF"], ["RS (cm/s)", "OF (°/s)"]):
            if param == "RS":
                lim_set = 0
            else:
                lim_set = 1
            ax = axes[i]
            _ = plot_speed_trace(
                trials_df=trials_df[trials_df.closed_loop == closed_loop],
                trial_list=trial_list,
                param=param,
                fs=fs,
                ax=ax,
                ylabel=ylabel,
                linecolor="k",
                linewidth=linewidth,
                plot=True,
                ylim=(
                    np.max([ylims[lim_set][0], ylims[lim_set + 2][0]]),
                    np.max([ylims[lim_set][1], ylims[lim_set + 2][1]]),
                ),
                fontsize_dict=fontsize_dict,
            )
            xlims.append(ax.get_xlim())

            # if param == "RS":
            #     ax.set_title(title, fontsize=fontsize_dict["title"])
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=fontsize_dict["tick"],
                length=2,
                pad=5,
            )
            if (i == 0) or (i == 2):
                labelbottom = False
            else:
                labelbottom = True
            if (i == 2) or (i == 3):
                labelleft = False
            else:
                labelleft = True
            ax.tick_params(
                labelbottom=labelbottom,
                labelleft=labelleft,
                left=True,
                bottom=True,
                pad=2,
            )
            if (i == 1) or (i == 3):
                ax.set_xlabel("Time (s)", fontsize=fontsize_dict["label"])
            if closed_loop == 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=0)
            i += 1
    # axes[0].set_ylim(ylims[2])
    # axes[2].set_ylim(ylims[2])
    # axes[1].set_ylim(ylims[3])
    # axes[3].set_ylim(ylims[3])

    # # Add scalebar
    # positions_2 = axes[2].get_position()
    # ax_scalebar1 = plt.gcf().add_axes([positions_2.x0+positions_2.width*0.8, positions_2.y0-positions_2.height*0.1, positions_2.width, positions_2.height])
    # common_utils.draw_axis_scalebars(ax_scalebar1, 0, 0, 25, 50, scalebar_labels=["25 s", "50 cm/s"], xlim=xlims[0], ylim=ylims[0], label_fontsize=fontsize_dict["legend"], linewidth=1, right=True, bottom=False)
    # positions_3 = axes[3].get_position()
    # ax_scalebar2 = plt.gcf().add_axes([positions_3.x0+positions_3.width*0.8, positions_3.y0-positions_3.height*0.3, positions_3.width, positions_3.height])
    # common_utils.draw_axis_scalebars(ax_scalebar2, 0, 0, 25, 100, scalebar_labels=["25 s", "100 \u00B0/s"], xlim=xlims[0], ylim=ylims[0], label_fontsize=fontsize_dict["legend"], linewidth=1, right=True, bottom=True)

    # # Add ylabel
    # axes[0].text(-15,
    #              np.sum(ylims[0])/2,
    #              "RS", fontsize=fontsize_dict["tick"], rotation=0, ha="left", va="center",)
    # axes[1].text(-15,
    #              1,
    #              "OF", fontsize=fontsize_dict["tick"], rotation=0, ha="left", va="center",)


def plot_openloop_rs_correlation_alldepths(
    results,
    depth_list,
    fontsize_dict,
    ax1,
    ax2,
    linewidth=3,
    elinewidth=3,
    jitter=0.2,
    scatter_markersize=2,
    scatter_alpha=0.5,
    capsize=3,
    capthick=10,
    ylim=None,
):
    results = results[results["rs_correlation_rval_openloop"].notnull().values]
    r_all = results["rs_correlation_rval_openloop"].values.astype(float)
    r_alldepths = np.vstack(
        [j for i in results["rs_correlation_rval_openloop_alldepths"].values for j in i]
    )

    CI_low_all, CI_high_all = common_utils.get_bootstrap_ci(r_all.T, sig_level=0.05)
    CI_low, CI_high = common_utils.get_bootstrap_ci(r_alldepths.T, sig_level=0.05)
    sns.stripplot(
        x=np.ones(r_all.shape[0]).flatten(),
        y=r_all.flatten(),
        jitter=jitter,
        edgecolor="white",
        color="k",
        alpha=scatter_alpha,
        ax=ax1,
        size=scatter_markersize,
    )
    ax1.plot(
        [-0.4, +0.4],
        [np.mean(r_all), np.mean(r_all)],
        linewidth=linewidth,
        color="k",
    )
    ax1.errorbar(
        x=0,
        y=np.mean(r_all),
        yerr=np.array(
            [np.mean(r_all) - CI_low_all, CI_high_all - np.mean(r_all)]
        ).reshape(2, 1),
        capsize=capsize,
        elinewidth=elinewidth,
        ecolor="k",
        capthick=capthick,
    )

    for idepth in range(len(depth_list)):
        color = plotting_utils.get_color(
            value=depth_list[idepth],
            value_min=np.min(depth_list),
            value_max=np.max(depth_list),
            cmap=cm.cool.reversed(),
            log=True,
        )
        sns.stripplot(
            x=np.ones(r_alldepths.shape[0]) * idepth,
            y=r_alldepths[:, idepth],
            jitter=jitter,
            edgecolor="white",
            color=color,
            alpha=scatter_alpha,
            ax=ax2,
            size=scatter_markersize,
        )
        ax2.plot(
            [idepth - 0.05 * len(depth_list), idepth + 0.05 * len(depth_list)],
            [np.mean(r_alldepths[:, idepth]), np.mean(r_alldepths[:, idepth])],
            linewidth=linewidth,
            color=color,
        )
        ax2.errorbar(
            x=idepth,
            y=np.mean(r_alldepths[:, idepth]),
            yerr=np.array(
                [
                    np.mean(r_alldepths[:, idepth]) - CI_low[idepth],
                    CI_high[idepth] - np.mean(r_alldepths[:, idepth]),
                ]
            ).reshape(2, 1),
            capsize=capsize,
            elinewidth=elinewidth,
            ecolor=color,
            capthick=capthick,
        )
    ax2.get_yaxis().set_visible(False)
    ax2.set_xticklabels(
        np.round((depth_list * 100)).astype("int"), fontsize=fontsize_dict["label"]
    )
    if ylim is None:
        ax1.set_ylim(-0.1, 1)
        ax2.set_ylim(-0.1, 1)
    else:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
    ax1.set_xlim([-1, 1])
    ax2.set_xlabel("Depth (cm)", fontsize=fontsize_dict["label"])
    ax2.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    sns.despine(ax=ax2, left=True)
    ax1.set_xticklabels(["All"], fontsize=fontsize_dict["label"])
    ax1.set_ylabel(
        "Correlation between actual\nand virtual running speed",
        fontsize=fontsize_dict["label"],
    )
    ax1.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    sns.despine(ax=ax1)


def plot_histogram_overlay(
    ax,
    fig,
    rs,
    depth_list,
    nbins,
    scaling_factor=0.01,
    facecolor="g",
    edgecolor="k",
    alpha=0.5,
    ylim=[1, 2e3],
    xlim=None,
):
    """Plot histogram overlay of OF distribution for different depths on top of the preferred OF-depth scatter plot.

    Args:
        ax (matplotlib.axes): Axes object to plot the histogram overlay
        fig (matplotlib.figure): Figure object to plot the histogram overlay
        rs (numpy.array): Running speed array
        depth_list (list): List of depths for which to plot the histogram overlay
        nbins (int): Number of bins for the histogram
        scaling_factor (float): Scaling factor for the histogram
        facecolor (str): Facecolor of the histogram
        edgecolor (str): Edgecolor of the histogram
        alpha (float): Transparency of the histogram
        ylim (list): Y-axis limits
        xlim (list): X-axis limits
    """
    ax2 = fig.add_axes(
        [
            ax.get_position().x0,
            ax.get_position().y0,
            ax.get_position().width,
            ax.get_position().height,
        ]
    )
    ax2.set_facecolor("none")
    if xlim is None:
        xlim = ax.get_xlim()
    ax2.set_xlim(xlim)
    ax2.set_xscale("log")
    for idepth, depth in enumerate(depth_list):
        of = np.degrees(rs / depth)
        bins = np.geomspace(np.nanmin(of), np.nanmax(of), nbins)
        n, _ = np.histogram(of, bins=bins)
        # Calculate bin widths
        bin_width = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]

        # Plot histogram manually
        bottom = (np.log10(depth * 100) - np.log10(ylim[0])) / (
            np.log10(ylim[1]) - np.log10(ylim[0])
        ) * (ylim[1] - ylim[0]) + ylim[0]
        ax2.bar(
            bins[:-1],
            (n * scaling_factor),
            width=bin_width,
            align="edge",
            facecolor=facecolor,
            edgecolor=edgecolor,
            bottom=bottom,
            alpha=alpha,
        )
        ax2.set_ylim(ylim)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
    sns.despine(ax=ax2)


def plot_treadmill_vs_closedloop_matrix(
    trials_df_tread,
    trials_df_sphere,
    roi,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
    is_closed_loop_tread=1,
    is_closed_loop_sphere=1,
    title_tread="Treadmill",
    title_sphere="Closed-loop",
    figsize=(12, 5),
    axes=None,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    max_abs_rs2motor_diff_ratio=0.3,
    split_tread_half=False,
    **kwargs,
):
    """
    Plot the RS-OF matrix for treadmill and closed-loop recordings side-by-side.

    Args:
        trials_df_tread (pd.DataFrame): Dataframe for treadmill trials.
        trials_df_sphere (pd.DataFrame): Dataframe for sphere (closed-loop) trials.
        roi (int): ROI index to plot.
        log_range (dict, optional): Log range for the heatmap. Defaults to None.
        is_closed_loop_tread (int, optional): 1 for closed loop, 0 for open loop for treadmill. Defaults to 0.
        is_closed_loop_sphere (int, optional): 1 for closed loop, 0 for open loop for sphere. Defaults to 1.
        title_tread (str, optional): Title for treadmill plot. Defaults to "Treadmill".
        title_sphere (str, optional): Title for sphere plot. Defaults to "Closed-loop".
        figsize (tuple, optional): Figure size. Defaults to (12, 5).
        axes (np.ndarray, optional): Axes object to plot the heatmap. Defaults to None.
        fontsize_dict (dict, optional): Dictionary of fontsizes.
        max_abs_rs2motor_diff_ratio (float, optional): Maximum absolute rs2motor diff
            ratio to consider. Defaults to 0.3.
        split_tread_half (bool, optional): Whether to split the treadmill into two halves. Defaults to False.
        **kwargs: Additional arguments passed to plot_RS_OF_matrix.
    """
    if axes is None:
        if split_tread_half:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = axes[0].get_figure()

    max_acc_ratio = kwargs.get("max_acc_ratio", None)
    rs_bins = kwargs.get("rs_bins", None)
    of_bins = kwargs.get("of_bins", None)
    vmin, vmax = 0, 1e-4
    for idx_df, trials_df in enumerate([trials_df_tread, trials_df_sphere]):
        if rs_bins is None:
            rs_bins = (
                np.logspace(
                    log_range["rs_bin_log_min"],
                    log_range["rs_bin_log_max"],
                    num=log_range["rs_bin_num"],
                    base=log_range["log_base"],
                )
                # / 100
            )
            rs_bins = np.insert(rs_bins, 0, 0)
        if of_bins is None:
            of_bins = np.logspace(
                log_range["of_bin_log_min"],
                log_range["of_bin_log_max"],
                num=log_range["of_bin_num"],
                base=log_range["log_base"],
            )
            of_bins = np.insert(of_bins, 0, 0)

        rs_arr = np.array([j for i in trials_df.RS_stim.values for j in i]) * 100
        of_arr = np.degrees([j for i in trials_df.OF_stim.values for j in i])
        acc_max_ratio = np.array(
            [j for i in trials_df.acceleration_ratio_max_stim.values for j in i]
        )
        dff_arr = np.vstack(trials_df.dff_stim.values)[:, roi]

        if max_acc_ratio is not None:
            idx = acc_max_ratio < max_acc_ratio
            rs_arr = rs_arr[idx]
            of_arr = of_arr[idx]
            dff_arr = dff_arr[idx]
        if (not idx_df) and max_abs_rs2motor_diff_ratio is not None:
            # apply filter on treadmill only
            rs2motor_diff_ratio = np.array(
                [
                    j
                    for i in trials_df.max_abs_rs2motor_diff_ratio_stim.values
                    for j in i
                ]
            )
            idx = rs2motor_diff_ratio < max_abs_rs2motor_diff_ratio
            rs_arr = rs_arr[idx]
            of_arr = of_arr[idx]
            dff_arr = dff_arr[idx]

        bin_means, rs_edges, of_egdes, _ = scipy.stats.binned_statistic_2d(
            x=rs_arr,
            y=of_arr,
            values=dff_arr,
            statistic="mean",
            bins=[rs_bins, of_bins],
        )
        vmax = max(vmax, np.round(np.nanmax(bin_means[1:-1, 1:-1]), 2))
        vmin = min(vmin, np.round(np.nanmin(bin_means[1:-1, 1:-1].flatten()), 2))
    vmin = max(0, vmin)

    # Plot sphere matrix
    # Note: We use the same vmin/vmax for consistent comparison
    plot_RS_OF_matrix(
        trials_df_sphere,
        roi,
        log_range=log_range,
        is_closed_loop=is_closed_loop_sphere,
        title=title_sphere,
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        fontsize_dict=fontsize_dict,
        cbar_width=None,
        **kwargs,
    )

    # Plot treadmill matrix
    if split_tread_half:
        trials_df_tread_first_half = trials_df_tread.copy()
        trials_df_tread_second_half = trials_df_tread.copy()
        for idx in trials_df_tread.index:
            dff = trials_df_tread.loc[idx, "dff_stim"]
            npts = len(dff)
            # put nan in either the first or second half
            dff_first_half = dff.copy()
            dff_first_half[npts // 2 :] = np.nan
            trials_df_tread_first_half.at[idx, "dff_stim"] = dff_first_half
            dff_second_half = dff.copy()
            dff_second_half[: npts // 2] = np.nan
            trials_df_tread_second_half.at[idx, "dff_stim"] = dff_second_half

        plot_RS_OF_matrix(
            trials_df_tread_first_half,
            roi,
            log_range=log_range,
            is_closed_loop=is_closed_loop_tread,
            title=title_tread + " first half",
            vmin=vmin,
            vmax=vmax,
            ax=axes[1],
            fontsize_dict=fontsize_dict,
            max_abs_rs2motor_diff_ratio=max_abs_rs2motor_diff_ratio,
            cbar_width=None,
            **kwargs,
        )
        axes[1].set_ylabel("")
        axes[1].set_yticklabels([])
        plot_RS_OF_matrix(
            trials_df_tread_second_half,
            roi,
            log_range=log_range,
            is_closed_loop=is_closed_loop_tread,
            title=title_tread + " second half",
            vmin=vmin,
            vmax=vmax,
            ax=axes[2],
            fontsize_dict=fontsize_dict,
            max_abs_rs2motor_diff_ratio=max_abs_rs2motor_diff_ratio,
            **kwargs,
        )
        axes[2].set_ylabel("")
        axes[2].set_yticklabels([])
    else:
        plot_RS_OF_matrix(
            trials_df_tread,
            roi,
            log_range=log_range,
            is_closed_loop=is_closed_loop_tread,
            title=title_tread,
            vmin=vmin,
            vmax=vmax,
            ax=axes[1],
            fontsize_dict=fontsize_dict,
            max_abs_rs2motor_diff_ratio=max_abs_rs2motor_diff_ratio,
            **kwargs,
        )
        axes[1].set_ylabel("")
    plt.tight_layout()
    return fig, axes


def plot_rsof_slice(
    ax,
    b_s,
    b_e,
    tav_df,
    of_bins,
    gaussian_func_=None,
    lower_bounds=None,
    upper_bounds=None,
    niter=10,
    of_min=2**-8,
    of_max=2**12,
    plot_trials=True,
    plot_fit_mean=True,
    plot_rs_label=True,
    color="darkorchid",
    linewidth=2,
    capsize=3,
    markersize=None,
    scatter_size=20,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 10},
):
    """
    Filters data for a specific running speed bin, fits a 1D Gaussian to optic flow responses,
    and plots the raw data, fit, and binned mean with bootstrap CI.

    If `plot_fit_mean`, a downward triangle marks the preferred OF (mean of the
    gaussian fit) at the top of the axes.

    color, linewidth, capsize, markersize and scatter_size are cosmetic and default
    to values suited to full-page axes. Reduce them for small panels.
    """
    if gaussian_func_ is None:
        gaussian_func_ = partial(fit_gaussian_blob.gaussian_1d, min_sigma=0.25)
    if lower_bounds is None:
        lower_bounds = [-np.inf, np.log(of_min), -np.inf, -np.inf]
    if upper_bounds is None:
        upper_bounds = [np.inf, np.log(of_max), np.inf, np.inf]

    # 1. Filter data
    mid_val = np.sqrt(b_s * b_e)
    if plot_rs_label:
        ax.text(
            1,
            0.8,
            f"RS: {int(mid_val)}",
            transform=ax.transAxes,
            horizontalalignment="right",
            fontsize=fontsize_dict.get("legend", 10),
        )
    ok_speed = (tav_df.rs > b_s) & (tav_df.rs < b_e)

    if not np.any(ok_speed):
        print(f"No trials found for RS {b_s:.1f}-{b_e:.1f}")
        return
    of = tav_df[ok_speed].of.values
    dff = tav_df[ok_speed].dff.values

    # Remove NaNs
    valid = ~(np.isnan(of) | np.isnan(dff))
    of = of[valid]
    dff = dff[valid]

    if len(of) == 0:
        return
    if plot_trials:
        # 2. Scatter raw data
        ax.scatter(
            of, dff, color="k", s=scatter_size, alpha=0.3, zorder=5, clip_on=False
        )

    # 3. Perform 1D Gaussian Fit in log-space
    # Initial guess: centre on the bin with the highest mean response
    of_bins = np.asarray(of_bins, dtype=float)
    m, _, _ = scipy.stats.binned_statistic(of, dff, bins=of_bins, statistic="mean")
    # geometric midpoint: bins are log-spaced and drawn on a log x-axis
    bin_mid = np.sqrt(of_bins[:-1] * of_bins[1:])

    def p0_func():
        # of_bins may start at 0 (catch-all bin), which has no log midpoint
        m_pos = np.where(bin_mid > 0, m, np.nan)
        best_of = bin_mid[np.nanargmax(m_pos)]
        return np.array(
            [
                np.random.normal(),  # log_amplitude
                np.log(best_of),  # x0 (log OF)
                np.random.normal(),  # log_sigma_x2
                np.random.normal(),  # offset
            ]
        )

    popt, rsq = common_utils.iterate_fit(
        func=gaussian_func_,
        X=np.log(of),
        y=dff,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        niter=niter,
        p0_func=p0_func,
    )

    # Plot the fit
    x_fine = np.linspace(np.log(of_min), np.log(of_max), 300)
    resp_pred = gaussian_func_(x_fine, *popt)
    ax.plot(
        np.exp(x_fine),
        resp_pred,
        color=color,
        lw=linewidth,
        label=f"pref OF={np.exp(popt[1]):.2f}, R²={rsq:.2f}",
    )
    # 4. Calculate Binned Stats & Bootstrap CI
    bin_ids = np.digitize(of, of_bins) - 1

    ci_low = []
    ci_high = []
    for i in range(len(bin_mid)):
        samples = dff[bin_ids == i]
        if len(samples) > 1:  # Bootstrap requires at least 2 samples
            low, high = common_utils.get_bootstrap_ci(samples, n_bootstraps=1000)
            ci_low.append(low[0])
            ci_high.append(high[0])
        else:
            # Fallback for bins with 0 or 1 samples
            val = samples[0] if len(samples) == 1 else np.nan
            ci_low.append(val)
            ci_high.append(val)

    # 5. Plot Errorbars
    err = [m - ci_low, ci_high - m]
    ax.errorbar(
        bin_mid,
        m,
        yerr=err,
        fmt="o",
        color=color,
        label="Binned mean & 95% CI",
        capsize=capsize,
        markersize=markersize,
        zorder=10,
    )
    # 6. Axis Styling
    ax.set_xscale("log")
    ax.set_ylabel(r"$\Delta$F/F", fontsize=fontsize_dict["label"])
    ax.axhline(0, color="grey", lw=0.5, zorder=-10)
    ax.set_xlim(of_bins[0], of_bins[-1])

    # 7. Mark the mean of the gaussian fit. y is in axes coordinates so the marker
    # stays at the top of the panel whatever ylim the caller sets afterwards.
    if plot_fit_mean:
        ax.plot(
            np.exp(popt[1]),
            1,
            marker="v",
            color=color,
            markersize=markersize if markersize is not None else 8,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=11,
        )
