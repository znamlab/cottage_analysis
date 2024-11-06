import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import scipy
import seaborn as sns
import flexiznam as flz
from scipy.stats import pearsonr
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    common_utils,
    size_control,
    fit_gaussian_blob,
)
from cottage_analysis.plotting import plotting_utils
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import rf_plots
from cottage_analysis.analysis import roi_location, common_utils


def plot_raster_all_depths(
    trials_df,
    roi,
    is_closed_loop,
    corridor_length=6,
    blank_length=0,
    nbins=60,
    vmax=1,
    plot=True,
    cbar_width=0.05,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    position=(0, 0, 1, 1),
):
    """Raster plot for neuronal activity for each depth for one ROI.

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        corridor_length (float, optional): length of the corridor in meters. Defaults to 6.
        blank_length (float, optional): length of the blank period to be plotted at each end of the corridor. Defaults to 0.
        nbins (int, optional): number of bins to bin the activity. Defaults to 60.
        vmax (int, optional): vmax to plot the heatmap. Defaults to 1.
        plot (bool, optional): whether to plot the raster or just to get the values. Defaults to True.
        cbar_width (float, optional): width of the colorbar. Defaults to 0.05.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
        position (tuple, optional): position of the plot, (x, y, width, height). Defaults to (0, 0, 1, 1).

    """
    # choose the trials with closed or open loop to visualize
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped = trials_df.groupby(by="depth")
    trial_number = len(trials_df) // len(depth_list)

    # bin dff according to distance travelled for each trial
    dffs_binned = np.zeros((len(depth_list), trial_number, nbins))
    min_distance = -blank_length
    max_distance = corridor_length + blank_length
    bins = np.linspace(
        start=min_distance, stop=max_distance, num=nbins + 1, endpoint=True
    )
    for idepth, depth in enumerate(depth_list):
        for itrial in np.arange(trial_number):
            dff = np.concatenate(
                (
                    grouped.get_group(depth).dff_blank_pre.values[itrial][:, roi],
                    grouped.get_group(depth).dff_stim.values[itrial][:, roi],
                    grouped.get_group(depth).dff_blank.values[itrial][:, roi],
                )
            )
            pos_arr = np.concatenate(
                (
                    grouped.get_group(depth).mouse_z_harp_blank_pre.values[itrial],
                    grouped.get_group(depth).mouse_z_harp_stim.values[itrial],
                    grouped.get_group(depth).mouse_z_harp_blank.values[itrial],
                )
            )
            pos_arr -= grouped.get_group(depth).mouse_z_harp_stim.values[itrial][0]
            bin_means, _, _ = scipy.stats.binned_statistic(
                x=pos_arr,
                values=dff,
                statistic="mean",
                bins=bins,
            )
            dffs_binned[idepth, itrial, :] = bin_means

    # colormap
    WhRdcmap = plotting_utils.generate_cmap(cmap_name="WhRd")

    # plot all depths as one heatmap
    if plot:
        plot_x, plot_y, plot_width, plot_height = position
        plot_prop = 0.9
        each_plot_width = (plot_width - cbar_width) / len(depth_list)
        ax = plt.gcf().add_axes([plot_x, plot_y, plot_width, plot_height])
        im = ax.imshow(
            np.swapaxes(dffs_binned, 0, 1).reshape(-1, nbins * len(depth_list)),
            aspect="auto",
            cmap=WhRdcmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        # Plot vertical lines to separate different depths
        ndepths = len(depth_list)
        for i in range(ndepths - 1):
            ax.axvline((i + 1) * nbins, color="k", linewidth=0.5, linestyle="dotted")
        # Change y ticks to trial number
        ax.set_ylabel("Trial number", fontsize=fontsize_dict["label"], labelpad=-5)
        ax.set_yticks([-0.5, dffs_binned.shape[1] - 0.5])
        ax.set_yticklabels([1, dffs_binned.shape[1]])
        ax.tick_params(axis="y", labelsize=fontsize_dict["tick"])
        # Change xticks positions to the middle of current ticks and show depth at the tick position
        blank_prop = blank_length / (corridor_length + blank_length * 2)
        xticks = np.linspace(nbins / 2, nbins * (ndepths - 1 / 2), ndepths)
        ax.set_xticks(xticks)
        ax.set_xticklabels((np.array(depth_list) * 100).astype("int"))
        ax.set_xlabel("Virtual depth (cm)", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="x", labelsize=fontsize_dict["tick"], rotation=0)

        # # for aligning with the scalebar
        # ax.vlines(blank_prop*nbins, 0, dffs_binned.shape[1], color="k", linestyle="--", linewidth=0.5)
        # ax.vlines(nbins-blank_prop*nbins, 0, dffs_binned.shape[1], color="k", linestyle="--", linewidth=0.5)

        ax2 = plt.gcf().add_axes(
            [
                plot_x + plot_width + 0.01,
                plot_y,
                cbar_width * 0.8,
                plot_height / 3,
            ]
        )
        # set colorbar
        cbar = plt.colorbar(im, cax=ax2, label="\u0394F/F")
        ax2.tick_params(labelsize=fontsize_dict["tick"])
        cbar.set_ticks([0, vmax])
        ax2.set_ylabel("\u0394F/F", fontsize=fontsize_dict["legend"])

    return dffs_binned, ax


def plot_depth_tuning_curve(
    neurons_df,
    trials_df,
    roi,
    param="depth",
    use_col="depth_tuning_popt_closedloop",
    min_sigma=0.5,
    folds=None,
    rs_thr=None,
    rs_thr_max=None,
    still_only=False,
    still_time=0,
    frame_rate=15,
    plot_fit=True,
    plot_smooth=False,
    linewidth=3,
    linecolor="k",
    markersize=5,
    markeredgecolor="k",
    closed_loop=1,
    label=None,
    ylim=None,
    ylim_precision_base=1,
    ylim_precision=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """
    Plot depth tuning curve for one neuron.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number.
        param (str, optional): parameter to plot ("depth" or "size"). Defaults to "depth".
        use_col (str, optional): column name to use for fitting. Defaults to "depth_tuning_popt_closedloop".
        min_sigma (float, optional): minimum sigma for gaussian fit. Defaults to 0.5.
        folds (int, optional): number of folds for cross-validation. Defaults to None. If folds is not None, it will plot depth tuning curve for each fold.
        rs_thr (float, optional): Min threshold of running speed for imaging frames. Defaults to None. (m/s)
        rs_thr_max (float, optional): Max threshold of running speed for imaging frames. Defaults to None. (m/s)
        still_only (bool, optional): Whether to use only frames when the mouse stay still for x frames. Defaults to False.
        still_time (int, optional): Number of seconds before a certain frame when the mouse stay still. Defaults to 0.
        frame_rate (int, optional): Imaging frame rate. Defaults to 15.
        plot_fit (bool, optional): Whether to plot fitted tuning curve or not. Defaults to True.
        plot_smooth (bool, optional): Whether to plot smoothed tuning curve or not. Defaults to False.
        linewidth (int, optional): linewidth. Defaults to 3.
        linecolor (str, optional): linecolor of true data. Defaults to "k".
        markersize (int, optional): markersize. Defaults to 5.
        markeredgecolor (str, optional): markeredgecolor. Defaults to "k".
        closed_loop (int, optional): whether it's closedloop or openloop. Defaults to 1.
        label (str, optional): label for the plot. Defaults to None.
        ylim (list, optional): y-axis limits. Defaults to None.
        ylim_precision_base (int, optional): base for setting y-axis limits. Defaults to 1.
        ylim_precision (int, optional): precision for setting y-axis limits. Defaults to 1.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """

    # Load average activity and confidence interval for this roi
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    if param == "depth":
        param_list = np.array(find_depth_neurons.find_depth_list(trials_df))
    elif param == "size":
        trials_df = size_control.get_physical_size(
            trials_df, use_cols=["size", "depth"], k=1
        )
        param_list = np.sort(trials_df["physical_size"].unique())
    log_param_list = np.log(param_list)
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
        trials_df=trials_df,
        rs_thr=rs_thr,
        rs_thr_max=rs_thr_max,
        still_only=still_only,
        still_time=still_time,
        frame_rate=frame_rate,
        closed_loop=closed_loop,
        param=param,
    )[:, :, roi]
    CI_low, CI_high = common_utils.get_bootstrap_ci(mean_dff_arr)
    mean_arr = np.nanmean(mean_dff_arr, axis=1)
    ax = plt.gca()
    ax.errorbar(
        log_param_list,
        mean_arr,
        yerr=(mean_arr - CI_low, CI_high - mean_arr),
        fmt=".",
        color=linecolor,
        markeredgecolor=markeredgecolor,
        markeredgewidth=0.3,
        markerfacecolor=linecolor,
        ls="none",
        fillstyle="full",
        linewidth=linewidth,
        markersize=markersize,
    )

    if plot_smooth:
        # calculate a tuning curve using gaussian smoothing over depths
        xs = np.linspace(log_param_list[0], log_param_list[-1], num=100)
        sd = 0.75
        ys = np.zeros((len(xs)))
        for i, x in enumerate(xs):
            weights = np.exp(-((log_param_list - x) ** 2) / (2 * sd**2))
            ys[i] = np.sum(weights * mean_arr) / np.sum(weights)
        plt.plot(
            xs,
            ys,
            color=linecolor,
            label=label,
            linewidth=linewidth,
        )
    # Load gaussian fit params for this roi
    if plot_fit:
        x = np.geomspace(param_list[0], param_list[-1], num=100)
        if folds is not None:
            for fold in np.arange(folds):
                [a, x0, log_sigma, b] = neurons_df.loc[roi, use_col][fold]
                gaussian_arr = fit_gaussian_blob.gaussian_1d(
                    np.log(x), a, x0, log_sigma, b, min_sigma
                )
                plt.plot(
                    np.log(x),
                    gaussian_arr,
                    color=linecolor,
                    linewidth=linewidth,
                    label=label,
                )
        else:
            [a, x0, log_sigma, b] = neurons_df.loc[roi, use_col]
            gaussian_arr = fit_gaussian_blob.gaussian_1d(
                np.log(x), a, x0, log_sigma, b, min_sigma
            )
            plt.plot(
                np.log(x),
                gaussian_arr,
                color=linecolor,
                linewidth=linewidth,
                label=label,
            )
    if ylim is None:
        ylim = [
            plt.gca().get_ylim()[0],
            common_utils.ceil(np.max(CI_high), ylim_precision_base, ylim_precision),
        ]
        plt.ylim(ylim)
        plt.yticks(
            [
                0,
                common_utils.ceil(np.max(CI_high), ylim_precision_base, ylim_precision),
            ],
            fontsize=fontsize_dict["tick"],
        )
    else:
        plt.ylim([ylim[0], ylim[1]])
        plt.yticks([np.round(ylim[0], 1), ylim[1]], fontsize=fontsize_dict["tick"])

    if param == "depth":
        plt.xticks(
            log_param_list,
            (np.round(np.array(param_list) * 100)).astype("int"),
        )
        plt.xlabel(f"Virtual depth (cm)", fontsize=fontsize_dict["label"])
    elif param == "size":
        plt.xticks(
            log_param_list,
            np.round(np.array(param_list) * 0.87 / 10 * 20, 1),
        )
        plt.xlabel(f"Virtual radius (cm)", fontsize=fontsize_dict["label"])
    sns.despine(ax=plt.gca(), offset=3, trim=True)
    plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"], labelpad=-5)
    plt.xticks(
        rotation=45,
    )
    plt.gca().tick_params(axis="both", labelsize=fontsize_dict["tick"])


def plot_running_stationary_depth_tuning(
    roi,
    roi_num,
    i,
    neurons_df,
    trials_df,
    ax,
    depth_tuning_kwargs,
    fontsize_dict,
    fov_ax=None,
    ops=None,
    stat=None,
    legend_loc="upper right",
    text_pos="upper_left",
):
    """Plot depth tuning curve for one neuron when the mouse is running vs stationary.

    Args:
        roi (int): ROI number.
        roi_num (int): number of total ROIs plotted in a set of graphs.
        i (int): index of the current ROI.
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        ax (plt.Axes): axes to plot the depth tuning curve.
        depth_tuning_kwargs (dict): dictionary of kwargs for plotting depth tuning curve.
        fontsize_dict (dict): dictionary of fontsize for title, label and tick.
        fov_ax (plt.Axes, optional): axes to plot the FOV. Defaults to None.
        ops (dict, optional): ops from suite2p. Defaults to None.
        stat (dict, optional): stat from suite2p. Defaults to None.
        legend_loc (str, optional): location of the legend. Defaults to "upper right".
        text_pos (str, optional): position of the text. Defaults to "upper_left".
    """
    ylims = []
    for (
        rs_thr,
        rs_thr_max,
        still_only,
        still_time,
        i_running,
        linecolor,
        label,
        use_col,
    ) in zip(
        [0.05, None],
        [None, 0.05],
        [0, 1],
        [0, 1],
        [0, 1],
        ["royalblue", "gray"],
        ["Running", "Stationary"],
        [
            "depth_tuning_popt_closedloop_running",
            "depth_tuning_popt_closedloop_notrunning",
        ],
    ):
        # calculate ylim
        mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
            trials_df=trials_df,
            rs_thr=rs_thr,
            rs_thr_max=rs_thr_max,
            still_only=still_only,
            still_time=still_time,
            frame_rate=15,
            closed_loop=1,
            param="depth",
        )[:, :, roi]
        CI_low, CI_high = common_utils.get_bootstrap_ci(mean_dff_arr)
        ylim = (np.nanmin(CI_low), np.nanmax(CI_high))
        ylims.append(ylim)

    ylim = (
        min([i[0] for i in ylims]),
        common_utils.ceil(max([i[1] for i in ylims]), 1),
    )
    # plot
    for (
        rs_thr,
        rs_thr_max,
        still_only,
        still_time,
        i_running,
        linecolor,
        label,
        use_col,
    ) in zip(
        [0.05, None],
        [None, 0.05],
        [0, 1],
        [0, 1],
        [0, 1],
        ["royalblue", "gray"],
        ["Running", "Stationary"],
        [
            "depth_tuning_popt_closedloop_running",
            "depth_tuning_popt_closedloop_notrunning",
        ],
    ):
        depth_tuning_running_kwargs = depth_tuning_kwargs.copy()
        depth_tuning_running_kwargs["rs_thr"] = rs_thr
        depth_tuning_running_kwargs["rs_thr_max"] = rs_thr_max
        depth_tuning_running_kwargs["still_only"] = still_only
        depth_tuning_running_kwargs["still_time"] = still_time
        depth_tuning_running_kwargs["linecolor"] = linecolor
        depth_tuning_running_kwargs["use_col"] = use_col
        plot_depth_tuning_curve(
            neurons_df=neurons_df,
            trials_df=trials_df,
            roi=roi,
            **depth_tuning_running_kwargs,
            ylim=ylim,
            label=label,
        )

        if i != 1:
            plt.ylabel("")
        if (i % 3 != 2) and roi_num != 1:
            plt.xlabel("")
            ax.set_xticklabels([])
        if i_running == 0:
            if text_pos == "upper_left":
                x_label = plt.xlim()[0] + 0.05 * (plt.xlim()[1] - plt.xlim()[0])
            elif text_pos == "upper_right":
                x_label = plt.xlim()[1] - 0.3 * (plt.xlim()[1] - plt.xlim()[0])
            plt.text(
                x_label,
                ylim[1],
                f"Cell {roi_num}",
                fontsize=fontsize_dict["legend"],
            )
            if fov_ax:
                fov_ax.annotate(
                    f"{roi_num}",
                    (
                        ops["meanImg"].shape[0] - stat[roi]["med"][1],
                        stat[roi]["med"][0],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="w",
                    fontsize=fontsize_dict["label"],
                    arrowprops=dict(facecolor="w", edgecolor="w", arrowstyle="->"),
                )
    if i == 0:
        plt.legend(
            loc=legend_loc,
            fontsize=fontsize_dict["legend"],
            framealpha=1,
            borderpad=0,
            frameon=False,
            handlelength=0.5,
        )


def get_PSTH(
    roi,
    psth=[],
    depth_list=[],
    is_closed_loop=1,
    trials_df=None,
    use_col="dff",
    rs_thr_min=None,  # m/s
    rs_thr_max=None,  # m/s
    still_only=False,
    still_time=1,  # s
    max_distance=6,  # m
    min_distance=0,
    nbins=20,
    bins=[],  # if bins are provided, nbins is ignored
    frame_rate=15,
    compute_ci=True,
):
    # confidence interval z calculation
    ci_range = 0.95

    if len(bins) == 0:
        if len(depth_list) > 0:
            all_ci = np.zeros((2, len(depth_list) + 1, nbins))
        bins = np.linspace(
            start=min_distance, stop=max_distance, num=nbins + 1, endpoint=True
        )
        bin_centers = (bins[1:] + bins[:-1]) / 2
    else:
        nbins = len(bins) - 1
        if len(depth_list) > 0:
            all_ci = np.zeros((2, len(depth_list) + 1, len(bins) - 1))
        bin_centers = (bins[1:] + bins[:-1]) / 2
    if len(psth) == 0:
        # choose the trials with closed or open loop to visualize
        trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

        depth_list = find_depth_neurons.find_depth_list(trials_df)
        grouped = trials_df.groupby(by="depth")
        # trial_number = len(trials_df) // len(depth_list)

        # bin dff according to distance travelled for each trial
        all_means = np.zeros((len(depth_list) + 1, nbins))
        all_ci = np.zeros((2, len(depth_list) + 1, nbins))

        all_trial_numbers = []
        for idepth, depth in enumerate(depth_list):
            all_trial_numbers.append(len(grouped.get_group(depth)))
        trial_number = np.min(all_trial_numbers)

        for idepth, depth in enumerate(depth_list):
            all_dff = []
            for itrial in np.arange(trial_number):
                # concatenate dff_blank_pre, dff, and dff_blank
                if use_col == "dff":
                    dff = np.concatenate(
                        (
                            grouped.get_group(depth)[f"{use_col}_blank_pre"].values[
                                itrial
                            ][:, roi],
                            grouped.get_group(depth)[f"{use_col}_stim"].values[itrial][
                                :, roi
                            ],
                            grouped.get_group(depth)[f"{use_col}_blank"].values[itrial][
                                :, roi
                            ],
                        )
                    )
                else:
                    dff = np.concatenate(
                        (
                            grouped.get_group(depth)[f"{use_col}_blank_pre"].values[
                                itrial
                            ],
                            grouped.get_group(depth)[f"{use_col}_stim"].values[itrial],
                            grouped.get_group(depth)[f"{use_col}_blank"].values[itrial],
                        )
                    )
                rs_arr = np.concatenate(
                    (
                        grouped.get_group(depth).RS_blank_pre.values[itrial],
                        grouped.get_group(depth).RS_stim.values[itrial],
                        grouped.get_group(depth).RS_blank.values[itrial],
                    )
                )
                pos_arr = np.concatenate(
                    (
                        grouped.get_group(depth).mouse_z_harp_blank_pre.values[itrial],
                        grouped.get_group(depth).mouse_z_harp_stim.values[itrial],
                        grouped.get_group(depth).mouse_z_harp_blank.values[itrial],
                    )
                )
                pos_arr -= grouped.get_group(depth).mouse_z_harp_stim.values[itrial][0]

                take_idx = apply_rs_threshold(
                    rs_arr, rs_thr_min, rs_thr_max, still_only, still_time, frame_rate
                )

                dff = dff[take_idx]
                # bin dff according to distance travelled
                dff, _, _ = scipy.stats.binned_statistic(
                    x=pos_arr,
                    values=dff,
                    statistic="mean",
                    bins=bins,
                )
                all_dff.append(dff)
            all_means[idepth, :] = np.nanmean(all_dff, axis=0)
            if compute_ci:
                all_ci[0, idepth, :], all_ci[1, idepth, :] = (
                    common_utils.get_bootstrap_ci(
                        np.array(all_dff).T, sig_level=1 - ci_range
                    )
                )
    else:
        all_dff = psth
        all_means = np.nanmean(all_dff, axis=0)
        for idepth, depth in enumerate(depth_list):
            if compute_ci:
                all_ci[0, idepth, :], all_ci[1, idepth, :] = (
                    common_utils.get_bootstrap_ci(
                        np.array(all_dff[:, idepth, :]).T, sig_level=1 - ci_range
                    )
                )

    return all_means, all_ci, bin_centers


def apply_rs_threshold(
    rs_arr,
    rs_thr_min,
    rs_thr_max,
    still_only,
    still_time,
    frame_rate,
):
    """Apply running speed threshold to select frames.

    Args:
        rs_arr (np.array): running speed array.
        rs_thr_min (float): min threshold of running speed for imaging frames.
        rs_thr_max (float): max threshold of running speed for imaging frames.
        still_only (bool): whether to use only frames when the mouse stay still for x frames.
        still_time (int): number of seconds before a certain frame when the mouse stay still.
        frame_rate (int): imaging frame rate.
    """
    # threshold running speed according to rs_thr
    if not still_only:  # take running frames
        if (rs_thr_min is None) and (rs_thr_max is None):  # take all frames
            take_idx = np.arange(len(rs_arr))
        else:
            if rs_thr_max is None:  # take frames with running speed > rs_thr
                take_idx = rs_arr > rs_thr_min
            elif rs_thr_min is None:  # take frames with running speed < rs_thr
                take_idx = rs_arr < rs_thr_max
            else:  # take frames with running speed between rs_thr_min and rs_thr_max
                take_idx = (rs_arr > rs_thr_min) & (rs_arr < rs_thr_max)
    else:  # take still frames
        if rs_thr_max is None:  # use not running data but didn't set rs_thr
            print(
                "ERROR: calculating under not_running condition without rs_thr_max to determine max speed"
            )
        else:  # take frames with running speed < rs_thr for x seconds
            take_idx = common_utils.find_thresh_sequence(
                array=rs_arr,
                threshold_max=rs_thr_max,
                length=int(still_time * frame_rate),
                shift=int(still_time * frame_rate),
            )
    return take_idx


def plot_PSTH(
    roi,
    is_closed_loop,
    trials_df=None,
    psth=[],
    depth_list=[],
    use_col="dff",
    corridor_length=6,
    blank_length=0,
    nbins=20,
    bins=[],  # if bins are provided, nbins is ignored
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    frame_rate=15,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    linewidth=3,
    legend_on=False,
    legend_loc="lower right",
    legend_bbox_to_anchor=(1.4, -0.6),
    show_ci=True,
    ylim=(None, None),
):
    """PSTH of a neuron for each depth and blank period.

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        max_distance (int, optional): max distance for each trial in meters. Defaults to 6.
        nbins (int, optional): number of bins to bin the activity. Defaults to 20.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
    """
    max_distance = corridor_length + blank_length
    min_distance = -blank_length
    if trials_df is not None:
        depth_list = find_depth_neurons.find_depth_list(trials_df)

    all_means, all_ci, bin_centers = get_PSTH(
        trials_df=trials_df,
        psth=psth,
        depth_list=depth_list,
        roi=roi,
        is_closed_loop=is_closed_loop,
        use_col=use_col,
        rs_thr_min=rs_thr_min,
        rs_thr_max=rs_thr_max,
        still_only=still_only,
        still_time=still_time,
        min_distance=min_distance,
        max_distance=max_distance,
        nbins=nbins,
        bins=bins,
        frame_rate=frame_rate,
    )

    if use_col == "RS":
        all_means = all_means * 100  # convert m/s to cm/s
    for idepth, depth in enumerate(depth_list):
        linecolor = plotting_utils.get_color(
            depth_list[idepth],
            value_min=np.min(depth_list),
            value_max=np.max(depth_list),
            log=True,
            cmap=cm.cool.reversed(),
        )
        plt.plot(
            bin_centers,
            all_means[idepth, :],
            color=linecolor,
            label=f"{(np.round(depth_list[idepth] * 100)).astype('int')} cm",
            linewidth=linewidth,
        )
        if show_ci:
            plt.fill_between(
                bin_centers,
                y1=all_ci[0, idepth, :],
                y2=all_ci[1, idepth, :],
                color=linecolor,
                alpha=0.3,
                edgecolor=None,
                rasterized=False,
            )

    plt.xlabel("Corridor position (m)", fontsize=fontsize_dict["label"])
    plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
    plt.xticks(
        [0, corridor_length],
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(fontsize=fontsize_dict["tick"])
    if (ylim[0] is None) and (ylim[1] is None):
        ylim = plt.gca().get_ylim()
        ylim = [ylim[0], plt_common_utils.ceil(ylim[1], 1)]
    elif ylim[0] is not None:
        if ylim[1] is None:
            ylim = (ylim[0], plt_common_utils.ceil(plt.gca().get_ylim()[1], 1))
            plt.ylim(ylim)
        else:
            ylim = ylim
        plt.ylim(ylim)
    elif (ylim[1] is not None) and (ylim[0] is None):
        ylim = (plt.gca().get_ylim()[0], ylim[1])
        plt.ylim(ylim)
    plt.yticks([ylim[0], ylim[1]], fontsize=fontsize_dict["tick"])
    plt.plot([0, 0], ylim, "k", linestyle="dotted", linewidth=0.5, label="_nolegend_")
    plt.plot(
        [corridor_length, corridor_length],
        ylim,
        "k",
        linestyle="dotted",
        linewidth=0.5,
        label="_nolegend_",
    )

    if legend_on:
        plt.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=fontsize_dict["legend"],
            frameon=False,
            handlelength=1,
        )
    plotting_utils.despine()


def plot_preferred_depth_hist(
    results_df,
    use_col="preferred_depth_closedloop",
    nbins=50,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """Plot histogram of preferred depth.

    Args:
        results_df (pd.DataFrame): dataframe with analyzed info of all rois.
        use_col (str, optional): column name to use for plotting. Defaults to "preferred_depth_closedloop".
        nbins (int, optional): number of bins for histogram. Defaults to 50.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """
    results_df = results_df[results_df["iscell"] == 1].copy()
    # convert to cm
    results_df[use_col] = results_df[use_col].apply(lambda x: np.log(x * 100))
    min_depth = np.nanmin(results_df[use_col])
    max_depth = np.nanmax(results_df[use_col])
    depth_bins = np.linspace(
        min_depth,
        max_depth,
        num=nbins,
    )
    tol = 1e-4
    # set rows where use_col = min or max to -inf and inf
    results_df[use_col] = results_df[use_col].apply(
        lambda x: -np.inf if x < min_depth + tol else x
    )
    results_df[use_col] = results_df[use_col].apply(
        lambda x: np.inf if x > max_depth - tol else x
    )

    n, _, _ = plt.hist(
        results_df[use_col],
        bins=depth_bins,
        weights=np.ones(len(results_df)) / len(results_df),
        color="cornflowerblue",
        edgecolor="royalblue",
    )
    # plot proportion of rows with -inf and inf values as separate bars at min_depth/2 and max_depth*2
    plt.bar(
        min_depth - 1,
        np.sum(results_df[use_col] == -np.inf) / len(results_df),
        color="cornflowerblue",
        edgecolor="royalblue",
        width=(max_depth - min_depth) / nbins,
    )
    plt.bar(
        max_depth + 1,
        np.sum(results_df[use_col] == np.inf) / len(results_df),
        color="cornflowerblue",
        edgecolor="royalblue",
        width=(max_depth - min_depth) / nbins,
    )

    ax = plt.gca()
    ax.set_ylabel("Proportion of neurons", fontsize=fontsize_dict["label"])
    ax.set_xlabel("Preferred virtual depth (cm)", fontsize=fontsize_dict["label"])
    tick_pos = [10, 100, 1000]
    ax.set_xticks(
        np.log(
            np.concatenate(
                (
                    np.arange(2, 9, 1),
                    np.arange(2, 9, 1) * 10,
                    np.arange(2, 9, 1) * 100,
                    [
                        2000,
                    ],
                )
            )
        ),
        minor=True,
    )
    plt.xticks(
        np.concatenate([[min_depth - 1], np.log(tick_pos), [max_depth + 1]]),
        labels=np.concatenate([["N.P."], tick_pos, ["F.P."]]),
        fontsize=fontsize_dict["tick"],
    )

    plt.ylim([0, np.round(np.max(n), 2)])
    plt.yticks([0, np.round(np.max(n), 2)], fontsize=fontsize_dict["tick"])
    plotting_utils.despine()


def plot_psth_raster(
    results_df,
    depth_list,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    vmax=2,
):
    """Plot PSTH raster for all neurons.

    Args:
        results_df (pd.DataFrame): dataframe with analyzed info of all rois from all sessions.
        depth_list (list): list of depths.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
        vmax (int, optional): maximum value for the colorbar. Defaults to 2.
    """
    psths = np.stack(results_df["psth_crossval"])[:, :-1, 10:-10]  # exclude blank
    ndepths = psths.shape[1]
    nbins = psths.shape[2]
    # Sort neurons by preferred depth
    preferred_depths = results_df["preferred_depth_closedloop_crossval"].values
    psths = psths[preferred_depths.argsort()]
    psths = psths.reshape(psths.shape[0], -1)
    # zscore each row
    normed_psth = (psths - np.nanmean(psths, axis=1)[:, np.newaxis]) / (
        np.nanstd(psths, axis=1)[:, np.newaxis]
    )
    # Plot PSTHs
    ax = plt.gca()
    im = ax.imshow(
        normed_psth,
        aspect="auto",
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax,
    )
    # Plot vertical lines to separate different depths
    for i in range(ndepths):
        ax.axvline((i + 1) * nbins, color="k", linewidth=0.5, linestyle="dotted")
    # Change xticks positions to the middle of current ticks and show depth at the tick position
    xticks = (np.arange(ndepths) + 0.5) * nbins
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(depth_list).astype("int"))
    ax.set_xlabel("Virtual depth (cm)", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="x", labelsize=fontsize_dict["tick"], rotation=60)
    ax.set_ylabel("Neuron number", fontsize=fontsize_dict["label"], labelpad=-5)
    ax.set_yticks([1, len(results_df)])
    ax.tick_params(axis="y", labelsize=fontsize_dict["tick"])
    ax.set_xlim([0, ndepths * nbins])

    # # for aligning with the scalebar
    # ax.vlines(1/4*60-10, -10, 9000, color="k", linestyle="--", linewidth=0.5)
    # ax.vlines(60-1/4*60-10, -10, 9000, color="k", linestyle="--", linewidth=0.5)

    ax_pos = ax.get_position()
    ax2 = plt.gcf().add_axes(
        [
            ax_pos.x1 + ax_pos.width * 0.05,
            ax_pos.y0,
            0.01,
            ax_pos.height / 2,
        ]
    )
    cbar = plt.colorbar(mappable=im, cax=ax2)
    cbar.set_label("Z-score", fontsize=fontsize_dict["legend"])
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"])


def plot_depth_neuron_perc_hist(
    results_df,
    bins=50,
    ylim=None,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """Plot histogram of proportion of depth-tuned neurons for each session.

    Args:
        results_df (pd.DataFrame): dataframe with analyzed info of all rois from all sessions.
        bins (int, optional): number of bins for histogram. Defaults to 50.
        ylim (tuple, optional): y-axis limits. Defaults to None.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """
    session_prop = results_df.groupby("session").agg({"depth_tuned": "mean"})
    plt.hist(
        session_prop["depth_tuned"],
        bins=bins,
        color="cornflowerblue",
        edgecolor="royalblue",
    )
    ax = plt.gca()
    xlim = ax.get_xlim()
    ax.set_xlim([0, xlim[1]])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(
        "Proportion of \ndepth-tuned neurons", fontsize=fontsize_dict["label"]
    )
    ax.set_ylabel("Number of sessions", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])
    # plot median proportion as a triangle along the top of the histogram
    median_prop = np.median(session_prop["depth_tuned"])
    print("Median proportion of depth-tuned neurons:", median_prop)
    print(
        "Range of proportions of depth-tuned neurons:",
        np.min(session_prop["depth_tuned"]),
        "to",
        np.max(session_prop["depth_tuned"]),
    )
    print("Number of sessions:", len(session_prop))
    ax.plot(
        median_prop,
        ax.get_ylim()[1] * 0.95,
        marker="v",
        markersize=5,
        markerfacecolor="cornflowerblue",
        markeredgecolor="royalblue",
    )
    plotting_utils.despine()


def plot_example_fov(
    neurons_df,
    stat,
    ops,
    ndepths=8,
    col="preferred_depth",
    cmap=cm.cool.reversed(),
    background_color=np.array([0.133, 0.545, 0.133]),
    n_std=6,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    fov_width=572.867,
):
    rf_plots.find_rf_centers(
        neurons_df,
        ndepths=ndepths,
        frame_shape=(16, 24),
        is_closed_loop=1,
        resolution=5,
    )
    roi_location.find_roi_centers(neurons_df, stat)
    if col == "preferred_depth":
        select_neurons = neurons_df[
            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
            & (neurons_df["iscell"] == 1)
        ]
        null_neurons = neurons_df[
            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] >= 0.05)
            & (neurons_df["iscell"] == 1)
        ]
    else:
        coef = np.stack(neurons_df[f"rf_coef_closedloop"].values)
        coef_ipsi = np.stack(neurons_df[f"rf_coef_ipsi_closedloop"].values)
        sig, _ = spheres.find_sig_rfs(
            np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
            np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
            n_std=n_std,
        )
        select_neurons = neurons_df[(sig == 1) & (neurons_df["iscell"] == 1)]
        null_neurons = neurons_df[(sig == 0) & (neurons_df["iscell"] == 1)]
    # Find neuronal masks and assign on the background image
    im = (
        np.ones((ops["Ly"], ops["Lx"], 3)) * background_color[np.newaxis, np.newaxis, :]
    )
    azi_min = select_neurons["rf_azi"].quantile(0.1)
    azi_max = select_neurons["rf_azi"].quantile(0.9)
    ele_min = select_neurons["rf_ele"].quantile(0.1)
    ele_max = select_neurons["rf_ele"].quantile(0.9)
    for i, n in null_neurons.iterrows():
        ypix = stat[n.roi]["ypix"][~stat[n.roi]["overlap"]]
        xpix = stat[n.roi]["xpix"][~stat[n.roi]["overlap"]]
        if len(xpix) > 0 and len(ypix) > 0:
            im[ypix, xpix, :] = 0.3
    for _, n in select_neurons.iterrows():
        ypix = stat[n.roi]["ypix"][~stat[n.roi]["overlap"]]
        xpix = stat[n.roi]["xpix"][~stat[n.roi]["overlap"]]
        if col == "preferred_depth_closedloop":
            rgba_color = plotting_utils.get_color(
                n[col],
                0.02,
                20,
                log=True,
                cmap=cmap,
            )
        elif col == "rf_azi":
            rgba_color = plotting_utils.get_color(
                n[col],
                azi_min,
                azi_max,
                log=False,
                cmap=cmap,
            )
        elif col == "rf_ele":
            rgba_color = plotting_utils.get_color(
                n[col],
                ele_min,
                ele_max,
                log=False,
                cmap=cmap,
            )
        im[ypix, xpix, :] = rgba_color[np.newaxis, :3]
    # Plot spatial distribution
    plt.imshow(im)
    if col != "preferred_depth_closedloop":
        # find the gradient of col w.r.t. center_x and center_y
        slope_x = scipy.stats.linregress(
            x=select_neurons["center_x"], y=select_neurons[col]
        ).slope
        slope_y = scipy.stats.linregress(
            x=select_neurons["center_y"], y=select_neurons[col]
        ).slope
        norm = np.linalg.norm(np.array([slope_x, slope_y]))
        slope_x /= norm
        slope_y /= norm
        arrow_length = 100
        # draw an arrow in the direction of the gradient
        plt.arrow(
            x=im.shape[1] / 2 - slope_x * arrow_length / 2,
            y=im.shape[0] / 2 - slope_y * arrow_length / 2,
            dx=slope_x * arrow_length,
            dy=slope_y * arrow_length,
            color="white",
            width=2,
            head_width=50,
        )
        print(f"{col}: slope x {slope_x}, slope y {slope_y}")
    plt.axis("off")
    # Add a colorbar for the dummy plot with the new colormap
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar.set_ticks(np.linspace(0, 1, 3))
    if col == "preferred_depth_closedloop":
        cbar.set_ticklabels((np.geomspace(0.02, 20, 3) * 100).astype("int"))
    elif col == "rf_azi":
        cbar.set_ticklabels(np.round(np.linspace(azi_min, azi_max, 3), 1))
    elif col == "rf_ele":
        cbar.set_ticklabels(np.round(np.linspace(ele_min, ele_max, 3), 1))
    cbar.ax.tick_params(labelsize=fontsize_dict["legend"])
    cbar_pos = np.array(plt.gca().get_position().bounds)
    cbar_pos[0] = cbar_pos[0] + cbar_pos[2] + 0.01
    cbar_pos[2] = 0.25
    cbar_pos[3] = cbar_pos[3] * 0.3
    cbar.ax.set_position(cbar_pos)
    cbar.ax.tick_params(axis="y", length=1.5)
    # Add scalebar
    scalebar_length_px = im.shape[0] / fov_width * 100  # Scale bar length in pixels
    rect = plt.Rectangle(
        (800, im.shape[0] * 0.95),
        scalebar_length_px,
        scalebar_length_px * 0.05,
        color="white",
    )
    plt.gca().invert_xaxis()
    plt.gca().add_patch(rect)
    return im


def plot_fov_mean_img(im, vmax=700, fov_width=572.867):
    plt.imshow(np.flip(im, axis=1), vmax=vmax, cmap="gray")
    plt.axis("off")
    cbar = plt.colorbar()
    cbar_pos = np.array(plt.gca().get_position().bounds)
    cbar_pos[0] = cbar_pos[0] + cbar_pos[2] + 0.005
    cbar_pos[2] = 0.15
    cbar_pos[3] = cbar_pos[3] * 0.3
    cbar.ax.set_position(cbar_pos)
    cbar.ax.tick_params(axis="y", length=1.5)
    cbar.remove()
    # Add scalebar
    scalebar_length_px = im.shape[0] / fov_width * 100  # Scale bar length in pixels
    rect = plt.Rectangle(
        (40, im.shape[0] * 0.93), scalebar_length_px, 20, color="white"
    )
    plt.gca().add_patch(rect)


def plot_mean_running_speed_alldepths(
    results,
    depth_list,
    fontsize_dict,
    param="RS",
    ylim=None,
    of_threshold=0.01,
    linewidth=3,
    elinewidth=3,
    jitter=0.2,
    scatter_markersize=2,
    scatter_alpha=0.5,
    capsize=3,
    capthick=10,
):
    ax = plt.gca()
    if param == "RS":
        rs_means = (
            np.vstack([j for i in results.rs_mean_closedloop.values for j in i]) * 100
        )
    elif param == "OF":
        rs_means = np.degrees(
            np.vstack([j for i in results.rs_mean_closedloop.values for j in i])
            / depth_list.reshape(1, -1)
        )
        rs_means[rs_means < of_threshold] = of_threshold
    CI_low, CI_high = common_utils.get_bootstrap_ci(rs_means.T, sig_level=0.05)
    for idepth in range(len(depth_list)):
        color = plotting_utils.get_color(
            value=depth_list[idepth],
            value_min=np.min(depth_list),
            value_max=np.max(depth_list),
            cmap=cm.cool.reversed(),
            log=True,
        )
        sns.stripplot(
            x=np.ones(rs_means.shape[0]) * idepth,
            y=rs_means[:, idepth],
            jitter=jitter,
            edgecolor="white",
            color=color,
            alpha=scatter_alpha,
            size=scatter_markersize,
        )
        plt.plot(
            [idepth - 0.4, idepth + 0.4],
            [np.mean(rs_means[:, idepth]), np.mean(rs_means[:, idepth])],
            linewidth=linewidth,
            color=color,
        )
        plt.errorbar(
            x=idepth,
            y=np.mean(rs_means[:, idepth]),
            yerr=np.array(
                [
                    np.mean(rs_means[:, idepth]) - CI_low[idepth],
                    CI_high[idepth] - np.mean(rs_means[:, idepth]),
                ]
            ).reshape(2, 1),
            capsize=capsize,
            elinewidth=elinewidth,
            ecolor=color,
            capthick=capthick,
        )
    ax.set_xticklabels(
        np.round((depth_list * 100)).astype("int"), fontsize=fontsize_dict["label"]
    )
    if param == "RS":
        ax.set_ylabel("Average running\nspeed (cm/s)", fontsize=fontsize_dict["label"])
        ax.set_ylim(0, ax.get_ylim()[1])
    elif param == "OF":
        ax.set_ylabel(
            "Average optic flow\nspeed (degrees/s)", fontsize=fontsize_dict["label"]
        )
        ax.set_yscale("log")
    ax.set_xlabel("Depth (cm)", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    if ylim is not None:
        ax.set_ylim(ylim)
        if param == "RS":
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 4))
    sns.despine(ax=ax)
