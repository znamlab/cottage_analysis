import os
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
from cottage_analysis.plotting import plotting_utils, grating_plots
from cottage_analysis.analysis import (
    find_depth_neurons,
    common_utils,
    fit_gaussian_blob,
    size_control,
)

# TODO:
# 1. plot depth tuning curve with the smoothing tuning method

def plot_depth_neuron_distribution(
    neurons_df,
    iscell,
    trials_df,
    protocol="SpheresPermTubeReward",
    depth_min=0.02,
    depth_max=20,
    bin_number=50,
    mode="discrete",
    color="gray",
    alpha=1,
):
    """
    Plot distribution of neurons' depth preferences in one session.

    Args:
        neurons_df (pd.DataFrame): Dataframe containing analyzed info of all rois
        iscell(np.1darray): 1d array of 0s and 1s indicating whether rois in neurons_df are cells or not.
        trials_df (pd.DataFrame): Dataframe containing info of all trials
        protocol (str, optional): protocol name. Defaults to 'SpheresPermTubeReward'.
        depth_min (float, optional): minimum fitted depth. Defaults to 0.02.
        depth_max (int, optional): maximum fitted depth. Defaults to 20.
        bin_number (int, optional): number of bins for continuous distribution. Defaults to 50.
        mode (str, optional): 'discrete' for bar graph, 'continuous' for histogram. Defaults to 'discrete'.
    """
    # Reload iscell file and filter out non-neuron rois
    neurons_df.is_cell = iscell
    neurons_df = neurons_df[neurons_df.is_cell == 1]

    if mode == "continuous":
        all_preferred_depths = (
            neurons_df[neurons_df.is_depth_neuron == 1]
        ).preferred_depth_closed_loop
        bins = np.geomspace(depth_min, depth_max, num=bin_number)
        plt.hist(
            all_preferred_depths,
            bins=bins,
            color=color,
            alpha=alpha,
            weights=np.ones(len(all_preferred_depths))
            / len(neurons_df[neurons_df.is_depth_neuron == 1]),
        )
        plt.xscale("log")
        plt.xlabel("Preferred depth (m)")
        plt.ylabel("Depth neuron%")

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


def get_depth_color(depth, depth_list, cmap=cm.cool.reversed(), log=True):
    """
    Calculate the color for a certain depth out of a depth list

    Args:
        depth (float): preferred depth of a certain neuron.
        depth_list (float): list of all depths.
        cmap (colormap, optional): colormap used. Defaults to cm.cool.reversed().

    Returns:
        rgba_color: tuple of 3 with RGB color values.
    """
    if log:
        norm = mpl.colors.Normalize(
            vmin=np.log(min(depth_list)), vmax=np.log(max(depth_list))
        )
        rgba_color = cmap(norm(np.log(depth)), bytes=True)
    else:
        norm = mpl.colors.Normalize(vmin=min(depth_list), vmax=max(depth_list))
        rgba_color = cmap(norm(depth), bytes=True)
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
    neurons_df.is_cell = iscell

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
        neurons_df[(neurons_df.is_cell == 1) & (neurons_df.is_depth_neuron == 1)]
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
        neurons_df[(neurons_df.is_cell == 1) & (neurons_df.is_depth_neuron != 1)]
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
    rs_thr_max=None,
    still_only=False,
    still_time=0,
    frame_rate=15,
    plot_fit=True,
    linewidth=3,
    linecolor="k",
    fit_linecolor="r",
    closed_loop=1,
    param="depth",
    use_col="depth_tuning_popt_closedloop",
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
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
        closed_loop (int, optional): 1 for closed loop, 0 for open loop. Defaults to 1.
        param (str, optional): 'depth' for virtual depth, 'size' for physical size. Defaults to 'depth'.
        use_col (str, optional): column name of the gaussian fit parameters. Defaults to 'depth_tuning_popt_closedloop'.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """

    # Load average activity and confidence interval for this roi
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    if param == "depth":
        param_list = np.array(find_depth_neurons.find_depth_list(trials_df))
    elif param == "size":
        param_list = np.sort(trials_df["physical_size"].unique())
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(trials_df=trials_df,
                                                  rs_thr=rs_thr, 
                                                  rs_thr_max=rs_thr_max, 
                                                  still_only=still_only, 
                                                  still_time=still_time, 
                                                  frame_rate=frame_rate, 
                                                  closed_loop=closed_loop, 
                                                  param=param)[:, :, roi]
    CI_low, CI_high = common_utils.get_confidence_interval(mean_dff_arr)
    mean_arr = np.nanmean(mean_dff_arr, axis=1)

    # Load gaussian fit params for this roi
    if plot_fit:
        min_sigma = 0.5
        [a, x0, log_sigma, b] = neurons_df.loc[roi, use_col]
        x = np.geomspace(param_list[0], param_list[-1], num=100)
        gaussian_arr = find_depth_neurons.gaussian_func(
            np.log(x), a, x0, log_sigma, b, min_sigma
        )

    # Plotting
    plt.plot(np.log(param_list), mean_arr, color=linecolor, linewidth=linewidth)
    plt.fill_between(
        np.log(param_list),
        CI_low,
        CI_high,
        color=linecolor,
        alpha=0.3,
        edgecolor=None,
        rasterized=False,
    )
    if plot_fit:
        plt.plot(np.log(x), gaussian_arr, color=fit_linecolor, linewidth=linewidth)
    if param=="depth":
        plt.xticks(
            np.log(param_list),
            (np.array(param_list) * 100).astype("int"),
            fontsize=fontsize_dict["tick"],
            rotation=45,
        )
        plt.xlabel(f"Virtual depth (cm)", fontsize=fontsize_dict["label"])
    elif param=="size":
        plt.xticks(
            np.log(param_list),
            (np.array(param_list)*0.87/10),
            fontsize=fontsize_dict["tick"],
            rotation=45,
        )
        plt.xlabel(f"Actual radius (cm)", fontsize=fontsize_dict["label"])
    plt.yticks(fontsize=fontsize_dict["tick"])
    plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"])

    plotting_utils.despine()


def plot_depth_tuning_curve_smooth(
    neurons_df,
    trials_df,
    roi,
    rs_thr=0.2,
    plot_fit=True,
    linewidth=3,
    linecolor="k",
    fit_linecolor="r",
    closed_loop=1,
    smoothing_sd=1,
):
    # Load average activity and confidence interval for this roi
    depth_list = np.array(find_depth_neurons.find_depth_list(trials_df))
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
        trials_df, rs_thr=rs_thr
    )[:, :, roi]
    mean_arr = np.mean(mean_dff_arr, axis=1)
    x = np.repeat(depth_list, mean_dff_arr.shape[1])

    tuning = plotting_utils.get_tuning_function(
        means=mean_arr,
        counts=np.repeat(mean_dff_arr.shape[1], len(depth_list)),
        smoothing_sd=smoothing_sd,
    )

    # ci_range = 0.95
    # z = scipy.stats.norm.ppf(1 - ((1 - ci_range) / 2))
    # ci = z * bin_stds / bin_counts
    # ci[np.isnan(ci)] = 0
    plt.plot(np.log(depth_list), mean_arr)
    plt.plot(
        np.linspace(np.log(depth_list[0]), np.log(depth_list[-1]), len(tuning)), tuning
    )


def generate_cmap(cmap_name="WhRd"):
    """Generate common colormap

    Args:
        cmap_name (str, optional): color map mame. Defaults to 'WhRd'.

    Returns:
        cmap (matplotlib.cmap object): matplotlib colormap
    """
    if cmap_name == "WhRd":
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(1, 1, N)
        vals[:, 1] = np.linspace(1, 0, N)
        vals[:, 2] = np.linspace(1, 0, N)
        cmap = ListedColormap(vals)

    return cmap


def plot_raster_all_depths(
    neurons_df,
    trials_df,
    roi,
    is_closed_loop,
    max_distance=6,
    nbins=60,
    frame_rate=15,
    vmax=1,
    plot=True,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """Raster plot for neuronal activity for each depth

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        max_distance (int, optional): max distance for each trial in meters. Defaults to 6.
        nbins (int, optional): number of bins to bin the activity. Defaults to 60.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
        vmax (int, optional): vmax to plot the heatmap. Defaults to 1.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """
    # choose the trials with closed or open loop to visualize
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")
    trial_number = len(trials_df) // len(depth_list)

    # bin dff according to distance travelled for each trial
    dffs_binned = np.zeros((len(depth_list), trial_number, nbins))
    for idepth, depth in enumerate(depth_list):
        all_dffs = grouped_trials.get_group(depth).dff_stim.values
        all_rs = grouped_trials.get_group(depth).RS_stim.values
        for itrial in np.arange(len(all_dffs)):
            if itrial < trial_number:
                dff = all_dffs[itrial][:, roi]
                rs_arr = all_rs[itrial]
                distance = np.cumsum(rs_arr / frame_rate)
                bins = np.linspace(
                    start=0, stop=max_distance, num=nbins + 1, endpoint=True
                )
                bin_means, _, _ = scipy.stats.binned_statistic(
                    x=distance,
                    values=dff,
                    statistic="mean",
                    bins=nbins,
                )
                dffs_binned[idepth, itrial, :] = bin_means

    # colormap
    WhRdcmap = generate_cmap(cmap_name="WhRd")

    # plot each depth as a heatmap
    if plot:
        for idepth, depth in enumerate(depth_list):
            plt.subplot(1, len(depth_list), idepth + 1)
            plt.imshow(
                dffs_binned[idepth], aspect="auto", cmap=WhRdcmap, vmin=0, vmax=vmax
            )
            plt.xticks(
                np.linspace(0, nbins, 3),
                (np.linspace(0, max_distance, 3) * 100).astype("int"),
                fontsize=fontsize_dict["tick"],
                rotation=45,
            )
            if idepth == 0:
                plt.ylabel("Trial no.", fontsize=fontsize_dict["label"])
                plt.tick_params(
                    left=True,
                    right=False,
                    labelleft=True,
                    labelbottom=True,
                    bottom=True,
                )
            else:
                plt.tick_params(
                    left=True,
                    right=False,
                    labelleft=False,
                    labelbottom=True,
                    bottom=True,
                )
            # plt.xlabel("Virtual distance (m)", fontsize=fontsize_dict["label"])
        # plt.tight_layout()
        # add_colorbar()
    return dffs_binned


def add_colorbar():
    cbar_pos = [
        1.02,
        plt.gca().get_position().y0,
        0.02,
        plt.gca().get_position().height,
    ]
    plt.axes(cbar_pos)
    plt.colorbar(cax=plt.gca(), label="\u0394F/F")


def plot_speed_tuning(
    neurons_df,
    trials_df,
    roi,
    is_closed_loop,
    nbins=20,
    which_speed="RS",
    speed_min=0.01,
    speed_max=1.5,
    speed_thr=0.01,
    smoothing_sd=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """Plot a neuron's speed tuning to either running speed or optic flow speed.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        nbins (int, optional): number of bins to bin the tuning curve. Defaults to 20.
        which_speed (str, optional): 'RS': running speed; 'OF': optic flow speed. Defaults to 'RS'.
        speed_min (float, optional): min RS speed for the bins (m/s). Defaults to 0.01.
        speed_max (float, optional): max RS speed for the bins (m/s). Defaults to 1.5.
        speed_thr (float, optional): thresholding RS for logging (m/s). Defaults to 0.01.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")

    if which_speed == "RS":
        speed_tuning = np.zeros(((len(depth_list) + 1), nbins))
        speed_ci = np.zeros(((len(depth_list) + 1), nbins))
        bins = (
            np.linspace(start=speed_min, stop=speed_max, num=nbins + 1, endpoint=True)
            * 100
        )

    elif which_speed == "OF":
        speed_tuning = np.zeros(((len(depth_list)), nbins))
        speed_ci = np.zeros(((len(depth_list)), nbins))
    bin_centers = np.zeros(((len(depth_list)), nbins))

    # Find all speed and dff of this ROI for a specific depth
    for idepth, depth in enumerate(depth_list):
        all_speed = grouped_trials.get_group(depth)[f"{which_speed}_stim"].values
        speed_arr = np.array([j for i in all_speed for j in i])
        all_dff = grouped_trials.get_group(depth)["dff_stim"].values
        dff_arr = np.array([j for i in all_dff for j in i[:, roi]])

        if which_speed == "OF":
            speed_arr = np.degrees(speed_arr)  # rad --> degrees
        if which_speed == "RS":
            speed_arr = speed_arr * 100  # m/s --> cm/s
        # threshold speed
        dff_arr = dff_arr[speed_arr > speed_thr]
        speed_arr = speed_arr[speed_arr > speed_thr]

        if which_speed == "OF":
            bins = np.geomspace(
                start=np.nanmin(speed_arr),
                stop=np.nanmax(speed_arr),
                num=nbins + 1,
                endpoint=True,
            )
        bin_centers[idepth] = (bins[:-1] + bins[1:]) / 2

        # calculate speed tuning
        bin_means, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="mean",
            bins=bins,
        )

        bin_stds, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="std",
            bins=bins,
        )

        bin_counts, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="count",
            bins=bins,
        )

        tuning = plotting_utils.get_tuning_function(
            bin_means, bin_counts, smoothing_sd=smoothing_sd
        )

        ci_range = 0.95
        z = scipy.stats.norm.ppf(1 - ((1 - ci_range) / 2))
        ci = z * bin_stds / np.sqrt(bin_counts)
        ci[np.isnan(ci)] = 0

        speed_tuning[idepth] = tuning
        speed_ci[idepth] = ci

    # Find tuning for blank period for RS
    if which_speed == "RS":
        all_speed = trials_df[f"{which_speed}_blank"].values
        speed_arr = np.array([j for i in all_speed for j in i]) * 100
        all_dff = trials_df["dff_blank"].values
        dff_arr = np.array([j for i in all_dff for j in i[:, roi]])

        # threshold speed
        dff_arr = dff_arr[speed_arr > speed_thr]
        speed_arr = speed_arr[speed_arr > speed_thr]

        # calculate speed tuning
        bin_means, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="mean",
            bins=bins,
        )

        bin_stds, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="std",
            bins=bins,
        )

        bin_counts, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="count",
            bins=bins,
        )

        tuning = plotting_utils.get_tuning_function(bin_means, bin_counts)

        ci = z * bin_stds / np.sqrt(bin_counts)
        ci[np.isnan(ci)] = 0

        speed_tuning[-1] = tuning
        speed_ci[-1] = ci

    # Plotting
    for idepth, depth in enumerate(depth_list):
        linecolor = get_depth_color(depth, depth_list, cmap=cm.cool.reversed())
        plt.plot(
            bin_centers[idepth, :],
            speed_tuning[idepth, :],
            color=linecolor,
            label=f"{int(depth_list[idepth] * 100)} cm",
        )
        plt.errorbar(
            x=bin_centers[idepth, :],
            y=speed_tuning[idepth, :],
            yerr=speed_ci[idepth, :],
            fmt="o",
            color=linecolor,
            ls="none",
        )

        if which_speed == "OF":
            plt.xscale("log")
            plt.xlabel("Optic flow speed (degrees/s)", fontsize=fontsize_dict["label"])

        # Plot tuning to gray period
        if which_speed == "RS":
            plt.plot(
                bin_centers[-1, :],
                speed_tuning[-1, :],
                color="gray",
                label=f"{int(depth_list[idepth] * 100)} cm",
            )
            plt.errorbar(
                x=bin_centers[-1, :],
                y=speed_tuning[-1, :],
                yerr=speed_ci[-1, :],
                fmt="o",
                color="gray",
                ls="none",
            )
            plt.xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
        plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
        plt.xticks(fontsize=fontsize_dict["tick"])
        plt.yticks(fontsize=fontsize_dict["tick"])
    plotting_utils.despine()


def plot_PSTH(
    neurons_df,
    trials_df,
    roi,
    is_closed_loop,
    max_distance=6,
    nbins=20,
    frame_rate=15,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    linewidth=3,
):
    """PSTH of a neuron for each depth and blank period.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        max_distance (int, optional): max distance for each trial in meters. Defaults to 6.
        nbins (int, optional): number of bins to bin the activity. Defaults to 20.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
    """

    # confidence interval z calculation
    ci_range = 0.95
    z = scipy.stats.norm.ppf(1 - ((1 - ci_range) / 2))

    # choose the trials with closed or open loop to visualize
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")
    trial_number = len(trials_df) // len(depth_list)

    # bin dff according to distance travelled for each trial
    all_means = np.zeros(((len(depth_list) + 1), nbins))
    all_ci = np.zeros(((len(depth_list) + 1), nbins))
    for idepth, depth in enumerate(depth_list):
        all_dff = []
        all_distance = []
        for itrial in np.arange(trial_number):
            dff = grouped_trials.get_group(depth).dff_stim.values[itrial][:, roi]
            rs_arr = grouped_trials.get_group(depth).RS_stim.values[itrial]
            distance = np.cumsum(rs_arr / frame_rate)
            all_dff.append(dff)
            all_distance.append(distance)

        all_dff = np.array([j for i in all_dff for j in i])
        all_distance = np.array([j for i in all_distance for j in i])
        bins = np.linspace(start=0, stop=max_distance, num=nbins + 1, endpoint=True)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        # calculate speed tuning
        bin_means, _, _ = scipy.stats.binned_statistic(
            x=all_distance,
            values=all_dff,
            statistic="mean",
            bins=nbins,
        )

        bin_stds, _, _ = scipy.stats.binned_statistic(
            x=all_distance,
            values=all_dff,
            statistic="std",
            bins=nbins,
        )

        bin_counts, _, _ = scipy.stats.binned_statistic(
            x=all_distance,
            values=all_dff,
            statistic="count",
            bins=nbins,
        )

        all_means[idepth, :] = bin_means

        ci = z * bin_stds / np.sqrt(bin_counts)
        ci[np.isnan(ci)] = 0
        all_ci[idepth, :] = ci

    # Blank dff
    dff = trials_df.dff_blank.values
    rs = trials_df.RS_blank.values

    all_dff = np.array([j for i in dff for j in i[:, roi]])
    rs_arr = np.array([j for i in rs for j in i])
    all_distance = np.cumsum(rs_arr / frame_rate)

    # calculate speed tuning
    bin_means, _, _ = scipy.stats.binned_statistic(
        x=all_distance,
        values=all_dff,
        statistic="mean",
        bins=nbins,
    )

    bin_stds, _, _ = scipy.stats.binned_statistic(
        x=all_distance,
        values=all_dff,
        statistic="std",
        bins=nbins,
    )

    bin_counts, _, _ = scipy.stats.binned_statistic(
        x=all_distance,
        values=all_dff,
        statistic="count",
        bins=nbins,
    )

    all_means[-1, :] = bin_means
    ci = z * bin_stds / np.sqrt(bin_counts)
    ci[np.isnan(ci)] = 0
    all_ci[-1, :] = ci

    for idepth, depth in enumerate(depth_list):
        linecolor = get_depth_color(depth, depth_list, cmap=cm.cool.reversed())
        plt.plot(
            bin_centers,
            all_means[idepth, :],
            color=linecolor,
            label=f"{int(depth_list[idepth] * 100)} cm",
            linewidth=linewidth,
        )

        plt.fill_between(
            bin_centers,
            y1=all_means[idepth, :] - all_ci[idepth, :],
            y2=all_means[idepth, :] + all_ci[idepth, :],
            color=linecolor,
            alpha=0.3,
            edgecolor=None,
            rasterized=False,
        )

    plt.plot(
        bin_centers,
        all_means[-1, :],
        color="gray",
        label=f"{int(depth_list[idepth] * 100)} cm",
        linewidth=linewidth,
    )
    plt.fill_between(
        bin_centers,
        y1=all_means[-1, :] - all_ci[-1, :],
        y2=all_means[-1, :] + all_ci[-1, :],
        color="gray",
        alpha=0.3,
        edgecolor=None,
        rasterized=False,
    )

    plt.xlabel("Virtual distance (m)", fontsize=fontsize_dict["label"])
    plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
    plt.xticks(
        # np.linspace(0, nbins, 3),
        # (np.linspace(0, max_distance, 3) * 100).astype("int"),
        fontsize=fontsize_dict["tick"],
        rotation=45,
    )
    plt.yticks(fontsize=fontsize_dict["tick"])
    plotting_utils.despine()


def basic_vis_session(neurons_df, trials_df, neurons_ds, **kwargs):
    rois = neurons_df.roi.values
    for is_closedloop in np.sort(trials_df.closed_loop.unique()):
        if is_closedloop:
            sfx = "closedloop"
        else:
            sfx = "openloop"
        os.makedirs(neurons_ds.path_full.parent / "plots" / f"basic_vis_{sfx}", exist_ok=True)

        plot_rows = 10
        plot_cols = 5

        params = dict(
            rs_thr=0.2,
            rs_curve=dict(speed_min=0.001, speed_max=1, nbins=10, speed_thr=0.001),
        )
        params.update(kwargs)
        for i in tqdm(range(int(len(rois) // plot_rows + 1))):
            if i * plot_rows < len(rois) - 1:
                plt.figure(figsize=(3 * plot_cols, 3 * plot_rows))
                for iroi, roi in enumerate(
                    rois[i * plot_rows : np.min([(i + 1) * plot_rows, len(rois)])]
                ):
                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 0))
                    plot_depth_tuning_curve(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        rs_thr=params["rs_thr"],
                        plot_fit=False,
                        linewidth=3,
                        linecolor="k",
                        fit_linecolor="r",
                        closed_loop=is_closedloop,
                    )
                    plt.title(f"roi{roi}")

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 1))
                    plot_speed_tuning(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        is_closed_loop=is_closedloop,
                        which_speed="RS",
                        smoothing_sd=1,
                        **params["rs_curve"],
                    )

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 2))
                    plot_speed_tuning(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        is_closed_loop=is_closedloop,
                        nbins=10,
                        which_speed="OF",
                        speed_min=0.01,
                        speed_max=1.5,
                        speed_thr=0.01,
                        smoothing_sd=1,
                    )

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 3))
                    plot_PSTH(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        is_closed_loop=is_closedloop,
                        max_distance=6,
                        nbins=20,
                        frame_rate=15,
                    )
                    plt.tight_layout()

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 4))
                    log_range = {
                        "rs_bin_log_min": 0,
                        "rs_bin_log_max": 2.5,
                        "rs_bin_num": 6,
                        "of_bin_log_min": -1.5,
                        "of_bin_log_max": 3.5,
                        "of_bin_num": 11,
                        "log_base": 10,
                    }
                    log_range.update(kwargs["RS_OF_matrix_log_range"])
                    plot_RS_OF_matrix(
                        trials_df=trials_df[trials_df.closed_loop == is_closedloop],
                        roi=roi,
                        log_range=log_range,
                    )

                plt.savefig(
                    neurons_ds.path_full.parent
                    / "plots"
                    / f"basic_vis_{sfx}"
                    / f"roi{rois[i*10]}- {np.min([(i+1)*10, len(rois)])}.png",
                    dpi=100,
                )
                
                plt.close()


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
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    extended_matrix = np.zeros((log_range["rs_bin_num"], log_range["of_bin_num"]))
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

    of_bins = np.logspace(
        log_range["of_bin_log_min"],
        log_range["of_bin_log_max"],
        num=log_range["of_bin_num"],
        base=log_range["log_base"],
    )
    of_bins = np.insert(of_bins, 0, 0)

    rs_arr = np.array([j for i in trials_df.RS_stim.values for j in i]) * 100
    of_arr = np.degrees([j for i in trials_df.OF_stim.values for j in i])
    dff_arr = np.vstack(trials_df.dff_stim.values)[:, roi]

    bin_means, rs_edges, of_egdes, _ = scipy.stats.binned_statistic_2d(
        x=rs_arr, y=of_arr, values=dff_arr, statistic="mean", bins=[rs_bins, of_bins]
    )

    plt.imshow(
        bin_means[1:, 1:].T,
        origin="lower",
        aspect="equal",
        # cmap=generate_cmap(cmap_name="WhRd"),
        cmap="Reds",
        vmin=0,
        vmax=np.nanmax(bin_means[1:, 1:]),
    )
    plt.colorbar()
    ticks_select1, ticks_select2, bin_edges1, bin_edges2 = get_RS_OF_heatmap_axis_ticks(
        log_range=log_range, fontsize_dict=fontsize_dict
    )
    plt.xticks(
        ticks_select1,
        bin_edges1,
        rotation=60,
        ha="center",
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(ticks_select2, bin_edges2, fontsize=fontsize_dict["tick"])
    plt.xlabel("Running spped (cm/s)", fontsize=fontsize_dict["label"])
    plt.ylabel("Optical flow speed (degrees/s)", fontsize=fontsize_dict["label"])

    # ax_main = plt.gca()
    # ax_main.tick_params(
    #     top=False,
    #     bottom=True,
    #     left=True,
    #     right=False,
    #     labelleft=False,
    #     labelbottom=False,
    # )
    # plt.xlabel("")
    # plt.ylabel("")
    # divider = make_axes_locatable(ax_main)
    # ax1 = divider.append_axes("left", size="20%", pad=0.05)
    # ax1.imshow(
    #     extended_matrix.T[1:, 0].reshape(-1, 1),
    #     cmap="Reds",
    #     origin="lower",
    #     vmin=0,
    #     vmax=np.nanmax(bin_means[1:, 1:])
    # )
    # plt.xticks(
    #     [ax1.get_xticks()[len(ax1.get_xticks()) // 2]],
    #     ["<1"],
    #     fontsize=fontsize_dict["tick"],
    # )

    extended_matrix = bin_means
    return extended_matrix


def get_RS_OF_heatmap_axis_ticks(log_range, fontsize_dict, playback=False, log=True):
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
    # if log == False:
    #     _, _ = plt.xticks(np.arange(bin_numbers[0]), bin_centers1, rotation=60, ha='center',
    #                       fontsize=fontsize_dict['xticks'])
    #     _, _ = plt.yticks(np.arange(bin_numbers[1]), bin_centers2, fontsize=fontsize_dict['yticks'])
    else:
        ticks_select1 = (np.arange(-1, bin_numbers[0] * 2, 1) / 2)[0::2]
        ticks_select2 = (np.arange(-1, bin_numbers[1] * 2, 1) / 2)[0::2]
        # _, _ = plt.xticks(
        #     ticks_select1,
        #     bin_edges1,
        #     rotation=60,
        #     ha="center",
        #     fontsize=fontsize_dict["tick"],
        # )
        # _, _ = plt.yticks(ticks_select2, bin_edges2, fontsize=fontsize_dict["tick"])

    return ticks_select1, ticks_select2, bin_edges1, bin_edges2


def plot_RS_OF_fitted_tuning(
    neurons_df,
    roi,
    model="gaussian_2d",
    min_sigma=0.25,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
):

    """
    Plot the fitted tuning of a neuron.
    """
    rs = (
        np.logspace(
            log_range["rs_bin_log_min"], log_range["rs_bin_log_max"], 100, base=10
        )
        / 100
    )  # cm/s --> m/s
    of = np.logspace(
        log_range["of_bin_log_min"], log_range["of_bin_log_max"], 100, base=10
    )  # deg/s

    rs_grid, of_grid = np.meshgrid(np.log(rs), np.log(of))
    if model == "gaussian_2d":
        resp_pred = fit_gaussian_blob.gaussian_2d(
            (rs_grid, of_grid),
            *neurons_df["rsof_popt_closedloop_g2d"].iloc[roi],
            min_sigma=0.25,
        )
    elif model == "gaussian_additive":
        resp_pred = fit_gaussian_blob.gaussian_additive(
            (rs_grid, of_grid),
            *neurons_df["rsof_popt_closedloop_gadd"].iloc[roi],
            min_sigma=0.25,
        )
    elif model == "gaussian_OF":
        resp_pred = fit_gaussian_blob.gaussian_1d(
            of_grid, *neurons_df["rsof_popt_closedloop_gof"].iloc[roi], min_sigma=0.25
        )
    resp_pred = resp_pred.reshape((len(of), len(rs)))

    plt.imshow(
        resp_pred,
        origin="lower",
        extent=[rs.min() * 100, rs.max() * 100, of.min(), of.max()],
        aspect=rs.max()
        * 100
        / of.max()
        * log_range["of_bin_num"]
        / log_range["rs_bin_num"],
        cmap="Reds",
    )


def size_control_session(neurons_df, trials_df, neurons_ds, **kwargs):
    rois = neurons_df.roi.values
    trials_df = trials_df[trials_df.closed_loop == 1]
    trials_df = size_control.get_physical_size(trials_df, use_cols=["size", "depth"], k=1)
    os.makedirs(neurons_ds.path_full.parent / "plots" / f"size_control_basic_vis", exist_ok=True)

    plot_rows = 10
    plot_cols = 3

    for i in tqdm(range(int(len(rois) // plot_rows + 1))):
        if i * plot_rows < len(rois) - 1:
            plt.figure(figsize=(3 * plot_cols, 3 * plot_rows))
            for iroi, roi in enumerate(
                rois[i * plot_rows : np.min([(i + 1) * plot_rows, len(rois)])]
            ):
                plt.subplot2grid((plot_rows, plot_cols), (iroi, 0))
                plot_depth_tuning_curve(
                    neurons_df=neurons_df,
                    trials_df=trials_df,
                    roi=roi,
                    rs_thr=None,
                    rs_thr_max=None,
                    still_only=False,
                    still_time=0,
                    frame_rate=15,
                    plot_fit=True,
                    linewidth=3,
                    linecolor="k",
                    fit_linecolor="r",
                    closed_loop=1,
                    param="depth",
                    use_col="depth_tuning_popt_closedloop",
                    fontsize_dict={"title": 15, "label": 10, "tick": 10},
                )
                
                plt.subplot2grid((plot_rows, plot_cols), (iroi, 1))
                linecolors = ["aqua", "b", "midnightblue"]
                for isize, size in enumerate(np.sort(trials_df["size"].unique())):
                    plot_depth_tuning_curve(
                        neurons_df=neurons_df,
                        trials_df=trials_df[trials_df["size"]==size],
                        roi=roi,
                        rs_thr=None,
                        rs_thr_max=None,
                        still_only=False,
                        still_time=0,
                        frame_rate=15,
                        plot_fit=False,
                        linewidth=3,
                        linecolor=linecolors[isize],
                        fit_linecolor="r",
                        closed_loop=1,
                        param="depth",
                        use_col="depth_tuning_popt_closedloop",
                        fontsize_dict={"title": 15, "label": 10, "tick": 10},
                    )
                    
                plt.subplot2grid((plot_rows, plot_cols), (iroi, 2))
                plot_depth_tuning_curve(
                    neurons_df=neurons_df,
                    trials_df=trials_df,
                    roi=roi,
                    rs_thr=None,
                    rs_thr_max=None,
                    still_only=False,
                    still_time=0,
                    frame_rate=15,
                    plot_fit=True,
                    linewidth=3,
                    linecolor=linecolors[isize],
                    fit_linecolor="r",
                    closed_loop=1,
                    param="size",
                    use_col="size_tuning_popt_closedloop",
                    fontsize_dict={"title": 15, "label": 10, "tick": 10},
                )
            
            plt.savefig(
                neurons_ds.path_full.parent 
                / "plots" 
                / f"size_control_basic_vis"
                / f"roi{rois[i*10]}- {np.min([(i+1)*10, len(rois)])}.png",
                dpi=100,
            )
            
            plt.close()
                
                
                