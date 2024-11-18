import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import scipy
import seaborn as sns

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
    fig,
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
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    legend_on=False,
    ci_range=0.95,
    ylim=None,
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
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
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


def plot_RS_OF_matrix(
    fig,
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
    ylabel="Optical flow speed \n(degrees/s)",
    title="",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
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

    if vmin is None:
        vmin = np.nanmax([0, np.percentile(bin_means[1:, 1:].flatten(), 1)])
    if vmax is None:
        vmax = np.round(np.nanmax(bin_means[1:, 1:].flatten()), 1)
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    im = ax.imshow(
        bin_means[1:, 1:].T,
        origin="lower",
        aspect="equal",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=fontsize_dict["title"])
    plot_x, plot_y, plot_width, plot_height = (
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width,
        ax.get_position().height,
    )

    ticks_select1, ticks_select2, bin_edges1, bin_edges2 = get_RS_OF_heatmap_axis_ticks(
        log_range=log_range, fontsize_dict=fontsize_dict
    )
    plt.xticks(
        ticks_select1[0::2],
        bin_edges1[0::2],
        fontsize=fontsize_dict["tick"],
    )

    plt.yticks(ticks_select2[1::2], bin_edges2[1::2], fontsize=fontsize_dict["tick"])

    if is_closed_loop:
        ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"], labelpad=0)
        ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"], labelpad=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax_left = fig.add_axes(
            [
                plot_x - plot_width / (log_range["rs_bin_num"] - 1) * 1.5,
                plot_y,
                plot_width / (log_range["rs_bin_num"] - 1),
                plot_height,
            ]
        )
        ax_left.imshow(
            bin_means[0, 1:].reshape(1, -1).T,
            origin="lower",
            aspect="equal",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        plt.yticks(
            ticks_select2[1::2], bin_edges2[1::2], fontsize=fontsize_dict["tick"]
        )
        plt.xticks([])

        ax_down = fig.add_axes(
            [
                plot_x,
                plot_y - plot_height / (log_range["of_bin_num"] - 1) * 1.5,
                plot_width,
                plot_height / (log_range["of_bin_num"] - 1),
            ]
        )
        ax_down.imshow(
            bin_means[1:, 0].reshape(-1, 1).T,
            origin="lower",
            aspect="equal",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xticks(
            ticks_select1[0::2], bin_edges1[0::2], fontsize=fontsize_dict["tick"]
        )
        plt.yticks([])

        ax_corner = fig.add_axes(
            [
                plot_x - plot_width / (log_range["rs_bin_num"] - 1) * 1.5,
                plot_y - plot_height / (log_range["of_bin_num"] - 1) * 1.5,
                plot_width / (log_range["rs_bin_num"] - 1),
                plot_height / (log_range["of_bin_num"] - 1),
            ]
        )
        ax_corner.imshow(
            bin_means[0, 0].reshape(1, 1),
            origin="lower",
            aspect="equal",
            cmap="Reds",
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
        ax2 = fig.add_axes(
            [plot_x + plot_width * 1.1, plot_y, plot_width * 0.05, plot_height / 2]
        )
        cbar = fig.colorbar(im, cax=ax2, label="\u0394F/F")
        ax2.tick_params(labelsize=fontsize_dict["legend"], length=2, pad=2)
        ax2.set_ylabel(
            "\u0394F/F", rotation=270, fontsize=fontsize_dict["legend"], labelpad=4
        )
        cbar.set_ticks([vmin, vmax])

    return vmin, vmax


def plot_RS_OF_fit(
    fig,
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
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    xlabel="Running speed (cm/s)",
    ylabel="Optical flow speed \n(degrees/s)",
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
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
    resp_pred = funcs[model](
        params,
        *neurons_df[f"rsof_popt_closedloop_{model}"].iloc[roi],
        min_sigma=min_sigma,
    ).reshape((len(of), len(rs)))

    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    im = ax.imshow(
        resp_pred,
        origin="lower",
        extent=[
            log_range["rs_bin_log_min"],
            log_range["rs_bin_log_max"],
            log_range["of_bin_log_min"],
            log_range["of_bin_log_max"],
        ],
        aspect="equal",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xticks(
        [0, 1, 2],
        labels=["1", "10", "100"],
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(
        [-1, 0, 1, 2, 3],
        labels=["0.1", "1", "10", "100", "1000"],
        fontsize=fontsize_dict["tick"],
    )
    if cbar_width is not None:
        ax2 = fig.add_axes(
            [plot_x + plot_width * 0.75, plot_y, cbar_width, plot_height * 0.9]
        )
        fig.colorbar(im, cax=ax2, label="\u0394F/F")
        ax2.tick_params(labelsize=fontsize_dict["legend"])
        ax2.set_ylabel("\u0394F/F", rotation=270, fontsize=fontsize_dict["legend"])
    plt.title(
        model_label,
        fontdict={"fontsize": fontsize_dict["label"]},
    )
    plt.text(
        x=log_range["rs_bin_log_min"] + 0.2,
        y=log_range["of_bin_log_max"] - 0.7,
        s=f"$R^2$ = {neurons_df[f'rsof_test_rsq_closedloop_{model}'].iloc[roi]:.2f}",
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
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
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
                neurons_df.groupby("session")
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
                    jitter=0.4,
                    edgecolor="white",
                    color=sns.color_palette("Set1")[i],
                )
                plt.plot(
                    [i - 0.4, i + 0.4],
                    [np.median(props[i]), np.median(props[i])],
                    linewidth=3,
                    color=color,
                )
                if ci is not None:
                    plt.fill_between(
                        [i - 0.4, i + 0.4],
                        [ci[i][0], ci[i][0]],
                        [ci[i][1], ci[i][1]],
                        color=sns.color_palette("Set1")[i],
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
            print(
                f"{labels[0]} vs {labels[1]}: {scipy.stats.wilcoxon(props[0],props[1])}"
            )
            print(
                f"{labels[0]} vs {labels[2]}: {scipy.stats.wilcoxon(props[0],props[2])}"
            )
            print(
                f"{labels[1]} vs {labels[2]}: {scipy.stats.wilcoxon(props[1],props[2])}"
            )
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
            X, y, s=s, alpha=alpha, c=color, edgecolors=edgecolors, linewidths=0.5
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
    # ax.tick_params(axis='both', which='minor', bottom=True, labelsize=fontsize_dict["tick"])
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
    ylabel="Optic flow speed (degree/s)",
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
    xlim=(0,100),
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
            trials_df[f"{param}_merged"] = trials_df.apply(
                lambda x: np.concatenate(
                    [x[f"{param}_stim"], np.full(len(x[f"{param}_blank"]), np.nan)]
                ),
                axis=1,
            )
        param_trace = np.concatenate(
            [row[f"{param}_merged"] for _, row in trials_df.iloc[trial_list].iterrows()]
        )
        if OF_to_degree: # to convert OF from rads/s to degrees/s
            param_trace = np.degrees(param_trace)
        if trial_list[0] == 0:
            blank_start = trials_df.iloc[trial_list[0]][f"{param[:2]}_blank_pre"][
                -int(fs * 10) :
            ]
            param_trace = np.concatenate([np.full(int(fs * 10), np.nan), param_trace])
        else:
            blank_start = trials_df.iloc[trial_list[0]][f"{param[:2]}_blank_pre"]
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
            len(row[f"{param}_stim"]) for _, row in trials_df.iloc[trial_list].iterrows()
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
                ax.set_yticks([0, ylim[1]//10*10])
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
            ax.set_xticks((np.array(trial_starts)+np.array(trial_lengths)/2)/fs)
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
        for param, ylabel in zip(["RS", "OF"], ["RS\n(cm/s)", "OF\n(degrees/s)"]):
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
            )
            ylims.append(ylim)

    i = 0
    for closed_loop, title in zip([1, 0], ["Closed loop", "Open loop"]):
        for param, ylabel in zip(["RS", "OF"], ["RS\n(cm/s)", "OF\n(degrees/s)"]):
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


def plot_histogram_overlay(ax, 
                           fig,
                           rs, 
                           depth_list, 
                           nbins, 
                           scaling_factor = 0.01,
                           facecolor="g", 
                           edgecolor="k",
                           alpha=0.5, 
                           ylim=[1,2e3], 
                           xlim=None, 
                           ):
    '''Plot histogram overlay of OF distribution for different depths on top of the preferred OF-depth scatter plot. 

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
    '''
    ax2 = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y0,
                        ax.get_position().width,
                        ax.get_position().height,])
    ax2.set_facecolor('none')
    if xlim is None:
        xlim = ax.get_xlim()
    ax2.set_xlim(xlim)
    ax2.set_xscale("log")
    for idepth, depth in enumerate(depth_list):
        of = np.degrees(rs / depth)
        bins = np.geomspace(np.nanmin(of), np.nanmax(of), nbins)
        n, _ = np.histogram(of, bins=bins);
        # Calculate bin widths
        bin_width = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]

        # Plot histogram manually
        bottom = (np.log10(depth*100)-np.log10(ylim[0]))/(np.log10(ylim[1])-np.log10(ylim[0]))*(ylim[1]-ylim[0])+ylim[0]
        ax2.bar(bins[:-1], 
                (n*scaling_factor), 
                width=bin_width, 
                align='edge',
                facecolor=facecolor, 
                edgecolor=edgecolor, 
                bottom=bottom,
                alpha=alpha)
        ax2.set_ylim(ylim)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)