import numpy as np
import pandas as pd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle
import os
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel

import flexiznam as flz
from znamutils import slurm_it
from cottage_analysis.analysis import spheres, common_utils
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import plotting_utils
from cottage_analysis.summary_analysis import depth_decoder_stats

CONDA_ENV = "2p_analysis_cottage2"


def bar_plot_ttest(
    group1, group2, labels, fig, plot_x, plot_y, plot_width, plot_height
):
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1), np.std(group2)

    # Calculate t-test
    t_statistic, p_value = ttest_rel(group1, group2)

    # Plot the data
    bar1 = ax.bar(0, mean1, yerr=std1, capsize=10, width=0.25, label=labels[0])
    bar2 = ax.bar(0.3, mean2, yerr=std2, capsize=10, width=0.25, label=labels[1])

    # Add bracket and asterisk if significant
    if p_value < 0.05:
        star = "*"
        if p_value < 0.01:
            star = "**"
        elif p_value < 0.001:
            star = "***"
        # Add square bracket
        x0, x1 = (
            bar1[0].get_x() + bar1[0].get_width() / 2,
            bar2[0].get_x() + bar2[0].get_width() / 2,
        )
        y, h, col = max([mean1 + std1]), 0.05, "k"
        ax.plot([x0, x0, x1, x1], [y + h, y + 2 * h, y + 2 * h, y + h], lw=1.5, c=col)
        # Add asterisk on top of bracket
        ax.text((x0 + x1) * 0.5, y + 2 * h, star, ha="center", va="bottom", fontsize=24)

    # Set the x-axis labels and title
    ax.set_xticks([0, 0.3])
    ax.set_xticklabels([labels[0], labels[1]])

    plotting_utils.despine()


def decoder_accuracy(
    decoder_results,
    plot_type="bar",
    markersize=5,
    colors=["b", "g"],
    markers=["o", "^"],
    markerfacecolors=["k", "none"],
    linewidth=0.5,
    xlabel=["Closed loop", "Open loop"],
    ylabel="Classification accuracy",
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    mode="accuracy",
    n_boots=10000,
    xlim=None,
    ylim=None,
):
    ndepths_list = decoder_results["ndepths"].unique()
    if plot_type == "bar":
        for ndepths, color, marker, markerfacecolor in zip(
            ndepths_list, colors, markers, markerfacecolors
        ):
            this_ndepths = decoder_results[decoder_results["ndepths"] == ndepths]
            plt.plot(
                [1, 2],
                [this_ndepths[f"{mode}_closedloop"], this_ndepths[f"{mode}_openloop"]],
                color=color,
                marker=marker,
                alpha=0.7,
                label=f"{ndepths} depths",
                markersize=markersize,
                linewidth=linewidth,
                markerfacecolor=markerfacecolor,
            )
            if mode == "accuracy":
                plt.axhline(
                    y=1 / ndepths, color=color, linestyle="dotted", linewidth=linewidth
                )

            plt.plot(
                [1, 2],
                [
                    np.median(this_ndepths[f"{mode}_closedloop"]),
                    np.median(this_ndepths[f"{mode}_openloop"]),
                ],
                color=color,
                marker=marker,
                alpha=0.7,
                markersize=0,
                linewidth=linewidth * 2,
                markerfacecolor=markerfacecolor,
            )
            handles, legend_labels = plotting_utils.get_unique_labels(plt.gca())
            plt.legend(
                handles,
                legend_labels,
                fontsize=fontsize_dict["legend"],
                frameon=False,
                loc="upper right",
                handlelength=3,
            )
            for icol, col in enumerate([f"{mode}_closedloop", f"{mode}_openloop"]):
                plt.plot(
                    [icol + 0.8, icol + 1.2],
                    [np.median(this_ndepths[col]), np.median(this_ndepths[col])],
                    color,
                    lw=2,
                )

            print(
                f"decoder accuracy {ndepths} depths 2-sided: {wilcoxon(this_ndepths[f'{mode}_closedloop'], this_ndepths[f'{mode}_openloop'])}"
            )
            print(
                f"decoder accuracy {ndepths} depths 1-sided greater: {wilcoxon(this_ndepths[f'{mode}_closedloop'], this_ndepths[f'{mode}_openloop'], alternative='greater',)}"
            )
        print(
            f"decoder accuracy all 2-sided: {wilcoxon(decoder_results[f'{mode}_closedloop'], decoder_results[f'{mode}_openloop'])}"
        )
        print(
            f"decoder accuracy all 1-sided greater: {wilcoxon(decoder_results[f'{mode}_closedloop'], decoder_results[f'{mode}_openloop'],alternative='greater',)}"
        )
        plotting_utils.despine()
        plt.xticks(
            [1, 2], xlabel, fontsize=fontsize_dict["label"], rotation=0, ha="center"
        )
        plt.yticks(fontsize=fontsize_dict["tick"])
        plt.xlim([0.5, 2.5])
        if mode == "accuracy":
            plt.ylim([0, 1])
        plt.ylabel(ylabel, fontsize=fontsize_dict["label"])

    elif plot_type == "scatter":
        ax = plt.gca()
        for ndepths, color, marker, label in zip(
            [5, 8], colors, markers, ["5 depths", "8 depths"]
        ):
            decoder_results_ndepths = decoder_results[
                decoder_results["ndepths"] == ndepths
            ]
            print(
                f"decoder accuracy {ndepths} depths: {len(decoder_results_ndepths)} sessions)"
            )

            # plot scatter of closed loop vs open loop decoding accuracy
            ax.scatter(
                decoder_results_ndepths["accuracy_closedloop"].values.flatten(),
                decoder_results_ndepths["accuracy_openloop"].values.flatten(),
                marker=marker,
                s=markersize,
                alpha=0.5,
                color=color,
                label=label,
            )
            ax.set_aspect("equal")
            if xlim is None:
                xlim = [0, 1]
            if ylim is None:
                ylim = [0, 1]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            ax.plot(
                xlim,
                ylim,
                color="k",
                linestyle="dotted",
                linewidth=linewidth,
            )
        ax.legend(
            fontsize=fontsize_dict["legend"],
            loc="upper left",
            frameon=False,
        )

        ax.set_xlabel("Closed loop decoding accuracy", fontsize=fontsize_dict["label"])
        ax.set_ylabel("Open loop decoding accuracy", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        sns.despine(ax=ax)


def calculate_average_confusion_matrix(
    decoder_results, col="", recording_types=["closedloop", "openloop"]
):
    conmat_mean = {}
    for recording_type in recording_types:
        if col == "":
            col_new = f"conmat_{recording_type}"
        else:
            print(f"Using {col} for confusion matrix")
            col_new = col
        decoder_results[f"conmat_prop_{recording_type}"] = decoder_results[
            col_new
        ].apply(lambda x: x / np.nansum(x))
        conmat_mean[recording_type] = np.nanmean(
            np.stack(decoder_results[f"conmat_prop_{recording_type}"]), axis=0
        )
    return conmat_mean


def plot_confusion_matrix(
    conmat,
    ax,
    vmax,
    fontsize_dict,
    depths=np.logspace(np.log2(5), np.log2(640), 8, base=2),
    colorbar_on=False,
    xtick_rotation=0,
):
    im = ax.imshow(conmat, interpolation="nearest", cmap="magma", vmax=vmax, vmin=0)
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        display = conmat[i, j]
        if display > 1:
            display = f"{int(display)}"
        else:
            display = f"{display:.2f}"
        # plt.text(
        #     j,
        #     i,
        #     display,
        #     horizontalalignment="center",
        #     color="black" if conmat[i, j] > vmax / 2 else "white",
        #     fontsize=fontsize_dict["tick"],
        # )
    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels(depths, fontsize=fontsize_dict["tick"], rotation=xtick_rotation)
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels(depths, fontsize=fontsize_dict["tick"])
    ax.set_xlabel("Predicted virtual depth (cm)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("True virtual depth (cm)", fontsize=fontsize_dict["label"])

    plot_x, plot_y, plot_width, plot_height = (
        ax.get_position().x0,
        ax.get_position().y0,
        ax.get_position().width,
        ax.get_position().height,
    )
    if colorbar_on:
        ax2 = plt.gcf().add_axes(
            [
                plot_x + plot_width + 0.01,
                plot_y,
                plot_width * 0.05,
                plot_height / 3,
            ]
        )
        # set colorbar
        cbar = plt.colorbar(im, cax=ax2, label="Accuracy")
        cbar.set_ticks([0, vmax])
        ax2.tick_params(labelsize=fontsize_dict["tick"])
        ax2.set_ylabel("Porportion of frames", fontsize=fontsize_dict["legend"])
    return im


def plot_closed_open_conmat(
    conmat_mean,
    normalize,
    fig,
    ax1,
    ax2,
    fontsize_dict,
    vmax=None,
):
    conmat_closed = conmat_mean["closedloop"]
    conmat_open = conmat_mean["openloop"]
    if normalize:
        conmat_closed = conmat_closed / conmat_closed.sum(axis=1)[:, np.newaxis]
        conmat_open = conmat_open / conmat_open.sum(axis=1)[:, np.newaxis]
    if vmax is None:
        vmax = np.round(
            np.max([conmat_closed, conmat_open]), 2
        )  # round vmax to the closest multiple of 0.05
    if len(conmat_closed) == 8:
        depths = np.logspace(np.log2(5), np.log2(640), 8, base=2)
    else:
        depths = np.logspace(np.log2(6), np.log2(600), 5, base=2)
    depths = np.round(depths).astype(int)
    plot_confusion_matrix(conmat_closed, ax1, vmax, fontsize_dict, depths=depths)
    ax1.set_title("Closed loop", fontsize=fontsize_dict["label"])
    im = plot_confusion_matrix(conmat_open, ax2, vmax, fontsize_dict, depths=depths)
    ax2.set_title("Open loop", fontsize=fontsize_dict["label"])
    ax1.set_xlabel("Predicted virtual depth (cm)", fontsize=fontsize_dict["label"])
    ax1.set_ylabel("True virtual depth (cm)", fontsize=fontsize_dict["label"])
    ax2.set_ylabel("True virtual depth (cm)", fontsize=fontsize_dict["label"])
    ax1.tick_params(labelsize=fontsize_dict["tick"])
    ax2.tick_params(labelsize=fontsize_dict["tick"])
    bounds = ax2.get_position().bounds
    cbar_ax = fig.add_axes(
        [bounds[0] + bounds[2] + 0.01, bounds[1], 0.01, bounds[3] / 2]
    )
    cbar = fig.colorbar(
        ax=ax2,
        mappable=im,
        cax=cbar_ax,
    )
    cbar_ax.tick_params(labelsize=fontsize_dict["tick"])
    cbar.set_ticks([0, vmax])


def plot_decoder_err_by_speeds(
    decoder_df,
    all_speed_bin_edges,
    axes,
    highest_bin=6,
    closed_loop=1,
    linecolor="k",
    linecolor_chance="r",
    linewidth=1,
    linewidth_chance=1,
    alpha=1,
    alpha_chance=0.5,
    markersize=5,
    fontsize_dict={"title": 10, "label": 5, "tick": 5, "legend": 5},
):
    # find decoder_df for 5 depth session
    use_nbins = len(all_speed_bin_edges) + 1
    speed_bins = all_speed_bin_edges[:use_nbins]
    bin_centers = (speed_bins[1:] + speed_bins[:-1]) / 2
    if closed_loop:
        sfx = "closedloop"
    else:
        sfx = "openloop"
    err_speed_bins = np.zeros((len(decoder_df), use_nbins))
    for i in range(use_nbins):
        err_speed_bins[:, i] = decoder_df[f"error_speed_bins_{sfx}_{i}"]

    # plot stationary
    mean_err = np.nanmean(err_speed_bins[:, 0])
    CI_low, CI_high = common_utils.get_bootstrap_ci(err_speed_bins[:, 0].T)
    axes[0].errorbar(
        x=0,
        y=mean_err,
        c=linecolor,
        yerr=np.concatenate(
            [[mean_err - CI_low.reshape(1)], [CI_high.reshape(1) - mean_err]]
        ),
        marker=".",
        alpha=alpha,
        linewidth=linewidth,
        markeredgewidth=linewidth,
        markersize=markersize,
    )

    # bins that are below the highest bin (<1m/s)
    mean_err = np.nanmean(err_speed_bins[:, 1:highest_bin], axis=0)
    CI_low, CI_high = common_utils.get_bootstrap_ci(err_speed_bins[:, 1:highest_bin].T)
    # bins that are within the highest bin (>1m/s)
    mean_err = np.concatenate(
        [mean_err, [np.nanmean(np.nanmean(err_speed_bins[:, highest_bin:].T, axis=0))]]
    )  # be careful with this array as it has a lot of nans
    CI_low_highest, CI_high_highest = common_utils.get_bootstrap_ci(
        np.nanmean(err_speed_bins[:, highest_bin:].T, axis=0)
    )
    CI_low = np.concatenate([CI_low, CI_low_highest])
    CI_high = np.concatenate([CI_high, CI_high_highest])
    axes[1].plot(
        bin_centers[:highest_bin],
        mean_err.flatten(),
        f"{linecolor}-",
        marker=".",
        alpha=alpha,
        linewidth=linewidth,
        markeredgewidth=linewidth,
        markersize=markersize,
    )

    axes[1].fill_between(
        bin_centers[:highest_bin],
        CI_low,
        CI_high,
        color=linecolor,
        alpha=0.2,
        edgecolor="none",
    )

    # add chance level error
    axes[1].axhline(
        np.nanmean(decoder_df[f"error_chance_{sfx}"]),
        color=linecolor_chance,
        linestyle="dotted",
        linewidth=linewidth,
        alpha=alpha_chance,
    )

    # set axis of the stationary plot
    axes[0].set_xlim([-0.5, 0.5])
    ylim = axes[0].get_ylim()
    axes[0].set_ylim([0, ylim[1]])
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["stationary"], rotation=60)
    axes[0].tick_params(axis="both", labelsize=fontsize_dict["tick"])
    axes[0].set_ylabel(
        r"Mean of $|log_2\frac{predicted\, depth}{true \, depth}|$",
        fontsize=fontsize_dict["label"],
    )
    # set axis of the speed bins plot
    xticks = speed_bins[: (highest_bin + 1)]
    axes[1].set_xticks(speed_bins[: (highest_bin + 1)])
    new_tick_labels = (
        np.round(speed_bins[: (highest_bin + 1)] * 100).astype("int").tolist()
    )
    new_tick_labels[-1] = f"> {np.round(speed_bins[highest_bin-1]*100).astype('int')}"
    axes[1].set_xticklabels(new_tick_labels, rotation=60)
    axes[1].set_xlim([0, speed_bins[highest_bin]])
    axes[1].set_ylim([0, ylim[1]])
    axes[1].tick_params(axis="both", labelsize=fontsize_dict["tick"])
    axes[1].set_xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
    # make y-axis invisible
    sns.despine(ax=axes[0])
    axes[1].spines["left"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].set_yticks([])
    return err_speed_bins


def plot_decoder_acc_by_speeds(
    decoder_df,
    all_speed_bin_edges,
    linecolors=["r", "orange"],
    main_linewidth=1,
    minor_linewidth=1,
    main_alpha=1,
    minor_alpha=0.3,
    markersize=5,
    fontsize_dict={"title": 10, "label": 5, "tick": 5, "legend": 5},
):
    # find decoder_df for 5 depth session
    decoder_df_5depths = decoder_df[
        decoder_df["session"].str.contains(f"PZAH6.4b|PZAG3.4f", regex=True)
    ]
    decoder_df_8depths = decoder_df[
        decoder_df["session"].str.contains(f"PZAH8.2|PZAH10.2", regex=True)
    ]
    use_nbins = len(all_speed_bin_edges) + 1
    speed_bins = all_speed_bin_edges[:use_nbins]
    ax = plt.gca()
    for i, (decoder_df_sub, label) in enumerate(
        zip([decoder_df_5depths, decoder_df_8depths], ["5 depths", "8 depths"])
    ):
        # cut every trial to the same length
        decoder_df_sub["acc_speed_bins_closedloop_cut"] = decoder_df_sub.apply(
            lambda x: x.acc_speed_bins_closedloop[:use_nbins], axis=1
        )
        acc_speed_bins = np.stack(decoder_df_sub.acc_speed_bins_closedloop_cut)
        mean_acc_speed_bins = np.nanmean(acc_speed_bins, axis=0)

        CI_low, CI_high = common_utils.get_bootstrap_ci(acc_speed_bins.T)
        ax.errorbar(
            x=np.arange(1),
            y=mean_acc_speed_bins.flatten()[0],
            yerr=(mean_acc_speed_bins - CI_low)[0],
            fmt=".",
            color=linecolors[i],
            ls="none",
            fillstyle="none",
            linewidth=main_linewidth,
            markeredgewidth=main_linewidth,
            markersize=markersize,
        )

        ax.errorbar(
            x=np.linspace(1, use_nbins - 1, use_nbins - 1),
            y=mean_acc_speed_bins.flatten()[1:use_nbins],
            yerr=(
                (mean_acc_speed_bins - CI_low)[1:use_nbins],
                (CI_high - mean_acc_speed_bins)[1:use_nbins],
            ),
            fmt=".",
            color=linecolors[i],
            ls="-",
            fillstyle="none",
            linewidth=main_linewidth,
            markeredgewidth=main_linewidth,
            markersize=markersize,
            label=label,
        )
        xticks = np.arange(use_nbins).tolist()
        ax.set_xticks(xticks)

        new_tick_labels = []
        for i, tick in enumerate(xticks):
            if i == 0:
                new_tick_labels.append("stationary")
            else:
                new_tick_labels.append(np.round(all_speed_bin_edges[i - 1], 1))
        ax.set_xticklabels(new_tick_labels)
        plt.tick_params(axis="both", labelsize=fontsize_dict["tick"])
    ax.set_xlabel("Running speed (m/s)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Decoder accuracy", fontsize=fontsize_dict["label"])
    plt.legend(frameon=False, fontsize=fontsize_dict["legend"])
    plotting_utils.despine()


# -- OLD --

# def plot_confusion_matrix(conmat, acc, normalize, fontsize_dict, title_sfx=""):
#     # Normalize confusion matrix with true labels
#     if normalize:
#         conmat = conmat / conmat.sum(axis=1)[:, np.newaxis]

#     plt.imshow(conmat, interpolation="nearest", cmap="Blues")
#     fmt = "d"
#     thresh = conmat.max() / 2.0
#     for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
#         display = conmat[i, j]
#         if display >= 1:
#             display = str(int(display))
#         else:
#             display = f"{display:.2f}"
#         plt.text(
#             j,
#             i,
#             display,
#             horizontalalignment="center",
#             color="white" if conmat[i, j] > thresh else "black",
#             fontsize=fontsize_dict["text"],
#         )
#     plt.xlabel("Predicted depth class", fontsize=fontsize_dict["label"])
#     plt.ylabel("True depth class", fontsize=fontsize_dict["label"])
#     plt.title(f"Accuracy {acc:.2f} {title_sfx}", fontsize=fontsize_dict["title"])
#     plt.xticks(fontsize=fontsize_dict["tick"])
#     plt.yticks(fontsize=fontsize_dict["tick"])


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={
        "mem": "32G",
        "time": "18:00:00",
        "partition": "ncpu",
    },
    print_job_id=True,
)
def plot_decoder_session(
    decoder_dict_path,
    save_path,
    session_name,
    project,
    photodiode_protocol,
    speed_bins,
):
    flexilims_session = flz.get_flexilims_session(project)
    _, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    with open(decoder_dict_path, "rb") as f:
        decoder_dict = pickle.load(f)
    # Plot confusion matrix
    print("Plotting confusion matrices...")
    plt.figure()
    for i, closed_loop in enumerate(np.sort(trials_df_all.closed_loop.unique())):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        plt.subplot(1, 2, i + 1)
        plot_confusion_matrix(
            decoder_dict[f"conmat{sfx}"],
            decoder_dict[f"accuracy{sfx}"],
            normalize=True,
            fontsize_dict={"text": 10, "label": 10, "title": 10, "tick": 5},
        )
        plt.title(sfx[1:], fontsize=10)
    os.makedirs(Path(save_path) / "plots" / "depth_decoder", exist_ok=True)
    plt.savefig(
        Path(save_path) / "plots" / "depth_decoder" / "confusion_matrix.png",
        dpi=300,
    )
    print("Confusion matrix plotted.")

    plt.figure(figsize=(3 * 2, 3 * (len(speed_bins) + 1)))
    for i, closed_loop in enumerate(np.sort(trials_df_all.closed_loop.unique())):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        for ispeed, speed_bin in enumerate(speed_bins):
            if ispeed == 0:
                title_sfx = f"still{sfx}"
            else:
                title_sfx = f"speed{sfx} {speed_bins[ispeed-1]:.1f}-{speed_bin:.1f} m/s"
            if len(decoder_dict[f"conmat_speed_bins{sfx}"][ispeed]) > 0:
                plt.subplot2grid((len(speed_bins) + 1, 2), (ispeed, i))
                plot_confusion_matrix(
                    decoder_dict[f"conmat_speed_bins{sfx}"][ispeed],
                    decoder_dict[f"acc_speed_bins{sfx}"][ispeed],
                    normalize=True,
                    fontsize_dict={"text": 5, "label": 5, "title": 10, "tick": 5},
                    title_sfx=title_sfx,
                )
        for speed_bin in [speed_bins[-1]]:
            if len(decoder_dict[f"conmat_speed_bins{sfx}"][-1]) > 0:
                title_sfx = f"speed > {speed_bin:.1f} m/s"
                plt.subplot2grid((len(speed_bins) + 1, 2), (len(speed_bins), i))
                plot_confusion_matrix(
                    decoder_dict[f"conmat_speed_bins{sfx}"][-1],
                    decoder_dict[f"acc_speed_bins{sfx}"][-1],
                    normalize=True,
                    fontsize_dict={"text": 5, "label": 5, "title": 10, "tick": 5},
                    title_sfx=title_sfx,
                )
    plt.tight_layout()
    plt.savefig(
        Path(save_path)
        / "plots"
        / "depth_decoder"
        / f"confusion_matrix_speed_bins.png",
        dpi=300,
    )
    print("Confusion matrix for different speed bins plotted.")
