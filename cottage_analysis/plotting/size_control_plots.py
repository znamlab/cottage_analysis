import numpy as np
import pandas as pd
import scipy
import flexiznam as flz
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter

from cottage_analysis.plotting import plotting_utils
from cottage_analysis.plotting.style import FONTSIZE_DICT


def plot_depth_size_fit_comparison(
    fig,
    neurons_df,
    filter=None,
    use_cols={
        "depth_fit_r2": "depth_tuning_test_rsq_closedloop",
        "size_fit_r2": "size_tuning_test_rsq_closedloop",
        "depth_fit_pval": "depth_tuning_test_spearmanr_pval_closedloop",
        "size_fit_pval": "size_tuning_test_spearmanr_pv,al_closedloop",
    },
    plot_type="scatter",
    s=5,
    c="k",
    alpha=0.5,
    nbins=20,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict=None,
):
    if fontsize_dict is None:
        fontsize_dict = FONTSIZE_DICT

    # Plot scatter of r2 depth vs. size
    if filter is None:
        filtered_neurons_df = neurons_df
    else:
        filtered_neurons_df = neurons_df[filter]

    print(
        scipy.stats.wilcoxon(
            filtered_neurons_df[use_cols["depth_fit_r2"]],
            filtered_neurons_df[use_cols["size_fit_r2"]],
        )
    )
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])

    if plot_type == "scatter":
        ax.scatter(
            filtered_neurons_df[use_cols["depth_fit_r2"]],
            filtered_neurons_df[use_cols["size_fit_r2"]],
            s=s,
            c=c,
            alpha=alpha,
            edgecolors="none",
        )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], c="k", linestyle="dotted", linewidth=1)
        ax.set_aspect("equal")

        ax.set_xlabel("Depth fit r-squared", fontsize=fontsize_dict["label"])
        ax.set_ylabel("Size fit r-squared", fontsize=fontsize_dict["label"])
        # ax.set_xscale("log")
        # ax.set_yscale("log")

    elif plot_type == "hist":
        diff = (
            filtered_neurons_df[use_cols["depth_fit_r2"]]
            - filtered_neurons_df[use_cols["size_fit_r2"]]
        )
        weights = np.ones_like(diff) / len(diff)
        ax.hist(diff, bins=nbins, color=c, alpha=alpha, weights=weights)
        ax.set_xlabel(
            "Difference between depth and \nsize tuning r-squared",
            fontsize=fontsize_dict["label"],
        )
        ax.set_ylabel("Proportion of neurons", fontsize=fontsize_dict["label"])
        ylim = ax.get_ylim()
        ax.vlines(0, 0, ylim[1], color="r", linestyle="dotted", linewidth=1)
        ax.set_title(
            f"median {np.median(diff):.4f}, p = {scipy.stats.wilcoxon(diff)[1]:.2e}",
            fontsize=fontsize_dict["title"],
        )
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        print(f"median {np.median(diff)}")

    plotting_utils.despine()


def plot_preferred_depths_sizes_scatter(
    neurons_df,
    sizes,
    plot_x,
    plot_y,
    plot_width,
    plot_height,
    fontsize_dict=None,
    max_y=80,
):
    if fontsize_dict is None:
        fontsize_dict = FONTSIZE_DICT
    fig = plt.gcf()
    for i, (size_x, size_y) in enumerate(
        zip([sizes[0], sizes[0], sizes[1]], [sizes[1], sizes[2], sizes[2]])
    ):
        ax = fig.add_axes(
            [plot_x + i * plot_width, plot_y, plot_width * 0.58, plot_height * 0.58]
        )
        ax.scatter(
            neurons_df[f"preferred_depth_size{size_x}"],
            neurons_df[f"preferred_depth_size{size_y}"],
            s=5,
            c="k",
            alpha=0.25,
            edgecolors="none",
        )
        xlim = (0.02, 50)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.plot(xlim, xlim, "k:", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(
            f"Preferred depth with \n{size_x} degree spheres (cm)",
            fontsize=fontsize_dict["label"],
            labelpad=1,
        )
        ax.set_ylabel(
            f"Preferred depth with \n{size_y} degree spheres (cm)",
            fontsize=fontsize_dict["label"],
            labelpad=1,
        )
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax.set_aspect("equal")
        sns.despine(ax=ax)

        # Add diagonal histogram
        min_x = -2
        max_x = 2
        span_x = max_x - min_x
        aspect_ratio_hist = 0.45
        tr = Affine2D().scale(1, (span_x * aspect_ratio_hist) / max_y).rotate_deg(-45)

        ticks_x = [-2, -1, 0, 1, 2]
        tick_labels_x = {t: f"$10^{{{t}}}$" for t in ticks_x}
        grid_locator1 = FixedLocator(ticks_x)
        tick_formatter1 = DictFormatter(tick_labels_x)

        ticks_y = [0, max_y]
        tick_labels_y = {0: "0", max_y: f"{max_y}"}
        grid_locator2 = FixedLocator(ticks_y)
        tick_formatter2 = DictFormatter(tick_labels_y)

        grid_helper = floating_axes.GridHelperCurveLinear(
            tr,
            extremes=(min_x, max_x, 0, max_y),
            grid_locator1=grid_locator1,
            tick_formatter1=tick_formatter1,
            grid_locator2=grid_locator2,
            tick_formatter2=tick_formatter2,
        )

        hist_w = plot_width * 0.48
        hist_h = plot_height * 0.48
        hist_x = plot_x + i * plot_width + plot_width * 0.39
        hist_y = plot_y + plot_height * 0.39

        ax_hist = floating_axes.FloatingAxes(
            fig,
            [hist_x, hist_y, hist_w, hist_h],
            grid_helper=grid_helper,
        )
        fig.add_axes(ax_hist)

        ax_hist.axis["bottom"].set_visible(True)
        ax_hist.axis["left"].set_visible(True)
        ax_hist.axis["top"].set_visible(False)
        ax_hist.axis["right"].set_visible(False)

        aux_ax = ax_hist.get_aux_axes(tr)

        ratio = (
            neurons_df[f"preferred_depth_size{size_x}"]
            / neurons_df[f"preferred_depth_size{size_y}"]
        )
        log_ratio = np.log10(ratio)
        bins = np.linspace(min_x, max_x, 21)
        counts, bin_edges = np.histogram(log_ratio, bins=bins)

        width = bins[1] - bins[0]
        aux_ax.bar(
            bins[:-1], counts, width=width, align="edge", color="k", edgecolor="none"
        )
        aux_ax.vlines(0, 0, max_y, color="white", linestyle="dotted", linewidth=1)
        med_val = np.log10(np.median(ratio))
        aux_ax.plot(
            med_val,
            max_y + (max_y * 0.08),
            marker=(3, 0, 15),
            markersize=3.5,
            color="k",
            clip_on=False,
        )

        ax_hist.axis["left"].label.set_text("Number of neurons")
        ax_hist.axis["left"].label.set_fontsize(fontsize_dict["tick"])
        ax_hist.axis["left"].major_ticklabels.set_fontsize(fontsize_dict["tick"])
        ax_hist.axis["bottom"].major_ticklabels.set_fontsize(fontsize_dict["tick"])

        print(
            f"spearmarnr {spearmanr(neurons_df[f'preferred_depth_size{size_x}'], neurons_df[f'preferred_depth_size{size_y}'])},\
              median {np.median(ratio.values)}"
        )
