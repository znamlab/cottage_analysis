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

from cottage_analysis.plotting import plotting_utils


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
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):

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
    neurons_df, sizes, plot_x, plot_y, plot_width, plot_height, fontsize_dict
):
    fig = plt.gcf()
    for i, (size_x, size_y) in enumerate(
        zip([sizes[0], sizes[0], sizes[1]], [sizes[1], sizes[2], sizes[2]])
    ):
        ax = fig.add_axes(
            [plot_x + i * plot_width, plot_y, plot_width * 0.6, plot_height * 0.6]
        )
        ax.scatter(
            neurons_df[f"preferred_depth_size{size_x}"],
            neurons_df[f"preferred_depth_size{size_y}"],
            s=1,
            c="k",
            alpha=0.2,
        )
        xlim = ax.get_xlim()
        # add diagonal line
        ax.plot(xlim, xlim, "k", linestyle="dotted", linewidth=1)
        # set labels
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
        sns.despine()

        # add histogram
        ax2 = fig.add_axes(
            [
                plot_x + (i + 0.1) * plot_width,
                plot_y + plot_height * 0.8,
                plot_width * 0.5,
                plot_height * 0.3,
            ]
        )
        ratio = (
            neurons_df[f"preferred_depth_size{size_x}"]
            / neurons_df[f"preferred_depth_size{size_y}"]
        )
        n, _, _ = ax2.hist(
            ratio,
            bins=np.geomspace(1e-2, 1e2, 21),
            color="k",
            alpha=1,
        )
        ax2.plot(
            np.median(ratio),
            150,
            marker="v",
            markersize=3,
            color="k",
        )
        ax2.vlines(1, 0, 150, color="white", linestyle="dotted", linewidth=1)
        ax2.set_xscale("log")
        ax2.set_xlim(1e-2, 1e2)
        ax2.set_xticks([0.01, 0.1, 1, 10, 100])
        ax2.set_ylim(0, 160)
        ax2.set_yticks([0, 160])
        ax2.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax2.set_ylabel("Number of neurons", fontsize=fontsize_dict["tick"])
        sns.despine()
        print(
            f"spearmarnr {spearmanr(neurons_df[f'preferred_depth_size{size_x}'], neurons_df[f'preferred_depth_size{size_y}'])},\
              median {np.median(ratio.values)}"
        )
