import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib import cm
from sklearn.metrics import mutual_info_score
from typing import Sequence, Dict, Any
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm

from cottage_analysis.analysis.common_utils import get_confidence_interval


def get_color(value, value_min, value_max, log=False, cmap=cm.cool.reversed()):
    if log:
        norm = mpl.colors.Normalize(
            vmin=np.log(value_min), vmax=np.log(value_max)
        )
        rgba_color = cmap(norm(np.log(value)), bytes=True)
    else:
        norm = mpl.colors.Normalize(vmin=value_min, vmax=value_max)
        rgba_color = cmap(norm(value), bytes=True)
    rgba_color = np.array([it / 255 for it in rgba_color])

    return rgba_color


def get_unique_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    return unique.values(), unique.keys()


def draw_axis_scalebars(
    ax,
    scalebar_x,
    scalebar_y,
    scalebar_width,
    scalebar_height,
    scalebar_labels,
    xlim=None,
    ylim=None,
    label_fontsize=5,
    linewidth=1,
    right=True,
    bottom=True,
):
    rect = patches.Rectangle(
        (scalebar_x, scalebar_y),
        scalebar_width,
        scalebar_height,
        linewidth=linewidth,
        edgecolor="none",
        facecolor="none",
    )
    ax.add_patch(rect)
    if right:
        right_edge = patches.FancyBboxPatch(
            (scalebar_x + scalebar_width, scalebar_y),
            0,
            scalebar_height,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(right_edge)
        ax.text(
            scalebar_x + scalebar_width * 1.2,
            scalebar_y + scalebar_height / 2,
            scalebar_labels[1],
            fontsize=label_fontsize,
            ha="left",
            va="center",
        )
    ax.set_ylim(ylim)
    if bottom:
        bottom_edge = patches.FancyBboxPatch(
            (scalebar_x, scalebar_y),
            scalebar_width,
            0,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(bottom_edge)
        ax.text(
            scalebar_x,
            scalebar_y + scalebar_height * 0.1,
            scalebar_labels[0],
            fontsize=label_fontsize,
            ha="left",
            va="bottom",
        )
    ax.set_xlim(xlim)
    ax.axis("off")


def plot_white_rectangle(x0, y0, width, height):
    ax = plt.gcf().add_axes([x0, y0, width, height])
    # Define the rectangle's bottom-left corner, width, and height
    rectangle = patches.Rectangle((0, 0), 1, 1, edgecolor="white", facecolor="white")

    # Add the rectangle to the plot
    ax.add_patch(rectangle)

    # Set plot limits to better visualize the rectangle
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_aspect("equal")
    # fig.patch.set_facecolor('gray')


# --- OLD ---
def segment_arr(arr_idx, segment_size):
    """
    Segment an input array into small chunks with defined size.

    :param arr_idx: np.ndarray. 1D. Array of indices of a target array.
    :param segment_size: int. Number of elements desired for each segment.
    :return: segment_starts: list. List of starting indices for all segments.
             segment_ends: list. List of end indices for all segments.
    """
    batch_num = len(arr_idx) // segment_size
    segment_starts = np.arange(0, batch_num * segment_size + 1, segment_size)
    segment_ends = np.arange(
        segment_size, batch_num * segment_size + segment_size, segment_size
    )
    if len(arr_idx) % segment_size != 0:
        segment_ends = np.concatenate((segment_ends, (arr_idx[-1] + 1).reshape(-1)))
    segment_starts = (segment_starts + arr_idx[0])[: len(segment_ends)]
    segment_ends = segment_ends + arr_idx[0]
    return segment_starts, segment_ends


def plot_raster(
    arr,
    vmin,
    vmax,
    cmap,
    fontsize_dict,
    blank_period=5,
    frame_rate=30,
    title=None,
    suffix=None,
    title_on=False,
    extent=[],
    set_nan_cmap=True,
    colorbar_on=True,
    ax=None,
):
    """
    Raster plot of input params. Row: trials. Column: time.

    :param arr: np.ndarray. 2D array, trials x time.
    :param vmin: vmin for plt.imshow heatmap.
    :param vmax: vmax for plt.imshow heatmap.
    :param cmap: color map object.
    :param blank_period: float, blank period in seconds.
    :param frame_rate: int, frame rate for imaging.
    :param title: str, title of the plots.
    :param suffix: str, suffix added to the end of the title.
    :param title_on: bool, whether put title + suffix on or only put the suffix.
    :param fontsize: int, fontsize of title.
    :param extent: list, [extent_x_min, extent_x_max, extent_y_min, extent_y_max]. Min/max values for heatmap x and y axes.
    :param set_nan_cmap: bool, whether to set nan values in colormap as a different colour (silver)
    """
    current_cmap = mpl.cm.get_cmap(cmap).copy()
    if set_nan_cmap:
        current_cmap.set_bad(color="silver")
    if len(extent) > 0:
        extent = extent
    else:
        extent = [
            -blank_period,
            -blank_period + arr.shape[1] / frame_rate,
            arr.shape[0],
            1,
        ]
    plt.imshow(
        arr,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=current_cmap,
        extent=extent,
        rasterized=False,
        interpolation="none",
    )
    ax = plt.gca()
    if title_on:
        plt.title(title + " " + suffix, fontsize=fontsize_dict["title"])
    else:
        plt.title(suffix, fontsize=fontsize_dict["title"])
    plt.ylim([arr.shape[0], 1])
    return ax


def plot_trial_onset_offset(onset, offset, ymin, ymax):
    """
    Plot a dashed line for trial onset and offset (based on x axis) on an existing plot.

    :param onset: float, onset time in whatever unit x axis of existing plot is.
    :param offset: float, onset time in whatever unit y axis of existing plot is.
    :param ymin: float, min value where dashed line starts at y axis.
    :param ymax: float, max value where dashed line starts at y axis.
    :return: None.
    """
    plt.vlines(
        x=onset,
        ymin=ymin,
        ymax=ymax,
        linestyles="dashed",
        colors="k",
        alpha=0.5,
        linewidth=1,
        rasterized=False,
    )
    plt.vlines(
        x=offset,
        ymin=ymin,
        ymax=ymax,
        linestyles="dashed",
        colors="k",
        alpha=0.5,
        linewidth=1,
        rasterized=False,
    )


def plot_line_with_error(
    arr,
    CI_low,
    CI_high,
    linecolor,
    label=None,
    marker="-",
    markersize=None,
    xarr=[],
    xlabel=None,
    ylabel=None,
    title_on=False,
    title=None,
    suffix=None,
    fontsize=10,
    axis_fontsize=10,
    linewidth=0.5,
    ax=None,
):
    if ax == None:
        if len(xarr) == 0:
            plt.plot(
                arr,
                marker,
                c=linecolor,
                linewidth=linewidth,
                label=label,
                alpha=1,
                markersize=markersize,
                rasterized=False,
            )
            plt.fill_between(
                np.arange(len(arr)),
                CI_low,
                CI_high,
                color=linecolor,
                alpha=0.3,
                edgecolor=None,
                rasterized=False,
            )
        else:
            plt.plot(
                xarr,
                arr,
                marker,
                c=linecolor,
                linewidth=linewidth,
                label=label,
                alpha=1,
                markersize=markersize,
                rasterized=False,
            )
            plt.fill_between(
                xarr,
                CI_low,
                CI_high,
                color=linecolor,
                alpha=0.3,
                edgecolor=None,
                rasterized=False,
            )
        plt.xlabel(xlabel, fontsize=axis_fontsize)
        plt.ylabel(ylabel, fontsize=axis_fontsize)
        if title_on:
            plt.title(title + " " + suffix, fontsize=fontsize)
        else:
            plt.title(suffix, fontsize=fontsize)
    else:
        if len(xarr) == 0:
            ax.plot(
                arr,
                marker,
                c=linecolor,
                linewidth=linewidth,
                label=label,
                alpha=1,
                markersize=markersize,
                rasterized=False,
            )
            ax.fill_between(
                np.arange(len(arr)),
                CI_low,
                CI_high,
                color=linecolor,
                alpha=0.3,
                edgecolor=None,
                rasterized=False,
            )
        else:
            ax.plot(
                xarr,
                arr,
                marker,
                c=linecolor,
                linewidth=linewidth,
                label=label,
                alpha=1,
                markersize=markersize,
                rasterized=False,
            )
            ax.fill_between(
                xarr,
                CI_low,
                CI_high,
                color=linecolor,
                alpha=0.3,
                edgecolor=None,
                rasterized=False,
            )
        ax.set_xlabel(xlabel, fontsize=axis_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        if title_on:
            ax.set_title(title + " " + suffix, fontsize=fontsize)
        else:
            ax.set_title(suffix, fontsize=fontsize)


def plot_scatter(
    x,
    y,
    xlim,
    ylim,
    markercolor,
    xlabel=None,
    ylabel=None,
    title_on=False,
    title=None,
    label=None,
):
    plt.plot(x, y, "o", markersize=3, c=markercolor, rasterized=False, label=label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if title_on:
        plt.title(title)


def get_binned_stats(xarr, yarr, bin_number):
    binned_stats = {
        "bin_edge_min": [],
        "bin_edge_max": [],
        "bins": [],
        "bin_means": np.zeros((xarr.shape[0], bin_number)),
        "bin_stds": np.zeros((xarr.shape[0], bin_number)),
        "bin_counts": np.zeros((xarr.shape[0], bin_number)),
        "bin_centers": np.zeros((xarr.shape[0], bin_number)),
        "bin_edges": np.zeros((xarr.shape[0], bin_number + 1)),
        "binnumber": [],
    }
    binned_stats["bin_edge_min"] = np.round(np.nanmin(xarr), 1)
    binned_stats["bin_edge_max"] = np.round(np.nanmax(xarr), 1) + 0.1
    binned_stats["bins"] = np.linspace(
        start=binned_stats["bin_edge_min"],
        stop=binned_stats["bin_edge_max"],
        num=bin_number + 1,
        endpoint=True,
    )

    for i in range(xarr.shape[0]):
        bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(
            xarr[i].flatten(),
            yarr[i].flatten(),
            statistic="mean",
            bins=binned_stats["bins"],
        )
        bin_stds, _, _ = scipy.stats.binned_statistic(
            xarr[i].flatten(),
            yarr[i].flatten(),
            statistic="std",
            bins=binned_stats["bins"],
        )
        bin_counts, _, _ = scipy.stats.binned_statistic(
            xarr[i].flatten(),
            yarr[i].flatten(),
            statistic="count",
            bins=binned_stats["bins"],
        )
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        binned_stats["bin_means"][i, :] = bin_means
        binned_stats["bin_stds"][i, :] = bin_stds
        binned_stats["bin_counts"][i, :] = bin_counts
        binned_stats["bin_centers"][i, :] = bin_centers
        binned_stats["bin_edges"][i, :] = bin_edges
        binned_stats["binnumber"].append(binnumber.reshape(xarr.shape[1], -1))

    return binned_stats


def get_binned_arr(xarr, yarr, bin_number, bin_edge_min=0, bin_edge_max=6):
    """
    assume xarr and yarr are 3D arrays
    """
    binned_stats = {}
    binned_yrr = np.zeros((yarr.shape[0], yarr.shape[1], bin_number))
    bins = np.linspace(
        start=bin_edge_min, stop=bin_edge_max, num=bin_number + 1, endpoint=True
    )

    for i in range(xarr.shape[0]):
        for j in range(xarr.shape[1]):
            bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(
                x=xarr[i, j, :][~np.isnan(xarr[i, j, :])],
                values=yarr[i, j, :][~np.isnan(yarr[i, j, :])],
                statistic="mean",
                bins=bins,
            )
            binned_yrr[i, j, :] = bin_means
    binned_stats["bins"] = bins
    binned_stats["binned_yrr"] = binned_yrr

    return binned_stats


def get_tuning_function(means, counts, smoothing_sd=1):
    totals = means * counts
    valid = ~np.isnan(means)
    totals_smooth = scipy.ndimage.gaussian_filter1d(
        totals[valid], sigma=smoothing_sd, mode="nearest"
    )
    occupancy_smooth = scipy.ndimage.gaussian_filter1d(
        counts[valid], sigma=smoothing_sd, mode="nearest"
    )
    tuning = np.zeros(means.shape)
    tuning[:] = np.nan
    tuning[valid] = totals_smooth / occupancy_smooth

    return tuning


def get_binned_stats_2d(
    xarr1,
    xarr2,
    yarr,
    bin_number=None,
    bin_size=None,
    bin_edges=[],
    log=False,
    log_base=np.e,
):
    if len(bin_edges) > 0:
        binned_stats = {
            "bin_edge_min": [],
            "bin_edge_max": [],
            "bins": [],
            "bin_means": [],
            #             'bin_sems':np.zeros((bin_number-1,bin_number-1)),
            #             'bin_counts':np.zeros((bin_number-1,bin_number-1)),
            "bin_centers": [],
            "bin_edges": [],
            "bin_numbers": [],
        }
        if log == True:
            bin_edges[0] = np.array(bin_edges[0])
            bin_edges[1] = np.array(bin_edges[1])
        binned_stats["bin_edge_min"].append(np.nanmin(bin_edges[0]))
        binned_stats["bin_edge_min"].append(np.nanmin(bin_edges[1]))
        binned_stats["bin_edge_max"].append(np.nanmax(bin_edges[0]))
        binned_stats["bin_edge_max"].append(np.nanmax(bin_edges[1]))
        binned_stats["bins"] = bin_edges
        binned_stats["bin_edges"].append(bin_edges[0])
        binned_stats["bin_edges"].append(bin_edges[1])
        binned_stats["bin_means"], _, _, _ = scipy.stats.binned_statistic_2d(
            xarr1.flatten(),
            xarr2.flatten(),
            yarr.flatten(),
            statistic="mean",
            bins=binned_stats["bins"],
            expand_binnumbers=True,
        )

        for iaxis in range(2):
            bin_width = (
                binned_stats["bin_edges"][iaxis][1]
                - binned_stats["bin_edges"][iaxis][0]
            )
            binned_stats["bin_centers"].append(
                binned_stats["bin_edges"][iaxis][1:] - bin_width / 2
            )  # THIS IS WRONG!! THIS DOESN'T TAKE INTO ACCOUNT THE LOGGED NUMBER
            binned_stats["bin_numbers"].append(
                len(binned_stats["bin_edges"][iaxis][1:] - bin_width / 2)
            )

    return binned_stats


def set_RS_OF_heatmap_axis_ticks(binned_stats):
    bin_numbers = binned_stats["bin_numbers"]
    bin_edges1 = np.array(binned_stats["bin_edges"][0], dtype="object")
    bin_edges2 = np.array(binned_stats["bin_edges"][1], dtype="object")
    ctr = 0
    for it in bin_edges1:
        if it >= 1:
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
    ticks_select1 = (np.arange(-1, bin_numbers[0] * 2, 1) / 2)[0::2]
    ticks_select2 = (np.arange(-1, bin_numbers[1] * 2, 1) / 2)[0::2]
    _, _ = plt.xticks(ticks_select1, bin_edges1, rotation=60, ha="center")
    _, _ = plt.yticks(ticks_select2, bin_edges2)


def get_mutual_info(arr1, arr2):
    arr1_mask = ~np.isnan(arr1)
    arr2_mask = ~np.isnan(arr2)
    mutual_mask = arr1_mask * arr2_mask
    arr1 = arr1[mutual_mask]
    arr2 = arr2[mutual_mask]
    mutual_info = mutual_info_score(arr1, arr2)

    return mutual_info


def despine():
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


# Make an array for ROI mask for each ROI
def stats_to_array(
    stats: Sequence[Dict[str, Any]], Ly: int, Lx: int, label_id: bool = False
):
    """
    converts stats sequence of dictionaries to an array
    :param stats: sequence of dictionaries from stat.npy
    :param Ly: number of pixels along dim Y from ops dictionary
    :param Lx: number of pixels along dim X
    :param label_id: keeps ROI indexing
    :return: numpy stack of arrays, each containing x and y pixels for each ROI
    """
    arrays = []
    for i, stat in enumerate(stats):
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[stat["ypix"], stat["xpix"]] = 1
        if label_id:
            arr *= i + 1
        arrays.append(arr)
    return np.stack(arrays)


def find_roi_center(cells_mask, roi):
    x_min = np.min(np.where(cells_mask[roi] > 0)[0])
    x_max = np.max(np.where(cells_mask[roi] > 0)[0])
    y_min = np.min(np.where(cells_mask[roi] > 0)[1])
    y_max = np.max(np.where(cells_mask[roi] > 0)[1])
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    radius = (x_range + y_range) / 2 / 2

    return int(x_center), int(y_center), int(radius)


def plot_sta(sta, extent=[-120, 120, -40, 40], clim=None):
    ndepths = sta.shape[0]
    fig, axes = plt.subplots(ndepths, 1, sharex="all", sharey="all", figsize=(5, 7.5))
    ims = []
    if not clim:
        clim = np.max(sta)
    for idepth in range(ndepths):
        ims.append(
            axes[idepth].imshow(
                sta[idepth].T,
                extent=extent,
                cmap="RdBu_r",
                vmin=-clim,
                vmax=clim,
            )
        )
        plt.xticks([extent[0], 0, extent[1]])
        plt.yticks([extent[2], 0, extent[3]])

    fig.supylabel("Elevation (degrees)")
    fig.supxlabel("Azimuth (degrees)")
    return fig, axes, ims


def linear_regression(X, y, x2, model=LinearRegression()):
    reg = model.fit(X, y)
    y_pred = reg.predict(x2)
    return reg.score(X, y), reg.coef_, reg.intercept_, y_pred


def scatter_plot_fit_line(
    X,
    y,
    x2,
    xlabel,
    ylabel,
    n_boots=10000,
    s=1,
    alpha=0.8,
    c="k",
    label=None,
    log=True,
    fit_line=True,
    model=LinearRegression(),
):
    if log:
        score, coef, intercept, y_pred_original = linear_regression(
            X=np.log(X), y=np.log(y), x2=np.log(x2), model=model
        )
        # print(coef, intercept)
        plt.scatter(X, y, s=s, alpha=alpha, c=c, label=label, edgecolors="none")

        y_pred_exp = []
        if fit_line:
            for _ in range(n_boots):
                sample_index = np.random.choice(range(0, len(X)), len(X))

                X_samples = X[sample_index]
                y_samples = y[sample_index]

                score, coef, intercept, y_pred = linear_regression(
                    X=np.log(X_samples), y=np.log(y_samples), x2=np.log(x2), model=model
                )
                y_pred_exp.append(np.exp(y_pred))
            y_pred_exp = np.array(y_pred_exp)
            lower_CI = np.percentile(y_pred_exp, 2.5, axis=0)
            higher_CI = np.percentile(y_pred_exp, 97.5, axis=0)
            plt.plot(x2, np.exp(y_pred_original), color=c, linewidth=1)
            plt.fill_between(
                x=x2.reshape(-1),
                y1=lower_CI.reshape(-1),
                y2=higher_CI.reshape(-1),
                color=c,
                alpha=0.25,
                zorder=0.01,
                edgecolor=None,
            )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale("log")
        plt.yscale("log")
        despine()


def set_aspect_ratio(ax, ratio=1):
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)


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
