import os
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import scipy
from cottage_analysis.plotting import plotting_utils
from cottage_analysis.analysis import (
    find_depth_neurons,
    common_utils,
    fit_gaussian_blob,
    size_control,
)
from v1_depth_map.figure_utils import closed_loop_rsof


# REPLACE
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
            rgba_color = plotting_utils.get_color(
                value=neurons_df.loc[n, "preferred_depth_closed_loop"],
                value_min=np.min(depth_list),
                value_max=np.max(depth_list),
                cmap=cmap,
                log=True,
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


def add_colorbar():
    cbar_pos = [
        1.02,
        plt.gca().get_position().y0,
        0.02,
        plt.gca().get_position().height,
    ]
    plt.axes(cbar_pos)
    plt.colorbar(cax=plt.gca(), label="\u0394F/F")


def basic_vis_session(neurons_df, trials_df, neurons_ds, **kwargs):
    rois = neurons_df.roi.values
    for is_closedloop in np.sort(trials_df.closed_loop.unique()):
        if is_closedloop:
            sfx = "closedloop"
        else:
            sfx = "openloop"
        os.makedirs(
            neurons_ds.path_full.parent / "plots" / f"basic_vis_{sfx}", exist_ok=True
        )

        plot_rows = 10
        plot_cols = 11

        params = dict(
            rs_thr=0.2,
            rs_curve=dict(speed_min=0.001, speed_max=1, nbins=10, speed_thr=0.001),
        )
        params.update(kwargs)
        for i in tqdm(range(int(len(rois) // plot_rows + 1))):
            if i * plot_rows < len(rois) - 1:
                fig = plt.figure(figsize=(3 * plot_cols, 3 * plot_rows))
                for iroi, roi in enumerate(
                    rois[i * plot_rows : np.min([(i + 1) * plot_rows, len(rois)])]
                ):
                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 0))
                    plot_depth_tuning_curve(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        rs_thr=None,
                        plot_fit=is_closedloop,
                        linewidth=3,
                        linecolor="k",
                        fit_linecolor="r",
                        closed_loop=is_closedloop,
                        use_col="depth_tuning_popt_closedloop",
                    )
                    plt.title(f"roi{roi}")
                    
                    # plt.subplot2grid((plot_rows, plot_cols), (iroi, 1))
                    # plot_depth_tuning_curve(
                    #     neurons_df=neurons_df,
                    #     trials_df=trials_df,
                    #     roi=roi,
                    #     rs_thr=0.05,
                    #     plot_fit=is_closedloop,
                    #     linewidth=3,
                    #     linecolor="k",
                    #     fit_linecolor="r",
                    #     closed_loop=is_closedloop,
                    #     use_col="depth_tuning_popt_closedloop_running",
                    # )
                    
                    # plt.subplot2grid((plot_rows, plot_cols), (iroi, 2))
                    # plot_depth_tuning_curve(
                    #     neurons_df=neurons_df,
                    #     trials_df=trials_df,
                    #     roi=roi,
                    #     rs_thr=None,
                    #     rs_thr_max=0.05,
                    #     still_only=True,
                    #     still_time=1,
                    #     frame_rate=15,
                    #     plot_fit=is_closedloop,
                    #     linewidth=3,
                    #     linecolor="k",
                    #     fit_linecolor="r",
                    #     closed_loop=is_closedloop,
                    #     use_col="depth_tuning_popt_closedloop_notrunning",
                    # )
                    

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 3))
                    plot_speed_tuning(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        is_closed_loop=is_closedloop,
                        which_speed="RS",
                        smoothing_sd=1,
                        **params["rs_curve"],
                    )

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 4))
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

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 5))
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

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 6))
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
                    extended_matrix = plot_RS_OF_matrix(
                        trials_df=trials_df[trials_df.closed_loop == is_closedloop],
                        roi=roi,
                        log_range=log_range,
                    )
                    
                    models = ["gof", "gadd", "g2d", "gratio"]
                    model_labels = ["OF only", "Additive", "Conjunctive", "Pure depth"]
                    for imodel, (model, model_label) in enumerate(zip(models, model_labels)):
                        if imodel == 0:
                            ylabel = "Optic flow speed (degrees/s)"
                        else:
                            ylabel = ""
                        if imodel == 1:
                            xlabel = "Running speed (cm/s)"
                        else:
                            xlabel = ""

                        ax = plt.subplot2grid((plot_rows, plot_cols), (iroi, 7+imodel), fig=fig)
                        vmin = np.nanmax([0, np.percentile(extended_matrix[1:, 1:].flatten(), 1)])
                        vmax = np.nanmax(extended_matrix[1:, 1:].flatten())
                        plot_RS_OF_fit(
                            fig=fig,
                            ax=ax,
                            neurons_df=neurons_df,
                            roi=roi,
                            model=model,
                            model_label=model_label,
                            min_sigma=0.25,
                            vmin=vmin,
                            vmax=vmax,
                            log_range={
                                "rs_bin_log_min": 0,
                                "rs_bin_log_max": 2.5,
                                "rs_bin_num": 6,
                                "of_bin_log_min": -1.5,
                                "of_bin_log_max": 3.5,
                                "of_bin_num": 11,
                                "log_base": 10,
                            },
                            plot_x=0.24 + 0.1 * imodel,
                            plot_y=0.64 - 0.43 * iroi,
                            plot_width=0.15,
                            plot_height=0.15,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            fontsize_dict={"title": 5, "label": 5, "tick": 5},
                            cbar_width=None,
                        )
                        if imodel > 0:
                            plt.gca().set_yticklabels([])

                plt.savefig(
                    neurons_ds.path_full.parent
                    / "plots"
                    / f"basic_vis_{sfx}"
                    / f"roi{rois[i*10]}- {np.min([(i+1)*10, len(rois)])}.png",
                    dpi=100,
                )

                plt.close()



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
    trials_df = size_control.get_physical_size(
        trials_df, use_cols=["size", "depth"], k=1
    )
    os.makedirs(
        neurons_ds.path_full.parent / "plots" / f"size_control_basic_vis", exist_ok=True
    )

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
                        trials_df=trials_df[trials_df["size"] == size],
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
