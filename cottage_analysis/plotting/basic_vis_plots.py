import os
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from cottage_analysis.plotting import plotting_utils, rsof_plots
from cottage_analysis.plotting import depth_selectivity_plots as dsp
from cottage_analysis.analysis import (
    find_depth_neurons,
    fit_gaussian_blob,
    size_control,
)


# REPLACE?
def plot_spatial_distribution(
    neurons_df, trials_df, ops, stat, iscell, cmap=cm.cool.reversed()
):
    """
        Plot spatial distribution of depth preference of a session.

    #     Args:
    #         neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
    #         trials_df (pd.DataFrame): dataframe with info of all trials.
    #         ops (np.ndarray): suite2p ops.
    #         stat (np.ndarray): suite2p stat.
    #         iscell (bool): suite2p iscell file (needs to reload before the plotting)
    #         cmap (matplotlib object, optional): Matplotlib colormao. Defaults to cm.cool.reversed().
    #"""

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
    trials_df["is_multidepth"] = trials_df.recording_name.str.contains("multidepth")
    if trials_df.is_multidepth.any():
        print(
            "trials_df contains multidepth recordings. Ignoring them for basic_vis_session"
        )
        trials_df = trials_df[~trials_df.is_multidepth]
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
                    dsp.plot_depth_tuning_curve(
                        neurons_df=neurons_df,
                        trials_df=trials_df,
                        roi=roi,
                        rs_thr=None,
                        plot_fit=is_closedloop,
                        linewidth=3,
                        linecolor="k",
                        closed_loop=is_closedloop,
                        use_col="depth_tuning_popt_closedloop",
                    )
                    plt.title(f"roi{roi}")

                    # plt.subplot2grid((plot_rows, plot_cols), (iroi, 1))
                    # dsp.plot_depth_tuning_curve(
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
                    # dsp.plot_depth_tuning_curve(
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
                    rsof_plots.plot_speed_tuning(
                        trials_df=trials_df,
                        roi=roi,
                        is_closed_loop=is_closedloop,
                        which_speed="RS",
                        smoothing_sd=1,
                        **params["rs_curve"],
                    )

                    plt.subplot2grid((plot_rows, plot_cols), (iroi, 4))
                    rsof_plots.plot_speed_tuning(
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
                    dsp.plot_PSTH(
                        trials_df=trials_df,
                        roi=roi,
                        is_closed_loop=is_closedloop,
                        nbins=20,
                        frame_rate=15,
                    )

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
                    vmin, vmax = rsof_plots.plot_RS_OF_matrix(
                        trials_df=trials_df[trials_df.closed_loop == is_closedloop],
                        roi=roi,
                        log_range=log_range,
                    )

                    models = ["gof", "gadd", "g2d", "gratio"]
                    model_labels = ["OF only", "Additive", "Conjunctive", "Pure depth"]
                    for imodel, (model, model_label) in enumerate(
                        zip(models, model_labels)
                    ):
                        if imodel == 0:
                            ylabel = "Optic flow speed (degrees/s)"
                        else:
                            ylabel = ""
                        if imodel == 1:
                            xlabel = "Running speed (cm/s)"
                        else:
                            xlabel = ""

                        ax = plt.subplot2grid(
                            (plot_rows, plot_cols), (iroi, 7 + imodel), fig=fig
                        )
                        col_name = f"rsof_popt_closedloop_{model}"
                        if col_name not in neurons_df.columns:
                            print(f"Not data for model {model}")
                            ax.axis("off")
                            continue
                        rsof_plots.plot_RS_OF_fit(
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
                            # plot_x=0.24 + 0.1 * imodel,
                            # plot_y=0.64 - 0.43 * iroi,
                            # plot_width=0.15,
                            # plot_height=0.15,
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
                    / f"roi{rois[i*10]}-{np.min([(i+1)*10, len(rois)])}.png",
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
                dsp.plot_depth_tuning_curve(
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
                    dsp.plot_depth_tuning_curve(
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
                dsp.plot_depth_tuning_curve(
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


def _plot_treadmill_speed_tuning(
    trials_df,
    roi,
    is_closed_loop,
    which_speed="RS",
    nbins=10,
    speed_min=0.01,
    speed_max=1.5,
    speed_thr=0.01,
    of_min=0.1,
    of_max=1000.0,
    smoothing_sd=1,
    fontsize_dict=None,
):
    """Plot speed tuning for a treadmill session (no depth grouping).

    In a treadmill session there is no ``depth`` column, so the standard
    :func:`rsof_plots.plot_speed_tuning` cannot be used directly (it groups
    by depth). This helper concatenates all stimulus frames across trials and
    computes a single pooled tuning curve.

    Args:
        trials_df (pd.DataFrame): Treadmill trials dataframe. Must contain
            ``dff_stim``, ``RS_stim``, and ``OF_stim`` columns.
        roi (int): ROI index.
        is_closed_loop (int): 1 for closed-loop, 0 for open-loop.
        which_speed (str): ``"RS"`` (running speed) or ``"OF"`` (optic flow).
        nbins (int): Number of bins for the tuning curve. Defaults to 10.
        speed_min (float): Minimum speed for RS bins (m/s). Defaults to 0.01.
        speed_max (float): Maximum speed for RS bins (m/s). Defaults to 1.5.
        speed_thr (float): Speed threshold below which frames are excluded (m/s).
            Defaults to 0.01.
        of_min (float): Minimum OF speed for bins (deg/s). Defaults to 0.1.
        of_max (float): Maximum OF speed for bins (deg/s). Defaults to 1000.
        smoothing_sd (float): SD of Gaussian kernel for smoothing. Defaults to 1.
        fontsize_dict (dict or None): Font size dict with keys ``"label"`` and
            ``"tick"``. If None a sensible default is used.
    """
    import scipy.stats
    import seaborn as sns

    if fontsize_dict is None:
        fontsize_dict = {"title": 10, "label": 8, "tick": 7}

    ax = plt.gca()
    df = trials_df[trials_df.closed_loop == is_closed_loop]
    if len(df) == 0:
        ax.axis("off")
        return

    # Concatenate all frames across trials
    dff_arr = np.concatenate([v[:, roi] for v in df.dff_stim.values if v.ndim == 2])
    if which_speed == "RS":
        speed_arr = np.concatenate(df.RS_stim.values) * 100  # m/s -> cm/s
        bins = np.linspace(speed_min * 100, speed_max * 100, nbins + 1)
        xlabel = "Running speed (cm/s)"
    else:
        speed_arr = np.degrees(np.concatenate(df.OF_stim.values))  # rad/s -> deg/s
        bins = np.geomspace(of_min, of_max, nbins + 1)
        xlabel = "Optic flow speed (deg/s)"

    if which_speed == "RS":
        mask = speed_arr > speed_thr * 100
    else:
        mask = speed_arr > 0
    speed_arr = speed_arr[mask]
    dff_arr = dff_arr[mask]

    bin_means, _, _ = scipy.stats.binned_statistic(
        x=speed_arr, values=dff_arr, statistic="mean", bins=bins
    )
    bin_counts, _, _ = scipy.stats.binned_statistic(
        x=speed_arr, values=dff_arr, statistic="count", bins=bins
    )
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Gaussian smoothing over bins (weighted by count)
    weights = np.where(np.isnan(bin_means), 0, bin_counts)
    smooth = np.zeros(nbins)
    for i in range(nbins):
        w = np.exp(-((np.arange(nbins) - i) ** 2) / (2 * smoothing_sd**2))
        w *= weights
        if w.sum() > 0:
            smooth[i] = np.nansum(w * bin_means) / w.sum()

    ax.plot(bin_centers, smooth, color="k", linewidth=1.5)
    ax.errorbar(
        bin_centers,
        bin_means,
        fmt="o",
        color="k",
        markersize=3,
        linewidth=1,
        markeredgewidth=0.3,
        markeredgecolor="w",
        ls="none",
    )
    if which_speed == "OF":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"], labelpad=-3)
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])
    sns.despine(ax=ax, offset=3, trim=True)


def basic_treadmill_session(neurons_df, trials_df, neurons_ds, **kwargs):
    """Basic visualisation plots for a treadmill-only session (no sphere stimulus).

    Because the treadmill protocol has no depth axis, this function skips depth
    tuning curves and instead produces an RS speed tuning, an OF speed tuning,
    an RS-OF heatmap and up to four RS-OF model-fit panels for every ROI.
    Fit panels are silently skipped when the corresponding column is absent from
    *neurons_df* (e.g. before fitting has been run).

    The output figures are saved under::

        <neurons_ds parent>/plots/basic_treadmill/

    Args:
        neurons_df (pd.DataFrame): Neurons dataframe (one row per ROI).
        trials_df (pd.DataFrame): Treadmill trials dataframe. Must contain
            ``dff_stim``, ``RS_stim``, ``OF_stim``, and ``closed_loop`` columns.
        neurons_ds: Flexilims neurons dataset (used for the output path).
        **kwargs: Optional overrides.

            ``rs_curve`` (dict): Passed to the RS speed tuning helper.
                Keys: ``speed_min``, ``speed_max``, ``nbins``, ``speed_thr``.
            ``of_curve`` (dict): Passed to the OF speed tuning helper.
                Keys: ``of_min``, ``of_max``, ``nbins``.
            ``RS_OF_matrix_log_range`` (dict): Log-range dict for
                :func:`rsof_plots.plot_RS_OF_matrix`.
            ``treadmill_sfx`` (str): Suffix appended to the RS-OF fit column
                names, e.g. ``"_treadmill"``. Defaults to ``"_treadmill"``.
    """
    rois = neurons_df.roi.values
    is_closed_loop = 1  # treadmill recordings are always closed-loop

    save_dir = neurons_ds.path_full.parent / "plots" / "basic_treadmill"
    os.makedirs(save_dir, exist_ok=True)

    treadmill_sfx = kwargs.get("treadmill_sfx", "_treadmill")

    rs_curve_defaults = dict(speed_min=0.01, speed_max=1.5, nbins=10, speed_thr=0.01)
    rs_curve_kwargs = {**rs_curve_defaults, **kwargs.get("rs_curve", {})}

    of_curve_defaults = dict(of_min=0.1, of_max=1000.0, nbins=10)
    of_curve_kwargs = {**of_curve_defaults, **kwargs.get("of_curve", {})}

    log_range_defaults = {
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    }
    log_range = {**log_range_defaults, **kwargs.get("RS_OF_matrix_log_range", {})}

    fontsize_dict = {"title": 10, "label": 8, "tick": 7, "legend": 6}

    # Layout per row: RS tuning | OF tuning | Depth tuning | RS-OF matrix | g2d | gadd | gof | gratio
    models = ["g2d", "gadd", "gof", "gratio"]
    model_labels = ["2D Gaussian", "Additive", "OF only", "RS/OF ratio"]
    plot_rows = 10
    plot_cols = 8

    for i in tqdm(range(int(len(rois) // plot_rows + 1))):
        roi_batch = rois[i * plot_rows : np.min([(i + 1) * plot_rows, len(rois)])]
        if len(roi_batch) == 0:
            break

        fig = plt.figure(figsize=(3 * plot_cols, 3 * plot_rows))

        for iroi, roi in enumerate(roi_batch):
            # ------ Column 0: RS speed tuning ------
            plt.subplot2grid((plot_rows, plot_cols), (iroi, 0))
            _plot_treadmill_speed_tuning(
                trials_df=trials_df,
                roi=roi,
                is_closed_loop=is_closed_loop,
                which_speed="RS",
                fontsize_dict=fontsize_dict,
                speed_min=rs_curve_kwargs["speed_min"],
                speed_max=rs_curve_kwargs["speed_max"],
                nbins=rs_curve_kwargs["nbins"],
                speed_thr=rs_curve_kwargs["speed_thr"],
            )
            plt.title(f"roi{roi}", fontsize=fontsize_dict["title"])

            # ------ Column 1: OF speed tuning ------
            plt.subplot2grid((plot_rows, plot_cols), (iroi, 1))
            _plot_treadmill_speed_tuning(
                trials_df=trials_df,
                roi=roi,
                is_closed_loop=is_closed_loop,
                which_speed="OF",
                fontsize_dict=fontsize_dict,
                of_min=of_curve_kwargs["of_min"],
                of_max=of_curve_kwargs["of_max"],
                nbins=of_curve_kwargs["nbins"],
            )

            # ------ Column 2: Depth tuning ------
            ax_depth = plt.subplot2grid((plot_rows, plot_cols), (iroi, 2))
            if "depth" in trials_df.columns:
                depth_col = f"depth_tuning_popt_closedloop{treadmill_sfx}"
                if depth_col not in neurons_df.columns:
                    depth_col = "depth_tuning_popt_closedloop"
                dsp.plot_depth_tuning_curve(
                    neurons_df=neurons_df,
                    trials_df=trials_df,
                    roi=roi,
                    rs_thr=None,
                    plot_fit=True,
                    linewidth=1.5,
                    linecolor="k",
                    closed_loop=is_closed_loop,
                    use_col=depth_col,
                    ax=ax_depth,
                    fontsize_dict=fontsize_dict,
                    markersize=3,
                )
            else:
                ax_depth.axis("off")

            # ------ Column 3: RS-OF heatmap ------
            ax_matrix = plt.subplot2grid((plot_rows, plot_cols), (iroi, 3))
            vmin, vmax = rsof_plots.plot_RS_OF_matrix(
                trials_df=trials_df[trials_df.closed_loop == is_closed_loop],
                roi=roi,
                log_range=log_range,
                is_closed_loop=is_closed_loop,
                fontsize_dict=fontsize_dict,
                cbar_width=None,
                ax=ax_matrix,
            )

            # ------ Columns 4-7: RS-OF model fits ------
            for imodel, (model, model_label) in enumerate(zip(models, model_labels)):
                ax = plt.subplot2grid((plot_rows, plot_cols), (iroi, 4 + imodel))
                col_name = f"rsof_popt_closedloop_{model}{treadmill_sfx}"
                if col_name not in neurons_df.columns:
                    ax.set_title(
                        f"{model_label}\n(not fitted)",
                        fontsize=fontsize_dict["title"],
                    )
                    ax.axis("off")
                    continue
                rsof_plots.plot_RS_OF_fit(
                    neurons_df=neurons_df,
                    roi=roi,
                    model=model,
                    model_label=model_label,
                    min_sigma=0.25,
                    vmin=vmin,
                    vmax=vmax,
                    log_range=log_range,
                    fontsize_dict=fontsize_dict,
                    cbar_width=None,
                    ax=ax,
                    sfx=treadmill_sfx,
                )
                if imodel > 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])

        plt.savefig(
            save_dir / f"roi{roi_batch[0]}-{roi_batch[-1]}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()
