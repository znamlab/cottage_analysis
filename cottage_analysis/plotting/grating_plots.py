import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from cottage_analysis.plotting import basic_vis_plots
from functools import partial


def plot_sftf_fit(
    neurons_df,
    roi,
    sf_range,
    tf_range,
    sf_ticks=None,
    tf_ticks=None,
    min_sigma=0.25,
    plot_grid=True,
    grid_rows=3,
    grid_cols=3,
    grid_x=0,
    grid_y=0,
    colorbar=True,
    vmax=None,
    fontsize_dict={"title": 10, "label": 10, "tick": 10},
):
    """Plot SFTF fit for a given ROI

    Args:
        neurons_df (pd.Dataframe): neurons_df
        roi (int): roi
        sf_range (list): list of log sf range
        tf_range (list): list of log tf range
        sf_ticks (list, optional): list of sf range. Defaults to None.
        tf_ticks (list, optional): list of tf range. Defaults to None.
        min_sigma (float, optional): min sigma for fitting 2D gaussian. Defaults to 0.25.
        plot_grid (bool, optional): whether to plot with subplots or grids. Defaults to True.
        grid_rows (int, optional): grid plot row number. Defaults to 3.
        grid_cols (int, optional): grid plot col number. Defaults to 3.
        grid_x (int, optional): grid plot start row index. Defaults to 0.
        grid_y (int, optional): grid plot start col index. Defaults to 0.
        colorbar (bool, optional): whether to add colorbar. Defaults to True.
        fontsize_dict (dict, optional): dict containing fontsize settings. Defaults to {"title": 10, "label": 10, "tick": 10}.
    """
    from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning

    neuron_series = neurons_df.iloc[roi]
    grating_tuning_ = partial(grating_tuning, min_sigma=min_sigma)
    popt = GratingParams(
        log_amplitude=neuron_series["log_amplitude"],
        sf0=neuron_series["sf0"],
        tf0=neuron_series["tf0"],
        log_sigma_x2=neuron_series["log_sigma_x2"],
        log_sigma_y2=neuron_series["log_sigma_y2"],
        theta=neuron_series["theta"],
        offset=neuron_series["offset"],
        alpha0=neuron_series["alpha0"],
        log_kappa=neuron_series["log_kappa"],
        dsi=neuron_series["dsi"],
    )
    # plot polar plot of direction tuning at the preferred SF and TF
    if plot_grid:
        plt.subplot2grid(
            (grid_rows, grid_cols), (grid_x + 1, grid_y + 1), projection="polar"
        )
    else:
        plt.subplot(3, 3, 5, projection="polar")
    angles = np.linspace(0, 2 * np.pi, 100)
    dir_tuning = grating_tuning_(
        (
            np.ones_like(angles) * neuron_series["sf0"],
            np.ones_like(angles) * neuron_series["tf0"],
            angles,
        ),
        *popt,
    )
    plt.plot(angles, dir_tuning, color="k", linewidth=2)
    plt.fill_between(angles, dir_tuning)
    sfs, tfs = np.meshgrid(
        np.linspace(sf_range[0], sf_range[1], 100),
        np.linspace(tf_range[0], tf_range[1], 100),
    )
    plt.xticks(fontsize=fontsize_dict["tick"])
    yticks = plt.gca().get_yticks()
    plt.yticks([yticks[0], yticks[-2]], fontsize=fontsize_dict["tick"])

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angle_pos = [6, 3, 2, 1, 4, 7, 8, 9]
    if vmax is None:
        vmax = np.exp(popt.log_amplitude) + popt.offset
    for i, angle in enumerate(angles):
        responses = grating_tuning_(
            (sfs, tfs, np.ones_like(sfs) * angle),
            *popt,
        )
        if plot_grid:
            plt.subplot2grid(
                (grid_rows, grid_cols),
                (grid_x + (angle_pos[i] - 1) // 3, grid_y + (angle_pos[i] - 1) % 3),
            )
        else:
            plt.subplot(3, 3, angle_pos[i])
        plt.imshow(
            responses,
            extent=[sf_range[0], sf_range[1], tf_range[0], tf_range[1]],
            origin="lower",
            vmax=vmax,
            vmin=popt.offset,
            cmap="magma",
        )
        if sf_ticks is not None:
            plt.xticks(
                np.log(sf_ticks), sf_ticks, rotation=90, fontsize=fontsize_dict["tick"]
            )
        if tf_ticks is not None:
            plt.yticks(np.log(tf_ticks), tf_ticks, fontsize=fontsize_dict["tick"])
    # add a colorbar aligned to the bottom subplots without changing axis locations
    if plot_grid:
        plt.tight_layout(pad=0.005)
    if not plot_grid:
        plt.tight_layout(pad=0.4)
    if colorbar:
        add_colorbar()
    return vmax


def plot_sftf_tuning(
    trials_df,
    roi,
    plot_grid=True,
    grid_rows=3,
    grid_cols=3,
    grid_x=0,
    grid_y=0,
    colorbar=True,
    vmax=None,
    fontsize_dict={"title": 10, "label": 10, "tick": 10},
):
    """Plot SFTF tuning for a given ROI with raw data

    Args:
        trials_df (pd.DataFrame): trials_df
        roi (int): roi index
        plot_grid (bool, optional): whether to plot with subplots or grids. Defaults to True.
        grid_rows (int, optional): grid plot row number. Defaults to 3.
        grid_cols (int, optional): grid plot col number. Defaults to 3.
        grid_x (int, optional): grid plot start row index. Defaults to 0.
        grid_y (int, optional): grid plot start col index. Defaults to 0.
        colorbar (bool, optional): whether to add colorbar. Defaults to True.
        fontsize_dict (dict, optional): dict containing fontsize settings. Defaults to {"title": 10, "label": 10, "tick": 10}.
    """

    # Select the mean dff values for all ROIs from trials_df
    numerical_columns = trials_df.loc[:, trials_df.columns.str.isnumeric().isna()]
    named_columns = trials_df[["Angle", "SpatialFrequency", "TemporalFrequency"]]
    dff_mean = pd.concat([numerical_columns, named_columns], axis=1)

    # Find the stimuli SF/Tf/angle ranges
    sf_range = np.sort(np.unique(trials_df.SpatialFrequency))
    tf_range = np.sort(np.unique(trials_df.TemporalFrequency))
    angle_range = np.sort(np.unique(trials_df.Angle))

    sf_range = np.sort(dff_mean.SpatialFrequency.unique())
    tf_range = np.sort(dff_mean.TemporalFrequency.unique())
    angle_range = np.sort(dff_mean.Angle.unique())
    angle_pos = [6, 3, 2, 1, 4, 7, 8, 9]
    all_angles = np.zeros((len(angle_range), len(sf_range), len(tf_range)))
    for i, angle in enumerate(angle_range):
        this_angle_df = dff_mean[dff_mean.Angle == angle]
        # create a matrix of responses as a function of SpatialFrequency and TemporalFrequency
        this_angle = np.zeros(
            (
                len(sf_range),
                len(tf_range),
            )
        )
        for isf, sf in enumerate(sf_range):
            for itf, tf in enumerate(tf_range):
                this_angle[isf, itf] = np.mean(
                    this_angle_df[
                        (this_angle_df.SpatialFrequency == sf)
                        & (this_angle_df.TemporalFrequency == tf)
                    ][roi]
                )
                all_angles[i, isf, itf] = this_angle[isf, itf]
    if vmax is None:
        # vmax=dff_mean[roi].max()
        vmax = np.nanmax(all_angles)

    # plot a polar plot of dff as a function of angle
    dir_tuning = np.max(all_angles, axis=(1, 2))
    dir_tuning = np.append(dir_tuning, dir_tuning[0])
    dirs = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0])

    if plot_grid:
        plt.subplot2grid(
            (grid_rows, grid_cols), (grid_x + 1, grid_y + 1), projection="polar"
        )
    else:
        plt.subplot(3, 3, 5, projection="polar")
    plt.plot(
        np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0]),
        dir_tuning,
        color="k",
    )
    plt.fill_between(
        np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0]),
        dir_tuning,
    )
    plt.xticks(fontsize=fontsize_dict["tick"])
    yticks = plt.gca().get_yticks()
    plt.yticks([yticks[0], yticks[-3]], fontsize=fontsize_dict["tick"])

    for i, angle in enumerate(angle_range):
        # create a subplot for each angle
        if plot_grid:
            plt.subplot2grid(
                (grid_rows, grid_cols),
                (grid_x + (angle_pos[i] - 1) // 3, grid_y + (angle_pos[i] - 1) % 3),
            )
        else:
            plt.subplot(3, 3, angle_pos[i])
        # plot a colormap of this_angle with tick labels
        plt.imshow(
            all_angles[i].T,
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        plt.yticks(
            range(len(tf_range)),
            tf_range,
            fontsize=fontsize_dict["tick"],
        )
        plt.xticks(
            range(len(sf_range)),
            sf_range,
            rotation=90,
            fontsize=fontsize_dict["tick"],
        )
        plt.gca().invert_yaxis()
    # add a colorbar aligned to the bottom subplots without changing axis locations
    if plot_grid:
        plt.tight_layout(pad=0.005)
    if not plot_grid:
        plt.tight_layout(pad=0.4)
    if colorbar:
        add_colorbar(y=0.5)
    return vmax


def add_colorbar(y=0):
    """Add colorbar to the current axis"""
    cbar_pos = [
        1.02,
        plt.gca().get_position().y0 + y,
        0.02,
        plt.gca().get_position().height,
    ]
    plt.axes(cbar_pos)
    plt.colorbar(cax=plt.gca(), label="dF/F")


def plot_tuning_stats(neurons_df, trials_df, rsq_thresh=0.1):
    neurons_df["speed0"] = neurons_df["tf0"] - neurons_df["sf0"]
    included_neurons = neurons_df[neurons_df["rsq"] > rsq_thresh]
    # plot distribution of preferred SF, TF, and DSI
    # also plot a scatter plot of preferred SF vs TF colorcoded by DSI
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(included_neurons["sf0"], bins=20)
    axes[0].set_xlabel("Preferred SF (cyc/deg)")
    axes[0].set_xticks(
        np.log(trials_df["SpatialFrequency"].unique()),
        labels=trials_df["SpatialFrequency"].unique(),
    )
    axes[0].set_ylabel("Number of neurons")

    axes[1].hist(included_neurons["tf0"], bins=20)
    axes[1].set_xlabel("Preferred TF (Hz)")
    axes[1].set_xticks(
        np.log(trials_df["TemporalFrequency"].unique()),
        labels=trials_df["TemporalFrequency"].unique(),
    )
    axes[1].set_ylabel("Number of neurons")

    axes[2].hist(included_neurons["dsi"], bins=20)
    axes[2].set_xlabel("DSI")
    axes[2].set_ylabel("Number of neurons")

    plt.figure(figsize=(5, 5))
    plt.scatter(
        included_neurons["sf0"],
        included_neurons["tf0"],
        c=included_neurons["dsi"],
        s=10,
        alpha=0.8,
    )
    plt.gca().set_xlabel("Preferred SF (cyc/deg)")
    plt.gca().set_xticks(
        np.log(trials_df["SpatialFrequency"].unique()),
        labels=trials_df["SpatialFrequency"].unique(),
        rotation=90,
    )
    plt.gca().set_ylabel("Preferred TF (Hz)")
    plt.gca().set_yticks(
        np.log(trials_df["TemporalFrequency"].unique()),
        labels=trials_df["TemporalFrequency"].unique(),
    )
    plt.gca().set_aspect("equal")
    # add a colorbar for the scatter plot
    cbar = plt.colorbar(plt.gca().collections[0], ax=plt.gca())
    cbar.set_label("DSI")
    plt.tight_layout()

    # select neurons tuned to SF and TF within a certain range
    sf_range = [np.log(0.005), np.log(0.64)]
    tf_range = [np.log(0.25), np.log(32)]
    included_neurons = included_neurons[
        (included_neurons["sf0"] > sf_range[0])
        & (included_neurons["sf0"] < sf_range[1])
        & (included_neurons["tf0"] > tf_range[0])
        & (included_neurons["tf0"] < tf_range[1])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(
        included_neurons["dsi"],
        included_neurons["speed0"],
        alpha=0.8,
    )
    axes[0].set_xlabel("DSI")
    axes[0].set_ylabel("Speed (deg/s)")
    axes[0].set_yticks(np.log([1, 10, 100, 1000]), labels=[1, 10, 100, 1000])


def basic_vis_SFTF_roi(
    neurons_df,
    trials_df_depth,
    trials_df_sftf,
    roi,
    add_depth=False,
    fontsize_dict={"title": 10, "label": 10, "tick": 10},
):
    """Basic visualisation plots of ROI's depth and SFTF features

    Args:
        neurons_df (pd.DataFrame): neurons_df
        trials_df_depth (pd.DataFrame): trials_df for depth recordings
        trials_df_sftf (pd.DataFrame): trials_df for SFTF recordings
        roi (int): roi index
        add_depth (bool, optional): whether to add depth tunings & speed tunings. Defaults to False.
        fontsize_dict (dict, optional): dict containing fontsize settings. Defaults to {'title': 10, 'label': 10, 'tick': 10}.
    """
    from cottage_analysis.analysis.fit_gaussian_blob import grating_tuning
    plot_rows = 3
    if add_depth:
        plot_cols = 7
        grating_tuning_grid_y = 1
        grating_fit_grid_y = 4
    else:
        plot_cols = 6
        grating_tuning_grid_y = 0
        grating_fit_grid_y = 3

    plt.figure(figsize=(plot_cols * 3, plot_rows * 3))
    if add_depth:
        # Plot depth tuning
        plt.subplot2grid((plot_rows, plot_cols), (0, 0), colspan=1)
        basic_vis_plots.plot_depth_tuning_curve(
            neurons_df=neurons_df,
            trials_df=trials_df_depth,
            roi=roi,
            rs_thr=0.2,
            plot_fit=False,
            linewidth=3,
            linecolor="k",
            fit_linecolor="r",
            closed_loop=1,
            fontsize_dict=fontsize_dict,
        )
        plt.title(
            f"roi{roi}, depth rsq{neurons_df.depth_tuning_test_rsq_closedloop[roi]:.2f}, SFTF rsq{neurons_df.rsq[roi]:.2f}",
            fontsize=fontsize_dict["title"],
        )

        # Plot speed tuning
        plt.subplot2grid((plot_rows, plot_cols), (1, 0), colspan=1)
        basic_vis_plots.plot_speed_tuning(
            neurons_df=neurons_df,
            trials_df=trials_df_depth,
            roi=roi,
            is_closed_loop=1,
            nbins=10,
            which_speed="RS",
            speed_min=0.01,
            speed_max=1.5,
            speed_thr=0.01,
            smoothing_sd=1,
            fontsize_dict=fontsize_dict,
        )

        plt.subplot2grid((plot_rows, plot_cols), (2, 0), colspan=1)
        basic_vis_plots.plot_speed_tuning(
            neurons_df=neurons_df,
            trials_df=trials_df_depth,
            roi=roi,
            is_closed_loop=1,
            nbins=10,
            which_speed="OF",
            speed_min=0.01,
            speed_max=1.5,
            speed_thr=0.01,
            smoothing_sd=1,
            fontsize_dict=fontsize_dict,
        )

    # Plot grating fit
    vmax_fit = plot_sftf_fit(
        neurons_df=neurons_df,
        roi=roi,
        sf_range=[
            np.log(np.sort(np.unique(trials_df_sftf.SpatialFrequency))[0]),
            np.log(np.sort(np.unique(trials_df_sftf.SpatialFrequency))[-1]),
        ],
        tf_range=[
            np.log(np.sort(np.unique(trials_df_sftf.TemporalFrequency))[0]),
            np.log(np.sort(np.unique(trials_df_sftf.TemporalFrequency))[-1]),
        ],
        sf_ticks=np.sort(trials_df_sftf.SpatialFrequency.unique()),
        tf_ticks=np.sort(trials_df_sftf.TemporalFrequency.unique()),
        min_sigma=0.25,
        plot_grid=True,
        grid_rows=plot_rows,
        grid_cols=plot_cols,
        grid_x=0,
        grid_y=grating_fit_grid_y,
        fontsize_dict=fontsize_dict,
        colorbar=True,
    )

    # Plot grating tuning with the raw data
    vmax_tuning = plot_sftf_tuning(
        trials_df=trials_df_sftf,
        roi=roi,
        plot_grid=True,
        grid_rows=plot_rows,
        grid_cols=plot_cols,
        grid_x=0,
        grid_y=grating_tuning_grid_y,
        fontsize_dict=fontsize_dict,
        colorbar=True,
        vmax=None,
    )


def basic_vis_SFTF_session(
    neurons_df,
    trials_df_depth,
    trials_df_sftf,
    rois=[],
    add_depth=False,
    save_dir=None,
    fontsize_dict={"title": 10, "label": 10, "tick": 10},
):
    """Plot basic visuation of a whole session, or a list of ROIs
    Args:
        neurons_df (pd.DataFrame): neurons_df
        trials_df_depth (pd.DataFrame): trials_df for depth recordings
        trials_df_sftf (pd.DataFrame): trials_df for SFTF recordings
        roi (int): roi index
        save_dir (str, optional): save directory for plots. Defaults to None.
        fontsize_dict (dict, optional): dict containing fontsize settings. Defaults to {'title': 10, 'label': 10, 'tick': 10}.
    """
    if len(rois) == 0:
        rois = neurons_df.roi.values

    if save_dir is not None:
        os.makedirs(save_dir / "plots" / "basic_vis_sftf", exist_ok=True)

    for i in tqdm(rois):
        basic_vis_SFTF_roi(
            neurons_df=neurons_df,
            trials_df_depth=trials_df_depth,
            trials_df_sftf=trials_df_sftf,
            roi=rois[i],
            add_depth=add_depth,
            fontsize_dict={"title": 15, "label": 10, "tick": 10},
        )
        if save_dir is not None:
            plt.savefig(
                save_dir / "plots" / "basic_vis_sftf" / f"roi{rois[i]}.png",
                dpi=100,
                bbox_inches="tight",
            )
