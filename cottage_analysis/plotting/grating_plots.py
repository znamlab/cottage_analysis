import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning
from functools import partial


def plot_sftf_fit(
    neuron_series,
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
    fontsize_dict={"title": 10, "label": 10, "tick": 10},
):
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
    plt.yticks(fontsize=fontsize_dict["tick"])

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angle_pos = [6, 3, 2, 1, 4, 7, 8, 9]
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
            vmax=(np.exp(popt.log_amplitude) + popt.offset),
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


def plot_sftf_tuning(
    trials_df,
    roi,
    plot_grid=True,
    grid_rows=3,
    grid_cols=3,
    grid_x=0,
    grid_y=0,
    colorbar=True,
    fontsize_dict={"title": 10, "label": 10, "tick": 10},
):
    # Select the mean dff values for all ROIs from trials_df
    numerical_columns = trials_df.loc[:, trials_df.columns.str.isnumeric().isna()]
    named_columns = trials_df[['Angle','SpatialFrequency','TemporalFrequency']]
    dff_mean = pd.concat([numerical_columns, named_columns], axis=1)

    # Find the stimuli SF/Tf/angle ranges
    sf_range = np.sort(np.unique(trials_df.SpatialFrequency))
    tf_range = np.sort(np.unique(trials_df.TemporalFrequency))
    angle_range = np.sort(np.unique(trials_df.Angle))

    # plot a polar plot of dff as a function of angle
    dir_tuning = dff_mean.groupby("Angle").max()
    # concatenate the first row to the end to close the circle
    dir_tuning = pd.concat([dir_tuning, dir_tuning.iloc[0:1]])
    if plot_grid:
        plt.subplot2grid(
            (grid_rows, grid_cols), (grid_x + 1, grid_y + 1), projection="polar"
        )
    else:
        plt.subplot(3, 3, 5, projection="polar")
    plt.plot(
        np.deg2rad(dir_tuning.index.values),
        dir_tuning[roi].values,
        color="k",
    )
    plt.fill_between(
        np.deg2rad(dir_tuning.index),
        dir_tuning[roi],
    )
    plt.xticks(fontsize=fontsize_dict["tick"])
    plt.yticks(fontsize=fontsize_dict["tick"])

    sf_range = np.sort(dff_mean.SpatialFrequency.unique())
    tf_range = np.sort(dff_mean.TemporalFrequency.unique())
    angle_range = np.sort(dff_mean.Angle.unique())
    angle_pos = [6, 3, 2, 1, 4, 7, 8, 9]
    for i, angle in enumerate(angle_range):
        # this_angle = dff_mean[dff_mean.Angle == angle]
        # # create a matrix of responses as a function of SpatialFrequency and TemporalFrequency
        # this_angle = this_angle.pivot(
        #     index="SpatialFrequency", columns="TemporalFrequency", values=roi
        # )
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
            this_angle.T,
            cmap="magma",
            vmin=0,
            vmax=dff_mean[roi].max(),
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
        add_colorbar()


def add_colorbar():
    cbar_pos = [
        1.02,
        plt.gca().get_position().y0,
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


def basic_vis_SFTF_session(neurons_df, trials_df, neurons_ds):
    rois = neurons_df[neurons_df.is_depth_neuron == 1].roi.values
    os.makedirs(neurons_ds.path_full.parent / "plots" / "basic_vis", exist_ok=True)

    plot_rows = 10
    plot_cols = 3

    for i in tqdm(range(int(len(rois) // plot_rows + 1))):
        plt.figure(figsize=(3 * plot_cols, 3 * plot_rows))
        iroi = 0
        for roi in rois[i * plot_rows : np.min([(i + 1) * plot_rows, len(rois)])]:
            plt.subplot2grid((plot_rows, plot_cols), (iroi, 0))
            plot_depth_tuning_curve(
                neurons_df=neurons_df,
                trials_df=trials_df,
                roi=roi,
                rs_thr=0.2,
                plot_fit=False,
                linewidth=3,
                linecolor="k",
                fit_linecolor="r",
                closed_loop=1,
            )
            plt.title(f"roi{roi}")

            plt.subplot2grid((plot_rows, plot_cols), (iroi, 1))
            plot_speed_tuning(
                neurons_df=neurons_df,
                trials_df=trials_df,
                roi=roi,
                is_closed_loop=1,
                nbins=10,
                which_speed="RS",
                speed_min=0.01,
                speed_max=1.5,
                speed_thr=0.01,
                smoothing_sd=1,
            )

            plt.subplot2grid((plot_rows, plot_cols), (iroi, 2))
            plot_speed_tuning(
                neurons_df=neurons_df,
                trials_df=trials_df,
                roi=roi,
                is_closed_loop=1,
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
                is_closed_loop=1,
                max_distance=6,
                nbins=20,
                frame_rate=15,
            )
            plt.tight_layout()

            iroi += 1
        plt.savefig(
            neurons_ds.path_full.parent
            / "plots"
            / "basic_vis"
            / f"roi{rois[i*10]}- {np.min([(i+1)*10, len(rois)])}.png",
            dpi=100,
        )
