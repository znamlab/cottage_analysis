import numpy as np
from matplotlib import pyplot as plt
from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning
from functools import partial


def plot_sftf_fit(
    neuron_series, sf_range, tf_range, sf_ticks=None, tf_ticks=None, min_sigma=0.25
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
        log_kappa=neuron_series["log_dir_width"],
        dsi=neuron_series["dsi"],
    )
    # plot polar plot of direction tuning at the preferred SF and TF
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
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angle_pos = [6, 3, 2, 1, 4, 7, 8, 9]
    for i, angle in enumerate(angles):
        responses = grating_tuning_(
            (sfs, tfs, np.ones_like(sfs) * angle),
            *popt,
        )
        plt.subplot(3, 3, angle_pos[i])
        plt.imshow(
            responses,
            extent=[sf_range[0], sf_range[1], tf_range[0], tf_range[1]],
            origin="lower",
            vmax=np.exp(popt.log_amplitude) + popt.offset,
            vmin=popt.offset,
            cmap="magma",
        )
        if sf_ticks is not None:
            plt.xticks(np.log(sf_ticks), sf_ticks, rotation=90)
        if tf_ticks is not None:
            plt.yticks(np.log(tf_ticks), tf_ticks)
    # add a colorbar aligned to the bottom subplots without changing axis locations
    plt.tight_layout(pad=0.4)
    add_colorbar()


def plot_sftf_tuning(dff_mean, roi):
    # plot a polar plot of dff as a function of angle
    plt.subplot(3, 3, 5, projection="polar")
    dir_tuning = dff_mean.groupby("Angle").max()
    # append the first angle to the end to close the circle
    dir_tuning = dir_tuning.append(dir_tuning.iloc[0])
    plt.plot(
        np.deg2rad(dir_tuning.index),
        dir_tuning[roi],
        color="k",
    )
    plt.fill_between(
        np.deg2rad(dir_tuning.index),
        dir_tuning[roi],
    )

    angle_pos = [6, 3, 2, 1, 4, 7, 8, 9]
    for i, angle in enumerate(dff_mean.Angle.unique()):
        this_angle = dff_mean[dff_mean.Angle == angle]
        # create a matrix of responses as a function of SpatialFrequency and TemporalFrequency
        this_angle = this_angle.pivot(
            index="SpatialFrequency", columns="TemporalFrequency", values=roi
        )
        # create a subplot for each angle
        plt.subplot(3, 3, angle_pos[i])
        # plot a colormap of this_angle with tick labels
        plt.imshow(
            this_angle.T,
            cmap="magma",
            vmin=0,
            vmax=dff_mean[roi].max(),
        )
        plt.yticks(range(len(this_angle.columns)), this_angle.columns)
        plt.xticks(range(len(this_angle.index)), this_angle.index, rotation=90)
        plt.gca().invert_yaxis()
    # add a colorbar aligned to the bottom subplots without changing axis locations
    plt.tight_layout(pad=0.4)
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
