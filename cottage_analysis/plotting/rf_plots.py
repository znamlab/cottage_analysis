import numpy as np
import matplotlib.pyplot as plt
import scipy

from cottage_analysis.analysis.spheres import rf_fitting, rf_analysis
from cottage_analysis.plotting import plotting_utils
import plotly.graph_objects as go


def plot_stimulus_frame(
    frame,
    idepth,
    depths,
    position=(0, 0, 1, 1),
    plot_prop=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """
    Plot the stimulus reconstruction across depths.

    Parameters:
    - frame: 2D or 3D array of shape (height, width) or (ndepths, height, width)
    - idepth: int or None, index of the presented depth (if None, assume multidepth stimulus)
    - depths: list of depths corresponding to the frames
    - position: tuple, position of the plot [x, y, width, height]
    - plot_prop: float, proportion of the plot height to use
    - fontsize_dict: dict, font sizes for title, label, and tick

    """
    plot_x, plot_y, plot_width, plot_height = position
    ndepths = len(depths)
    for i in range(ndepths):
        ax = plt.gcf().add_axes(
            [
                plot_x,
                plot_y - plot_height / ndepths * i,
                plot_width,
                plot_height / ndepths * plot_prop,
            ]
        )
        if idepth is None:
            # multidepth stimulus
            ax.imshow(
                frame[i].squeeze(),
                cmap="gray_r",
                origin="lower",
                extent=(0, 120, -40, 40),
                aspect="equal",
                vmin=-1,
                vmax=1,
            )
        elif i == idepth:
            frame = frame.astype(float)
            frame[frame == 0] = 0.5
            ax.imshow(
                frame,
                cmap="gray_r",
                origin="lower",
                extent=[0, 120, -40, 40],
                aspect="equal",
                vmax=1,
                vmin=0,
            )
        else:
            ax.imshow(
                np.ones_like(frame) * 0.5,
                cmap="gray_r",
                origin="lower",
                extent=[0, 120, -40, 40],
                aspect="equal",
                vmax=1,
                vmin=0,
            )
        # add text indicating the depth
        ax.text(
            5,
            15,
            f"{int(depths[i] * 100)} cm",
            fontsize=fontsize_dict["tick"],
            color="white",
            fontdict={"weight": "bold"},
        )
        if i == ndepths - 1:
            ax.set_xlabel("Azimuth (degrees)", fontsize=fontsize_dict["label"])
        elif i == ndepths // 2:
            ax.set_ylabel("Elevation (degrees)", fontsize=fontsize_dict["label"])
        if i != ndepths - 1:
            ax.set_xticklabels([])
        ax.set_xticks([0, 60, 120])
        ax.tick_params(axis="both", labelsize=fontsize_dict["tick"], length=1.5)


def plot_rf(
    neurons_df,
    roi,
    is_closed_loop=1,
    ndepths=8,
    frame_shape=(16, 24),
    position=[0, 0, 1, 1],
    plot_prop=0.9,
    xlabel="Azimuth (deg)",
    ylabel="Elevation (deg)",
    fontsize_dict={"title": 15, "label": 10, "tick": 5, "legend": 8},
    use_ipsi=False,
    use_multidepth=False,
    extent=(0, 120, -40, 40),
    clim=None,
):
    if use_ipsi:
        sfx = "_ipsi"
    else:
        sfx = ""
    if is_closed_loop:
        sfx += "_closedloop"
    else:
        sfx += "_openloop"
    if use_multidepth:
        sfx += "_multidepth"
    coef = neurons_df.loc[roi, f"rf_coef{sfx}"][:, :-1].copy()
    coef = coef.reshape(coef.shape[0], ndepths, frame_shape[0], frame_shape[1])
    coef_mean = np.nanmean(coef, axis=0)
    coef_max = np.nanmax(coef_mean)
    if clim is None:
        clim = max(np.round(coef_max, 1), 0.1)
    plot_x, plot_y, plot_width, plot_height = position
    axes = []
    for i in range(ndepths):
        ax = plt.gcf().add_axes(
            [
                plot_x,
                plot_y - plot_height / ndepths * i,
                plot_width,
                plot_height / ndepths * plot_prop,
            ]
        )
        im = plt.imshow(
            coef_mean[i, :, :],
            origin="lower",
            cmap="bwr",
            extent=extent,
            vmin=-clim,
            vmax=clim,
        )
        if i != ndepths - 1:
            plt.gca().set_xticklabels([])
        if i == ndepths // 2:
            ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
        if i == ndepths - 1:
            ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", labelsize=fontsize_dict["tick"], length=1.5)
        ax.set_xticks([0, 60, 120])
        axes.append(ax)
        if i == ndepths - 1:
            ax_pos = ax.get_position()
            ax2 = plt.gcf().add_axes(
                [
                    ax_pos.x1 + ax_pos.width * 0.05,
                    ax_pos.y0,
                    0.005,
                    ax_pos.height / 2,
                ]
            )
            cbar = plt.colorbar(mappable=im, cax=ax2)
            # cbar.set_label("Z-score", fontsize=fontsize_dict["legend"])
            cbar.ax.tick_params(labelsize=fontsize_dict["legend"], length=2, pad=1)
            cbar.set_ticks([-np.round(coef_max, 1), 0, np.round(coef_max, 1)])
    return axes


def plot_rf_centers(
    fig,
    results,
    is_closed_loop=1,
    colors=["r", "b"],
    ndepths=8,
    frame_shape=(16, 24),
    n_stds=6,
    plot_x=0,
    plot_y=1,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 5},
):
    if is_closed_loop:
        sfx = "_closedloop"
    else:
        sfx = "_openloop"

    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    sessions = results.session.unique()

    for i in range(len(sessions)):
        # Get the coef and ipsi+_coef from each session
        session = sessions[i]
        results_sess = results[results.session == session]
        azi, ele, _, coef = rf_analysis.find_rf_centers(
            neurons_df=results_sess,
            is_closed_loop=is_closed_loop,
            ndepths=ndepths,
            frame_shape=frame_shape,
            resolution=5,
        )

        coef_ipsi = np.stack(results_sess[f"rf_coef_ipsi{sfx}"].values)

        # Find cells with significant RF
        sig, _ = rf_fitting.find_sig_rfs(
            np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
            np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
            n_std=n_stds,
        )

        # Plot
        cell_idx = sig & (results_sess.iscell == 1)
        ax.scatter(
            azi[cell_idx] + np.random.rand(np.sum(cell_idx)) * 4 - 2,
            ele[cell_idx] + np.random.rand(np.sum(cell_idx)) * 4 - 2,
            c=colors[i],
            edgecolors="none",
            s=10,
            alpha=0.3,
        )
        ax.set_aspect("equal", adjustable="box")
        plotting_utils.despine()
        ax.set_xlabel("Azimuth (degrees)", fontsize=fontsize_dict["label"])
        ax.set_ylabel("Elevation (degrees)", fontsize=fontsize_dict["label"])
        ax.set_xlim([0, 120])
        ax.set_ylim([-40, 40])
        ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])


def plot_sig_rf_perc(
    all_sig,
    all_sig_ipsi,
    plot_type="bar",
    bar_color="k",
    hist_color="k",
    hist_edgecolor="k",
    scatter_color="k",
    scatter_size=10,
    scatter_alpha=0.3,
    bins=10,
    fontsize_dict={"title": 15, "label": 10, "tick": 5},
):
    if plot_type == "bar":
        plt.bar(
            x=[0, 1],
            height=[np.mean(all_sig), np.mean(all_sig_ipsi)],
            yerr=[scipy.stats.sem(all_sig), scipy.stats.sem(all_sig_ipsi)],
            capsize=10,
            color=bar_color,
            alpha=0.5,
        )
        plt.scatter(
            x=np.zeros(len(all_sig)),
            y=all_sig,
            color=scatter_color,
            s=scatter_size,
            alpha=scatter_alpha,
        )
        plt.scatter(
            x=np.ones(len(all_sig_ipsi)),
            y=all_sig_ipsi,
            color=scatter_color,
            s=scatter_size,
            alpha=scatter_alpha,
        )
        plt.xticks(
            [0, 1], ["Contralateral", "Ipsilateral"], fontsize=fontsize_dict["label"]
        )
        plt.ylabel(
            "Proportion of neurons \nwith significant RFs",
            fontsize=fontsize_dict["label"],
        )
        plt.ylim([0, 1])
    elif plot_type == "hist":
        n, _, _ = plt.hist(
            all_sig, bins=bins, color=hist_color, edgecolor=hist_edgecolor, linewidth=1
        )
        plt.xlabel(
            "Proportion of neurons \nwith significant RFs",
            fontsize=fontsize_dict["label"],
        )
        plt.ylabel("Number of sessions", fontsize=fontsize_dict["label"])
        plt.xlim([0, 1])
        plt.ylim([0, (np.floor_divide(np.nanmax(n), 5) + 1) * 5])
    # plot median proportion as a triangle along the top of the histogram
    median_prop = np.median(all_sig)
    print("Median proportion of sig RF out of depth-tuned neurons:", median_prop)
    print(
        "Range of proportion of sig RF out of depth-tuned neurons:",
        np.min(all_sig),
        "to",
        np.max(all_sig),
    )
    print("Number of sessions:", len(all_sig))
    plt.plot(
        median_prop,
        plt.ylim()[1] * 0.95,
        marker="v",
        markersize=7,
        color=hist_color,
        markeredgecolor=hist_edgecolor,
        markeredgewidth=1,
    )
    plotting_utils.despine()
    plt.tick_params(labelsize=fontsize_dict["tick"])


def plot_rf_3d(neurons_df, rois, depths, savepath, fontsize_dict):
    depth, ele, azi = np.mgrid[0:8, -37.5:37.5:16j, 2.5:117.5:24j]

    data = []
    for roi, rf_color, line_color in zip(
        rois, ["Reds", "Greens", "Blues"], ["red", "green", "blue"]
    ):
        coef = neurons_df.loc[roi, f"rf_coef_closedloop"][:, :-1].copy()
        coef = coef.reshape(coef.shape[0], len(depths), 16, 24)
        coef_mean = np.mean(coef, axis=0)
        data.append(
            go.Volume(
                x=depth.flatten(),
                y=azi.flatten(),
                z=ele.flatten(),
                value=coef_mean.flatten(),
                isomin=coef_mean.max() / 2,
                isomax=coef_mean.max(),
                opacity=0.2,  # needs to be small to see through all surfaces
                surface_count=11,  # needs to be a large number for good volume rendering
                colorscale=rf_color,
            )
        )
        rf_center = np.argmax(coef_mean)
        # add a line from the centre of the RF to the edge along each axis
        data.append(
            go.Scatter3d(
                x=depth.flatten()[[rf_center, 0]],
                y=azi.flatten()[[rf_center, rf_center]],
                z=ele.flatten()[[rf_center, rf_center]],
                mode="lines",
                line=dict(color=line_color, width=2),
            )
        )
        data.append(
            go.Scatter3d(
                x=depth.flatten()[[rf_center, rf_center]],
                y=azi.flatten()[[rf_center, 0]],
                z=ele.flatten()[[rf_center, rf_center]],
                mode="lines",
                line=dict(color=line_color, width=2),
            )
        )
        data.append(
            go.Scatter3d(
                x=depth.flatten()[[rf_center, rf_center]],
                y=azi.flatten()[[rf_center, rf_center]],
                z=ele.flatten()[[rf_center, 0]],
                mode="lines",
                line=dict(color=line_color, width=2),
            )
        )
        data.append(
            go.Scatter3d(
                x=[
                    depth.flatten()[rf_center],
                ],
                y=[
                    azi.flatten()[rf_center],
                ],
                z=[
                    ele.flatten()[rf_center],
                ],
                mode="markers",
                marker=dict(color=line_color, size=5, symbol="circle"),
            )
        )
    fig = go.Figure(data=data)
    font_params = dict(
        title_font_family="Arial",
        title_font_size=fontsize_dict["label"],
        tickfont=dict(
            family="Arial",
            size=fontsize_dict["tick"],
        ),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Depth (cm)",
                tickmode="array",
                tickvals=np.arange(0, 8),
                ticktext=depths * 100,
                **font_params,
            ),
            yaxis=dict(
                title="Azimuth<br>(degrees)",
                nticks=4,
                range=[0, 90],
                tickvals=[0, 45, 90],
                **font_params,
            ),
            zaxis=dict(
                title="Elevation<br>(degrees)",
                nticks=4,
                range=[-20, 20],
                tickvals=[-15, 0, 15],
                **font_params,
            ),
            camera=dict(
                eye=dict(x=1.75, y=1.75, z=1.75),
            ),
        ),
        showlegend=False,
    )
    fig.update_coloraxes(showscale=False)

    fig.show()
    fig.write_image(savepath)


def plot_gradient_polar(
    angles,
    nbins=36,
    plot_x=0,
    plot_y=0,
    plot_width=0.2,
    plot_height=0.2,
    edgecolor="k",
    facecolor="k",
    alpha=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 5},
):
    ax = plt.gcf().add_axes([plot_x, plot_y, plot_width, plot_height], polar=True)
    # Create a histogram
    counts, bins = np.histogram(angles, bins=np.linspace(0, 360, nbins + 1))

    # Convert bins to centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Create the bar plot on the polar axis
    ax.bar(
        np.radians(bin_centers),
        counts,
        width=(2 * np.pi) / nbins,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )

    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(["M", "A", "L", "P"], fontsize=fontsize_dict["label"])
    ax.tick_params(axis="x", which="major", pad=-6)
    ax.set_yticks(ax.get_yticks())
    ax.tick_params(axis="y", which="major", labelsize=fontsize_dict["tick"], pad=0)
