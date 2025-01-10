import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import flexiznam as flz
from cottage_analysis.analysis import spheres
from cottage_analysis.analysis import roi_location
from cottage_analysis.plotting import plotting_utils
from cottage_analysis.pipelines import pipeline_utils
import plotly.graph_objects as go
import os


def plot_stimulus_frame(
    frame,
    idepth,
    depths,
    position=(0, 0, 1, 1),
    plot_prop=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
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
        if i == idepth:
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
    fontsize_dict={"title": 15, "label": 10, "tick": 5},
):
    if is_closed_loop:
        sfx = "_closedloop"
    else:
        sfx = "_openloop"
    coef = neurons_df.loc[roi, f"rf_coef{sfx}"][:, :-1]
    coef = coef.reshape(coef.shape[0], ndepths, frame_shape[0], frame_shape[1])
    coef_mean = np.mean(coef, axis=0)
    coef_max = np.nanmax(coef_mean)
    plot_x, plot_y, plot_width, plot_height = position
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
            extent=[0, 120, -40, 40],
            vmin=-np.round(coef_max, 1),
            vmax=np.round(coef_max, 1),
        )
        if i != ndepths - 1:
            plt.gca().set_xticklabels([])
        if i == ndepths // 2:
            ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
        if i == ndepths - 1:
            ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", labelsize=fontsize_dict["tick"], length=1.5)
        ax.set_xticks([0, 60, 120])

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


def get_rf_results(project, sessions, is_closed_loop=1):
    if is_closed_loop:
        sfx = "_closedloop"
    else:
        sfx = "_openloop"
    for i, session_name in enumerate(sessions):
        flexilims_session = flz.get_flexilims_session(project_id=project)
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        results = pd.DataFrame(
            {
                "session": np.nan,
                "roi": np.nan,
                "iscell": np.nan,
                "preferred_depth": np.nan,
                "preferred_depth_rsq": np.nan,
                "coef": [[np.nan]] * len(neurons_df),
                "coef_ipsi": [[np.nan]] * len(neurons_df),
            }
        )

        # Add roi, preferred depth, iscell to results
        results["roi"] = np.arange(len(neurons_df))
        results["session"] = session_name
        results["preferred_depth"] = neurons_df["preferred_depth_closedloop"]
        results["preferred_depth_rsq"] = neurons_df["depth_tuning_test_rsq_closedloop"]
        exp_session = flz.get_entity(
            datatype="session", name=session_name, flexilims_session=flexilims_session
        )
        suite2p_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=exp_session.name,
            dataset_type="suite2p_rois",
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            return_dataseries=False,
        )
        iscell = np.load(
            suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True
        )[:, 0]
        results["iscell"] = iscell

        # Add coef to results
        results[f"rf_coef{sfx}"] = neurons_df[f"rf_coef{sfx}"]
        results[f"rf_coef_ipsi{sfx}"] = neurons_df[f"rf_coef_ipsi{sfx}"]

        if i == 0:
            results_all = results
        else:
            results_all = pd.concat([results_all, results], axis=0, ignore_index=True)

    return results_all


def find_rf_centers(
    neurons_df,
    ndepths=8,
    frame_shape=(16, 24),
    is_closed_loop=1,
    resolution=5,
):
    if is_closed_loop:
        sfx = "_closedloop"
    else:
        sfx = "_openloop"
    coef = np.stack(neurons_df[f"rf_coef{sfx}"].values)
    coef_ = (coef[:, :, :-1]).reshape(
        coef.shape[0], coef.shape[1], ndepths, frame_shape[0], frame_shape[1]
    )
    coef_mean = np.mean(coef_, axis=1)

    # Find the center (index of maximum value of fitted RF)
    max_idx = [
        np.unravel_index(coef_mean[i, :, :].argmax(), coef_mean[0, :, :].shape)
        for i in range(coef_mean.shape[0])
    ]
    max_idx = np.array(max_idx)

    def index_to_deg(idx, resolution=resolution, n_ele=80):
        azi = (idx[:, 2] + 0.5) * resolution
        ele = (idx[:, 1] + 0.5 - n_ele / 2) * resolution
        return azi, ele

    azi, ele = index_to_deg(max_idx, n_ele=frame_shape[0])
    idepth = max_idx[:, 0]
    neurons_df["rf_azi"] = azi
    neurons_df["rf_ele"] = ele
    return azi, ele, idepth, coef


def plot_rf_centers(
    fig,
    results,
    is_closed_loop=1,
    colors=["r", "b"],
    ndepths=8,
    frame_shape=(16, 24),
    n_stds=5,
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
        azi, ele, _, coef = find_rf_centers(
            neurons_df=results_sess,
            is_closed_loop=is_closed_loop,
            ndepths=ndepths,
            frame_shape=frame_shape,
            resolution=5,
        )

        coef_ipsi = np.stack(results_sess[f"rf_coef_ipsi{sfx}"].values)

        # Find cells with significant RF
        sig, _ = spheres.find_sig_rfs(
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


def load_sig_rf(
    flexilims_session,
    session_list,
    use_cols=[
        "roi",
        "is_depth_neuron",
        "best_depth",
        "preferred_depth_closedloop",
        "preferred_depth_closedloop_crossval",
        "depth_tuning_test_rsq_closedloop",
        "depth_tuning_test_spearmanr_rval_closedloop",
        "depth_tuning_test_spearmanr_pval_closedloop",
        "rf_coef_closedloop",
        "rf_coef_ipsi_closedloop",
        "rf_rsq_closedloop",
        "rf_rsq_ipsi_closedloop",
        "preferred_RS_closedloop_g2d",
        "preferred_RS_closedloop_crossval_g2d",
        "preferred_OF_closedloop_g2d",
        "preferred_OF_closedloop_crossval_g2d",
        "rsof_test_rsq_closedloop_g2d",
        "rsof_rsq_closedloop_g2d",
        "rsof_popt_closedloop_g2d",
    ],
    n_std=5,
    verbose=1,
):
    all_sig = []
    all_sig_ipsi = []
    isess = 0
    neurons_df_all = []
    for session in session_list:
        # get session
        session_series = flz.get_entity(
            datatype="session", name=session, flexilims_session=flexilims_session
        )
        if (
            "exclude_reason" in session_series
            and session_series["exclude_reason"] == "not V1"
        ):
            v1 = False
        else:
            v1 = True
        # Load neurons_df
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        try:
            neurons_df = pd.read_pickle(neurons_ds.path_full)
        except FileNotFoundError:
            print(f"ERROR: SESSION {session}: neurons_df not found")
            continue

        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df
            else:
                neurons_df = neurons_df[use_cols]

            # Load iscell
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=session,
                dataset_type="suite2p_rois",
                filter_datasets={"anatomical_only": 3},
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = np.load(
                suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True
            )[:, 0]
            neurons_df["iscell"] = iscell
            neurons_df["session"] = session
            roi_location.determine_roi_locations(
                neurons_df, flexilims_session, session, suite2p_ds
            )
            # Load RF significant %
            coef = np.stack(neurons_df["rf_coef_closedloop"].values)
            coef_ipsi = np.stack(neurons_df["rf_coef_ipsi_closedloop"].values)
            if coef_ipsi.ndim == 3:
                sig, sig_ipsi = spheres.find_sig_rfs(
                    np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
                    np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
                    n_std=n_std,
                )
                neurons_df["rf_sig"] = sig
                neurons_df["rf_sig_ipsi"] = sig_ipsi
                select_neurons = (
                    (neurons_df["iscell"] == 1)
                    & (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
                    & (neurons_df["depth_tuning_test_spearmanr_rval_closedloop"] > 0.1)
                )
                sig = sig[select_neurons]
                sig_ipsi = sig_ipsi[select_neurons]
                all_sig.append(np.mean(sig))
                all_sig_ipsi.append(np.mean(sig_ipsi))
                if ("PZAH6.4b" in session) or ("PZAG3.4f" in session):
                    ndepths = 5
                else:
                    ndepths = 8
                azi, ele, _, _ = find_rf_centers(
                    neurons_df,
                    ndepths=ndepths,
                    frame_shape=(16, 24),
                    is_closed_loop=1,
                    resolution=5,
                )
                neurons_df["rf_azi"] = azi
                neurons_df["rf_ele"] = ele
                neurons_df["v1"] = v1
                neurons_df_all.append(neurons_df)
                if verbose:
                    print(f"SESSION {session} concatenated")
                isess += 1
            else:
                print(
                    f"ERROR: SESSION {session}: rf_coef_closedloop and rf_coef_ipsi_closedloop not all 3D"
                )

        else:
            print(f"ERROR: SESSION {session}: specified cols not all in neurons_df")
    neurons_df_all = pd.concat(neurons_df_all, axis=0, ignore_index=True)
    return all_sig, all_sig_ipsi, neurons_df_all


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
        coef = neurons_df.loc[roi, f"rf_coef_closedloop"][:, :-1]
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


def calculate_rf_gradient(
    flexilims_session,
    neurons_ds,
    neurons_df,
    session_name,
    n_std=6,
):
    # load iscell
    suite2p_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        allow_multiple=False,
        return_dataseries=False,
    )
    iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[
        :, 0
    ]
    neurons_df["iscell"] = iscell

    # calculate gradients of azimuth and elevation
    session_df = pd.DataFrame([session_name], columns=["session_name"])
    coef = np.stack(neurons_df[f"rf_coef_closedloop"].values)
    coef_ipsi = np.stack(neurons_df[f"rf_coef_ipsi_closedloop"].values)
    sig, _ = spheres.find_sig_rfs(
        np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
        np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
        n_std=n_std,
    )
    select_neurons = neurons_df[(sig == 1) & (neurons_df["iscell"] == 1)]
    null_neurons = neurons_df[(sig == 0) & (neurons_df["iscell"] == 1)]
    # find the gradient of col w.r.t. center_x and center_y
    for neurons, sfx in zip([select_neurons, null_neurons], ["rf_sig", "rf_null"]):
        for col, colname in zip(
            ["rf_azi", "rf_ele", "preferred_depth_closedloop"],
            ["azi", "ele", "depth"],
        ):
            slope_x = scipy.stats.linregress(
                x=neurons["center_x"], y=neurons[col]
            ).slope
            slope_y = scipy.stats.linregress(
                x=neurons["center_y"], y=neurons[col]
            ).slope
            norm = np.linalg.norm(np.array([slope_x, slope_y]))
            slope_x /= norm
            slope_y /= norm
            session_df[f"slope_x_{colname}_{sfx}"] = slope_x
            session_df[f"slope_y_{colname}_{sfx}"] = slope_y

    # save file
    session_df.to_pickle(neurons_ds.path_full.parent / "rf_gradients.pkl")
    return session_df


def calculate_rf_gradient_all_sessions(
    flexilims_session,
    session_list,
    neurons_df_all_aligned,
    filename="rf_gradients.pkl",
    conflicts="skip",
    verbose=False,
):
    for isess, session in enumerate(session_list):
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        if os.path.exists(neurons_ds.path_full.parent / filename) and (
            conflicts == "skip"
        ):
            # print(f"File {filename} already exists. Skipping calculation for session {session}")
            session_df = pd.read_pickle(neurons_ds.path_full.parent / filename)
        else:
            neurons_df = neurons_df_all_aligned[
                neurons_df_all_aligned.session == session
            ]
            session_df = calculate_rf_gradient(
                flexilims_session=flexilims_session,
                neurons_ds=neurons_ds,
                neurons_df=neurons_df,
                session_name=session,
                n_std=6,
            )
        if isess == 0:
            session_df_all = session_df
        else:
            session_df_all = pd.concat([session_df_all, session_df], ignore_index=True)
        if verbose:
            print(f"Finished concat rf gradient from session {session}")

    return session_df_all


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
