import flexiznam as flz
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm
import pickle
from scipy.stats import gaussian_kde, mannwhitneyu


PROJECT = "colasa_3d-vision_revisions"
flz_session = flz.get_flexilims_session(PROJECT)

processed_root = flz.get_data_root(
    "processed", project=PROJECT, flexilims_session=flz_session
)


def build_sessions_df(micelist):
    """
    Makes a df called processed which contains onw row for each processed session, and fields for interesting
    variables when it comes to plot stuff. It can also be used, while you're iterating, to generate
    useful session statistics plots for each session.

    Args:
        micelist (list of str): a list of mouse names in flexilims

    Out:
        processed (df): a pandas df, onw row per session.
    """

    # generate the barebones processed dataframe
    mouse_i = 0
    for mouse in micelist:
        if mouse_i == 0:
            processed = find_processed_sessions(mouse)
        else:
            processed = pd.concat(
                [find_processed_sessions(mouse), processed], ignore_index=True
            )
        mouse_i += 1

    # Initialize storage lists
    hist_counts = []
    bin_edges = []
    mean_log_depths = []
    sems = []
    spearman_r_dist = []
    proportion_depthtuned = []
    log_pref_depth = []

    for i, session in tqdm(processed.iterrows()):
        neurons_path = processed_root / session.path / "neurons_df.pickle"
        save_path = processed_root / session.path / "figures/"

        with open(neurons_path, "rb") as f:
            neurons_df = NumpyFixUnpickler(f).load()

        sig_neurons_df, proportion = save_significant_neurons(
            neurons_df, return_proportion=True
        )
        proportion_depthtuned.append(proportion)

        # Compute mean and SEM of log_pref_depth for significant neurons
        mean_log_depth = np.mean(sig_neurons_df["log_pref_depth"])
        sem = np.std(sig_neurons_df["log_pref_depth"], ddof=1) / np.sqrt(
            len(sig_neurons_df)
        )

        log_pref_depth.append(sig_neurons_df["log_pref_depth"])
        mean_log_depths.append(mean_log_depth)
        sems.append(sem)

        # Get pref depth histogram
        hist_count, bin_edge = save_histogram(
            sig_neurons_df, session.name, save_path, print_figure=False
        )
        hist_counts.append(hist_count)
        bin_edges.append(bin_edge)

        # Get Spearman r distribution
        spearman = sig_neurons_df["depth_tuning_test_spearmanr_rval_closedloop"]
        spearman_r_dist.append(spearman)

    # Add all computed columns to the DataFrame
    processed["hist_counts"] = hist_counts
    processed["bin_edges"] = bin_edges
    processed["mean_log_depth"] = mean_log_depths
    processed["sem_log_depth"] = sems
    processed["spearman_r_dist"] = spearman_r_dist
    processed["proportion_depthtuned"] = proportion_depthtuned
    processed["log_pref_depth"] = log_pref_depth

    # Add mice and date names
    mice = []
    dates = []
    for i, session in processed.iterrows():
        name = session["name"]
        namelist = name.split("_")
        mice.append(namelist[0])
        dates.append(namelist[1])

    processed["mouse"] = mice
    processed["date"] = dates

    # Add exposure dates

    # Ensure 'date' is treated as a string or int for sorting (yyyymmdd format is naturally sortable)
    processed = processed.sort_values(by=["mouse", "date"]).reset_index(drop=True)

    # Create the exposure_day column by grouping by mouse and ranking the date
    processed["exposure_day"] = (
        processed.groupby("mouse")["date"]
        .rank(method="dense")  # or method='first' if dates could repeat
        .astype(int)
        - 1  # Start from 0
    )

    return processed


def find_processed_sessions(mouse, protocol="SpheresPermTubeReward"):
    # Check all sessions
    sessions = flz.get_children(
        parent_name=mouse,
        children_datatype="session",
        project_id=PROJECT,
        flexilims_session=flz_session,
    )
    # print(sessions.name)
    # List the sessions that are SphereTube
    for i in sessions.name:
        SphereTube_recordings = flz.get_children(
            parent_name=i,
            children_datatype="recording",
            project_id=PROJECT,
            flexilims_session=flz_session,
        )
        SphereTube_recordings = SphereTube_recordings[
            SphereTube_recordings["protocol"] == "SpheresPermTubeReward"
        ]
        if len(SphereTube_recordings) == 0:
            sessions = sessions[sessions["name"] != i]

    # Keep the sessions that are processed
    for i, session in sessions.iterrows():
        # print(session)
        neurons_path = processed_root / session.path / "neurons_df.pickle"
        if not os.path.isfile(neurons_path):
            name_to_drop = session.name
            sessions = sessions[sessions["name"] != name_to_drop]

    return sessions


def save_significant_neurons(neurons_df, return_proportion=False):
    """ """
    # Filter significantly depth-tuned neurons
    sig_neurons_df = neurons_df[neurons_df["is_depth_neuron"]].copy()

    # Compute log preferred depth (in cm)
    sig_neurons_df["log_pref_depth"] = np.log10(
        sig_neurons_df["preferred_depth_closedloop"] * 100
    )

    if return_proportion:
        proportion = len(sig_neurons_df) / len(neurons_df)
        return sig_neurons_df, proportion
    else:
        return sig_neurons_df


def save_histogram(
    sig_neurons_df,
    session,
    path="figures/",
    print_figure=True,
    property="log_pref_depth",
):
    """
    Saves a histogram of a property of significantly depth-tuned neurons
    and returns an alternative histogram with 10 bins as a NumPy array.

    Args:
        neurons_df (pd.DataFrame): DataFrame containing neuron data.
        mouse (str): Mouse identifier.
        session (int): Session number.
        path (str, optional): Directory to save the figure. Defaults to "figures/".

    Returns:
        np.array: Histogram data with 10 bins.
    """

    # Ensure path exists
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    if print_figure:
        # Save the histogram with 100 bins
        plt.figure(figsize=(8, 6))
        plt.hist(sig_neurons_df[property], bins=100, color="blue", alpha=0.7)
        plt.ylabel("Frequency")
        if property == "log_pref_depth":
            plt.xlabel(f"Log Preferred Depth (cm)")
        else:
            plt.xlabel(property)
        plt.title(f"Session {session}, Significantly Depth-Tuned Neurons")

        # Save figure
        filename = save_path / f"session_{session}_{property}_hist.png"
        plt.savefig(filename, dpi=300)
        plt.close()

    # Generate alternative histogram with 10 bins
    hist_counts, bin_edges = np.histogram(sig_neurons_df[property], bins=10)

    return hist_counts, bin_edges


class NumpyFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric":
            module = "numpy.core.numeric"
        return super().find_class(module, name)


## STATS


def test_changes_over_days(processed, property="spearman_r_dist", mice=None):
    if mice is None:
        mice = processed["mouse"].unique()

    p_values = []
    u_values = []

    for mouse in mice:
        mouse_data = processed[processed["mouse"] == mouse]

        day_range = list(
            range(1, max(mouse_data["exposure_day"]) + 1)
        )  # To compare curr day with the day before
        mouse_p_values = []
        mouse_u_values = []

        for day in day_range:
            spearmans_today = np.concatenate(
                mouse_data[mouse_data["exposure_day"] == day]["spearman_r_dist"].values
            )
            spearmans_yesterday = np.concatenate(
                mouse_data[mouse_data["exposure_day"] == day - 1][
                    "spearman_r_dist"
                ].values
            )

            res = mannwhitneyu(spearmans_today, spearmans_yesterday)

            mouse_p_values.append(res.pvalue)
            mouse_u_values.append(res.statistic)

        p_values.append(mouse_p_values)
        u_values.append(mouse_u_values)

    return p_values, u_values


## PLOTTING


def plot_log_depth_over_days(processed, mice=None):
    """
    Plot mean log depth with SEM over exposure days for each mouse.

    Parameters:
    - processed: DataFrame with columns ['mouse', 'exposure_day', 'mean_log_depth', 'sem_log_depth']
    - mice: optional list of mouse IDs to plot (default: all unique mice in `processed`)
    """
    if mice is None:
        mice = processed["mouse"].unique()

    fig, ax = plt.subplots()

    for mouse in tqdm(mice, desc="Plotting mice"):
        mouse_data = processed[processed["mouse"] == mouse]

        ax.errorbar(
            mouse_data["exposure_day"],
            mouse_data["mean_log_depth"],
            yerr=mouse_data["sem_log_depth"],
            label=mouse,
            capsize=3,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    ax.set_xlabel("Exposure Day")
    ax.set_ylabel("Mean Log Depth")
    ax.set_title("Mean Log Depth over Exposure Days by Mouse")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig


def plot_spearman_r_kde_by_day(processed):
    """One subplot per exposure day showing KDEs of Spearman r values for all mice."""

    exposure_days = sorted(processed["exposure_day"].unique())
    mice = sorted(processed["mouse"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(mice)))  # consistent colors per mouse
    mouse_color_map = dict(zip(mice, colors))

    fig, axes = plt.subplots(
        len(exposure_days), 1, figsize=(10, 2.5 * len(exposure_days)), sharex=True
    )

    if len(exposure_days) == 1:
        axes = [axes]  # Make iterable if only one subplot

    for ax, day in zip(axes, exposure_days):
        day_data = processed[processed["exposure_day"] == day]

        for _, row in day_data.iterrows():
            mouse = row["mouse"]
            color = mouse_color_map[mouse]
            dist = row["spearman_r_dist"]

            if len(dist) < 2:
                continue  # can't KDE on 1 point

            kde = gaussian_kde(dist)
            r_range = np.linspace(-1, 1, 200)
            kde_vals = kde(r_range)

            # Plot KDE
            ax.plot(r_range, kde_vals, label=mouse, color=color, alpha=0.7)

            # Plot vertical line at median
            ax.axvline(np.median(dist), color=color, linestyle="--", alpha=0.7)

        ax.set_ylabel(f"Day {day}")
        ax.grid(True)

    axes[-1].set_xlabel("Spearman r (depth tuning)")
    axes[0].set_title("KDE of Spearman r by Mouse for Each Exposure Day")

    # Legend: only once
    handles = [plt.Line2D([0], [0], color=mouse_color_map[m], label=m) for m in mice]
    axes[0].legend(
        handles=handles, title="Mouse", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    plt.tight_layout()
    plt.show()

    return fig


def plot_selective_proportion_over_days(processed, mice=None):
    """
    Plot proportion of depth-tuned (selective) neurons over exposure days for each mouse.

    Parameters:
    - processed: DataFrame with columns ['mouse', 'exposure_day', 'proportion_depthtuned']
    - mice: optional list of mouse IDs to plot (default: all unique mice in `processed`)

    Returns:
    - fig: the matplotlib figure object
    """
    if mice is None:
        mice = processed["mouse"].unique()

    fig, ax = plt.subplots()

    for mouse in tqdm(mice, desc="Plotting mice"):
        mouse_data = processed[processed["mouse"] == mouse]

        ax.plot(
            mouse_data["exposure_day"],
            mouse_data["proportion_depthtuned"],
            label=mouse,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    ax.set_xlabel("Exposure Day")
    ax.set_ylabel("Proportion of Depth-Tuned Neurons")
    ax.set_title("Proportion of Selective Neurons over Exposure Days by Mouse")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig


def plot_pvalues_over_days(p_values, mice):
    """
    Plot p-values over exposure days for each mouse.

    Parameters:
    - p_values: list of lists, one per mouse
    - mice: list of mouse IDs in same order as p_values
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for mouse_pvals, mouse in zip(p_values, mice):
        ax.plot(
            range(1, len(mouse_pvals) + 1),
            mouse_pvals,
            label=mouse,
            marker="o",
            linestyle="-",
        )

    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_xlabel("Day")
    ax.set_ylabel("Mann-Whitney U p-value")
    ax.set_title("Day-to-Day Spearman r Distribution Changes (Mann-Whitney U)")
    ax.legend(title="Mouse")
    ax.set_yscale(
        "log"
    )  # Optional: log scale for better visibility if p-values vary widely
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig
