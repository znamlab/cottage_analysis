import flexiznam as flz
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
from itertools import combinations
import json
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
    is_depth_neuron_dist = []

    for i, session in tqdm(processed.iterrows()):
        neurons_path = processed_root / session.path / "neurons_df.pickle"
        save_path = processed_root / session.path / "figures/"

        with open(neurons_path, "rb") as f:
            neurons_df = NumpyFixUnpickler(f).load()

        # Save the is_depth_neuron field
        is_depth_neuron_dist.append(neurons_df["is_depth_neuron"])

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
        spearman = neurons_df["depth_tuning_test_spearmanr_rval_closedloop"]
        spearman_r_dist.append(spearman)

    # Add all computed columns to the DataFrame
    processed["hist_counts"] = hist_counts
    processed["bin_edges"] = bin_edges
    processed["mean_log_depth"] = mean_log_depths
    processed["sem_log_depth"] = sems
    processed["spearman_r_dist"] = spearman_r_dist
    processed["proportion_depthtuned"] = proportion_depthtuned
    processed["log_pref_depth"] = log_pref_depth
    processed["is_depth_neuron_dist"] = is_depth_neuron_dist

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


## HIERARCHICAL BOOTSTRAP

## ROICAT data

# roicat_sessions = {
#    'PZAG16.3c': ['S20250219', 'S20250313'],
#    'PZAG17.3a': ['S20250227', 'S20250228', 'S20250303', 'S20250305', 'S20250306'],
#    'PZAG16.3b': ['S20250224', 'S20250225', 'S20250226', 'S20250310', 'S20250313'],
#    'PZAH17.1e': ['S20250305', 'S20250311']
# }

roicat_sessions = {
    "PZAG16.3c": ["S20250219", "S20250313"],
    "PZAG17.3a": ["S20250227", "S20250228", "S20250303", "S20250305", "S20250306"],
    "PZAG16.3b": ["S20250224", "S20250225", "S20250226", "S20250310", "S20250313"],
    "PZAH17.1e": ["S20250305", "S20250311"],
}


def process_session_pair(
    session_a: str, session_b: str, mouse, save_dir: Path, matched_df, roicat_neuronsdf
):
    save_dir.mkdir(parents=True, exist_ok=True)

    total_matched = len(matched_df)
    useful_matched = filter_matched_df(matched_df)
    n_useful = len(useful_matched)

    # Write counts to a small summary file
    summary_path = save_dir / f"{session_a}__{session_b}__summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"mouse: {mouse}\n")
        f.write(f"sessions: {session_a}, {session_b}\n")
        f.write(f"matched_total: {total_matched}\n")
        f.write(f"useful_matched: {n_useful}\n")

    # If no useful matches, bail out loudly
    if n_useful == 0:
        print(
            f"NO USEFUL MATCHED ROWS FOR {mouse} {session_a} VS {session_b} — SKIPPING PLOTS"
        )
        return

    # Otherwise, make and save the figures
    sessionids = [f"{mouse}_{session_a}", f"{mouse}_{session_b}"]

    connect_figure = plot_connect_depth_preference_across_sessions(
        roicat_neuronsdf, useful_matched, sessionids
    )
    hist_figure = plot_difference_histogram(useful_matched)
    raw_depth_scatter_figure, _ = plot_pref_depth_scatter(useful_matched, use_log=False)
    log_depth_scatter_figure, _ = plot_pref_depth_scatter(useful_matched, use_log=True)

    connect_figure.savefig(save_dir / f"{session_a}__{session_b}__connect.png", dpi=300)
    raw_depth_scatter_figure.savefig(
        save_dir / f"{session_a}__{session_b}__raw_depth_scatter.png", dpi=300
    )
    log_depth_scatter_figure.savefig(
        save_dir / f"{session_a}__{session_b}__log_depth_scatter.png", dpi=300
    )
    hist_figure.savefig(save_dir / f"{session_a}__{session_b}__hist.png", dpi=300)

    print(
        f"Saved figures for {session_a} and {session_b} of mouse {mouse} in {save_dir}"
    )


def run_all_pairs(sessions_by_mouse: dict[str, list[str]]):
    flz_session = flz.get_flexilims_session(PROJECT)

    root_path = flz.get_data_root(
        "processed", project=PROJECT, flexilims_session=flz_session
    )
    root_path = root_path / PROJECT

    for mouse, sessions in sessions_by_mouse.items():
        # Sort for consistent ordering

        sessions = sorted(sessions)

        # for each mouse, generate mouse data:

        processed = find_processed_sessions(mouse)

        roicat_dict = load_roicat_data(mouse)

        roicat_processed = select_roicat_sessions(processed, mouse, roicat_sessions)
        roicat_neuronsdf = generate_roicat_neuronsdf(roicat_processed, roicat_dict)

        # Unique unordered pairs
        for sA, sB in combinations(sessions, 2):
            save_dir = root_path / mouse / "ROICaT" / f"{sA}__{sB}"
            save_dir.mkdir(parents=True, exist_ok=True)
            sessionids = [f"{mouse}_{sA}", f"{mouse}_{sB}"]
            matched_df = match_two_sessions(roicat_neuronsdf, sessionids)

            # ---- Run your analysis code ----
            print(sessionids)
            process_session_pair(sA, sB, mouse, save_dir, matched_df, roicat_neuronsdf)


#### So, the pipeline works as follows. You chose one mouse, then you get the roicat
# dict and the processed sections for that mouse, and then you generate a neuronsdf
# for those roicat processed sessions. The processed sessions are mouse-specific,
# and only then should one go around sessions. So, a nested structure, mice first and
# sessions after.

# An issue is that roicat_neuronsdf and roicat_dict are ordered differently


# make the neuronsdf a dictionary, make the roicat_dict[labels_bySession] a dictionary too
####
def load_roicat_data(mouse):
    # TODO: MAKE THE OUTPUT OF THE ROICAT PIPELINE ALREADY A DICT LIKE THIS
    # so that we don't rely on roicat_sessions being ocnsistnet with the order in ROICAT.
    # Or make ROICAT read from roicat_sessions.

    base_path = "/nemo/lab/znamenskiyp/home/shared/projects/colasa_3d-vision_revisions"
    roicat_path = os.path.join(
        base_path, mouse, "ROICaT", f"{mouse}.tracking.results_clusters.json"
    )

    with open(roicat_path, "r") as f:
        roicat_dict = json.load(f)

    print(f"Loaded ROICaT data for {mouse}")
    print(f"Data type: {type(roicat_dict)}")
    if isinstance(roicat_dict, dict):
        print(f"Top-level keys: {list(roicat_dict.keys())}")

    # make into a dict by sessions:
    labels_dict = {}
    roicat_labels = [f"{mouse}_{session}" for session in roicat_sessions[mouse]]
    for label in roicat_labels:
        labels_dict[label] = roicat_dict["labels_bySession"][roicat_labels.index(label)]
    roicat_dict["labels_bySession"] = labels_dict

    return roicat_dict


def select_roicat_sessions(processed, mouse, roicat_sessions):
    mouse_sessions = roicat_sessions[mouse]

    session_names = []
    for session in mouse_sessions:
        session_names.append(f"{mouse}_{session}")

    roicat_processed = processed[processed["name"].isin(session_names)]

    return roicat_processed


def generate_roicat_neuronsdf(roicat_processed, roicat_dict):
    roicat_neuronsdf = {}
    for i, session in tqdm(roicat_processed.iterrows()):
        neurons_path = processed_root / session.path / "neurons_df.pickle"

        with open(neurons_path, "rb") as f:
            neurons_df = NumpyFixUnpickler(f).load()
            roicat_neuronsdf[session.name] = neurons_df

    # Add the cluster labels and quality metrics to each neurons df

    start_neuron = 0  # running index over neurons
    start_cluster = 0  # running index over clusters

    for key in roicat_neuronsdf.keys():
        ndf = roicat_neuronsdf[key]  # neurons df for this session
        # ----------------- 1.  add cluster_id (already OK) -----------------
        ndf["cluster_id"] = roicat_dict["labels_bySession"][key]

        # ----------------- 2.  slice the two global arrays -----------------
        # how many neurons and how many clusters in this session?
        n_neurons = len(ndf)
        n_clusters = len(np.unique(ndf["cluster_id"]))

        # neuron-level quality metric (one per neuron)
        end_neuron = start_neuron + n_neurons
        ndf["sample_silhouette"] = roicat_dict["quality_metrics"]["sample_silhouette"][
            start_neuron:end_neuron
        ]

        # cluster-level quality metric (one per cluster)
        end_cluster = start_cluster + n_clusters
        clust_sil = roicat_dict["quality_metrics"]["cluster_silhouette"][
            start_cluster:end_cluster
        ]

        # ----------------- 3.  broadcast cluster silhouette to every neuron -----------------
        # map cluster_id → silhouette
        clust_sil_map = dict(
            enumerate(clust_sil)
        )  # assumes clusters are indexed 0..n_clusters-1
        ndf["cluster_silhouette"] = ndf["cluster_id"].map(clust_sil_map)

        # ----------------- 4.  advance the running indices -----------------
        start_neuron = end_neuron
        start_cluster = end_cluster

    return roicat_neuronsdf


def match_two_sessions(roicat_neuronsdf, sessionids=None):
    """
    Takes the info in roicat_dict to build a df with info of individual neurons
    over sessions as columns.
    """

    # Grab the two sessions
    if sessionids is not None:
        df0 = roicat_neuronsdf[sessionids[0]]  # session
        df1 = roicat_neuronsdf[sessionids[1]]
        # session
    else:
        df0 = roicat_neuronsdf[0]  # session 0
        df1 = roicat_neuronsdf[1]  # session 1

    # 1) clusters present in *both* sessions
    common_clusters = np.intersect1d(
        df0["cluster_id"].unique(), df1["cluster_id"].unique()
    )

    rows = []
    for cid in common_clusters:
        # row for this cluster in each session
        row0 = df0[df0["cluster_id"] == cid]
        row1 = df1[df1["cluster_id"] == cid]

        # guard: skip clusters that have more than one neuron per session
        if len(row0) != 1 or len(row1) != 1:
            continue  # or handle as you wish

        row0 = row0.iloc[0]
        row1 = row1.iloc[0]

        rows.append(
            {
                "cluster_id": cid,
                "neuron_idx_session0": row0.name,  # DataFrame index
                "neuron_idx_session1": row1.name,
                "sample_sil_session0": row0["sample_silhouette"],
                "sample_sil_session1": row1["sample_silhouette"],
                "cluster_silhouette": row0["cluster_silhouette"],  # same for both
                "spearman_r_session0": row0[
                    "depth_tuning_test_spearmanr_rval_closedloop"
                ],
                "spearman_r_session1": row1[
                    "depth_tuning_test_spearmanr_rval_closedloop"
                ],
                "pref_depth_session0": row0["preferred_depth_closedloop"],
                "pref_depth_session1": row1["preferred_depth_closedloop"],
                "is_depth_neuron_session0": row0["is_depth_neuron"],
                "is_depth_neuron_session1": row1["is_depth_neuron"],
            }
        )

    matched_df = pd.DataFrame(rows).reset_index(drop=True)
    matched_df.head()

    return matched_df


def get_log_depth(df, raw_key="preferred_depth_closedloop", log_key="log_pref_depth"):
    if log_key in df.columns:
        return df[log_key].values
    if raw_key in df.columns:
        return np.log10(df[raw_key].values)  # natural-log; use np.log10 if you prefer
    raise KeyError("Depth column not found")


def filter_matched_df(
    matched_df,
    cluster_silhouette_threshold=0.2,
    sample_silhouette_threshold=0.1,
    filter_by_depth=True,
):
    """
    Filter matched_df based on silhouette thresholds.

    Parameters:
    - matched_df: DataFrame containing matched clusters between sessions
    - cluster_silhouette_threshold: threshold for cluster silhouette score
    - sample_silhouette_threshold: threshold for sample silhouette score
    - filter_by_depth: whether to keep only clusters that are depth-tuned in both sessions

    Returns:
    -useful_matched: DataFrame with filtered matched clusters

    """
    useful_matched = matched_df[
        (matched_df["cluster_silhouette"] > cluster_silhouette_threshold)
        & (matched_df["sample_sil_session0"] > sample_silhouette_threshold)
        & (matched_df["sample_sil_session1"] > sample_silhouette_threshold)
    ]

    useful_matched["difference"] = (
        useful_matched["pref_depth_session1"] - useful_matched["pref_depth_session0"]
    )

    if filter_by_depth:
        # Keep only clusters that are depth-tuned in both sessions
        useful_matched = useful_matched[
            useful_matched["is_depth_neuron_session0"]
            & useful_matched["is_depth_neuron_session1"]
        ]

    return useful_matched


def plot_connect_depth_preference_across_sessions(
    roicat_neuronsdf, useful_matched, sessionids=None, filter_extreme_depths=False
):
    """
    Plot depth preference of neurons across two sessions, with connecting lines for
    matched clusters.

    Parameters:
    - roicat_neuronsdf: list of DataFrames for each session
    - useful_matched: DataFrame containing matched clusters between sessions,
    filtered by silhouette scores
    - filter_extreme_depths: whether to filter out extreme depth values. Extreme depths
    are log depths of 1 (10m) and -1.5 (0.03m)
    """

    if filter_extreme_depths:
        useful_matched = useful_matched[
            (np.log10(useful_matched["pref_depth_session0"]) < 1)
            & (np.log10(useful_matched["pref_depth_session0"]) > -1.5)
            & (np.log10(useful_matched["pref_depth_session1"]) < 1)
            & (np.log10(useful_matched["pref_depth_session1"]) > -1.5)
        ]

    if sessionids is not None:
        df0 = roicat_neuronsdf[sessionids[0]]  # session
        df1 = roicat_neuronsdf[sessionids[1]]
        # session
    else:
        df0 = roicat_neuronsdf[0]
        df1 = roicat_neuronsdf[1]

    y0 = get_log_depth(df0)
    y1 = get_log_depth(df1)

    # add a tiny horizontal jitter so points don’t overlap perfectly
    x0 = np.random.normal(loc=0, scale=0.04, size=len(y0))
    x1 = np.random.normal(loc=1, scale=0.04, size=len(y1))

    fig, ax = plt.subplots(figsize=(4, 6))

    ax.scatter(x0, y0, alpha=0.4, color="steelblue", s=10)
    ax.scatter(x1, y1, alpha=0.4, color="coral", s=10)

    # Plot connecting lines for every matched cluster
    for _, row in useful_matched.iterrows():
        # guarantee log depth
        d0 = np.log10(row["pref_depth_session0"])
        d1 = np.log10(row["pref_depth_session1"])
        ax.plot([0, 1], [d0, d1], color="gray", alpha=0.6, linewidth=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Session 0", "Session 1"])
    ax.set_ylabel("log (pref-depth)")
    if filter_extreme_depths:
        ax.set_title(
            "Depth preference of neurons across sessions\n(connected lines = same cluster)\nExtreme depths filtered out"
        )
    else:
        ax.set_title(
            "Depth preference of neurons across sessions\n(connected lines = same cluster)"
        )

    # make it pretty
    ax.spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    return fig


def plot_difference_histogram(useful_matched, n_shuffles=50):
    """
    Plot histogram of absolute differences in depth preference between sessions,
    overlaying histograms of null distributions from shuffled data.

    Parameters:
    - useful_matched: DataFrame containing matched clusters with depth preferences,
                    filtered by silhouette scores
    - n_shuffles: Number of shuffles to create null distributions (default: 50)

    Returns:
    - fig: matplotlib Figure object (so the caller can save it)
    """

    # Initialize
    null_differences = np.zeros((len(useful_matched), n_shuffles))

    # Extract original depth arrays
    pref0 = useful_matched["pref_depth_session0"].values
    pref1 = useful_matched["pref_depth_session1"].values

    # Shuffle independently
    for i in range(n_shuffles):
        shuffled_pref0 = np.random.permutation(pref0)
        shuffled_pref1 = np.random.permutation(pref1)
        null_differences[:, i] = shuffled_pref1 - shuffled_pref0

    # diffs: real differences (Session 1 - Session 0)
    diffs = useful_matched["difference"]

    # Create figure/axes explicitly
    fig, ax = plt.subplots(figsize=(6, 4))

    # 1. Plot histogram of the real absolute differences and get bins
    counts, bins, patches = ax.hist(np.abs(diffs), bins=50, color="blue", alpha=0.7)
    ax.set_xscale("log")

    # 2. Overlay very light red lines for each null distribution
    for i in range(null_differences.shape[1]):
        shuffled_diff = null_differences[:, i]
        ax.hist(
            np.abs(shuffled_diff), bins=bins, color="red", alpha=0.5, histtype="step"
        )

    ax.set_title("Absolute Difference in depth preference (real vs null)")
    ax.set_xlabel("Absolute Difference in Depth (m)")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    return fig


def plot_pref_depth_scatter(useful_matched, use_log=True, show=False):
    """
    Scatter of preferred depth between Session 0 and Session 1 with a regression line.
    - use_log=True: uses log10(pref_depth_*). Non-positive or non-finite values are dropped.
    - use_log=False: uses raw pref_depth_*.
    Returns: (fig, (slope, intercept, r_squared))
    """

    x_raw = useful_matched["pref_depth_session0"].to_numpy()
    y_raw = useful_matched["pref_depth_session1"].to_numpy()

    if use_log:
        mask = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0) & (y_raw > 0)
        x = np.log10(x_raw[mask])
        y = np.log10(y_raw[mask])
        xlabel = "Log10 Preferred Depth Session 0"
        ylabel = "Log10 Preferred Depth Session 1"
        title = "Log Preferred Depths of Matched Neurons Across Two Sessions"
    else:
        mask = np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x_raw[mask]
        y = y_raw[mask]
        xlabel = "Preferred Depth Session 0"
        ylabel = "Preferred Depth Session 1"
        title = "Preferred Depths of Matched Neurons Across Two Sessions"

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        x,
        y,
        alpha=0.5,
        c=useful_matched.loc[mask, "cluster_silhouette"],
        cmap="viridis",
        s=20,
    )

    # Fit line and R^2
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color="red")

    r = np.corrcoef(x, y)[0, 1]
    r_squared = float(r**2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate stats on the figure
    ax.text(
        0.05,
        0.95,
        f"Slope: {m:.3f}\nIntercept: {b:.3f}\nR²: {r_squared:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.2),
    )

    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cluster Silhouette")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, (m, b, r_squared)
