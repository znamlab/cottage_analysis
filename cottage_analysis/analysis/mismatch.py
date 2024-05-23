from functools import partial
import flexiznam as flz
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


print = partial(print, flush=True)
from sklearn.model_selection import StratifiedKFold, train_test_split

from cottage_analysis.preprocessing import synchronisation


def format_imaging_df(recording, imaging_df):
    """Format sphere params in imaging_df.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.

    Returns:
        DataFrame: contains information for each monitor frame and vis-stim.

    """

    # Indicate whether it's a closed loop or open loop session
    if "Playback" in recording.name:
        imaging_df["closed_loop"] = 0
    else:
        imaging_df["closed_loop"] = 1
    imaging_df["RS"] = (
        imaging_df.mouse_z_harp.diff() / imaging_df.mouse_z_harptime.diff()
    )
    # average RS eye for each imaging volume
    imaging_df["RS_eye"] = (
        imaging_df.mismatch_mouse_z.diff() / imaging_df.monitor_harptime.diff()
    )
    # imaging_df.depth = imaging_df.depth / 100  # convert cm to m
    # OF for each imaging volume
    # imaging_df["OF"] = imaging_df.RS_eye / imaging_df.depth
    return imaging_df


def generate_trials_df(recording, imaging_df):
    """Generate a DataFrame that contains information for each trial.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        imaging_df(pd.DataFrame): dataframe that contains info for each imaging volume.

    Returns:
        DataFrame: contains information for each trial.

    """

    trials_df = pd.DataFrame(
        columns=[
            "trial_no",
            "depth",
            "recording_name",
            "closed_loop",
            "imaging_harptime_stim_start",
            "imaging_harptime_stim_stop",
            "imaging_harptime_blank_start",
            "imaging_harptime_blank_stop",
            "imaging_stim_start",
            "imaging_stim_stop",
            "imaging_blank_start",
            "imaging_blank_stop",
            "imaging_blank_pre_start",
            "imaging_blank_pre_stop",
            "RS_stim",  # actual running speed, m/s
            "RS_blank",
            "RS_blank_pre",
            "RS_eye_stim",  # virtual running speed, m/s
            "OF_stim",  # optic flow speed = RS/depth, rad/s
            "dff_stim",
            "dff_blank",
            "dff_blank_pre",
            "mouse_z_harp_stim",
            "mouse_z_harp_blank",
            "mouse_z_harp_blank_pre",
        ]
    )

    # Find the change of depth
    imaging_df["stim"] = np.nan
    imaging_df.loc[imaging_df.depth.notnull(), "stim"] = 1
    imaging_df.loc[imaging_df.depth < 0, "stim"] = 0
    imaging_df_simple = imaging_df[
        (imaging_df["stim"].diff() != 0) & (imaging_df["stim"]).notnull()
    ].copy()
    imaging_df_simple.depth = np.round(imaging_df_simple.depth, 2)

    # Find frame or volume of imaging_df for trial start and stop
    # (depending on whether return_volume=True in generate_imaging_df)
    blank_time = 10
    start_volume_stim = imaging_df_simple[
        (imaging_df_simple["stim"] == 1)
    ].imaging_volume.values
    start_volume_blank = imaging_df_simple[
        (imaging_df_simple["stim"] == 0)
    ].imaging_volume.values
    if start_volume_blank[0] < start_volume_stim[0]:
        print("Warning: blank starts before stimulus starts! Double check!")
        start_volume_blank = start_volume_blank[1:]
        assert (
            start_volume_blank[0] > start_volume_stim[0]
        ), "Warning: 2 blank starts before stimulus starts! Double check!"

    if len(start_volume_stim) != len(
        start_volume_blank
    ):  # if trial start and blank numbers are different
        if (
            len(start_volume_stim) - len(start_volume_blank)
        ) == 1:  # last trial is not complete when stopping the recording
            stop_volume_blank = start_volume_stim[1:] - 1
            start_volume_stim = start_volume_stim[: len(start_volume_blank)]
        else:  # something is wrong
            print("Warning: incorrect stimulus trial structure! Double check!")
    else:  # if trial start and blank numbers are the same
        stop_volume_blank = start_volume_stim[1:] - 1
        last_blank_stop_time = (
            imaging_df.loc[start_volume_blank[-1]].imaging_harptime + blank_time
        )
        stop_volume_blank = np.append(
            stop_volume_blank,
            (np.abs(imaging_df.imaging_harptime - last_blank_stop_time)).idxmin(),
        )
    stop_volume_stim = start_volume_blank - 1
    start_volume_blank_pre = np.append(0, start_volume_blank[:-1])
    stop_volume_blank_pre = start_volume_stim - 1
    # Assign trial no, depth, start/stop time, start/stop imaging volume to trials_df
    # harptime are imaging trigger harp time
    trials_df.trial_no = np.arange(len(start_volume_stim))
    trials_df.depth = pd.Series(imaging_df.loc[start_volume_stim].depth.values)
    trials_df.imaging_harptime_stim_start = imaging_df.loc[
        start_volume_stim
    ].imaging_harptime.values
    trials_df.imaging_harptime_stim_stop = imaging_df.loc[
        stop_volume_stim
    ].imaging_harptime.values
    trials_df.imaging_harptime_blank_start = imaging_df.loc[
        start_volume_blank
    ].imaging_harptime.values
    trials_df.imaging_harptime_blank_stop = imaging_df.loc[
        stop_volume_blank
    ].imaging_harptime.values

    trials_df.imaging_stim_start = pd.Series(start_volume_stim)
    trials_df.imaging_stim_stop = pd.Series(stop_volume_stim)
    trials_df.imaging_blank_start = pd.Series(start_volume_blank)
    trials_df.imaging_blank_stop = pd.Series(stop_volume_blank)
    trials_df.imaging_blank_pre_start = pd.Series(start_volume_blank_pre)
    trials_df.imaging_blank_pre_stop = pd.Series(stop_volume_blank_pre)
    # If the blank stop of last trial is beyond the number of imaging frames
    if np.isnan(trials_df.imaging_blank_stop.iloc[-1]):
        trials_df.imaging_blank_stop.iloc[-1] = len(imaging_df) - 1
    # Get rid of the overlap of imaging frame no. between different trials
    mask = trials_df.imaging_stim_start == trials_df.imaging_blank_stop.shift(1)
    trials_df.loc[mask, "imaging_stim_start"] += 1

    # Assign protocol to trials_df
    if "Playback" in recording.name:
        trials_df.closed_loop = 0
    else:
        trials_df.closed_loop = 1

    def assign_values_to_df(trials_df, imaging_df, column_name, epoch):
        trials_df[f"{column_name}_{epoch}"] = trials_df.apply(
            lambda x: imaging_df[column_name]
            .loc[int(x[f"imaging_{epoch}_start"]) : int(x[f"imaging_{epoch}_stop"])]
            .values,
            axis=1,
        )
        return trials_df

    columns_to_assign = ["mouse_z_harp", "mouse_z_harp", "RS", "RS_eye", "OF"]
    for epoch in ["stim", "blank", "blank_pre"]:
        for column in columns_to_assign:
            if column != "OF" or column != "RS_eye" or epoch == "stim":
                trials_df = assign_values_to_df(trials_df, imaging_df, column, epoch)
        trials_df[f"dff_{epoch}"] = trials_df.apply(
            lambda x: np.stack(
                imaging_df.dffs.loc[
                    int(x[f"imaging_{epoch}_start"]) : int(x[f"imaging_{epoch}_stop"])
                ]
            ).squeeze(),
            axis=1,
        )

    # Add recording name
    trials_df.recording_name = recording.genealogy[-1]
    # Rename
    trials_df = trials_df.drop(columns=["imaging_blank_start"])

    return trials_df


def sync_all_recordings(
    session_name,
    flexilims_session=None,
    project=None,
    filter_datasets=None,
    recording_type="two_photon",
    protocol_base="SpheresPermTubeReward",
    photodiode_protocol=5,
    return_volumes=True,
    harp_is_in_recording=True,
    use_onix=False,
    conflicts="skip",
    sync_kwargs=None,
    ephys_kwargs=None,
):
    """Concatenate synchronisation results for all recordings in a session.

    Args:
        session_name (str): {mouse}_{session}
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.
        filter_datasets (dict): dictionary of filter keys and values to filter for the desired suite2p dataset (e.g. {'anatomical':3}) Default to None.
        recording_type (str, optional): Type of the recording. Defaults to "two_photon".
        protocol_base (str, optional): Base of the protocol. Defaults to "SpheresPermTubeReward".
        photodiode_protocol (int): number of photodiode quad colors used for monitoring frame refresh.
            Either 2 or 5 for now. Defaults to 5.
        return_volumes (bool): if True, return only the first frame of each imaging volume. Defaults to True.
        harp_is_in_recording (bool): if True, harp is in the same recording as the imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for synchronisation. Defaults to False.
        conflicts (str): how to handle conflicts. Defaults to "skip".
        sync_kwargs (dict): kwargs for synchronisation.generate_vs_df. Defaults to None.
        return_multiunit (bool): if True, process multiunit activity. Defaults to False.
        ephys_kwargs (dict): Keyword arguments for synchronisation.generate_spike_rate_df.
            `return_multiunit` or `exp_sd` for instance. Defaults to None.

    Returns:
        (pd.DataFrame, pd.DataFrame): tuple of two dataframes, one concatenated vs_df for all recordings, one concatenated trials_df for all recordings.
    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value=recording_type,
        flexilims_session=flexilims_session,
    )
    recordings = recordings[recordings.name.str.contains(protocol_base)]
    if "exclude_reason" in recordings.columns:
        recordings = recordings[recordings["exclude_reason"].isna()]

    load_onix = False if recording_type == "two_photon" else True

    # initialising
    vs_df_all = list(np.zeros(len(recordings.name)))
    imaging_df_all = list(np.zeros(len(recordings.name)))

    for i, recording_name in enumerate(recordings.name):
        print(f"Processing recording {i+1}/{len(recordings)}")
        print(recording_name)
        recording, harp_recording, onix_rec = get_relevant_recordings(
            recording_name, flexilims_session, harp_is_in_recording, load_onix
        )
        vs_df = synchronisation.generate_vs_df(
            recording=recording,
            photodiode_protocol=photodiode_protocol,
            flexilims_session=flexilims_session,
            harp_recording=harp_recording,
            onix_recording=onix_rec if use_onix else None,
            project=project,
            protocol_base=protocol_base,
            conflicts=conflicts,
            sync_kwargs=sync_kwargs,
        )

        if recording_type == "two_photon":
            imaging_df = synchronisation.generate_imaging_df(
                vs_df=vs_df,
                recording=recording,
                flexilims_session=flexilims_session,
                filter_datasets=filter_datasets,
                return_volumes=return_volumes,
            )
        else:
            imaging_df, unit_ids = synchronisation.generate_spike_rate_df(
                vs_df=vs_df,
                onix_recording=onix_rec,
                harp_recording=harp_recording,
                flexilims_session=flexilims_session,
                filter_datasets=filter_datasets,
                **ephys_kwargs,
            )

        imaging_df = format_imaging_df(imaging_df=imaging_df, recording=recording)

        vs_df_all[i] = vs_df
        imaging_df_all[i] = imaging_df

    print(f"Finished concatenating vs_df and trials_df")

    return vs_df_all, imaging_df_all, recordings


def get_relevant_recordings(
    recording_name, flexilims_session, harp_is_in_recording, use_onix
):
    """Get the recording, harp recording and onix recording for a given recording name.

    Args:
        recording_name (str): name of the recording.
        flexilims_session (flexilims_session): flexilims session.
        harp_is_in_recording (bool): if True, harp is in the same recording as the imaging. Defaults to True.
        use_onix (bool): if True, use onix recording for synchronisation. Defaults to False.

    Returns:
        (recording, harp_recording, onix_rec): tuple of recording, harp recording and onix recording.
    """
    recording = flz.get_entity(
        datatype="recording",
        name=recording_name,
        flexilims_session=flexilims_session,
    )

    if harp_is_in_recording:
        harp_recording = recording
    else:
        harp_recording = flz.get_children(
            parent_id=recording.origin_id,
            children_datatype="recording",
            flexilims_session=flexilims_session,
            filter=dict(protocol="harpdata"),
        )
        assert (
            len(harp_recording) == 1
        ), f"{len(harp_recording)} harp recording(s) found for {recording_name}"
        harp_recording = harp_recording.iloc[0]

    if use_onix:
        onix_rec = flz.get_children(
            parent_id=recording.origin_id,
            children_datatype="recording",
            flexilims_session=flexilims_session,
            filter=dict(protocol="onix"),
        )
        assert (
            len(onix_rec) == 1
        ), f"{len(onix_rec)} onix recording(s) found for {recording_name}"
        onix_rec = onix_rec.iloc[0]
    else:
        onix_rec = None

    return recording, harp_recording, onix_rec


def plot_mismatch_raster(closed_loop):
    closed_loop = find_mismatch(closed_loop)
    closed_loop, indices = create_mismatch_window(
        closed_loop, window_start=5, window_end=20
    )
    neurons, neurons_df = build_neurons_df(closed_loop)
    misperneuron = build_mismatches_per_neuron_list(
        neurons, neurons_df, window_start=5, window_end=20
    )
    mismatch_raster = raster(neurons, misperneuron, window_start=5, window_end=20)
    rand_raster = make_rand_raster(closed_loop, n_events=200, window_end=10)
    sorted_mismatch_raster = modulation_sort_raster(rand_raster, mismatch_raster)
    plot_raster(sorted_mismatch_raster)


def find_mismatch(closed_loop):
    # Find how running speed and displayed speed  change
    closed_loop["mousez_dif"] = np.zeros(len(closed_loop["mouse_z"]))
    closed_loop.loc[1:, "mousez_dif"] = np.diff(closed_loop["mouse_z"])

    # Locate points where they decouple
    closed_loop["mismz_dif"] = np.zeros(len(closed_loop["mouse_z"]))
    closed_loop.loc[1:, "mismz_dif"] = np.diff(closed_loop["mismatch_mouse_z"])
    closed_loop["mism_ratio"] = closed_loop["mousez_dif"] / closed_loop["mismz_dif"]
    closed_loop["mismatch"] = (
        (closed_loop["mism_ratio"] > 1.2) | (closed_loop["mism_ratio"] < -1000)
    ).astype(int)
    # To catch the fact that it's -inf sometimes during a mismatch

    closed_loop.drop(columns={"mismz_dif", "mousez_dif"})

    return closed_loop


def create_mismatch_window(
    closed_loop, window_start=5, window_end=10, event="mismatch"
):
    """
    Forms a mask in the region of the recording used to analyse mismatch responses.
    The mask goes from window_start to window_end, in frames. Needs input from find_mismatch.
    """

    # Create a new column initialized to 0
    closed_loop["range_indicator"] = 0

    # Make a diff to look at starting frames
    closed_loop["start_mismatch"] = np.zeros(len(closed_loop[event]))

    closed_loop.loc[1:, "start_mismatch"] = np.diff(closed_loop[event])

    # Find indices where 'indicator' is 1
    indices = closed_loop.index[closed_loop["start_mismatch"] == 1].tolist()
    print(f"Mismatches happen in: {indices}")

    # Set range_indicator to 1 for 5 rows before and after each index where 'indicator' is 1
    for idx in indices:
        start = max(idx - window_start, 0)
        end = min(idx + window_end, len(closed_loop[event]) - 1)
        # print((start,  end))
        closed_loop.loc[start:end, "range_indicator"] = 1

    return closed_loop, indices


def build_neurons_df(closed_loop):
    """
    Makes a dataframe with the mask for responses.
    """
    # Create the initial DataFrame with range_indicator
    neurons_df = pd.DataFrame(
        {"range_indicator": closed_loop["range_indicator"].copy()}
    )

    # Extract the number of neurons
    neurons = closed_loop["dffs"][0].shape[1]

    # Create a DataFrame for the neurons data
    neuron_data = pd.DataFrame(
        {
            f"neuron{neuron}": [
                closed_loop["dffs"][i][0][neuron]
                for i in range(len(closed_loop["mismatch"]))
            ]
            for neuron in tqdm(range(neurons))
        }
    )

    # Concatenate the range_indicator and neuron data
    neurons_df = pd.concat([neurons_df, neuron_data], axis=1)

    return neurons, neurons_df


def build_mismatches_per_neuron_list(
    neurons, neurons_df, window_start=5, window_end=10, indices=None
):
    mismatches_per_neuron = list(np.zeros(neurons))

    window = window_start + window_end

    neurons_df["start_mismatch"] = np.zeros(len(neurons_df["range_indicator"]))

    neurons_df.loc[1:, "start_mismatch"] = np.diff(neurons_df["range_indicator"])

    if indices is None:
        n_mismatches = len(
            neurons_df["start_mismatch"][neurons_df["start_mismatch"] == 1]
        )
    else:
        n_mismatches = len(indices)

    print(f"# mismatches: {n_mismatches}")

    for i in range(neurons):
        mismatches_per_neuron[i] = np.zeros((n_mismatches, window))

    # Initialize variables to track the start and end of intervals
    in_interval = False
    start_idx = None

    if indices is None:
        # Iterate through the DataFrame to identify intervals
        idx_mismatch = -1
        print(f"Building {n_mismatches} mismatches per neuron")
        for idx, row in tqdm(neurons_df.iterrows()):
            if row["range_indicator"] == 1 and not in_interval:
                # Start of a new interval
                start_idx = idx
                in_interval = True
                idx_mismatch += 1
                # print(f"This is mismatch {idx_mismatch}")
            elif row["range_indicator"] == 0 and in_interval:
                # End of the current interval
                end_idx = idx - 1
                for neuron in range(neurons):
                    mismatches_per_neuron[neuron][idx_mismatch, :] = neurons_df[
                        f"neuron{neuron}"
                    ][start_idx:end_idx]
                in_interval = False
                # print(f"start and end idx: {(start_idx, end_idx)}")

    else:
        print(f"Building {n_mismatches} mismatches per neuron")
        nframes = len(neurons_df)
        for idx_mismatch, idx in tqdm(enumerate(indices)):
            start_idx = max(0, idx - window_start)
            end_idx = min((nframes - 1), idx + window_end)
            for neuron in range(neurons):
                mismatches_per_neuron[neuron][idx_mismatch, :] = neurons_df[
                    f"neuron{neuron}"
                ][start_idx:end_idx]
            # print(f"start and end idx: {(start_idx, end_idx)}")

    return mismatches_per_neuron


def raster(
    neurons, mismatches_per_neuron, n_neurons=100, window_start=5, window_end=10
):
    window = window_start + window_end

    # initialise
    mismatch_raster = np.zeros((neurons, window))

    for i in range(neurons):
        mismatch_raster[i, :] = np.mean(mismatches_per_neuron[i], axis=0)

    # print(mismatch_raster.shape)

    # Calculate differences for each row
    # differences = np.apply_along_axis(calculate_difference, 1, mismatch_raster)
    # print(differences[0:10])

    return mismatch_raster


def plot_partial_raster(mismatch_raster, n_neurons=100, window_start=5, window_end=10):
    start = 0
    end = n_neurons

    fig = plt.figure(figsize=(30, 10), facecolor="w")
    ax = fig.add_subplot(111)
    im = ax.imshow(mismatch_raster[start:end])

    ax.set_title(f"Raster plot of first {end} neurons aligned to mismatch")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Neurons")
    fig.colorbar(im, label="dff")
    ax.axvline(window_start, color="grey")
    plt.show()


def plot_raster(mismatch_raster, vmin=2, vmax=-0.5):
    fig = plt.figure(figsize=(30, 10), facecolor="w")
    ax = fig.add_subplot(111)
    im = ax.imshow(
        mismatch_raster, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto"
    )

    ax.set_title(f"Raster plot of neurons aligned to mismatch")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Neurons")
    fig.colorbar(im, label="dff")
    ax.axvline(5, color="grey")
    plt.show()


def modulation_sort_raster(rand_raster, mismatch_raster):
    rand_avg = np.mean(rand_raster, axis=1)
    mismatch_avg = np.mean(mismatch_raster[:, 8:17], axis=1)
    modulation_raster = mismatch_avg - rand_avg
    sorted_indices = np.argsort(-modulation_raster)
    # print(sorted_indices[0:10])

    # Sort the array based on the calculated differences
    sorted_mismatch_raster = mismatch_raster[sorted_indices]

    return sorted_mismatch_raster


def make_rand_raster(closed_loop, n_events=100, window_end=5):
    closed_loop = attach_randevents_to_recording(closed_loop, n_events=n_events)
    rand_rec, indices = create_mismatch_window(
        closed_loop, window_start=0, window_end=window_end, event="randevents"
    )
    neurons, rand_neurodf = build_neurons_df(rand_rec)
    rand_misperneuron = build_mismatches_per_neuron_list(
        neurons, rand_neurodf, window_start=0, window_end=window_end, indices=indices
    )
    rand_raster = raster(
        neurons, rand_misperneuron, window_start=0, window_end=window_end
    )

    return rand_raster


def calculate_difference(row, window_start=5, window_end=10):
    window = window_start + window_end
    first_5_sum = np.sum(row[0:window_start])
    last_5_sum = np.sum(row[window_start + 1 : window - window_start])
    return last_5_sum - first_5_sum


def generate_random_events(n_frames, n_events=100):
    return [random.randint(0, n_frames - 1) for _ in range(n_events)]


def attach_randevents_to_recording(closed_loop, n_events=100):
    n_frames = len(closed_loop)
    events = generate_random_events(n_frames, n_events)
    closed_loop["randevents"] = 0
    closed_loop.loc[events, "randevents"] = 1

    return closed_loop
