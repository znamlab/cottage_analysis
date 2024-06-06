from functools import partial
import flexiznam as flz
import numpy as np
import pandas as pd
import random
from scipy.stats import zscore, mannwhitneyu
import matplotlib.pyplot as plt
from tqdm import tqdm


print = partial(print, flush=True)

from cottage_analysis.preprocessing import synchronisation

PROJECT = "663214d08993fd0b6e6b5f1d"
PROTOCOL = "KellerTube"
MESSAGES = "harpmessage.bin"


def analyse_session(session, flexilims_session=None):
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    exp_session = flz.get_entity(
        datatype="session", name=session, flexilims_session=flexilims_session
    )

    vs_df_all, imaging_df_all, recordings = sync_all_recordings(
        session_name=session,
        flexilims_session=flexilims_session,
        project=PROJECT,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base=PROTOCOL,
        photodiode_protocol=5,
        return_volumes=True,
    )

    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value="two_photon",
        flexilims_session=flexilims_session,
    )

    recordings = recordings[recordings.name.str.contains(PROTOCOL)]

    for i, recname in enumerate(recordings.name):
        recording = flz.get_entity(
            datatype="recording",
            name=recname,
            flexilims_session=flexilims_session,
        )

        # Add playback attribute
        is_playback = determine_if_playback(recording, flexilims_session)

        print(
            "###########################################################################"
        )
        print(f"Analysing recording {recording.name}")
        print(
            "###########################################################################"
        )

        closed_loop = imaging_df_all[i]

        analyse_recording(closed_loop, recording, flexilims_session)


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


def analyse_recording(
    closed_loop, recording, flexilims_session, null_mode="trial_structure", save=True
):
    """
    Main entry point for the quick analysis pipeline. null_mode
    determineshow to  calculate the null distribution to establish size
    or significance of mismatch responses: do you randomly sample all
    the trial or do you take into account that the mismatches only happen
    at some points during the trial.
    """

    is_playback = determine_if_playback(recording, flexilims_session)

    print("Estimating mismatch distribution")
    closed_loop = find_mismatch(closed_loop, is_playback)
    closed_loop, idxs = create_mismatch_window(
        closed_loop, window_start=5, window_end=20
    )
    neurons, neurons_df = build_neurons_df(closed_loop)
    misperneuron = build_mismatches_per_neuron_list(
        neurons, neurons_df, window_start=5, window_end=20
    )
    mismatch_raster = raster(neurons, misperneuron, window_start=5, window_end=20)

    if null_mode == "trial_structure":
        closed_loop = find_trials(closed_loop)
        closed_loop = define_window_for_mismatch(closed_loop)
        indices = generate_plausible_mismatch_indices(closed_loop)
    else:
        indices = None

    print("Estimating null distribution")
    rand_raster, rand_misperneuron = make_rand_raster(
        closed_loop, n_events=200, window_end=10, indices=indices
    )
    sorted_mismatch_raster, modulation_raster = modulation_sort_raster(
        rand_raster, mismatch_raster
    )
    sorted_p = calculate_significance(
        misperneuron, rand_misperneuron, modulation_raster
    )

    print("Plotting")
    rasterfig, rasterax = plot_raster(sorted_mismatch_raster)
    rasterfig, rasterax, rasterax2 = plot_significance(rasterfig, rasterax, sorted_p)
    plt.show()
    popfig, popax = plot_pop_response(sorted_mismatch_raster)
    plt.show()
    if not is_playback:
        sync_loop, trialfig, trialax = check_trials(
            flexilims_session, recording, closed_loop
        )
        plt.show()

    print("Saving...")

    if save:
        ## save stuff

        processed = flz.get_data_root("processed", flexilims_session=flexilims_session)
        path = processed / recording.path

        # getting unsorted significance of modulation
        p = calculate_significance(misperneuron, rand_misperneuron)

        # save dataframe
        mismatch_df = {"modulation_size": modulation_raster, "p_value": p}
        mismatch_df = pd.DataFrame(mismatch_df)
        mismatch_df.to_pickle(str(path / "mismatch_df.pkl"))

        # save mismatch_raster
        np.save(str(path / "mismatch_raster.npy"), mismatch_raster)

        # save figures
        rasterfig.savefig(str(path / "raster"))
        popfig.savefig(str(path / "population"))
        if not is_playback:
            trialfig.savefig(str(path / "trials"))


def find_mismatch(closed_loop, is_playback):
    """
    A mismatch is the coupling between running speed andopotic flow changing suddently
    We use that fact to find them in the data. This function adds  diffs  for mouse_z
    and mismatch_mouse_z, calculates the  ratio and thresholds it to generate a mismatch
    column in the closed_loop df.
    """
    # Find how running speed and displayed speed  change
    if is_playback:
        closed_loop["mousez_dif"] = np.zeros(len(closed_loop["eye_z"]))
        closed_loop.loc[1:, "mousez_dif"] = np.diff(closed_loop["eye_z"])
    else:
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


def build_neurons_df(closed_loop, do_zscore=True):
    """
    Makes a dataframe of timepoints x dffs. Can save the dffs as a z-score
    by default.
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

    if do_zscore:
        neuron_array = zscore(neuron_data.to_numpy())
        print("Z-scoring the dataframe")
        for i in tqdm(range(neurons)):
            neuron_data[f"neuron{i}"] = neuron_array[:, i]

    # Concatenate the range_indicator and neuron data
    neurons_df = pd.concat([neurons_df, neuron_data], axis=1)

    return neurons, neurons_df


def build_mismatches_per_neuron_list(
    neurons, neurons_df, window_start=5, window_end=10, indices=None
):
    """
    Builds a  really useful object: a listof the responses of all neurons
    around a time window to a particular event.
    """

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
                # You need to make sure the window always has the right size.
                slice_data = neurons_df[f"neuron{neuron}"][start_idx:end_idx].values
                if len(slice_data) < window:
                    slice_data = np.pad(
                        slice_data, (0, window - len(slice_data)), "constant"
                    )
                mismatches_per_neuron[neuron][idx_mismatch, :] = slice_data

    return mismatches_per_neuron


def raster(
    neurons, mismatches_per_neuron, n_neurons=100, window_start=5, window_end=10
):
    """
    Generates a raster array aligned to  an  event.
    """
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
    """
    Plots a raster in  which you can  see the first n neurons.
    """
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


def plot_raster(mismatch_raster, vmin=-1, vmax=1, do_zscore=True):
    """
    Plots a raster  of mismatch  responses with useful  defaults.
    """
    fig = plt.figure(figsize=(30, 10), facecolor="w")
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Adjust these values as needed
    im = ax.imshow(
        mismatch_raster, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto"
    )

    ax.set_title(f"Raster plot of neurons aligned to mismatch", size=30)
    ax.set_xlabel("Frames", size=25)
    ax.set_ylabel("Neurons", size=25)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=14)  # Change the colorbar tick labels size

    if do_zscore:
        # Optionally set colorbar label with custom size
        cbar.set_label("Z-score", fontsize=25)
    else:
        cbar.set_label("Dff", fontsize=25)
    ax.axvline(5, color="grey")

    return fig, ax


def plot_pop_response(sorted_mismatch_raster, how_many=None):
    fig, ax = plt.subplots()

    ax.plot(np.mean(sorted_mismatch_raster, axis=0), label="Total")

    if how_many is None:
        how_many = int(0.1 * len(sorted_mismatch_raster))

    ax.plot(
        np.mean(sorted_mismatch_raster[:how_many, :], axis=0), label=f"First {how_many}"
    )
    ax.plot(
        np.mean(sorted_mismatch_raster[-how_many:, :], axis=0),
        label=f"Bottom {how_many}",
    )

    ax.set_title("Population response")
    ax.set_ylabel("Z-score")
    ax.set_xlabel("Frames")
    ax.axvline(5, color="red", alpha=0.3, label="Mismatch onset")
    ax.axvline(15, color="green", alpha=0.3, label="End of response window")
    ax.axhline(0, color="green", alpha=0.3)

    ax.legend()

    return fig, ax


def modulation_sort_raster(rand_raster, mismatch_raster):
    """
    Using a raster of neurons aligned to randomly triggered events,
    sorts neurons based on how different is their response to mismatch
    compared  to their response to  random events. This is the Attinger, 2017
    way to asess mismatch modulation.
    """
    rand_avg = np.mean(rand_raster, axis=1)
    mismatch_avg = np.mean(mismatch_raster[:, 8:17], axis=1)
    modulation_raster = mismatch_avg - rand_avg
    sorted_indices = np.argsort(-modulation_raster)
    # print(sorted_indices[0:10])

    # Sort the array based on the calculated differences
    sorted_mismatch_raster = mismatch_raster[sorted_indices]

    return sorted_mismatch_raster, modulation_raster


def make_rand_raster(closed_loop, n_events=100, window_end=5, indices=None):
    """
    Builds an  object like mismatch_raster, buut for neurons aligned to
    randomly triggered events.
    """
    closed_loop = attach_randevents_to_recording(
        closed_loop, indices, n_events=n_events
    )
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

    return rand_raster, rand_misperneuron


def find_trials(closed_loop):
    """
    Calculate, based on corridor length and ITI, where
    there should be trial boundaries. Useful for  estimating  trial structure where
    it was not saved. Checks on check_trials. Assumes 6m-2s
    """

    # Add a new column for the trial indicator
    closed_loop["trial_indicator"] = 0

    # Define the thresholds
    distance_threshold = 6  # six meters
    time_threshold = 2  # 3

    # Initialize variables for trial tracking
    start_distance = 0
    current_trial = 1
    n_rows = len(closed_loop)
    print(f"Finding trials for {n_rows} rows")
    i = 0

    while i < n_rows:
        current_distance = closed_loop["mouse_z"].iloc[i] - start_distance
        # Assign trial indicator for the next six meters
        while i < n_rows and current_distance <= distance_threshold:
            closed_loop.loc[i, "trial_indicator"] = current_trial
            i += 1
            if i < n_rows:
                current_distance = closed_loop["mouse_z"].iloc[i] - start_distance

        # Calculate the ending harptime for the current trial
        if i < n_rows:
            end_harptime = closed_loop.loc[i - 1, "mouse_z_harptime"]

        # Assign 0 for the next three seconds
        while (
            i < n_rows
            and closed_loop.loc[i, "mouse_z_harptime"] <= end_harptime + time_threshold
        ):
            closed_loop.loc[i, "trial_indicator"] = 0
            i += 1

        # Move to the next trial
        current_trial += 1
        if i < n_rows:
            start_distance = closed_loop["mouse_z"].iloc[i]

    return closed_loop


def check_trials(flexilims_session, recording, closed_loop):
    """
    Checks the trial structure inferred by find_trials against what was saved
    in MismatchDebug. We use 6m-ITI parceling to delimit the trace into trials.
    Now, we have trials. Are they good? We expect the old mismatch window to
    end a consistent little bit before the trial ends.

    Out:
        sync_loop : closed_loop df with the added IsMismatch column.
        fig, ax :  for looking closely at the verification plot.
    """

    MismatchDebug = get_mismatch_debug_file(flexilims_session, recording)

    sync_loop = synchronize_dataframes(closed_loop, MismatchDebug)

    fig, ax = plot_synchronized_data(closed_loop, sync_loop)

    return sync_loop, fig, ax


def calculate_difference(row, window_start=5, window_end=10):
    window = window_start + window_end
    first_5_sum = np.sum(row[0:window_start])
    last_5_sum = np.sum(row[window_start + 1 : window - window_start])
    return last_5_sum - first_5_sum


def generate_random_events(n_frames, n_events=100, indices=None):
    if indices is None:
        values = [random.randint(0, n_frames - 1) for _ in range(n_events)]
    else:
        values = np.random.choice(indices, size=n_events)

    return values


def attach_randevents_to_recording(closed_loop, indices, n_events=100):
    n_frames = len(closed_loop)
    events = generate_random_events(n_frames, n_events, indices)
    closed_loop["randevents"] = 0
    closed_loop.loc[events, "randevents"] = 1

    return closed_loop


def get_mismatch_debug_file(flexilims_session, recording):
    raw = flz.get_data_root("raw", flexilims_session=flexilims_session)

    ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp",
        allow_multiple=False,
    )

    filename = ds.csv_files["DebugMismatchDis"]

    MismatchDebug = pd.read_csv(raw / recording.path / filename)

    return MismatchDebug


def synchronize_dataframes(df_a, df_b):
    df_a["IsMismatch"] = 0

    print("Iterating over {len(df_a)} rows for synchronization")

    for i, (idx, row) in tqdm(enumerate(df_a.iterrows())):
        value_a = row["mismatch_mouse_z"] * 100

        # Find the row in df_b where the value in 'variable1' is closest to value_a
        closest_row = df_b.iloc[(df_b["MismatchDistance"] - value_a).abs().argmin()]
        # print((closest_row["MismatchDistance"], value_a))

        # Combine the rows
        df_a["IsMismatch"][i] = closest_row["IsMismatch"]

    return df_a


def plot_synchronized_data(closed_loop, sync_loop, start=0, end=-1):
    fig, ax = plt.subplots(figsize=(40, 10))  # Very large figure

    ax.plot(
        closed_loop["mouse_z"][start:end],
        closed_loop["trial_indicator"][start:end],
        label="Trial indicator",
    )
    ax.plot(
        closed_loop["mouse_z"][start:end],
        closed_loop["mismatch"][start:end] * 30,
        label="Mismatch",
    )
    ax.plot(
        closed_loop["mouse_z"][start:end],
        sync_loop["IsMismatch"][start:end] * 30,
        label="Old mismatch window",
    )

    ax.legend()
    ax.set_xlabel("Mouse Z")
    ax.set_ylabel("Values")
    ax.set_title("Synchronized Data Plot")

    return fig, ax


def generate_plausible_mismatch_indices(closed_loop):
    """
    After define_window_for_mismatch, find the indices to choose
    the mismatch null distribution.
    Out:

        indices(list of int):  all  the possible indices where there
        could have been a mismatch given trial structure.
    """
    mis_closed_loop = closed_loop[closed_loop["mismatch_window"]]
    indices = mis_closed_loop.index.tolist()
    return indices


def define_window_for_mismatch(closed_loop, corridor_length=6):
    """
    Mismatches appear randomly drawn from a uniform distribution
    that is way bigger than the window of positions in which they
    can be displayed. The overlap between the two is  p(mismatch).

    No mismatches can be displayed in the first third of the
    corridor, or in the last 5/6. We aggregate all those indices,
    and then sample from the list of indices to generate the null
    distribution.

    This defines the mismatch window. Needs the trial_indicator column

    Out:
        closed_loop(df),  but with a column called mismatch_window (bool)
        which is true where there could have been a
    """

    # Add a new column for the trial indicator
    closed_loop["mismatch_window"] = False
    closed_loop["in_trial"] = np.where(closed_loop["trial_indicator"] > 0, True, False)

    # Define the thresholds
    beggining_threshold = corridor_length * (1 / 3)
    end_threshold = corridor_length * (5 / 6)

    # Initialize variables for trial tracking
    start_distance = 0
    n_rows = len(closed_loop)
    print(n_rows)
    i = 0

    while i < n_rows:
        current_distance = closed_loop["mouse_z"].iloc[i] - start_distance
        in_trial = closed_loop["in_trial"].iloc[i]

        if (
            closed_loop["in_trial"].iloc[i] == 1
            and closed_loop["in_trial"].iloc[i - 1] == 0
        ):
            start_distance = closed_loop["mouse_z"].iloc[i]

        # Assign mismatch window indicator
        while (
            i < n_rows
            and current_distance >= beggining_threshold
            and current_distance < end_threshold
        ):
            in_trial = closed_loop["in_trial"].iloc[i]

            if in_trial:
                closed_loop.loc[i, "mismatch_window"] = True
            i += 1
            if i < n_rows:
                current_distance = closed_loop["mouse_z"].iloc[i] - start_distance

        i += 1

    return closed_loop


def calculate_significance(misperneuron, rand_misperneuron, modulation_raster=None):
    """
    Calculates,  for each neuron, whether  it's significanttly modulated  by
    It compares the distribution of mean responses to mismatches and random  events, and
    applies a Mann-Whitney test to see if  they are drawn from the same distr. It outputs
    the p-values of all the comparisons, either sortedby response size (providing a modulation_raster)
    or with the order in neurons_df (leting modulation_raster default to None).
    """

    neuron_p = np.zeros(len(misperneuron))

    for neuron in tqdm(range(len(misperneuron))):
        mis_responses = misperneuron[neuron][:, 5:15]  # keep the rersponse part
        rand_responses = rand_misperneuron[neuron]

        mis_mean = np.mean(mis_responses, axis=1)
        rand_mean = np.mean(rand_responses, axis=1)
        p = mannwhitneyu(mis_mean, rand_mean)
        neuron_p[neuron] = p.pvalue

    if modulation_raster is not None:
        sorted_indices = np.argsort(-modulation_raster)
        sorted_p = neuron_p[sorted_indices]

        return sorted_p

    else:
        return neuron_p


def plot_significance(fig, ax, sorted_p, alpha=0.05):
    # Prepare for plotting,  threshold  p-values
    plot_sorted_p = np.where(sorted_p < alpha, 1, 0)
    plot_sorted_p = plot_sorted_p[np.newaxis, :]

    # Edit raster plot to add  significance.
    ax2 = fig.add_axes(
        [0.06, 0.1, 0.03, 0.8], sharey=ax
    )  # Adjust these values as needed
    ax2.imshow(plot_sorted_p.T, aspect="auto", cmap="binary")
    ax.yaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.set_ylabel("Neurons", size=25)

    return fig, ax, ax2


def determine_if_playback(recording, flexilims_session):
    ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp",
        allow_multiple=False,
    )
    attributes = ds.extra_attributes
    if "playback" in recording.name:
        attributes["playback"] = True
    else:
        attributes["playback"] = False

    ds.extra_attributes = attributes
    ds.update_flexilims(mode="overwrite")

    return attributes["playback"]
