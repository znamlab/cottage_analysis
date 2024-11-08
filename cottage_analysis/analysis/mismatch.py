from functools import partial
import flexiznam as flz
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
from pathlib import Path
from scipy.stats import zscore, mannwhitneyu, pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

print = partial(print, flush=True)

from cottage_analysis.preprocessing import synchronisation

PROJECT = "663214d08993fd0b6e6b5f1d"
PROTOCOL = "KellerTube"
MESSAGES = "harpmessage.bin"


def analyse_session(session, flexilims_session=None):
    """
    Entry point for anything non-multigain
    """
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

        print(">>>>>>>>>>>>>>> Z-SCORE ANALYSIS <<<<<<<<<<<<<<<<<<<<")
        analyse_recording(closed_loop, recording, flexilims_session)
        print(">>>>>>>>>>>>>>> DFF ANALYSIS <<<<<<<<<<<<<<<<<<<<")
        analyse_recording(closed_loop, recording, flexilims_session, do_zscore=False)


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
    closed_loop,
    recording,
    flexilims_session,
    null_mode="trial_structure",
    do_zscore=True,
    save=True,
):
    """
    Main entry point for the quick analysis pipeline. null_mode
    determines how to  calculate the null distribution to establish size
    or significance of mismatch responses: do you randomly sample all
    the trial or do you take into account that the mismatches only happen
    at some points during the trial.

    The do_zscore option implements an analysis  that looks like the one  in
    Attinger et. al, 2017: substracts the value of the pre-mismatch window
    for both null events and mismatches, and then uses those traces for plotting,
    saving and calculatingg significance w the dff.
    """

    is_playback = determine_if_playback(recording, flexilims_session)
    is_multimismatch = determine_if_multimismatch(recording, flexilims_session)

    print("Estimating mismatch distribution")
    closed_loop = find_mismatch(closed_loop, is_playback)
    closed_loop, idxs = create_mismatch_window(
        closed_loop, window_start=5, window_end=20
    )
    neurons, neurons_df = build_neurons_df(closed_loop, do_zscore=do_zscore)
    misperneuron = build_mismatches_per_neuron_list(
        neurons, neurons_df, window_start=5, window_end=20
    )
    mismatch_raster = raster(
        neurons, misperneuron, window_start=5, window_end=20, do_zscore=do_zscore
    )

    if null_mode == "trial_structure":
        closed_loop = find_trials(closed_loop)
        closed_loop = define_window_for_mismatch(closed_loop)
        indices = generate_plausible_mismatch_indices(closed_loop)
    else:
        indices = None

    print("Estimating null distribution")
    rand_raster, rand_misperneuron = make_rand_raster(
        closed_loop,
        n_events=100,
        window_start=5,
        window_end=10,
        indices=indices,
        do_zscore=do_zscore,
    )
    sorted_mismatch_raster, modulation_raster = modulation_sort_raster(
        rand_raster, mismatch_raster, window_start=5
    )
    sorted_p = calculate_significance(
        misperneuron, rand_misperneuron, modulation_raster
    )

    print("Plotting")
    if do_zscore:
        rasterfig, rasterax = plot_raster(sorted_mismatch_raster)
    else:
        rasterfig, rasterax = plot_raster(
            sorted_mismatch_raster, vmin=-0.35, vmax=0.35, do_zscore=do_zscore
        )
    rasterfig, rasterax, rasterax2 = plot_significance(rasterfig, rasterax, sorted_p)
    plt.show()
    popfig, popax = plot_pop_response(sorted_mismatch_raster, do_zscore=do_zscore)
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

        if do_zscore:
            naming = ""
        else:
            naming = "_dff"

        # getting unsorted significance of modulation
        p = calculate_significance(misperneuron, rand_misperneuron)

        # save dataframe
        mismatch_df = {"modulation_size": modulation_raster, "p_value": p}
        mismatch_df = pd.DataFrame(mismatch_df)
        mismatch_df.to_pickle(str(path / "mismatch_df.pkl"))

        # save mismatch_raster
        np.save(str(path / f"mismatch_raster{naming}.npy"), mismatch_raster)

        # save figures
        rasterfig.savefig(str(path / f"raster{naming}"))
        popfig.savefig(str(path / f"population{naming}"))
        if not is_playback:
            trialfig.savefig(str(path / "trials"))


def find_mismatch(closed_loop, is_playback):
    """
    A mismatch is the coupling between running speed and optic flow changing suddently
    We use that fact to find them in the data. This function adds  diffs  for mouse_z
    and mismatch_mouse_z, calculates the  ratio and thresholds it to generate a mismatch
    column in the closed_loop df.
    """

    closed_loop = mismatch_ratio(closed_loop, is_playback)

    closed_loop["mismatch"] = (
        (closed_loop["mism_ratio"] > 1.2) | (closed_loop["mism_ratio"] < -1000)
    ).astype(int)
    # To catch the fact that it's -inf sometimes during a mismatch

    closed_loop.drop(columns={"mismz_dif", "mousez_dif"})

    # Filter noise:
    print("Filtering blips")
    for i in tqdm(range(1, len(closed_loop))):  # can't happen in the first  value
        if (
            closed_loop["mismatch"][i] == 1
            and closed_loop["mismatch"][i - 1] == 0
            and closed_loop["mismatch"][i + 1] == 0
        ):
            closed_loop["mismatch"][i] = 0

    return closed_loop


def mismatch_ratio(closed_loop, is_playback):
    # Find how running speed and displayed speed change
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

    return closed_loop


def create_mismatch_window(
    closed_loop, window_start=5, window_end=20, event="mismatch"
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
    neurons, neurons_df, window_start=5, window_end=20, indices=None
):
    """
    Builds a  really useful object: a list of the responses of all neurons
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
    neurons,
    mismatches_per_neuron,
    n_neurons=100,
    window_start=5,
    window_end=9,
    do_zscore=True,
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

    if not do_zscore:
        mismatch_raster -= np.mean(mismatch_raster[:, 0:5], axis=1, keepdims=True)

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
        mismatch_raster,
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
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


def plot_pop_response(sorted_mismatch_raster, how_many=None, do_zscore=True):
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
    if do_zscore:
        ax.set_ylabel("Z-score")
    else:
        ax.set_ylabel("Dff")
    ax.set_xlabel("Frames")
    ax.axvline(5, color="red", alpha=0.3, label="Mismatch onset")
    ax.axvline(9, color="green", alpha=0.3, label="End of response window")
    ax.axhline(0, color="green", alpha=0.3)

    ax.legend()

    return fig, ax


def modulation_sort_raster(
    rand_raster, mismatch_raster, window_start=5, sort="diff_to_null"
):
    """
    Using a raster of neurons aligned to randomly triggered events,
    sorts neurons based on how different is their response to mismatch
    compared  to their response to  random events. This is the Attinger, 2017
    way to asess mismatch modulation.
    """
    rand_avg = np.mean(rand_raster[:, window_start:-1], axis=1)
    mismatch_avg = np.mean(mismatch_raster[:, 5:9], axis=1)
    modulation_raster = mismatch_avg - rand_avg
    if sort == "diff_to_null":
        sorted_indices = np.argsort(-modulation_raster)
    if sort == "response_in_mismatch":
        sorted_indices = np.argsort(-mismatch_avg)

    # print(sorted_indices[0:10])

    # Sort the array based on the calculated differences
    sorted_mismatch_raster = mismatch_raster[sorted_indices]

    return sorted_mismatch_raster, modulation_raster


def make_rand_raster(
    closed_loop,
    n_events=100,
    window_start=5,
    window_end=10,
    indices=None,
    do_zscore=True,
):
    """
    Builds an  object like mismatch_raster, buut for neurons aligned to
    randomly triggered events.
    """
    closed_loop = attach_randevents_to_recording(
        closed_loop, indices, n_events=n_events
    )
    rand_rec, indices = create_mismatch_window(
        closed_loop,
        window_start=window_start,
        window_end=window_end,
        event="randevents",
    )
    neurons, rand_neurodf = build_neurons_df(rand_rec, do_zscore=do_zscore)
    rand_misperneuron = build_mismatches_per_neuron_list(
        neurons,
        rand_neurodf,
        window_start=window_start,
        window_end=window_end,
        indices=indices,
    )
    rand_raster = raster(
        neurons,
        rand_misperneuron,
        window_start=window_start,
        window_end=window_end,
        do_zscore=do_zscore,
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
        while np.isnan(closed_loop["mouse_z"].iloc[i]):
            i += 1
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

    print(f"Iterating over {len(df_a)} rows for synchronization")

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
    mis_closed_loop = closed_loop[closed_loop["range_indicator"] == 0]
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
                closed_loop.loc[i : i + 7, "mismatch_window"] = True
                # Because mismatches can happen at the last moment of the window, we
                # add room for one more mismatch.
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
        mis_responses = misperneuron[neuron][:, 5:9]  # keep the rersponse part
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


# region depth
############

### DEPTH ANALYSIS

############
# This containns code for joining the mismatch with the depth analysis


def build_depth_df(session, index=0, flexilims_session=None):
    """
    Finds the adequate files for a particular session, then joins mismatch information
    with depth information.

    Args:
        session: the experimental session name
        index: which recording within the session is used (we don't duplicate neurons). Defaults
        to the first
        flexilims_session:

    Out:
        depth_df

    """

    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    exp_session = flz.get_entity(
        datatype="session", name=session, flexilims_session=flexilims_session
    )

    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value="two_photon",
        flexilims_session=flexilims_session,
    )

    recordings = recordings[recordings.name.str.contains(PROTOCOL)]
    i = index
    recname = recordings.name[i]

    recording = flz.get_entity(
        datatype="recording",
        name=recname,
        flexilims_session=flexilims_session,
    )

    processed = flz.get_data_root("processed", flexilims_session=flexilims_session)
    yiran_path = processed / exp_session.path / "neurons_df.pickle"
    yiran_df = pd.read_pickle(yiran_path)

    mismatch_path = processed / recording.path / "mismatch_df.pkl"
    mismatch_df = pd.read_pickle(mismatch_path)

    depth_df = {
        "preferred_depth_closedloop": yiran_df["preferred_depth_closedloop"],
        "depth_tuning_test_spearmanr_pval_closedloop": yiran_df[
            "depth_tuning_test_spearmanr_pval_closedloop"
        ],
        "depth_tuning_test_spearmanr_rval_closedloop": yiran_df[
            "depth_tuning_test_spearmanr_rval_closedloop"
        ],
        "mis_modulation_size": mismatch_df["modulation_size"],
        "mis_p_value": mismatch_df["p_value"],
    }

    depth_df = pd.DataFrame(depth_df)

    return depth_df


def read_mismatch_df(session, index, flexilims_session=None):
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    exp_session = flz.get_entity(
        datatype="session", name=session, flexilims_session=flexilims_session
    )

    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value="two_photon",
        flexilims_session=flexilims_session,
    )

    recordings = recordings[recordings.name.str.contains(PROTOCOL)]
    i = index
    recname = recordings.name[i]

    recording = flz.get_entity(
        datatype="recording",
        name=recname,
        flexilims_session=flexilims_session,
    )

    processed = flz.get_data_root("processed", flexilims_session=flexilims_session)

    mismatch_path = processed / recording.path / "mismatch_df.pkl"
    mismatch_df = pd.read_pickle(mismatch_path)

    return mismatch_df


def process_depth_df(depth_df, threshold=0.05):
    """
    Keeps only the neurons that are significantly modulated by depth and mismatch. Also splits
    mismatch cells into positive and negative populations.
    """
    sig_depth_df = depth_df[
        depth_df["depth_tuning_test_spearmanr_pval_closedloop"] < threshold
    ]
    sig_depth_df = sig_depth_df[
        sig_depth_df["depth_tuning_test_spearmanr_rval_closedloop"] > 0
    ]
    sig_depth_df = sig_depth_df[sig_depth_df["mis_p_value"] < threshold]

    positive_mismatch = sig_depth_df[sig_depth_df["mis_modulation_size"] > 0]
    negative_mismatch = sig_depth_df[sig_depth_df["mis_modulation_size"] < 0]

    return sig_depth_df, positive_mismatch, negative_mismatch


def plot_correlation_with_fit(sig_depth_df, all_points=False):
    """
    Calculates the Pearson correlation and produces a line of best fit in a depth x mismatch
    plot.
    """

    log_sig_preferred_depth = np.log10(sig_depth_df["preferred_depth_closedloop"])

    fig, ax = plt.subplots(facecolor="white")

    # Scatter plot
    if all_points:
        color_filter = (
            (sig_depth_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
            & (sig_depth_df["depth_tuning_test_spearmanr_rval_closedloop"] > 0)
            & (sig_depth_df["mis_p_value"] < 0.05)
        )

        # Filter indices for colors
        indices_blue = np.where(color_filter)[0]
        indices_gray = np.where(~color_filter)[0]

        # Plot blue points for condition met
        ax.scatter(
            log_sig_preferred_depth[indices_blue],
            sig_depth_df["mis_modulation_size"].iloc[indices_blue],
            marker=".",
            alpha=0.2,
            color="blue",
            label="Significant tuning",
        )

        # Plot gray points for condition not met
        ax.scatter(
            log_sig_preferred_depth[indices_gray],
            sig_depth_df["mis_modulation_size"].iloc[indices_gray],
            marker=".",
            alpha=0.2,
            color="gray",
            label="Non significant tuning",
        )
        ax.set_ylabel("Mismatch Modulation")
        ax.set_xlabel("Log Preferred Depth")
    else:
        ax.scatter(
            log_sig_preferred_depth,
            sig_depth_df["mis_modulation_size"],
            marker=".",
            alpha=0.2,
        )
        ax.set_ylabel("Mismatch Modulation")
        ax.set_xlabel("Log Preferred Depth")

    # Calculate correlation
    corr_coef, p_value = pearsonr(
        log_sig_preferred_depth, sig_depth_df["mis_modulation_size"]
    )

    # Linear fit
    fit = np.polyfit(log_sig_preferred_depth, sig_depth_df["mis_modulation_size"], 1)
    fit_fn = np.poly1d(fit)

    # Plot the best fit line
    ax.plot(
        log_sig_preferred_depth,
        fit_fn(log_sig_preferred_depth),
        color="red",
        label=f"Best fit line: y={fit[0]:.2f}x + {fit[1]:.2f}",
    )

    # Display correlation coefficient
    ax.set_title(f"Correlation: {corr_coef:.2f} (p-value: {p_value:.2e})")
    ax.legend()

    return fig, ax


def plot_differential_histograms(positive_mismatch, negative_mismatch):
    """
    Plots the distribution of depth preferences foor positive and negative mismatch cells. Tests
    that the two histograms are drawn from different distributions with a Mann-Whitney U.
    """

    fig, ax = plt.subplots(1, 2, sharex=True, facecolor="white")
    ax[0].hist(np.log10(positive_mismatch["preferred_depth_closedloop"]), bins=100)
    ax[0].set_title(f"Positive mismatch, n={len(positive_mismatch)}")
    ax[0].set_xlim((-2, 2))
    ax[0].axvline(
        np.median(np.log10(positive_mismatch["preferred_depth_closedloop"])),
        color="red",
    )

    ax[1].hist(np.log10(negative_mismatch["preferred_depth_closedloop"]), bins=100)
    ax[1].set_title(f"Negative mismatch,  n={len(negative_mismatch)}")
    ax[1].set_xlim((-2, 2))
    ax[1].axvline(
        np.median(np.log10(negative_mismatch["preferred_depth_closedloop"])),
        color="red",
    )

    ax[1].set_xlabel("log pref depth")
    ax[0].set_xlabel("log pref depth")
    ax[0].set_ylabel("Count")

    p = mannwhitneyu(
        np.log10(positive_mismatch["preferred_depth_closedloop"]),
        np.log10(negative_mismatch["preferred_depth_closedloop"]),
    )
    fig.suptitle(f"M-W U: p = {p.pvalue:.2e}")
    print(p)

    return fig, ax


def plot_differential_kdes(positive_mismatch, negative_mismatch):
    """
    Plots the kernel density estimation of depth preferences for positive and negative mismatch cells.
    Tests that the two distributions are drawn from different distributions with a Mann-Whitney U test.
    """

    # Prepare data
    positive_depths = np.log10(positive_mismatch["preferred_depth_closedloop"])
    negative_depths = np.log10(negative_mismatch["preferred_depth_closedloop"])

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, sharex=True, facecolor="white", figsize=(7, 5))

    # Plot KDE for positive mismatch
    sns.kdeplot(
        positive_depths, ax=ax, fill=True, color="blue", label="Positive mismatch"
    )
    ax.set_title(f"n={len(positive_mismatch)+len(negative_mismatch)}")
    ax.set_xlim((-2, 2))
    ax.axvline(
        np.median(positive_depths),
        color="red",
        linestyle="--",
        label=f"Positive median (n={len(positive_mismatch)})",
    )

    # Plot KDE for negative mismatch
    sns.kdeplot(
        negative_depths, ax=ax, fill=True, color="green", label="Negative mismatch"
    )
    ax.axvline(
        np.median(negative_depths),
        color="purple",
        linestyle="--",
        label=f"Negative median (n={len(negative_mismatch)})",
    )

    # Labels and titles
    ax.set_xlabel("Log Pref Depth")
    ax.set_ylabel("Density")
    ax.legend()

    # Perform Mann-Whitney U test
    p = mannwhitneyu(positive_depths, negative_depths)
    fig.suptitle(f"Mann-Whitney U: p = {p.pvalue:.2e}")
    print(p)

    return fig, ax


# endregion
# region multigain
###########
## MULTIGAIN
###########

# This contains code for analysing the particularities of the multigain case


def analyse_multimismatch(
    closed_loop, recording, is_playback=False, flexilims_session=None
):
    findfig, findax, closed_loop, gains_df = find_multimismatch(
        closed_loop, recording, is_playback, flexilims_session
    )

    closed_loop, idxs = create_mismatch_window(
        closed_loop, window_start=5, window_end=20
    )
    neurons, neurons_df = build_neurons_df(closed_loop)

    misperneuron = build_mismatches_per_neuron_list(
        neurons, neurons_df, window_start=5, window_end=20
    )

    tt_misperneuron = break_into_trial_types(misperneuron, gains_df)

    null = build_null_dist_per_trial_type(closed_loop)

    tt_raster, tt_rand_raster, tt_rand_misperneuron = make_tt_rasters(
        tt_misperneuron, null, neurons, closed_loop
    )

    tt_sorted_mismatch_raster, tt_modulation_raster = make_tt_sorted_raster(
        tt_rand_raster, tt_raster
    )

    tt_sorted_p = calculate_tt_significance(
        tt_misperneuron, tt_rand_misperneuron, tt_modulation_raster
    )

    fig, axes = plot_raster_grid(tt_sorted_mismatch_raster, tt_sorted_p)

    return tt_sorted_mismatch_raster, tt_sorted_p


def find_multimismatch(
    closed_loop, recording, is_playback=False, flexilims_session=None
):
    """
    Main entry point for the fascinating job of finding multimismatch events,
    which requires a different pipeline. It is called by find_mismatch. The trace it works with
    is the ratio of the increases in running speed and optic flow. It is filtered for
    one-frame noise, and then approximated to the closest gain out of the four possible.
    """

    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    closed_loop = mismatch_ratio(closed_loop, is_playback)

    ##Filter the gain  of optic flow

    # Filter out NaNs. Assume same value
    closed_loop["mism_ratio_filled"] = closed_loop["mism_ratio"].fillna(method="ffill")

    # List of target values
    target_values = [1, 0.25, 4]

    # Apply the function to the series
    print("Filling NaNs")
    filtered_series = closed_loop["mism_ratio_filled"].apply(
        lambda x: find_closest_value(x, target_values)
    )

    # filter blips
    print("Filtering blips")

    closed_loop["filtered_mism_ratio"] = filtered_series
    for i in range(1, len(closed_loop) - 1):
        if (
            closed_loop["filtered_mism_ratio"][i]
            != closed_loop["filtered_mism_ratio"][i - 1]
        ) and (
            closed_loop["filtered_mism_ratio"][i]
            != closed_loop["filtered_mism_ratio"][i + 1]
        ):
            closed_loop["filtered_mism_ratio"][i] = closed_loop["filtered_mism_ratio"][
                i - 1
            ]

    # build trial structure
    print("Processing trial structure")
    sync_loop = find_trials_from_log(closed_loop, flexilims_session, recording)

    # Put it back on the same df
    closed_loop["trial_indicator"] = sync_loop["in_trial"]
    closed_loop["estimated_mismatch"] = sync_loop["mismatch"]
    closed_loop["trial_start"] = sync_loop["trial_start"]
    closed_loop = define_window_for_mismatch(closed_loop)

    # Use mismatch window to find mismatches and note gain
    print("Finding mismatches")
    closed_loop, gains_df = find_multimismatches_in_window(closed_loop)

    # check
    print("Plotting")
    fig, ax = check_multimismatch(closed_loop)

    return fig, ax, closed_loop, gains_df


def check_multimismatch(closed_loop, beg=0, end=-1):
    """
    Plots reconstructed mismatches and trial structure as a sanity check.
    """

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the data
    ax.plot(
        closed_loop["mouse_z"][beg:end],
        closed_loop["filtered_mism_ratio"][beg:end],
        label="Filtered",
        alpha=0.5,
    )
    ax.plot(
        closed_loop["mouse_z"][beg:end],
        closed_loop["mism_ratio"][beg:end],
        label="Unfiltered",
        alpha=0.5,
    )
    ax.plot(
        closed_loop["mouse_z"][beg:end],
        closed_loop["mismatch"][beg:end],
        label="Mismatch",
        alpha=0.5,
    )
    # ax.plot(closed_loop["mouse_z"][beg:end], sync_loop["IsMismatch"][beg:end], label="IsMismatch", alpha=0.5)
    ax.plot(
        closed_loop["mouse_z"][beg:end],
        closed_loop["in_trial"][beg:end],
        label="In Trial",
        alpha=0.5,
    )
    ax.plot(
        closed_loop["mouse_z"][beg:end],
        closed_loop["mismatch_window"][beg:end],
        label="Mismatch window",
        alpha=0.5,
    )

    # Add legend
    ax.legend()

    # Set title and labels
    ax.set_title("Trial from index {} to {}".format(beg, end))
    ax.set_xlabel("Mouse position")
    ax.set_ylabel("Values")

    # Set white background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    return fig, ax


def find_multimismatches_in_window(closed_loop):
    """
    Finds mismatches taking advantage of the fact that they are all the changes in OF/RS coupling
    that happen within the window where mismatches are possible. Notes gain before-during.
    """

    closed_loop["mismatch"] = 0
    closed_loop["start_gain"] = 0
    closed_loop["end_gain"] = 0
    closed_loop["filtered_mism_diff"] = 0
    closed_loop["filtered_mism_diff"][1:] = abs(
        np.diff(closed_loop["filtered_mism_ratio"])
    )
    mismatch_starters = []
    mismatch_ends = []

    mismatch = False
    for i in range(1, len(closed_loop) - 1):
        if closed_loop["mismatch_window"][i] == True:
            prevent_double_tick = False  # If the first case  is true, it edits, then the second is true, prevent that!
            if closed_loop["filtered_mism_diff"][i] > 0 and mismatch:
                mismatch = False
                prevent_double_tick = True
            if (
                closed_loop["filtered_mism_diff"][i] > 0
                and not mismatch
                and not prevent_double_tick
            ):
                mismatch = True
                start_idx = i
                mismatch_starters.append(
                    closed_loop["filtered_mism_ratio"][start_idx - 1]
                )
                mismatch_ends.append(closed_loop["filtered_mism_ratio"][i + 1])

            if mismatch:
                closed_loop["mismatch"][i] = 1
                closed_loop["start_gain"][i] = closed_loop["filtered_mism_ratio"][
                    start_idx - 1
                ]
                closed_loop["end_gain"][i] = closed_loop["filtered_mism_ratio"][i]

    gains_df = {"start": mismatch_starters, "end": mismatch_ends}
    gains_df = pd.DataFrame(gains_df)

    return closed_loop, gains_df


def find_trials_from_log(closed_loop, flexilims_session, recording):
    """
    Second way to reconstruct trials. When you have an is_mismatch log for  every trial, you
    use the ending to estimate every trial. This assumes that the  trials are 6m long, that there
    are no mismatches from 0 to 2/3, and from 5/6 to the  end.
    """

    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    MismatchDebug = get_mismatch_debug_file(flexilims_session, recording)

    sync_loop = synchronize_dataframes(closed_loop, MismatchDebug)

    sync_loop["mismatch"] = 0
    sync_loop["trial_end"] = 0
    sync_loop["trial_start"] = 0
    sync_loop["in_trial"] = 0

    # We always start in-trial, but out of scanimage, so build first trial
    start_index = 0
    trial_end = 6
    closest_row = sync_loop.iloc[(sync_loop["mouse_z"] - trial_end).abs().argmin()]
    end_index = closest_row.name
    sync_loop.loc[start_index:end_index, "in_trial"] = 1
    sync_loop.at[start_index, "trial_start"] = 1
    sync_loop.at[end_index, "trial_end"] = 1

    for i in range(1, len(sync_loop) - 1):
        if (sync_loop["IsMismatch"][i] == False) and (
            sync_loop["IsMismatch"][i + 1] == True
        ):
            sync_loop.loc[i - 3 : i, "mismatch"] = 1
        if (sync_loop["IsMismatch"][i - 1] == True) and (
            sync_loop["IsMismatch"][i] == False
        ):
            trial_start = sync_loop["mouse_z"][i] - ((5 / 6) * 6)
            trial_end = sync_loop["mouse_z"][i] + ((1 / 6) * 6)
            closest_row = sync_loop.iloc[
                (sync_loop["mouse_z"] - trial_start).abs().argmin()
            ]
            start_index = closest_row.name
            sync_loop.at[start_index, "trial_start"] = 1
            closest_row = sync_loop.iloc[
                (sync_loop["mouse_z"] - trial_end).abs().argmin()
            ]
            end_index = closest_row.name
            sync_loop.at[end_index, "trial_end"] = 1

            sync_loop.loc[start_index:end_index, "in_trial"] = 1

    return sync_loop


def break_into_trial_types(misperneuron, gains_df):
    """
    We split misperneuron, which is a list of mismatches per neuron, into different trial
    types. For that, we use gains_df, which holds the begginning and end gains of all
    found mismatches.
    """
    tt_misperneuron = {"htm": [], "htl": [], "mtl": [], "mth": [], "ltm": [], "lth": []}

    # Dictionary to hold the indexes for each trial type
    tt_indices = {"htm": [], "htl": [], "mtl": [], "mth": [], "ltm": [], "lth": []}

    # Identifying indexes for each trial type
    tt_indices["htm"] = gains_df[
        (gains_df["start"] == 4) & (gains_df["end"] == 1)
    ].index.tolist()
    tt_indices["htl"] = gains_df[
        (gains_df["start"] == 4) & (gains_df["end"] == 0.25)
    ].index.tolist()
    tt_indices["mtl"] = gains_df[
        (gains_df["start"] == 1) & (gains_df["end"] == 0.25)
    ].index.tolist()
    tt_indices["mth"] = gains_df[
        (gains_df["start"] == 1) & (gains_df["end"] == 4)
    ].index.tolist()
    tt_indices["ltm"] = gains_df[
        (gains_df["start"] == 0.25) & (gains_df["end"] == 1)
    ].index.tolist()
    tt_indices["lth"] = gains_df[
        (gains_df["start"] == 0.25) & (gains_df["end"] == 4)
    ].index.tolist()

    # Mapping misperneuron values to tt_misperneuron based on tt_indices
    for trial_type in tt_misperneuron.keys():
        tt_misperneuron[trial_type] = list(np.zeros(len(misperneuron)))
        for neuron in range(len(misperneuron)):
            tt_misperneuron[trial_type][neuron] = misperneuron[neuron][
                tt_indices[trial_type]
            ]

    return tt_misperneuron


def build_null_dist_per_trial_type(closed_loop):
    """
    We use the code  for the  normal mismatch to find a set of null events in the
    region of  the trials (same starting gain)  where a mismatch could have happened
    but did not.
    """
    # Gain values
    tt_gains = {
        "h": 4,
        "m": 1,
        "l": 0.25,
    }
    tt_null = {"htm": [], "htl": [], "mtl": [], "mth": [], "ltm": [], "lth": []}
    # New dictionary to store filtered DataFrames
    null = {}

    for gain in tt_gains.keys():
        # Filter for moments with the correct gain
        trials = closed_loop[
            (closed_loop["filtered_mism_ratio"] == tt_gains[gain])
            & (closed_loop["range_indicator"] == 0)
        ]
        null[gain] = trials.index.tolist()

    for gain in null.keys():
        plt.figure()
        plt.title(f"Filtered Trials for Gain {gain}")
        plt.plot(null[gain], closed_loop["filtered_mism_ratio"][null[gain]], ".")
        plt.xlabel("Index")
        plt.ylabel("Filtered Mism Ratio")
        plt.show()

        tt_null = {
            "htm": null["h"],
            "htl": null["h"],
            "mtl": null["m"],
            "mth": null["m"],
            "ltm": null["l"],
            "lth": null["l"],
        }

    return tt_null


def make_tt_rasters(
    tt_misperneuron,
    null,
    neurons,
    closed_loop,
    read=False,
    recording=None,
    flexilims_session=None,
):
    """
    We use previous code tu turn tt_misperneuron into a raster for each trial type. We
    also use this, like before, as an entry point to generate the null distribution, whihch
    is akin to tt_misperneuron but holds responses to random events.

    A raster here is just an array that holds, for every neuron, their average response
    to a mismatch event.
    """

    if read:
        print("Reading")
        processed = flz.get_data_root("processed", flexilims_session=flexilims_session)
        path = processed / recording.path

        # Read dictionaries using pickle
        with open(path / "tt_raster.pkl", "rb") as f:
            tt_raster = pickle.load(f)
        with open(path / "tt_rand_raster.pkl", "rb") as f:
            tt_rand_raster = pickle.load(f)
        with open(path / "tt_rand_misperneuron.pkl", "rb") as f:
            tt_rand_misperneuron = pickle.load(f)
    else:
        tt_raster = {}
        tt_rand_raster = {}
        tt_rand_misperneuron = {}

        for gain in tt_misperneuron.keys():
            print(f"Calculating rasters for {gain}")
            tt_raster[gain] = raster(
                neurons, tt_misperneuron[gain], window_start=5, window_end=20
            )
            tt_rand_raster[gain], tt_rand_misperneuron[gain] = make_rand_raster(
                closed_loop, n_events=200, window_end=10, indices=null[gain]
            )

    return tt_raster, tt_rand_raster, tt_rand_misperneuron


def make_tt_sorted_raster(tt_rand_raster, tt_raster, sort="all"):
    """
    This function sorts rasters based on the size of their responses. It can sort them
    normally or it can use value of a previous trial type. This is to show how much the
    neuronal populations that are responsive to each trial type overlap. If you write htl
    in sort, you will sort all rasters according to the response in htl
    """
    tt_sorted_mismatch_raster = {}
    tt_modulation_raster = {}

    if sort == "all":
        for gain in tt_raster.keys():
            (
                tt_sorted_mismatch_raster[gain],
                tt_modulation_raster[gain],
            ) = modulation_sort_raster(tt_rand_raster[gain], tt_raster[gain])
    else:
        for gain in tt_raster.keys():
            rand_raster, mismatch_raster = tt_rand_raster[gain], tt_raster[gain]
            rand_avg = np.mean(rand_raster, axis=1)
            mismatch_avg = np.mean(mismatch_raster[:, 5:9], axis=1)
            modulation_raster = mismatch_avg - rand_avg
            tt_modulation_raster[gain] = modulation_raster

        modulation_raster = tt_modulation_raster[sort]

        for gain in tt_raster.keys():
            mismatch_raster = tt_raster[gain]
            sorted_indices = np.argsort(-modulation_raster)

            # Sort the array based on the calculated differences
            sorted_mismatch_raster = mismatch_raster[sorted_indices]

            tt_sorted_mismatch_raster[gain] = sorted_mismatch_raster

    return tt_sorted_mismatch_raster, tt_modulation_raster


def calculate_tt_significance(
    tt_misperneuron, tt_rand_misperneuron, tt_modulation_raster, sort="all"
):
    """
    Calculates the significance of the modulation using the null distribution like in the
    case of the simple mismatch.
    """
    tt_sorted_p = {}

    if sort == "all":
        for gain in tt_misperneuron.keys():
            tt_sorted_p[gain] = calculate_significance(
                tt_misperneuron[gain],
                tt_rand_misperneuron[gain],
                tt_modulation_raster[gain],
            )
    else:
        for gain in tt_misperneuron.keys():
            tt_sorted_p[gain] = calculate_significance(
                tt_misperneuron[gain],
                tt_rand_misperneuron[gain],
                tt_modulation_raster[sort],
            )

    return tt_sorted_p


def plot_raster_grid(tt_sorted_mismatch_raster, tt_sorted_p, vmin=-2, vmax=2):
    """
    Plots a 3x3 grid of raster plots for different gain combinations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(40, 40), facecolor="w")
    fig.subplots_adjust(hspace=0.32, wspace=0.5)

    gain_labels = ["high", "medium", "low"]
    gain_indices = ["h", "m", "l"]

    # Add column labels
    for ax, col_label in zip(axes[0], gain_labels):
        ax.set_title(f"End: {col_label}", size=60)

    # Add row labels
    for ax, row_label in zip(axes[:, 0], gain_labels):
        ax.set_ylabel(f"Start: {row_label}", size=60, rotation=90, labelpad=200)

    for i, gain_start in enumerate(gain_indices):
        for j, gain_end in enumerate(gain_indices):
            gain_combo = f"{gain_start}t{gain_end}"
            ax = axes[i, j]

            if gain_combo in tt_sorted_mismatch_raster:
                raster_data = tt_sorted_mismatch_raster[gain_combo]
                im = ax.imshow(
                    raster_data,
                    cmap="bwr",
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                    interpolation="nearest",
                )

                ax.set_xlabel("Frames", size=30)
                cbar = fig.colorbar(im, ax=ax, orientation="vertical")
                cbar.ax.tick_params(labelsize=30)
                cbar.set_label("Z-score" if True else "Dff", fontsize=30)
                ax.axvline(5, color="grey")

                # Plot significance
                plot_sorted_p = np.where(tt_sorted_p[gain_combo] < 0.05, 1, 0)
                plot_sorted_p = plot_sorted_p[np.newaxis, :]
                ax2 = fig.add_axes(
                    [0.08 + j * 0.285, 0.12 + (2 - i) * 0.28, 0.03, 0.2], sharey=ax
                )
                ax2.imshow(plot_sorted_p.T, aspect="auto", cmap="binary")
                ax2.xaxis.set_visible(False)
                ax2.set_ylabel("Neurons", size=30)

    return fig, axes


def find_closest_value(value, target_values):
    return min(target_values, key=lambda x: abs(x - value))


def determine_if_multimismatch(recording, flexilims_session):
    ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp",
        allow_multiple=False,
    )
    attributes = ds.extra_attributes
    if "multigain" in recording.name:
        attributes["multigain"] = True
    else:
        attributes["multigain"] = False

    ds.extra_attributes = attributes
    ds.update_flexilims(mode="overwrite")

    return attributes["multigain"]


def save_tt_rasters(
    tt_raster, tt_rand_raster, tt_rand_misperneuron, recording, flexilims_session=None
):
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    # Access the processed data root
    processed = Path(
        flz.get_data_root("processed", flexilims_session=flexilims_session)
    )
    # Construct the path using the recording.path
    path = processed / recording.path

    # Ensure the directory exists
    path.mkdir(parents=True, exist_ok=True)

    # Save dictionaries using pickle
    with open(path / "tt_raster.pkl", "wb") as f:
        pickle.dump(tt_raster, f)
    with open(path / "tt_rand_raster.pkl", "wb") as f:
        pickle.dump(tt_rand_raster, f)
    with open(path / "tt_rand_misperneuron.pkl", "wb") as f:
        pickle.dump(tt_rand_misperneuron, f)

    print(f"Files saved successfully to {path}")


################
# INDIVIDUAL MULTIGAIN PLOTTING
###################


def make_matrix_list(tt_misperneuron, tt_rand_misperneuron):
    by_neur_modulation = {}
    by_neur_rand = {}

    print("Response to mismatch events")

    for tt in tqdm(tt_misperneuron.keys()):
        by_neur_modulation[tt] = np.zeros(len(tt_misperneuron[tt]))
        for neuron in range(len(tt_misperneuron[tt])):
            by_neur_modulation[tt][neuron] = np.mean(
                tt_misperneuron[tt][neuron][:, 5:10]
            )

    null_tts = {  # Use the null distribution for real trialtypes
        "high": "htl",
        "medium": "mtl",
        "low": "lth",
    }

    print("Response to null mismatches")

    for tt in tqdm(null_tts.keys()):
        sample_trial = null_tts[tt]
        by_neur_rand[tt] = np.zeros(len(tt_rand_misperneuron[sample_trial]))
        for neuron in range(len(tt_rand_misperneuron[sample_trial])):
            by_neur_rand[tt][neuron] = np.mean(
                tt_rand_misperneuron[sample_trial][neuron][:, 0:5]
            )

    by_neur_modulation = pd.DataFrame(by_neur_modulation)
    by_neur_rand = pd.DataFrame(by_neur_rand)

    matrix_list = list(np.zeros(len(by_neur_modulation)))
    for neuron in range(len(by_neur_modulation)):
        matrix_list[neuron] = activity_matrix(
            by_neur_modulation.iloc[neuron, :], by_neur_rand.iloc[neuron, :]
        )

    return matrix_list


def activity_matrix(by_neur_modulation, by_neur_rand):
    activity_matrix = [
        [by_neur_rand["high"], by_neur_modulation["htm"], by_neur_modulation["htl"]],
        [by_neur_modulation["mth"], by_neur_rand["medium"], by_neur_modulation["mtl"]],
        [by_neur_modulation["lth"], by_neur_modulation["ltm"], by_neur_rand["low"]],
    ]
    return np.array(activity_matrix)


def plot_trace_matrix(neuron, tt_misperneuron, tt_rand_misperneuron):
    trial_plots = {
        "htm": (0, 1),
        "htl": (0, 2),
        "mtl": (1, 0),
        "mth": (1, 2),
        "ltm": (2, 1),
        "lth": (2, 0),
    }
    null_plots = {"high": (0, 0), "medium": (1, 1), "low": (2, 2)}
    null_tts = {  # Use the null distribution for real trialtypes
        "high": "htl",
        "medium": "mtl",
        "low": "lth",
    }
    fig, ax = plt.subplots(3, 3, facecolor="w", figsize=(10, 10))

    size = 14

    for axis, col_label in zip(ax[0], null_tts.keys()):
        axis.set_title(f"End: {col_label}", size=size)

    for axis, row_label in zip(ax[:, 0], null_tts.keys()):
        axis.set_ylabel(f"Start: {row_label}", size=size)

    fig.suptitle(f"Neuron {neuron}", size=size * 1.2)

    for tt in trial_plots.keys():
        for event in tt_misperneuron[tt][neuron]:
            ax[trial_plots[tt]].plot(event, alpha=0.5, color="red")
            ax[trial_plots[tt]].axvline(5, color="blue")
            ax[trial_plots[tt]].set_ylim((-3, 4))
            # ax[trial_plots[tt]].set_xticks([])  # Remove x-axis tick marks
            # ax[trial_plots[tt]].set_yticks([])  # Remove y-axis tick marks

    for tt in null_plots.keys():
        for event in tt_rand_misperneuron[null_tts[tt]][neuron]:
            ax[null_plots[tt]].plot(event, alpha=0.2, color="grey")
            ax[null_plots[tt]].axvline(5, color="blue")
            ax[null_plots[tt]].set_ylim((-3, 4))
            # ax[null_plots[tt]].set_xticks([])  # Remove x-axis tick marks
            # ax[null_plots[tt]].set_yticks([])  # Remove y-axis tick marks

    return fig, ax


def plot_matrix_grid(matrix_list, side=10, vmin=-1, vmax=1):
    """
    Plots a 3x3 grid of raster plots for different gain combinations.
    """
    data = matrix_list[0 : (side**2)]
    print(len(data))
    fig, axes = plt.subplots(10, 10, figsize=(40, 40), facecolor="w")

    for i in range(10):
        for j in range(10):
            ax = axes[i, j]
            if len(data) > ((i * side) + j):
                im = ax.imshow(
                    data[(i * side) + j],
                    cmap="bwr",
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                    interpolation="nearest",
                )

                cbar = fig.colorbar(im, ax=ax, orientation="vertical")
                cbar.ax.tick_params(labelsize=0)
                # cbar.set_label("Z-score" if True else "Dff", fontsize=15)

                # Set title for each subplot
                ax.set_title(f"{(i*side)+j}", fontsize=30)

            # Remove ticks and labels
            ax.set_xticks([])  # Remove x-axis tick marks
            ax.set_yticks([])  # Remove y-axis tick marks

    # Adjust layout to make room for colorbar and titles
    plt.tight_layout()
    return fig, axes


def plot_trace_matrix(neuron, tt_misperneuron, tt_rand_misperneuron):
    trial_plots = {
        "htm": (0, 1),
        "htl": (0, 2),
        "mtl": (1, 0),
        "mth": (1, 2),
        "ltm": (2, 1),
        "lth": (2, 0),
    }
    null_plots = {"high": (0, 0), "medium": (1, 1), "low": (2, 2)}
    null_tts = {  # Use the null distribution for real trialtypes
        "high": "htl",
        "medium": "mtl",
        "low": "lth",
    }
    fig, ax = plt.subplots(3, 3, facecolor="w", figsize=(10, 10))

    size = 14

    for axis, col_label in zip(ax[0], null_tts.keys()):
        axis.set_title(f"End: {col_label}", size=size)

    for axis, row_label in zip(ax[:, 0], null_tts.keys()):
        axis.set_ylabel(f"Start: {row_label}", size=size)

    fig.suptitle(f"Neuron {neuron}", size=size * 1.2)

    for tt in trial_plots.keys():
        event_list = []
        for event in tt_misperneuron[tt][neuron]:
            ax[trial_plots[tt]].plot(event, alpha=0.4, color="orange")
            ax[trial_plots[tt]].axvline(5, color="blue")
            ax[trial_plots[tt]].set_ylim((-3, 4))
            event_list.append(event)
            # ax[trial_plots[tt]].set_xticks([])  # Remove x-axis tick marks
            # ax[trial_plots[tt]].set_yticks([])  # Remove y-axis tick marks
        ax[trial_plots[tt]].plot(np.mean(event_list, axis=0), color="cyan")

    for tt in null_plots.keys():
        null_trial = null_tts[tt]
        for event in tt_rand_misperneuron[null_trial][neuron]:
            ax[null_plots[tt]].plot(event, alpha=0.2, color="grey")
            ax[null_plots[tt]].axvline(0, color="blue")
            ax[null_plots[tt]].set_ylim((-3, 4))
            # ax[null_plots[tt]].set_xticks([])  # Remove x-axis tick marks
            # ax[null_plots[tt]].set_yticks([])  # Remove y-axis tick marks

    return fig, ax


################
# MODEL TO EXPLAIN MATRICES
###################

# Priors to explain the response of each cell. Predictions for the value of z-score(dff)

pos_mismatch = [[0, 0.5, 1], [0, 0, 0.5], [0, 0, 0]]

neg_mismatch = [[0, 0, 0], [0.5, 0, 0], [1, 0.5, 0]]

near = [[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 0]]

mid = [[0.5, 1, 0.5], [0.5, 1, 0.5], [0.5, 1, 0.5]]

far = [[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]]

basis_matrices = [pos_mismatch, neg_mismatch, near, mid, far]

basis_labels = ["pos_mismatch", "neg_mismatch", "near", "mid", "far"]


def fit_models(matrix_list, basis_matrices=basis_matrices, basis_labels=basis_labels):
    model_df = {"matrix": matrix_list}

    model_df = pd.DataFrame(model_df)

    for hyp in basis_labels:
        model_df[hyp] = 0
        model_df[f"coeff_{hyp}"] = 0

    # Reshape the basis matrices into vectors and stack them to form a feature matrix
    features = np.stack([np.array(m).ravel() for m in basis_matrices]).T

    # Linear regression model
    model = LinearRegression()

    # Fit model for each target matrix
    for neuron, target in tqdm(enumerate(matrix_list)):
        target_vector = target.ravel()
        model.fit(features, target_vector)
        coefficients = model.coef_
        model_df.at[neuron, "total"] = model.score(features, target_vector)
        for idx, hyp in enumerate(basis_labels):
            model_df.at[neuron, f"coeff_{hyp}"] = coefficients[idx]

    # calculate coefficients of partial determination
    for neuron, target in tqdm(enumerate(matrix_list)):
        target_vector = target.ravel()
        cpds = calculate_cpd(features, target_vector)

        for idx, hyp in enumerate(basis_labels):
            model_df.at[neuron, hyp] = cpds[idx]

    return model_df


def calculate_cpd(features, target_vector):
    # Full model
    model_full = LinearRegression().fit(features, target_vector)
    r_squared_full = model_full.score(features, target_vector)

    cpds = []
    n_predictors = features.shape[1]

    # Reduced models
    for i in range(n_predictors):
        # Select all features except the one at index i
        features_reduced = np.delete(features, i, axis=1)
        model_reduced = LinearRegression().fit(features_reduced, target_vector)
        r_squared_reduced = model_reduced.score(features_reduced, target_vector)

        # Calculate CPD
        cpd = (r_squared_full - r_squared_reduced) / (r_squared_full)
        cpds.append(cpd)

    return cpds
