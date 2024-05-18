import warnings
import numpy as np
import pandas as pd
import scipy.signal
import flexiznam as flz
from functools import partial
from znamutils import slurm_it
from cottage_analysis.utilities.misc import get_str_or_recording

from cottage_analysis.io_module.harp import load_harpmessage
from cottage_analysis.io_module import onix as onix_io
from cottage_analysis.io_module.visstim import get_frame_log, get_param_log
from cottage_analysis.io_module.spikes import (
    load_kilosort_folder,
    get_smoothed_spike_rate,
)
from cottage_analysis.preprocessing import onix as onix_prepro
from cottage_analysis.preprocessing import find_frames
from cottage_analysis.imaging.common.find_frames import find_imaging_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers


print = partial(print, flush=True)


@slurm_it(conda_env="cottage_analysis")
def find_monitor_frames(
    vis_stim_recording,
    flexilims_session=None,
    photodiode_protocol=5,
    conflicts="skip",
    harp_recording=None,
    onix_recording=None,
    sync_kwargs=None,
    project=None,
):
    """Synchronise monitor frame using the find_frames.sync_by_correlation, and save them
    into monitor_frames_df.pickle and monitor_db_dict.pickle.

    Args:
        recording (str or pandas.Series): recording name or recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        photodiode_protocol (int): number of photodiode quad colors used for monitoring frame refresh.
            Either 2 or 5 for now. Defaults to 5.
        conflicts (str): how to deal with conflicts when updating flexilims. Defaults to "skip".
        harp_recording (str or pandas.Series): recording name or recording entry
            from flexilims containing the photodiode signal. If None use
            vis_stim_recording. Defaults to None.
        onix_recording (str or pandas.Series): recording name or recording entry
            from flexilims containing the analog photodiode signal. Defaults to None.
        sync_kwargs (dict): keyword arguments for the sync function. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if
            flexilims_session is None.

    Returns:
        DataFrame: contains information for each monitor frame.

    """
    assert conflicts in ["skip", "overwrite", "abort"]
    if flexilims_session is None:
        assert (
            project is not None
        ), "project must be provided if flexilims_session is None"
        flexilims_session = flz.get_flexilims_session(project_id=project)

    vis_stim_recording = get_str_or_recording(vis_stim_recording, flexilims_session)
    if harp_recording is None:
        harp_recording = vis_stim_recording
    else:
        harp_recording = get_str_or_recording(harp_recording, flexilims_session)
        onix_recording = get_str_or_recording(onix_recording, flexilims_session)

    # Create output and reload
    monitor_frames_ds = flz.Dataset.from_origin(
        origin_id=vis_stim_recording["id"],
        dataset_type="monitor_frames",
        flexilims_session=flexilims_session,
        conflicts=conflicts,
    )
    if monitor_frames_ds.flexilims_status() != "not online" and conflicts == "skip":
        print("Loading existing monitor frames...")
        return pd.read_pickle(monitor_frames_ds.path_full)

    monitor_frames_ds.path = monitor_frames_ds.path.parent / f"monitor_frames_df.pickle"

    # Get photodiode
    raw = flz.get_data_root("raw", flexilims_session=flexilims_session)
    harp_message, harp_ds = load_harpmessage(
        recording=harp_recording,
        flexilims_session=flexilims_session,
        conflicts="skip",
    )
    if onix_recording is None:
        # get the photodiode from harp directly
        photodiode = harp_message["photodiode"]
        analog_time = harp_message["analog_time"]
    else:
        onix_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=onix_recording.name,
            dataset_type="onix",
            allow_multiple=False,
        )
        breakout = onix_io.load_breakout(raw / onix_recording.path)
        onix_data = onix_prepro.preprocess_onix_recording(
            dict(breakout_data=breakout), harp_message=harp_message
        )
        if "aio_mapping" in onix_ds.extra_attributes:
            ch_pd = onix_ds.extra_attributes["aio_mapping"]["photodiode"]
        else:
            ch_pd = onix_prepro.ANALOG_INPUTS.index("photodiode")
        photodiode = onix_data["breakout_data"]["aio"][ch_pd, :]
        analog_time = onix_data["onix2harp"](onix_data["breakout_data"]["aio-clock"])
        # to make it faster, decimate the photodiode signal
        photodiode = scipy.signal.decimate(photodiode, 5)
        analog_time = analog_time[::5]

    # Get frame log
    frame_log = get_frame_log(
        flexilims_session,
        harp_recording=harp_recording,
        vis_stim_recording=vis_stim_recording,
    )

    recording_duration = frame_log.HarpTime.values[-1] - frame_log.HarpTime.values[0]
    frame_rate = 1 / frame_log.HarpTime.diff().median()
    print(f"Recording is {recording_duration:.0f} s long.")
    print(f"Frame rate is {frame_rate:.0f} Hz.")

    # If the last frame log row contains Nan, remove it and warn
    if frame_log.iloc[-1].isna().any():
        frame_log = frame_log[:-1]
        print(
            f"WARNING: Removed last row of FrameLog.csv with NaN probably due to bonsai crash."
        )

    # Get frames from photodiode trace, depending on the photodiode protocol is 2 or 5
    diagnostics_folder = (
        monitor_frames_ds.path_full.parent / "diagnostics" / "frame_sync"
    )
    diagnostics_folder.mkdir(parents=True, exist_ok=True)
    if photodiode_protocol == 2:
        params = dict(
            photodiode_sampling=1000,
            plot=True,
            plot_start=10000,
            plot_range=50,
            plot_dir=diagnostics_folder,
        )
        if sync_kwargs is not None:
            params.update(sync_kwargs)
        frames_df = find_frames.sync_by_frame_alternating(
            photodiode=photodiode,
            analog_time=analog_time,
            frame_rate=frame_rate,
            **params,
        )
    elif photodiode_protocol == 5:
        params = dict(
            time_column="HarpTime",
            sequence_column="PhotoQuadColor",
            num_frame_to_corr=6,
            maxlag=3.0 / frame_rate,
            expected_lag=2.0 / frame_rate,
            frame_rate=frame_rate,
            correlation_threshold=0.8,
            relative_corr_thres=0.02,
            frame_detection_height=0.1,
            minimum_lag=1.0 / frame_rate,
            do_plot=True,
            save_folder=diagnostics_folder,
            verbose=True,
            ignore_errors=True,
        )
        if sync_kwargs is not None:
            params.update(sync_kwargs)
        frames_df, _ = find_frames.sync_by_correlation(
            frame_log,
            photodiode_time=analog_time,
            photodiode_signal=photodiode,
            **params,
        )
    params["photodiode_protocol"] = photodiode_protocol
    # Save monitor frame dataframes
    frames_df.to_pickle(monitor_frames_ds.path_full)
    monitor_frames_ds.extra_attributes = params
    monitor_frames_ds.update_flexilims(mode="overwrite")
    return frames_df


def generate_vs_df(
    recording,
    photodiode_protocol=5,
    flexilims_session=None,
    project=None,
    harp_recording=None,
    onix_recording=None,
    conflicts="skip",
    sync_kwargs=None,
):
    """Generate a DataFrame that contains information for each monitor frame. This
    requires monitor frames to be synced first.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity
        photodiode_protocol (int): number of photodiode quad colors used for monitoring
            frame refresh. Either 2 or 5 for now. Defaults to 5.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to
            None.
        project (str, optional): project name. Defaults to None. Must be provided if
            flexilims_session is None.
        harp_recording (str or pandas.Series, optional): recording name or recording
            entry if different from (vis stim) recording. Defaults to None.
        onix_recording (str or pandas.Series, optional): recording name or recording
            entry if photodiode is recorded on onix. Defaults to None.
        conflicts (str, optional): how to deal with conflicts when updating flexilims.
            Defaults to "skip".
        sync_kwargs (dict, optional): keyword arguments for the sync function. Defaults
            to None.

    Returns:
        DataFrame: contains information for each monitor frame.

    """
    assert flexilims_session is not None or project is not None
    assert photodiode_protocol in [2, 5]
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)

    recording = get_str_or_recording(recording, flexilims_session)
    harp_recording = get_str_or_recording(harp_recording, flexilims_session)
    if harp_recording is None:
        harp_recording = recording

    monitor_frames_df = find_monitor_frames(
        vis_stim_recording=recording,
        flexilims_session=flexilims_session,
        photodiode_protocol=photodiode_protocol,
        harp_recording=harp_recording,
        onix_recording=onix_recording,
        conflicts=conflicts,
        sync_kwargs=sync_kwargs,
    )

    monitor_frames_df = monitor_frames_df[
        monitor_frames_df.closest_frame.notnull()
    ].copy()
    monitor_frames_df = find_frames.remove_frames_in_wrong_order(monitor_frames_df)
    monitor_frames_df.closest_frame = monitor_frames_df.closest_frame.astype("int")
    harp_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=harp_recording.name,
        dataset_type="harp",
        allow_multiple=False,
        return_dataseries=False,
    )
    if type(harp_ds.extra_attributes["csv_files"]) == str:
        harp_files = eval(harp_ds.extra_attributes["csv_files"])
    else:
        harp_files = harp_ds.extra_attributes["csv_files"]

    if photodiode_protocol == 5:
        # Merge MouseZ and EyeZ from FrameLog.csv to frame_df according to FrameIndex
        frame_log = get_frame_log(
            harp_ds.flexilims_session,
            harp_recording=harp_recording,
            vis_stim_recording=recording,
        )

        frame_log_z = frame_log[["FrameIndex", "HarpTime", "MouseZ", "EyeZ"]].copy()
        frame_log_z.rename(
            columns={
                "FrameIndex": "closest_frame",
                "HarpTime": "harptime_framelog",
                "MouseZ": "mouse_z",
                "EyeZ": "eye_z",
            },
            inplace=True,
        )

        if frame_log_z.closest_frame.isna().any():
            print(
                f"WARNING: {np.sum(frame_log_z.closest_frame.isna())} frames are missing from FrameLog.csv. This is likely due to bonsai crash at the end."
            )
            frame_log_z = frame_log_z[frame_log_z.closest_frame.notnull()]
            frame_log_z.closest_frame = frame_log_z.closest_frame.astype("int")

        merge_on = "closest_frame"
    else:
        # TODO account for display lag
        # Assume peak time is the same as onset time, as we don't know about onset time when photodiode quad color is only 2
        monitor_frames_df = monitor_frames_df.rename(
            columns={"peak_time": "onset_time"}
        )
        encoder_path = harp_ds.path_full / harp_files["RotaryEncoder"]
        frame_log_z = pd.read_csv(encoder_path)[["Frame", "HarpTime", "MouseZ", "EyeZ"]]
        frame_log_z = frame_log_z[frame_log_z.Frame.diff() != 0].copy()
        frame_log_z.rename(
            columns={"HarpTime": "onset_time", "MouseZ": "mouse_z", "EyeZ": "eye_z"},
            inplace=True,
        )
        frame_log_z.drop(columns={"Frame"}, inplace=True)
        merge_on = "onset_time"

    frame_log_z.mouse_z = frame_log_z.mouse_z / 100  # convert cm to m
    frame_log_z.eye_z = frame_log_z.eye_z / 100  # convert cm to m

    if monitor_frames_df[merge_on].dtype != frame_log_z[merge_on].dtype:
        # print a warning if the merge_on column is not the same type in both dataframes
        warnings.warn(
            f"WARNING: merge_on column {merge_on} is not the same type in both "
            + f"dataframes. monitor_frame_df is {monitor_frames_df[merge_on].dtype} and"
            + f"frame_log_z is {frame_log_z[merge_on].dtype}. Converting to int64."
        )
        # convert both to int64
        monitor_frames_df[merge_on] = monitor_frames_df[merge_on].astype("int64")
        frame_log_z[merge_on] = frame_log_z[merge_on].astype("int64")

    vs_df = pd.merge_asof(
        left=monitor_frames_df[["closest_frame", "onset_time"]],
        right=frame_log_z,
        on=merge_on,
        direction="backward",
        allow_exact_matches=True,
    )
    # Align paramLog with vs_df
    param_log = get_param_log(
        flexilims_session=flexilims_session,
        harp_recording=harp_recording,
        vis_stim_recording=recording,
    )
    # TODO COPY FROM RAW AND READ FROM PROCESSED INSTEAD
    param_log = param_log.rename(columns={"HarpTime": "stimulus_harptime"})
    if "Frameindex" in param_log.columns:
        if param_log.Frameindex.isna().any():
            print(
                f"WARNING: {np.sum(param_log.Frameindex.isna())} frames are missing from ParamLog.csv. This is likely due to bonsai crash at the end."
            )
            param_log = param_log[param_log.Frameindex.notnull()]
            param_log.Frameindex = param_log.Frameindex.astype("int")

    if photodiode_protocol == 5:
        vs_df = pd.merge_asof(
            left=vs_df,
            right=param_log,
            left_on="closest_frame",
            right_on="Frameindex",
            direction="backward",
            allow_exact_matches=True,
        )
    else:
        vs_df = pd.merge_asof(
            left=vs_df,
            right=param_log,
            left_on="onset_time",
            right_on="stimulus_harptime",
            direction="backward",
            allow_exact_matches=True,
        )
    # Rename
    vs_df.rename(
        columns={"closest_frame": "monitor_frame", "onset_time": "monitor_harptime"},
        inplace=True,
    )
    vs_df.drop(
        columns=[
            "harptime_framelog",
            "harptime_sphere",
            "harptime_imaging_trigger",
            "offset_time",
            "peak_time",
        ],
        errors="ignore",
        inplace=True,
    )
    return vs_df


def generate_imaging_df(
    vs_df, recording, flexilims_session, filter_datasets=None, return_volumes=True
):
    """
    Generate a DataFrame that contains information for each imaging volume / frame incorporating
    the monitor frame information.

    Args:
        vs_df (DataFrame): DataFrame, e.g. output of generate_vs_df
        recording (pandas.Series): recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        filter_datasets (dict, optional): filters to apply on choosing suite2p datasets. Defaults to None.
        return_volumes (bool): if True, return only the first frame of each imaging volume. Defaults to True.

    Returns:
        DataFrame: contains information for each imaging volume / frame.

    """
    # get the suite2p dataset to check the frame number, frame rate and number of planes
    suite2p_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="suite2p_traces",
        filter_datasets=filter_datasets,
        allow_multiple=False,
        return_dataseries=False,
    )
    if "nframes" in suite2p_ds.extra_attributes:
        volume_number = float(suite2p_ds.extra_attributes["nframes"])
    else:
        volume_number = float(
            np.load(suite2p_ds.path_full / "plane0" / "dff_ast.npy").shape[1]
        )
    nplanes = float(suite2p_ds.extra_attributes["nplanes"])
    fs = float(suite2p_ds.extra_attributes["fs"])
    harp_npz_path = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp_npz",
        allow_multiple=False,
        return_dataseries=False,
    ).path_full
    # frame period calculated based of the frame rate in ops.npy
    # subtracting 1 ms to account for the duration of the triggers
    imaging_df = find_imaging_frames(
        harp_message=format_loggers.format_img_frame_logger(
            harpmessage_file=harp_npz_path, register_address=32
        ),
        frame_number=int(volume_number * nplanes),
        frame_period=(1 / fs) / nplanes,
        register_address=32,
        frame_period_tolerance=0.001,
    )

    imaging_df = imaging_df[["HarpTime", "ImagingFrame"]]
    imaging_df.rename(
        columns={
            "HarpTime": "imaging_harptime",
            "ImagingFrame": "imaging_frame",
        },
        inplace=True,
    )
    imaging_df["imaging_volume"] = (
        (imaging_df.imaging_frame / nplanes).apply(np.floor).astype(int)
    )
    # if return_volumes is True, select rows where imaging_volume changes
    if return_volumes:
        volume_starts = imaging_df.imaging_volume.diff()
        volume_starts.iloc[0] = 1
        imaging_df = imaging_df[volume_starts != 0].copy()
    # add a column for the harptime at end the imaging volume
    imaging_df["imaging_harptime_end"] = imaging_df.imaging_harptime.shift(-1)
    # set the last value of imaging_harptime_end to the last value of imaging_harptime + median frame period
    imaging_df.at[imaging_df.index[-1], "imaging_harptime_end"] = (
        imaging_df["imaging_harptime"].iloc[-1]
        + imaging_df["imaging_harptime"].diff().median()
    )
    # select the last monitor frame before the end of each imaging volume / frame
    imaging_df = pd.merge_asof(
        left=imaging_df,
        right=vs_df,
        left_on="imaging_harptime_end",
        right_on="monitor_harptime",
        direction="backward",
        allow_exact_matches=True,
    )
    # Align mouse z extracted from harpmessage with frame (mouse z before the harptime of frame)
    harpmessage = np.load(harp_npz_path)
    mouse_z_harp_df = pd.DataFrame(
        {
            "mouse_z_harptime": harpmessage["analog_time"],
            "mouse_z_harp": np.cumsum(harpmessage["rotary_meter"]),
        }
    )
    # select the last mouse z before the end of each imaging volume / frame
    imaging_df = pd.merge_asof(
        left=imaging_df,
        right=mouse_z_harp_df,
        left_on="imaging_harptime_end",
        right_on="mouse_z_harptime",
        direction="backward",
        allow_exact_matches=True,
    )
    dff_fname = (
        "dff_ast.npy" if suite2p_ds.extra_attributes["ast_neuropil"] else "dff.npy"
    )
    spks_fname = (
        "spks_ast.npy" if suite2p_ds.extra_attributes["ast_neuropil"] else "spks.npy"
    )
    dffs = []
    spks = []
    for iplane in range(int(nplanes)):
        dffs.append(np.load(suite2p_ds.path_full / f"plane{iplane}" / dff_fname))
        spks.append(np.load(suite2p_ds.path_full / f"plane{iplane}" / spks_fname))
    dffs = np.vstack(dffs).T
    spks = np.vstack(spks).T
    # convert dffs to list of arrays
    if nplanes == 1.0 and dffs.shape[0] > imaging_df["imaging_frame"].idxmax() + 1:
        print(
            "Warning: The number of imaging frames from ScanImage is greater than the number of imaging frames synchronised with visual stimulus. Truncating the suite2p traces to match."
        )
        last_valid_frame = imaging_df["imaging_frame"].idxmax() + 1
        imaging_df["dffs"] = np.split(
            dffs[0:last_valid_frame, :], last_valid_frame, axis=0
        )
        imaging_df["spks"] = np.split(
            spks[0:last_valid_frame, :], last_valid_frame, axis=0
        )
    elif nplanes > 1.0 and dffs.shape[0] > imaging_df["imaging_volume"].idxmax() + 1:
        print(
            "Warning: The number of imaging frames from ScanImage is greater than the number of imaging frames synchronised with visual stimulus. Truncating the suite2p traces to match."
        )
        last_valid_volume = imaging_df["imaging_volume"].idxmax() + 1
        imaging_df["dffs"] = np.split(
            dffs[0:last_valid_volume, :], last_valid_volume, axis=0
        )
        imaging_df["spks"] = np.split(
            spks[0:last_valid_volume, :], last_valid_volume, axis=0
        )
    else:
        imaging_df["dffs"] = np.split(dffs, dffs.shape[0], axis=0)
        imaging_df["spks"] = np.split(spks, spks.shape[0], axis=0)

    return imaging_df


def generate_spike_rate_df(
    vs_df,
    onix_recording,
    harp_recording,
    flexilims_session,
    rate_bin=1 / 30.0,
    exp_sd=0.1,
    filter_datasets=None,
    return_multiunit=False,
    unit_list=None,
):
    """This is the equivalent of generate_imaging_df for spike rate data.

    Spike rate will be calculated for each bin of `rate_bin` secondes.

    Args:
        vs_df (DataFrame): DataFrame, e.g. output of generate_vs_df
        onix_recording (pandas.Series or str): recording entry from flexilims.
        harp_recording (pandas.Series or str): recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        rate_bin (int): bin size in s.
        exp_sd (float): standard deviation of the exponential filter to apply on the spike rate.
        filter_datasets (dict, optional): filters to apply on choosing onix datasets.
            Defaults to None.
        return_multiunit (bool): if True, process multiunits as well. Defaults to False.
        unit_list (list): list of units to process. Defaults to None.

    Returns:
        DataFrame: contains information for each neuron / frame.
    """
    onix_recording = get_str_or_recording(onix_recording, flexilims_session)
    harp_recording = get_str_or_recording(harp_recording, flexilims_session)
    spike_ds = flz.get_datasets(
        origin_id=onix_recording.id,
        dataset_type="kilosort2",
        allow_multiple=False,
        flexilims_session=flexilims_session,
    )

    out = load_kilosort_folder(spike_ds.path_full, return_multiunit=return_multiunit)
    if return_multiunit:
        ks_data, good_units, mua_units = out
        units = {**good_units, **mua_units}
    else:
        ks_data, units = out
    if unit_list is not None:
        print(f"Filtering {len(units)} units to {unit_list}")
        units = {k: v for k, v in units.items() if k in unit_list}
        print(f"Filtered to {len(units)} units")

    # Express spikes in harptime
    harp_message, harp_ds = load_harpmessage(
        recording=harp_recording,
        flexilims_session=flexilims_session,
        conflicts="skip",
    )
    onix_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=onix_recording.name,
        dataset_type="onix",
        allow_multiple=False,
    )
    breakout = onix_io.load_breakout(onix_ds.path_full)
    rhd = onix_io.load_rhd2164(onix_ds.path_full, cut_if_not_multiple=True)
    onix_data = onix_prepro.preprocess_onix_recording(
        dict(breakout_data=breakout), harp_message=harp_message, cut_onix=True
    )
    onix_data.keys()
    units_harp = {}
    for cl, spike_index in units.items():
        # if the recording has been interupted we can have spike_index further than
        # clock. Cut them
        spike_index = spike_index[spike_index < len(rhd["clock"])]
        spike_clock = rhd["clock"][spike_index]
        units_harp[cl] = onix_data["onix2harp"](spike_clock)

    # create a dataframe that looks like the imaging df, but using bins of rate_bin
    bins = np.arange(
        vs_df.monitor_harptime.min() - 10,
        vs_df.monitor_harptime.max() + 10 + rate_bin,
        rate_bin,
    )
    imaging_df = pd.DataFrame(
        dict(
            imaging_harptime=bins[:-1],
            imaging_harptime_end=bins[1:],
            imaging_frame=np.arange(len(bins) - 1),
        )
    )

    imaging_df = pd.merge_asof(
        left=imaging_df,
        right=vs_df,
        left_on="imaging_harptime_end",
        right_on="monitor_harptime",
        direction="backward",
        allow_exact_matches=True,
    )
    # Align mouse z extracted from harpmessage with frame (mouse z before the harptime of frame)
    harpmessage = load_harpmessage(
        recording=harp_recording, flexilims_session=flexilims_session
    )[0]
    mouse_z_harp_df = pd.DataFrame(
        {
            "mouse_z_harptime": harpmessage["analog_time"],
            "mouse_z_harp": np.cumsum(harpmessage["rotary_meter"]),
        }
    )
    # select the last mouse z before the end of each imaging volume / frame
    imaging_df = pd.merge_asof(
        left=imaging_df,
        right=mouse_z_harp_df,
        left_on="imaging_harptime_end",
        right_on="mouse_z_harptime",
        direction="backward",
        allow_exact_matches=True,
    )

    # get the spike rate for each units
    spks, unit_ids = get_smoothed_spike_rate(
        units_harp, bins, exp_sd=exp_sd, save_folder=spike_ds.path_full
    )
    # convert spks to list of arrays
    imaging_df["spks"] = np.split(spks, spks.shape[0], axis=0)
    imaging_df["dffs"] = np.split(spks, spks.shape[0], axis=0)
    imaging_df["unit_ids"] = [unit_ids] * len(imaging_df)

    return imaging_df, unit_ids


def fill_missing_imaging_volumes(df, nan_col="RS"):
    """
    Create a dataframe with a single row for each imaging volume, by forward filling
    the values from the previous imaging volume.

    Args:
        df (DataFrame): DataFrame, e.g. output of generate_vs_df
        nan_col (string): name of the colume for imaging_df where there are nan values due to frame drops

    Returns:
        DataFrame: DataFrame with a single row for each imaging volume

    """
    img_df = pd.DataFrame({"imaging_volume": np.arange(df["imaging_volume"].max())})
    # select rows of df where imaging_volume is not nan
    img_df = pd.merge_asof(
        left=img_df,
        right=df[df[nan_col].notna()],
        on="imaging_volume",
        direction="backward",
        allow_exact_matches=True,
    )
    return img_df


def load_imaging_data(recording_name, flexilims_session, filter_datasets=None):
    suite2p_traces = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording_name,
        dataset_type="suite2p_traces",
        allow_multiple=False,
        filter_datasets=filter_datasets,
    )
    dffs = []
    for iplane in range(int(float(suite2p_traces.extra_attributes["nplanes"]))):
        plane_path = suite2p_traces.path_full / f"plane{iplane}"
        dffs.append(np.load(plane_path / "dff_ast.npy"))
    return np.concatenate(dffs, axis=0).T
