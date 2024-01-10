import warnings
import shutil
import numpy as np
import pandas as pd
import flexiznam as flz
from functools import partial
from znamutils import slurm_it
from cottage_analysis.utilities.misc import get_str_or_recording
from cottage_analysis.io_module.harp import load_harpmessage
from cottage_analysis.io_module import onix as onix_io
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
        breakout = onix_io.load_breakout(raw / onix_recording.path)
        onix_data = onix_prepro.preprocess_onix_recording(
            dict(breakout_data=breakout), harp_message=harp_message
        )
        ch_pd = onix_prepro.ANALOG_INPUTS.index("photodiode")
        photodiode = onix_data["breakout_data"]["aio"][ch_pd, :]
        analog_time = onix_data["onix2harp"](onix_data["breakout_data"]["aio-clock"])

    # Get frame log
    if type(harp_ds.extra_attributes["csv_files"]) == str:
        # Some yaml info have been saved as string instead of dict
        # TODO: fix on flexilims and/or use yaml.safe_load
        frame_log = pd.read_csv(
            harp_ds.path_full / eval(harp_ds.extra_attributes["csv_files"])["FrameLog"]
        )
    else:
        frame_log = pd.read_csv(
            harp_ds.path_full / harp_ds.extra_attributes["csv_files"]["FrameLog"]
        )
    recording_duration = frame_log.HarpTime.values[-1] - frame_log.HarpTime.values[0]
    frame_rate = 1 / frame_log.HarpTime.diff().median()
    print(f"Recording is {recording_duration:.0f} s long.")
    print(f"Frame rate is {frame_rate:.0f} Hz.")

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
    raw = flz.get_data_root("raw", flexilims_session=flexilims_session)
    processed = flz.get_data_root("processed", flexilims_session=flexilims_session)
    if photodiode_protocol == 5:
        # Merge MouseZ and EyeZ from FrameLog.csv to frame_df according to FrameIndex
        frame_log = pd.read_csv(harp_ds.path_full / harp_files["FrameLog"])
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

    vs_df = pd.merge_asof(
        left=monitor_frames_df[["closest_frame", "onset_time"]],
        right=frame_log_z,
        on=merge_on,
        direction="backward",
        allow_exact_matches=True,
    )
    # Align paramLog with vs_df
    paramlog_path = harp_ds.path_full / harp_files["NewParams"]
    # TODO COPY FROM RAW AND READ FROM PROCESSED INSTEAD
    param_log = pd.read_csv(paramlog_path)
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
    imaging_df["imaging_harptime_end"].iloc[-1] = (
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
    vs_df, onix_recording, flexilims_session, rate_bin, filter_datasets=None
):
    """This is the equivalent of generate_imaging_df for spike rate data.

    Spike rate will be calculated for each bin of `rate_bin` secondes.

    Args:
        vs_df (DataFrame): DataFrame, e.g. output of generate_vs_df
        onix_recording (pandas.Series): recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        rate_bin (int): bin size in s.
        filter_datasets (dict, optional): filters to apply on choosing onix datasets.
            Defaults to None."""
    return


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
