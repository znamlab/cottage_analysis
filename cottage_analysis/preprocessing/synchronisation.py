import numpy as np
import pandas as pd
import flexiznam as flz
from cottage_analysis.io_module import harp
from cottage_analysis.preprocessing import find_frames
from cottage_analysis.imaging.common.find_frames import find_imaging_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers
from functools import partial

print = partial(print, flush=True)


def load_harpmessage(
    recording,
    flexilims_session,
    conflicts="skip",
    di_names=("frame_triggers", "lick_detection", "di2_encoder_initial_state"),
):
    """Save harpmessage into a npz file, or load existing npz file. Then load harpmessage file as a np arrray.

    Args:
        recording (str or pandas.Series): recording name or recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        conflicts (str): how to deal with conflicts when updating flexilims. Defaults to "skip".
        di_names (tuple): names of the digital inputs to rename harp meassage. Defaults
            to ("frame_triggers", "lick_detection", "di2_encoder_initial_state").

    Returns:
        np.array: loaded harpmessages as numpy array
        flz.Dataset: raw harp dataset

    """
    assert conflicts in ["skip", "overwrite", "abort"]
    if type(recording) == str:
        recording = flz.get_entity(
            datatype="recording", name=recording, flexilims_session=flexilims_session
        )

    npz_ds = flz.Dataset.from_origin(
        origin_id=recording["id"],
        dataset_type="harp_npz",
        flexilims_session=flexilims_session,
        conflicts=conflicts,
    )
    # find raw data
    harp_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording["name"],
        dataset_type="harp",
        allow_multiple=False,
        return_dataseries=False,
    )
    if (npz_ds.flexilims_status() != "not online") and (conflicts == "skip"):
        print("Loading existing harp_npz file...")
        return np.load(npz_ds.path_full), harp_ds

    # parse harp message
    print("Saving harp messages into npz...")
    params = dict(
        harp_bin=harp_ds.path_full / harp_ds.extra_attributes["binary_file"],
        di_names=di_names,
        verbose=False,
    )
    harp_messages = harp.load_harp(**params)

    # save npz
    npz_ds.path = npz_ds.path.parent / f"harpmessage.npz"
    npz_ds.path_full.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_ds.path_full, **harp_messages)

    # update flexilims
    npz_ds.extra_attributes.update(params)
    npz_ds.update_flexilims(mode="overwrite")

    print("Harp messages saved.")
    return harp_messages, harp_ds


def find_monitor_frames(
    recording, flexilims_session, photodiode_protocol=5, conflicts="skip"
):
    """Synchronise monitor frame using the find_frames.sync_by_correlation, and save them
    into monitor_frames_df.pickle and monitor_db_dict.pickle.

    Args:
        recording (str or pandas.Series): recording name or recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        photodiode_protocol (int): number of photodiode quad colors used for monitoring frame refresh.
            Either 2 or 5 for now. Defaults to 5.
        conflicts (str): how to deal with conflicts when updating flexilims. Defaults to "skip".

    Returns:
        DataFrame: contains information for each monitor frame.

    """
    assert conflicts in ["skip", "overwrite", "abort"]
    # Find paths
    if type(recording) == str:
        recording = flz.get_entity(
            datatype="recording", name=recording, flexilims_session=flexilims_session
        )
        if recording is None:
            raise ValueError(f"Recording {recording} does not exist.")
    # Load files
    monitor_frames_ds = flz.Dataset.from_origin(
        origin_id=recording["id"],
        dataset_type="monitor_frames",
        flexilims_session=flexilims_session,
        conflicts=conflicts,
    )
    if monitor_frames_ds.flexilims_status() != "not online" and conflicts == "skip":
        print("Loading existing monitor frames...")
        return pd.read_pickle(monitor_frames_ds.path_full)

    harp_messages, harp_ds = load_harpmessage(
        recording=recording, flexilims_session=flexilims_session, conflicts="skip"
    )
    monitor_frames_ds.path = monitor_frames_ds.path.parent / f"monitor_frames_df.pickle"

    frame_log = pd.read_csv(
        harp_ds.path_full / harp_ds.extra_attributes["csv_files"]["FrameLog"]
    )
    recording_duration = frame_log.HarpTime.values[-1] - frame_log.HarpTime.values[0]

    frame_rate = 1 / frame_log.HarpTime.diff().median()
    print(f"Recording is {recording_duration:.0f} s long.")
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
            plot_range=1000,
            plot_dir=diagnostics_folder,
        )
        frames_df = find_frames.sync_by_frame_alternating(
            photodiode=harp_messages["photodiode"],
            analog_time=harp_messages["analog_time"],
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
            minimum_lag=1.0 / frame_rate,
            do_plot=True,
            save_folder=diagnostics_folder,
            verbose=True,
        )
        frames_df, _ = find_frames.sync_by_correlation(
            frame_log,
            harp_messages["analog_time"],
            harp_messages["photodiode"],
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
):
    """Generate a DataFrame that contains information for each monitor frame. This requires
    monitor frames to be synced first.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity
        photodiode_protocol (int): number of photodiode quad colors used for monitoring
            frame refresh. Either 2 or 5 for now. Defaults to 5.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.

    Returns:
        DataFrame: contains information for each monitor frame.

    """
    assert flexilims_session is not None or project is not None
    assert photodiode_protocol in [2, 5]
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)
    monitor_frames_df = find_monitor_frames(
        recording=recording,
        flexilims_session=flexilims_session,
        photodiode_protocol=photodiode_protocol,
        conflicts="skip",
    )

    monitor_frames_df = monitor_frames_df[
        monitor_frames_df.closest_frame.notnull()
    ].copy()
    monitor_frames_df = find_frames.remove_frames_in_wrong_order(monitor_frames_df)
    monitor_frames_df.closest_frame = monitor_frames_df.closest_frame.astype("int")
    harp_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording.name,
        dataset_type="harp",
        allow_multiple=False,
        return_dataseries=False,
    )
    if photodiode_protocol == 5:
        # Merge MouseZ and EyeZ from FrameLog.csv to frame_df according to FrameIndex
        frame_log_path = harp_ds.path_full / harp_ds.csv_files["FrameLog"]
        frame_log = pd.read_csv(frame_log_path)
        frame_log_z = frame_log[["FrameIndex", "HarpTime", "MouseZ", "EyeZ"]]
        frame_log_z.rename(
            columns={
                "FrameIndex": "closest_frame",
                "HarpTime": "harptime_framelog",
                "MouseZ": "mouse_z",
                "EyeZ": "eye_z",
            },
            inplace=True,
        )
        merge_on = "closest_frame"
    else:
        # TODO account for display lag
        # Assume peak time is the same as onset time, as we don't know about onset time when photodiode quad color is only 2
        monitor_frames_df = monitor_frames_df.rename(
            columns={"peak_time": "onset_time"}
        )
        encoder_path = harp_ds.path_full / harp_ds.csv_files["RotaryEncoder"]
        frame_log_z = pd.read_csv(encoder_path)[["Frame", "HarpTime", "MouseZ", "EyeZ"]]
        frame_log_z = frame_log_z[frame_log_z.Frame.diff() != 0]
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
    paramlog_path = harp_ds.path_full / harp_ds.csv_files["NewParams"]
    # TODO COPY FROM RAW AND READ FROM PROCESSED INSTEAD
    param_log = pd.read_csv(paramlog_path)
    param_log = param_log.rename(columns={"HarpTime": "stimulus_harptime"})
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
        frame_number = float(suite2p_ds.extra_attributes["nframes"])
    else:
        frame_number = float(
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
        frame_number=int(frame_number * nplanes),
        frame_period=(1 / fs) / nplanes - 0.001,
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
    imaging_df.imaging_volume = (
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
    if dffs.shape[0] > imaging_df['imaging_frame'].idxmax():
        print("Warning: The number of imaging frames from ScanImage is greater than the number of imaging frames synchronised with visual stimulus. Truncating the suite2p traces to match.")
        last_valid_frame = imaging_df['imaging_frame'].idxmax()+1
        imaging_df["dffs"] = np.split(dffs[0:last_valid_frame,:], last_valid_frame, axis=0)
        imaging_df["spks"] = np.split(spks[0:last_valid_frame,:], last_valid_frame, axis=0)
    else:
        imaging_df["dffs"] = np.split(dffs, dffs.shape[0], axis=0)
        imaging_df["spks"] = np.split(spks, spks.shape[0], axis=0)
         
    return imaging_df


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
