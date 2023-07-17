import functools

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import flexiznam as flz
from cottage_analysis.io_module import harp
from cottage_analysis.preprocessing import find_frames
from cottage_analysis.filepath import generate_filepaths
from cottage_analysis.imaging.common import find_frames as find_img_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers


def load_harpmessage(recording, flexilims_session, conflicts="skip"):
    """Save harpmessage into a npz file, or load existing npz file. Then load harpmessage file as a np arrray.

    Args:
        recording (str or pandas.Series): recording name or recording entry from flexilims.
        flexilims_session (flexilims.Flexilims): flexilims session.
        conflicts (str): how to deal with conflicts when updating flexilims. Defaults to "skip".

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
    harp_ds = flz.get_child_datasets(
        flexilims_session,
        recording["name"],
        dataset_type="harp",
        allow_multiple=False,
        return_dataseries=False,
    )
    if npz_ds.flexilims_status() != "not online" and conflicts == "skip":
        print("Loading existing harp_npz file...")
        return np.load(npz_ds.path_full), harp_ds

    # parse harp message
    print("Saving harp messages into npz...")
    params = dict(
        harp_bin=harp_ds.path_full / harp_ds.extra_attributes["binary_file"],
        di_names=("frame_triggers", "lick_detection", "di2_encoder_initial_state"),
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
        recording=recording, flexilims_session=flexilims_session, conflicts=conflicts
    )
    monitor_frames_ds.path = monitor_frames_ds.path.parent / f"monitor_frames_df.pickle"

    frame_log = pd.read_csv(
        harp_ds.path_full.parent / harp_ds.extra_attributes["csv_files"]["FrameLog"]
    )
    recording_duration = frame_log.HarpTime.values[-1] - frame_log.HarpTime.values[0]

    frame_rate = 1 / frame_log.HarpTime.diff().median()
    print(f"Recording is {recording_duration:.0f} s long.")
    # Get frames from photodiode trace, depending on the photodiode protocol is 2 or 5
    if photodiode_protocol == 2:
        frames_df = find_frames.sync_by_frame_alternating(
            photodiode=harp_messages["photodiode"],
            analog_time=harp_messages["analog_time"],
            frame_rate=frame_rate,
            photodiode_sampling=1000,
            plot=True,
            plot_start=10000,
            plot_range=1000,
            plot_dir=monitor_frames_ds.path_full.parent,
        )
    elif photodiode_protocol == 5:
        frames_df, _ = find_frames.sync_by_correlation(
            frame_log,
            harp_messages["analog_time"],
            harp_messages["photodiode"],
            time_column="HarpTime",
            sequence_column="PhotoQuadColor",
            num_frame_to_corr=6,
            maxlag=3.0 / frame_rate,
            expected_lag=2.0 / frame_rate,
            frame_rate=frame_rate,
            correlation_threshold=0.8,
            relative_corr_thres=0.02,
            minimum_lag=1.0 / frame_rate,
            do_plot=False,
            verbose=True,
        )
    # Save monitor frame dataframes
    frames_df.to_pickle(monitor_frames_ds.path_full)
    monitor_frames_ds.extra_attributes["photodiode_protocol"] = photodiode_protocol
    monitor_frames_ds.update_flexilims(mode="overwrite")
    return frames_df


def generate_vs_df(
    recording,
    photodiode_protocol=5,
    flexilims_session=None,
    project=None,
    is_imaging=True,
):
    """Generate a DataFrame that contains information for each monitor frame. This requires monitor frames to be synced first.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project)
        photodiode_protocol (int): number of photodiode quad colors used for monitoring frame refresh. Either 2 or 5 for now. Defaults to 5.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        project (str): project name. Defaults to None. Must be provided if flexilims_session is None.
        is_imaging (bool): is the data imaging data? Defaults to True. If it is imaging data, imaging trigger log will be synced with vs_df.

    Returns:
        DataFrame: contains information for each monitor frame.

    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)
    raw_path = Path(flz.PARAMETERS["data_root"]["raw"]) / recording.path
    processed_path = Path(flz.PARAMETERS["data_root"]["processed"]) / recording.path
    monitor_frames_path = (
        processed_path / "sync" / "monitor_frames" / "monitor_frames_df.pickle"
    )
    monitor_frames_df = pd.read_pickle(monitor_frames_path)

    if photodiode_protocol == 5:
        # Find frames that are not skipped
        monitor_frame_valid = monitor_frames_df[
            monitor_frames_df.closest_frame.notnull()
        ][["closest_frame", "onset_time", "offset_time", "peak_time"]]
        monitor_frame_valid["closest_frame"] = monitor_frame_valid[
            "closest_frame"
        ].astype("int")
        monitor_frame_valid = monitor_frame_valid.sort_values("closest_frame")

        # Merge MouseZ and EyeZ from FrameLog.csv to frame_df according to FrameIndex
        frame_log = pd.read_csv(raw_path / "FrameLog.csv")
        frame_log_z = frame_log[["FrameIndex", "HarpTime", "MouseZ", "EyeZ"]]
        frame_log_z = frame_log_z.rename(
            columns={
                "FrameIndex": "closest_frame",
                "HarpTime": "harptime_framelog",
                "MouseZ": "mouse_z",
                "EyeZ": "eye_z",
            }
        )
        frame_log_z["mouse_z"] = frame_log_z["mouse_z"] / 100  # convert cm to m
        frame_log_z["eye_z"] = frame_log_z["eye_z"] / 100  # convert cm to m
        vs_df = pd.merge_asof(
            left=monitor_frame_valid,
            right=frame_log_z,
            on="closest_frame",
            direction="nearest",
            allow_exact_matches=True,
        )

    if photodiode_protocol == 2:
        monitor_frame_valid = monitor_frames_df[["closest_frame", "peak_time"]]
        # Assume peak time is the same as onset time, as we don't know about onset time when photodiode quad color is only 2
        monitor_frame_valid = monitor_frame_valid.rename(
            columns={"peak_time": "onset_time"}
        )
        monitor_frame_valid["closest_frame"] = monitor_frame_valid[
            "closest_frame"
        ].astype("int")
        monitor_frame_valid = monitor_frame_valid.sort_values("closest_frame")

        encoder_path = raw_path / "RotaryEncoder.csv"
        mouse_z_df = pd.read_csv(encoder_path)[["Frame", "HarpTime", "MouseZ", "EyeZ"]]
        mouse_z_df = mouse_z_df[mouse_z_df.Frame.diff() != 0]
        mouse_z_df = mouse_z_df.rename(
            columns={"HarpTime": "onset_time", "MouseZ": "mouse_z", "EyeZ": "eye_z"}
        )
        mouse_z_df = mouse_z_df.drop(columns={"Frame"})

        mouse_z_df["mouse_z"] = mouse_z_df["mouse_z"] / 100  # convert cm to m
        mouse_z_df["eye_z"] = mouse_z_df["eye_z"] / 100  # convert cm to m
        vs_df = pd.merge_asof(
            left=monitor_frame_valid,
            right=mouse_z_df,
            on="onset_time",
            direction="nearest",
            allow_exact_matches=True,
        )

    # Filter monitor frame index and get rid of frames when diff is negative
    frame_idx_dff = vs_df.monitor_frame.diff()
    frame_idx_dff[0] = 1
    vs_df = vs_df[~(frame_idx_dff.shift(-1) < 0)]

    vs_df = vs_df.sort_values("onset_time")

    # Align imaging frame time with monitor frame onset time (imaging frame time later than monitor frame onset time)
    if is_imaging:
        suite2p_dataset = get_child_dataset(
            flexilims_session, recording.name, "suite2p_traces"
        )
        save_folder = processed_path / "sync"
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        p_msg = processed_path / "sync" / "harpmessage.npz"
        frame_number = float(suite2p_dataset.extra_attributes["nframes"])
        nplanes = float(suite2p_dataset.extra_attributes["nplanes"])
        fs = float(suite2p_dataset.extra_attributes["fs"])
        # frame period calculated based of the frame rate in ops.npy
        # subtracting 1 ms to account for the duration of the triggers
        img_frame_logger = find_img_frames.find_imaging_frames(
            harp_message=format_loggers.format_img_frame_logger(
                harpmessage_file=p_msg, register_address=32
            ),
            frame_number=int(frame_number * nplanes),
            frame_period=(1 / fs) / nplanes - 0.001,
            register_address=32,
            frame_period_tolerance=0.001,
        )

        img_frame_logger = img_frame_logger[["HarpTime", "ImagingFrame"]]
        img_frame_logger.to_pickle(save_folder / "img_frame_logger.pickle")
        img_frame_logger = img_frame_logger.rename(
            columns={
                "HarpTime": "onset_time",
                "ImagingFrame": "imaging_frame",
            }
        )
        img_frame_logger["imaging_volume"] = (
            img_frame_logger["imaging_frame"] / nplanes
        ).astype(int)
        vs_df = pd.merge_asof(
            left=vs_df,
            right=img_frame_logger,
            on="onset_time",
            direction="forward",
            allow_exact_matches=True,
        )

    # Align mouse z extracted from harpmessage with frame (mouse z before the harptime of frame)
    harpmessage = np.load(p_msg)
    mouse_z_harp_df = pd.DataFrame(
        {
            "onset_time": harpmessage["analog_time"],
            "mouse_z_harp": np.cumsum(harpmessage["rotary_meter"]),
        }
    )
    vs_df = pd.merge_asof(
        left=vs_df,
        right=mouse_z_harp_df,
        on="onset_time",
        direction="backward",
        allow_exact_matches=True,
    )

    # Rename
    vs_df = vs_df.rename(columns={"closest_frame": "monitor_frame"})
    for col in ["harptime_framelog", "harptime_sphere", "harptime_imaging_trigger"]:
        if col in vs_df.columns:
            vs_df = vs_df.drop(columns=[col])
    for col in ["onset_time", "offset_time", "peak_time"]:
        if col in vs_df.columns:
            vs_df = vs_df.rename(
                columns={
                    "onset_time": "onset_harptime",
                    "offset_time": "offset_harptime",
                    "peak_time": "peak_harptime",
                }
            )

    return vs_df


def fill_missing_imaging_volumes(df):
    """
    Create a dataframe with a single row for each imaging volume, by forward filling
    the values from the previous imaging volume.

    Args:
        df (DataFrame): DataFrame, e.g. output of generate_vs_df

    Returns:
        DataFrame: DataFrame with a single row for each imaging volume

    """
    img_df = pd.DataFrame({"imaging_volume": np.arange(df["imaging_volume"].max())})
    # select rows of df where imaging_volume is not nan
    img_df = pd.merge_asof(
        left=img_df,
        right=df[df["imaging_volume"].notna()],
        on="imaging_volume",
        direction="forward",
        allow_exact_matches=True,
    )
    return img_df


def get_child_dataset(flz_session, parent_name, dataset_type):
    """
    Get the last dataset of a given type for a given parent entity.

    Args:
        flz_session (flexilims_session): flexilims session
        parent_name (str): name of the parent entity
        dataset_type (str): type of the dataset

    Returns:
        Dataset: the last dataset of the given type for the given parent entity

    """
    all_children = flz.get_children(
        parent_name=parent_name,
        children_datatype="dataset",
        flexilims_session=flz_session,
    )
    selected_datasets = all_children[all_children["dataset_type"] == dataset_type]
    if len(selected_datasets) == 0:
        raise ValueError(f"No {dataset_type} dataset found for session {parent_name}")
    elif len(selected_datasets) > 1:
        print(
            f"{len(selected_datasets)} {dataset_type} datasets found for session {parent_name}"
        )
        print("Will return the last one...")
    return flz.Dataset.from_dataseries(
        selected_datasets.iloc[-1], flexilims_session=flz_session
    )


def load_imaging_data(recording, flexilims_session):
    suite2p_traces = get_child_dataset(flexilims_session, recording, "suite2p_traces")
    dffs = []
    for iplane in range(int(float(suite2p_traces.extra_attributes["nplanes"]))):
        plane_path = suite2p_traces.path_full / f"plane{iplane}"
        dffs.append(np.load(plane_path / "dff_ast.npy"))
    return np.concatenate(dffs, axis=0).T


def fill_in_missing_index(df, value_col):
    # suggestion to use `fill_missing_imaging_volumes` instead?

    # Fill in the value of the missing imaging frame from vs_df, if the monitor frames drop for more than one imaging frame
    # reindex dataframe to create continuous index
    new_index = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + 1)
    df = df.reindex(new_index)

    # insert NaN rows where the original index was missing
    df.loc[df[value_col].isna(), :] = np.nan

    # fill the NaN row with the previous valid value
    df[value_col] = df[value_col].fillna(method="ffill")

    return df
