import functools

print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

import flexiznam as flz
from cottage_analysis.io_module import harp
from cottage_analysis.preprocessing import find_frames
from cottage_analysis.depth_analysis.filepath import generate_filepaths
from cottage_analysis.imaging.common import find_frames as find_img_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers


def load_harpmessage(project, mouse, session, protocol, irecording=0, redo=False):
    """Save harpmessage into a npz file, or load existing npz file. Then load harpmessage file as a np arrray.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        redo (bool, optional): re-transform harp bin file into npz file or not. Defaults to False.

    Returns:
        np.array: loaded harpmessages as numpy array
    """
    flexilims_session = flz.get_flexilims_session(project_id=project)
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        flexilims_session=flexilims_session,
    )
    # Find harpmessage path
    harpmessage_file = generate_filepaths.generate_logger_path(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=irecording,
        flexilims_session=flexilims_session,
        logger_name="harp_message",
    )

    (
        rawdata_folder,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    ) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=irecording,
        flexilims_session=flexilims_session,
    )

    # save harp message into npz, or load existing npz file
    msg = Path(str(harpmessage_file).replace("csv", "bin"))
    p_msg = protocol_folder / "sync"
    if not os.path.exists(p_msg):
        os.makedirs(p_msg)
    p_msg = p_msg / (msg.stem + ".npz")
    if (not p_msg.is_file()) or redo == True:
        print("Saving harp messages into npz...")
        harp_messages = harp.load_harp(
            msg,
            di_names=("frame_triggers", "lick_detection", "di2_encoder_initial_state"),
        )
        p_msg.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p_msg, **harp_messages)
        print("Harp messages saved.")
    elif (p_msg.is_file()) and (redo == False):
        print("harpmessage.npz already exists. Loading harpmessage.npz...")
    harp_messages = np.load(p_msg)
    print("harpmessage loaded.")

    return harp_messages


def find_monitor_frames(
    project, mouse, session, protocol, irecording=0, redo=True, redo_harpnpz=False
):
    """Synchronise monitor frame using the find_frames.sync_by_correlation, and save them into monitor_frames_df.pickle and monitor_db_dict.pickle under the path {trace_folder/'sync/monitor_frames/'}

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        redo (bool, optional): re-sync the monitor frames or not. Defaults to True.
        redo_harpnpz (bool, optional): re-transform harp bin file into npz file or not. Defaults to False.
    """
    # Find paths
    flexilims_session = flz.get_flexilims_session(project_id=project)
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        flexilims_session=flexilims_session,
    )
    (
        rawdata_folder,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    ) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=irecording,
        flexilims_session=flexilims_session,
    )

    if redo:
        # Load files
        harp_messages = load_harpmessage(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            all_protocol_recording_entries=all_protocol_recording_entries,
            irecording=irecording,
            flexilims_session=flexilims_session,
            redo=redo_harpnpz,
        )

        frame_log = pd.read_csv(rawdata_folder / "FrameLog.csv")
        expected_sequence = (
            pd.read_csv(
                rawdata_folder / "random_sequence_5values_alternate.csv", header=None
            )
            .loc[:, 0]
            .values
        )
        step_values = frame_log.PhotoQuadColor.unique()
        ao_time = harp_messages["analog_time"]
        photodiode = harp_messages["photodiode"]
        ao_sampling = 1 / np.mean(np.diff(ao_time))

        print("Data loaded.")
        print(
            "Recording is %d s long."
            % (frame_log.HarpTime.values[-1] - frame_log.HarpTime.values[0])
        )

        # Synchronisation
        frame_rate = 144
        frames_df, db_dict = find_frames.sync_by_correlation(
            frame_log,
            ao_time,
            photodiode,
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
            debug=True,
        )

        # Save monitor frame dataframes
        save_folder = protocol_folder / "sync/monitor_frames"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        frames_df.to_pickle(save_folder / "monitor_frames_df.pickle")
        with open(save_folder / "monitor_db_dict.pickle", "wb") as handle:
            pickle.dump(db_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_vs_df(project, mouse, session, protocol, irecording=0):
    """Generate a dataframe that contains information for each monitor frame. This requires monitor frames to be synced first.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.

    Returns:
        DataFrame: contains information for each monitor frame.
    """
    flexilims_session = flz.get_flexilims_session(project_id=project)
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        flexilims_session=flexilims_session,
    )
    sess_children = generate_filepaths.get_session_children(
        project=project,
        mouse=mouse,
        session=session,
        flexilims_session=flexilims_session,
    )
    sess_children_protocols = sess_children[
        sess_children["name"].str.contains("(SpheresPermTubeReward|Fourier|Retinotopy)")
    ]
    folder_no = sess_children_protocols.index.get_loc(
        sess_children_protocols[
            sess_children_protocols.id
            == all_protocol_recording_entries.iloc[irecording].id
        ].index[0]
    )
    (
        rawdata_folder,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    ) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=0,
    )
    save_folder = protocol_folder / "sync/monitor_frames/"
    with open(save_folder / "monitor_db_dict.pickle", "rb") as handle:
        monitor_db_dict = pickle.load(handle)
    with open(save_folder / "monitor_frames_df.pickle", "rb") as handle:
        monitor_frames_df = pickle.load(handle)

    # Find frames that are not skipped
    monitor_frame_valid = monitor_frames_df[monitor_frames_df.closest_frame.notnull()][
        ["closest_frame", "onset_time", "offset_time", "peak_time"]
    ]
    monitor_frame_valid["closest_frame"] = monitor_frame_valid["closest_frame"].astype(
        "int"
    )
    monitor_frame_valid = monitor_frame_valid.sort_values("closest_frame")

    # Merge MouseZ and EyeZ from FrameLog.csv to frame_df according to FrameIndex
    frame_log = pd.read_csv(rawdata_folder / "FrameLog.csv")
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

    # Align sphere parameter with the frame (harptime later than the logged sphere time)
    vs_df = vs_df.sort_values("onset_time")
    param_log = pd.read_csv(rawdata_folder / "NewParams.csv")
    param_log_simple = param_log[["HarpTime", "Radius"]]
    param_log_simple = param_log_simple.rename(
        columns={"HarpTime": "harptime_sphere", "Radius": "depth"}
    )
    param_log_simple["onset_time"] = param_log_simple["harptime_sphere"]
    vs_df = pd.merge_asof(
        left=vs_df,
        right=param_log_simple,
        on="onset_time",
        direction="backward",
        allow_exact_matches=False,
    )  # Does not allow exact match of sphere rendering time and frame onset time?
    vs_df["depth"] = vs_df["depth"] / 100  # convert cm to m

    # Align imaging frame time with monitor frame onset time (imaging frame time later than monitor frame onset time)
    save_folder = protocol_folder / "sync"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ops = np.load(suite2p_folder / "ops.npy", allow_pickle=True)
    ops = ops.item()
    harpmessage_file = generate_filepaths.generate_logger_path(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=irecording,
        flexilims_session=flexilims_session,
        logger_name="harp_message",
    )
    p_msg = protocol_folder / "sync/harpmessage.npz"
    img_frame_logger = format_loggers.format_img_frame_logger(
        harpmessage_file=p_msg, register_address=32
    )
    frame_number = ops["frames_per_folder"][folder_no]
    img_frame_logger = find_img_frames.find_imaging_frames(
        harp_message=img_frame_logger,
        frame_number=frame_number,
        exposure_time=0.0324 * 2,
        register_address=32,
        exposure_time_tolerance=0.001,
    )

    img_frame_logger = img_frame_logger[["HarpTime", "ImagingFrame"]]
    img_frame_logger.to_pickle(save_folder / "img_frame_logger.pickle")
    img_frame_logger = img_frame_logger.rename(
        columns={
            "HarpTime": "harptime_imaging_trigger",
            "ImagingFrame": "imaging_frame",
        }
    )
    img_frame_logger["onset_time"] = img_frame_logger["harptime_imaging_trigger"]
    vs_df = pd.merge_asof(
        left=vs_df,
        right=img_frame_logger,
        on="onset_time",
        direction="forward",
        allow_exact_matches=True,
    )

    # Indicate whether it's a closed loop or open loop session
    if "Playback" in protocol:
        vs_df["closed_loop"] = 0
    else:
        vs_df["closed_loop"] = 1

    # Rename
    vs_df = vs_df.rename(columns={"closest_frame": "monitor_frame"})
    vs_df = vs_df.drop(
        columns=["harptime_framelog", "harptime_sphere", "harptime_imaging_trigger"]
    )
    vs_df = vs_df.rename(
        columns={
            "onset_time": "onset_harptime",
            "offset_time": "offset_harptime",
            "peak_time": "peak_harptime",
        }
    )

    # Save df to pickle
    vs_df.to_pickle(save_folder / "vs_df.pickle")
    return vs_df


def fill_in_missing_index(df, value_col):
    # Fill in the value of the missing imaging frame from vs_df, if the monitor frames drop for more than one imaging frame
    # reindex dataframe to create continuous index
    new_index = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + 1)
    df = df.reindex(new_index)

    # insert NaN rows where the original index was missing
    df.loc[df[value_col].isna(), :] = np.nan

    # fill the NaN row with the previous valid value
    df[value_col] = df[value_col].fillna(method="ffill")

    return df


def generate_imaging_df(project, mouse, session, protocol, vs_df, irecording=0):
    flexilims_session = flz.get_flexilims_session(project_id=project)
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        flexilims_session=flexilims_session,
    )
    sess_children = generate_filepaths.get_session_children(
        project=project,
        mouse=mouse,
        session=session,
        flexilims_session=flexilims_session,
    )
    sess_children_protocols = sess_children[
        sess_children["name"].str.contains("(SpheresPermTubeReward|Fourier|Retinotopy)")
    ]
    folder_no = sess_children_protocols.index.get_loc(
        sess_children_protocols[
            sess_children_protocols.id
            == all_protocol_recording_entries.iloc[irecording].id
        ].index[0]
    )
    (
        rawdata_folder,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    ) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=0,
    )
    save_folder = protocol_folder / "sync"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Imaging_df: to find the RS/OF array for each imaging frame
    imaging_df = pd.DataFrame(
        columns=[
            "imaging_frame",
            "harptime_imaging_trigger",
            "depth",
            # "trial_no",
            "is_stim",
            "RS",  # actual running speed, m/s
            "RS_eye",  # virtual running speed, m/s
            "OF",  # optic flow speed = RS/depth, rad/s
            "dffs",
            "closed_loop",
        ]
    )

    # Imaging frame number
    grouped_vs_df = vs_df.groupby("imaging_frame")
    ops = np.load(suite2p_folder / "ops.npy", allow_pickle=True)
    ops = ops.item()
    frame_number = ops["frames_per_folder"][folder_no]
    max_frame_in_vs_df = np.nanmax(vs_df.imaging_frame)
    if frame_number != max_frame_in_vs_df + 1:
        print(
            f"WARNING: Last {(frame_number-1-max_frame_in_vs_df)} imaging frames might be dropped. Check vs_df!"
        )
    imaging_df.imaging_frame = np.arange(frame_number)

    # dffs for each imaging frame: ncells x 1 frame
    dffs = np.load(trace_folder / "dffs_ast.npy")
    imaging_df.dffs = dffs.T.tolist()

    # RS for each imaging frame: the speed of the previous imaging frame
    rs_img = (
        grouped_vs_df.apply(
            lambda x: (x["mouse_z"].iloc[-1] - x["mouse_z"].iloc[0])
            / (x["onset_harptime"].iloc[-1] - x["onset_harptime"].iloc[0])
        )
        .to_frame()
        .rename(columns={0: "RS"})
    )
    rs_img = fill_in_missing_index(rs_img, value_col="RS")
    rs_img = rs_img.RS.values
    rs_img = np.insert(rs_img, 0, 0)
    rs_img = rs_img[:-1]
    imaging_df.RS = rs_img

    # RS_eye for each imaging frame: the eye speed of the previous imaging frame
    rs_eye_img = (
        grouped_vs_df.apply(
            lambda x: (x["eye_z"].iloc[-1] - x["eye_z"].iloc[0])
            / (x["onset_harptime"].iloc[-1] - x["onset_harptime"].iloc[0])
        )
        .to_frame()
        .rename(columns={0: "RS_eye"})
    )
    rs_eye_img = fill_in_missing_index(rs_eye_img, value_col="RS_eye")
    rs_eye_img = rs_eye_img.RS_eye.values
    rs_eye_img = np.insert(rs_eye_img, 0, 0)
    rs_eye_img = rs_eye_img[:-1]
    imaging_df.RS_eye = rs_eye_img

    # depth for each imaging frame
    depth_img = grouped_vs_df.depth.min().to_frame().rename(columns={0: "depth"})
    depth_img = fill_in_missing_index(depth_img, value_col="depth")
    imaging_df.depth = depth_img
    imaging_df.is_stim = imaging_df.apply(lambda x: int(x.depth > 0), axis=1)
    imaging_df.depth[imaging_df.depth.isna()] = 0
    imaging_df.depth[imaging_df.depth < 0] = np.nan
    imaging_df.depth = imaging_df.depth.fillna(method="ffill")
    imaging_df.depth[imaging_df.depth == 0] = np.nan

    # OF for each imaging frame
    imaging_df.OF = imaging_df.RS_eye / imaging_df.depth
    imaging_df.OF[imaging_df.is_stim == 0] = np.nan

    # closed loop status for each imaging frame
    if "Playback" in protocol:
        imaging_df.closed_loop = 0
    else:
        imaging_df.closed_loop = 1

    # Find imaging frame trigger time
    if Path(save_folder / "img_frame_logger.pickle").is_file():
        with open(save_folder / "img_frame_logger.pickle", "rb") as handle:
            img_frame_logger = pickle.load(handle)
    else:
        p_msg = protocol_folder / "sync/harpmessage.npz"
        img_frame_logger = format_loggers.format_img_frame_logger(
            harpmessage_file=p_msg, register_address=32
        )
        img_frame_logger = find_img_frames.find_imaging_frames(
            harp_message=img_frame_logger,
            frame_number=frame_number,
            exposure_time=0.0324 * 2,
            register_address=32,
            exposure_time_tolerance=0.001,
        )
    imaging_df.harptime_imaging_trigger = img_frame_logger.HarpTime.values

    # Save df to pickle
    vs_df.to_pickle(save_folder / "imaging_df.pickle")

    return imaging_df


def generate_trials_df(project, mouse, session, protocol, vs_df, irecording=0):
    """Generate a dataframe that contains information for each trial. This requires monitor frames to be synced and vs_df to be generated first.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.
        vs_df (DataFrame): contains information for each monitor frame.

    Returns:
        DataFrame: contains information for each trial.
    """
    imaging_df = generate_imaging_df(
        project, mouse, session, protocol, vs_df, irecording
    )
    flexilims_session = flz.get_flexilims_session(project_id=project)
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        flexilims_session=flexilims_session,
    )
    sess_children = generate_filepaths.get_session_children(
        project=project,
        mouse=mouse,
        session=session,
        flexilims_session=flexilims_session,
    )
    sess_children_protocols = sess_children[
        sess_children["name"].str.contains("(SpheresPermTubeReward|Fourier|Retinotopy)")
    ]
    folder_no = sess_children_protocols.index.get_loc(
        sess_children_protocols[
            sess_children_protocols.id
            == all_protocol_recording_entries.iloc[irecording].id
        ].index[0]
    )
    (
        rawdata_folder,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    ) = generate_filepaths.generate_file_folders(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=0,
    )

    # trials_df
    trials_df = pd.DataFrame(
        columns=[
            "trial_no",
            "depth",
            "harptime_stim_start",
            "harptime_stim_stop",
            "harptime_blank_start",
            "harptime_blank_stop",
            "imaging_frame_stim_start",
            "imaging_frame_stim_stop",
            "imaging_frame_blank_start",
            "imaging_frame_blank_stop",
            "RS_stim",  # actual running speed, m/s
            "RS_blank",
            "RS_eye_stim",  # virtual running speed, m/s
            "OF_stim",  # optic flow speed = RS/depth, rad/s
            "dff_stim",
            "dff_blank",
            # "spheres_no",
            "closed_loop",
        ]
    )

    # Find the start and stop of each trial
    blank_time = 10  # s

    vs_df["stim"] = np.nan
    vs_df.loc[vs_df.depth.notnull(), "stim"] = 1
    vs_df.loc[vs_df.depth < 0, "stim"] = 0
    vs_df_simple = vs_df[(vs_df["stim"].diff() != 0) & (vs_df["stim"].notnull())]
    vs_df_simple.depth = np.round(vs_df_simple.depth, 2)

    start_idx_stim = vs_df_simple[(vs_df_simple["stim"] == 1)].index
    start_idx_blank = vs_df_simple[(vs_df_simple["stim"] == 0)].index
    if len(start_idx_stim) != len(start_idx_blank):
        if (len(start_idx_stim) - len(start_idx_blank)) == 1:
            stop_idx_blank = start_idx_stim[1:] - 1
            start_idx_stim = start_idx_stim[: len(start_idx_blank)]
        else:
            print("Warning: incorrect stimulus trial structure! Double check!")
    else:
        stop_idx_blank = start_idx_stim[1:] - 1
        last_blank_stop_time = (
            vs_df.loc[start_idx_blank[-1]].onset_harptime + blank_time
        )
        stop_idx_blank = np.append(
            stop_idx_blank,
            (np.abs(vs_df["onset_harptime"] - last_blank_stop_time)).idxmin(),
        )
    stop_idx_stim = start_idx_blank - 1

    # Assign trial no, depth, start/stop time, start/stop imaging frame to trials_df
    # Harptime for starts and stops are harptime for monitor frames, not corresponding to imaging trigger harptime
    trials_df.trial_no = np.arange(len(start_idx_stim))
    trials_df.depth = vs_df.loc[start_idx_stim].depth.values
    trials_df.harptime_stim_start = vs_df.loc[start_idx_stim].onset_harptime.values
    trials_df.harptime_stim_stop = vs_df.loc[stop_idx_stim].onset_harptime.values
    trials_df.harptime_blank_start = vs_df.loc[start_idx_blank].onset_harptime.values
    trials_df.harptime_blank_stop = vs_df.loc[stop_idx_blank].onset_harptime.values
    trials_df.imaging_frame_stim_start = vs_df.loc[start_idx_stim].imaging_frame.values
    trials_df.imaging_frame_blank_start = vs_df.loc[
        start_idx_blank
    ].imaging_frame.values
    trials_df.imaging_frame_blank_stop = vs_df.loc[stop_idx_blank].imaging_frame.values
    trials_df.imaging_frame_stim_stop = trials_df.imaging_frame_blank_start - 1

    mask = (
        trials_df.imaging_frame_stim_start
        == trials_df.imaging_frame_blank_stop.shift(1)
    )  # Get rid of the overlap of imaging frame no. between different trials
    trials_df.loc[mask, "imaging_frame_stim_start"] += 1

    if "Playback" in protocol:
        trials_df.closed_loop = 0
    else:
        trials_df.closed_loop = 1

    # Assign RS array from imaging_df back to trials_df
    trials_df.RS_stim = trials_df.apply(
        lambda x: imaging_df.RS.loc[
            x.imaging_frame_stim_start : x.imaging_frame_stim_stop
        ].values,
        axis=1,
    )

    trials_df.RS_blank = trials_df.apply(
        lambda x: imaging_df.RS.loc[
            x.imaging_frame_blank_start : x.imaging_frame_blank_stop
        ].values,
        axis=1,
    )

    trials_df.RS_eye_stim = trials_df.apply(
        lambda x: imaging_df.RS_eye.loc[
            x.imaging_frame_stim_start : x.imaging_frame_stim_stop
        ].values,
        axis=1,
    )

    # trials_df.OF_stim = trials_df.apply(
    #         lambda x: x["RS_eye_stim"] / x["depth"], axis=1
    #     )
    trials_df.OF_stim = trials_df.apply(
        lambda x: imaging_df.OF.loc[
            x.imaging_frame_stim_start : x.imaging_frame_stim_stop
        ].values,
        axis=1,
    )

    # Assign dffs array to trials_df
    dffs = np.load(trace_folder / "dffs_ast.npy")
    trials_df.dff_stim = trials_df.apply(
        lambda x: dffs[
            :, int(x.imaging_frame_stim_start) : int(x.imaging_frame_stim_stop)
        ],
        axis=1,
    )

    trials_df.dff_blank = trials_df.apply(
        lambda x: dffs[
            :, int(x.imaging_frame_blank_start) : int(x.imaging_frame_blank_stop)
        ],
        axis=1,
    )

    # Rename
    trials_df = trials_df.drop(columns=["imaging_frame_blank_start"])

    # Save df to pickle
    save_folder = protocol_folder / "sync"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    trials_df.to_pickle(save_folder / "trials_df.pickle")

    return trials_df, imaging_df
