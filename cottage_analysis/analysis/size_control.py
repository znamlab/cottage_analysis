import numpy as np
import pandas as pd
from tqdm import tqdm
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.imaging.common.find_frames import find_imaging_frames
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers
from cottage_analysis.analysis import spheres, find_depth_neurons
import flexiznam as flz
from pathlib import Path


def format_imaging_df(recording, imaging_df, original_size=0.087):
    """Format sphere params in imaging_df.

    Args:
        recording (Series): recording entry returned by flexiznam.get_entity(name=recording_name, project_id=project).
        imaging_df (pd.DataFrame): dataframe that contains info for each monitor frame.
        original_size (float, optional): original size of the sphere at 10 degrees. Defaults to 0.087.

    Returns:
        DataFrame: contains information for each monitor frame and vis-stim.

    """
    if "Radius" in imaging_df.columns:
        imaging_df = imaging_df.rename(columns={"Radius": "depth"})
    elif "Depth" in imaging_df.columns:
        imaging_df = imaging_df.rename(columns={"Depth": "depth"})
    imaging_df["size"] = imaging_df.OriginalSize / original_size * 10
    # Indicate whether it's a closed loop or open loop session
    if "Playback" in recording.name:
        imaging_df["closed_loop"] = 0
    else:
        imaging_df["closed_loop"] = 1
    imaging_df["RS"] = (
        imaging_df.mouse_z_harp.diff() / imaging_df.mouse_z_harptime.diff()
    )
    # average RS eye for each imaging volume
    imaging_df["RS_eye"] = imaging_df.eye_z.diff() / imaging_df.monitor_harptime.diff()
    # depth for each imaging volume
    imaging_df[imaging_df["depth"] == -9999].depth = np.nan
    imaging_df.depth = imaging_df.depth / 100  # convert cm to m
    # OF for each imaging volume
    imaging_df["OF"] = imaging_df.RS_eye / imaging_df.depth
    return imaging_df


def generate_trials_df(recording, imaging_df):
    trials_df = spheres.generate_trials_df(recording, imaging_df)

    imaging_df["stim"] = np.nan
    imaging_df.loc[imaging_df.depth.notnull(), "stim"] = 1
    imaging_df.loc[imaging_df.depth < 0, "stim"] = 0
    imaging_df_simple = imaging_df[
        (imaging_df["stim"].diff() != 0) & (imaging_df["stim"]).notnull()
    ]
    start_volume_stim = imaging_df_simple[
        (imaging_df_simple["stim"] == 1)
    ].imaging_frame.values
    trials_df["size"] = pd.Series(imaging_df.loc[start_volume_stim]["size"].values)

    return trials_df


def sync_all_recordings(
    session_name,
    flexilims_session=None,
    project=None,
    filter_datasets=None,
    recording_type="two_photon",
    protocol_base="SizeControl",
    photodiode_protocol=5,
    return_volumes=True,
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

    for i, recording_name in enumerate(recordings.name):
        recording = flz.get_entity(
            datatype="recording",
            name=recording_name,
            flexilims_session=flexilims_session,
        )

        print(f"Processing recording {i+1}/{len(recordings)}")
        vs_df = synchronisation.generate_vs_df(
            recording=recording,
            photodiode_protocol=photodiode_protocol,
            flexilims_session=flexilims_session,
            project=project,
        )

        imaging_df = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=recording,
            flexilims_session=flexilims_session,
            filter_datasets=filter_datasets,
            return_volumes=return_volumes,
        )

        imaging_df = format_imaging_df(recording=recording, imaging_df=imaging_df)

        trials_df = generate_trials_df(recording=recording, imaging_df=imaging_df)

        if i == 0:
            vs_df_all = vs_df
            trials_df_all = trials_df
        else:
            vs_df_all = pd.concat([vs_df_all, vs_df], ignore_index=True)
            trials_df_all = pd.concat([trials_df_all, trials_df], ignore_index=True)
    print(f"Finished concatenating vs_df and trials_df")

    return vs_df_all, trials_df_all


def find_best_size(
    trials_df,
    neurons_df,
    neurons_ds,
    rs_thr=None,
    rs_thr_max=None,
    is_closedloop=1,
    still_only=False,
    still_time=0,
    frame_rate=15,
):
    """Find the size with max mean response for each neuron.

    Args:
        trials_df (pd.DataFrame): trials dataframe.
        neurons_df (pd.DataFrame): neurons dataframe.
        neurons_ds (neurons_ds): neurons dataset.
        rs_thr (float, optional): minimum running speed threshold. Defaults to None.
        rs_thr_max (float, optional): maximum running speed threshold. Defaults to None.
        is_closedloop (int, optional): 1 for closed loop, 0 for open loop. Defaults to 1.
        still_only (bool, optional): if True, only use still trials. Defaults to False.
        still_time (int, optional): still time in seconds. Defaults to 0.
        frame_rate (int, optional): frame rate. Defaults to 15.
    """
    trials_df = trials_df[trials_df.closed_loop == is_closedloop]
    trials_df = get_physical_size(trials_df, use_cols=["size", "depth"], k=1)
    size_list = np.sort(trials_df["physical_size"].unique())
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
        trials_df=trials_df,
        rs_thr=rs_thr,
        rs_thr_max=rs_thr_max,
        still_only=still_only,
        still_time=still_time,
        frame_rate=frame_rate,
        closed_loop=is_closedloop,
        param="size",
    )
    for roi in np.arange(len(neurons_df)):
        neurons_df.loc[roi, "best_size"] = size_list[
            np.argmax(np.nanmean(mean_dff_arr[:, :, roi], axis=1))
        ]

    return neurons_df, neurons_ds


def get_physical_size(trials_df, use_cols=["size", "depth"], k=1):
    """Get physical size of the stimulus.

    Args:
        trials_df (pd.DataFrame): trials dataframe.
        use_cols (list): list of columns to use. (["size", "depth"])
        k (int, optional): scaling factor. Defaults to 1.

    Returns:
        pd.DataFrame: trials dataframe with physical size column.
    """
    trials_df["physical_size"] = trials_df[use_cols[0]] * trials_df[use_cols[1]] * k
    return trials_df
