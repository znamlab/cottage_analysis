"""Utility functions to load visual stimulation data from the database."""

import pandas as pd
import numpy as np
import flexiznam as flz

from cottage_analysis.utilities.misc import get_str_or_recording
from cottage_analysis.io_module.harp import get_harp_dataset


def get_frame_log(flexilims_session, harp_recording=None, vis_stim_recording=None):
    """Get frame log from visual stimulation recording.

    This will load the frame log using harp_recording or vis_stim_recording depending on
    which one is provided.

    Args:
        flexilims_session (flexilims.Flexilims, optional): Flexilims session.
        harp_recording (str or pandas.Series, optional): HARP recording. Defaults to None.
        vis_stim_recording (str or pandas.Series, optional): Visual stimulation recording. Defaults to None.

    Returns:
        pandas.DataFrame: frame log.
    """
    vis_stim_ds = get_visstim_ds(flexilims_session, harp_recording, vis_stim_recording)

    if type(vis_stim_ds.extra_attributes["csv_files"]) == str:
        # Some yaml info have been saved as string instead of dict
        # TODO: fix on flexilims and/or use yaml.safe_load
        frame_log = pd.read_csv(
            vis_stim_ds.path_full
            / eval(vis_stim_ds.extra_attributes["csv_files"])["FrameLog"]
        )
    else:
        frame_log = pd.read_csv(
            vis_stim_ds.path_full
            / vis_stim_ds.extra_attributes["csv_files"]["FrameLog"]
        )
    return frame_log


def get_param_log(
    flexilims_session,
    harp_recording=None,
    vis_stim_recording=None,
    log_name=None,
    multidepth=False,
):
    """Get param log from visual stimulation recording.

    This will load the frame log using harp_recording or vis_stim_recording depending on
    which one is provided.

    Args:
        flexilims_session (flexilims.Flexilims, optional): Flexilims session.
        harp_recording (str or pandas.Series, optional): HARP recording. Defaults to None.
        vis_stim_recording (str or pandas.Series, optional): Visual stimulation recording. Defaults to None.
        log_name (str, optional): Name of the log to load. If None, will load
            "ParamLog.csv" if it exists, "NewParams.csv" otherwise. Defaults to None.
        multidepth (bool, optional): Whether to load the multidepth log. Defaults to False.

    Returns:
        pandas.DataFrame: frame log.
    """
    vis_stim_ds = get_visstim_ds(flexilims_session, harp_recording, vis_stim_recording)

    if log_name is None:
        if "ParamLog" in vis_stim_ds.extra_attributes["csv_files"]:
            log_name = "ParamLog"
        else:
            log_name = "NewParams"

    if type(vis_stim_ds.extra_attributes["csv_files"]) == str:
        # Some yaml info have been saved as string instead of dict
        # TODO: fix on flexilims and/or use yaml.safe_load
        param_log = pd.read_csv(
            vis_stim_ds.path_full
            / eval(vis_stim_ds.extra_attributes["csv_files"])[log_name]
        )
        return param_log

    if not multidepth:
        param_log = pd.read_csv(
            vis_stim_ds.path_full / vis_stim_ds.extra_attributes["csv_files"][log_name]
        )
        return param_log

    # multidepth case
    csvs = vis_stim_ds.extra_attributes["csv_files"]
    dfs_by_depth = {}
    for csv_id, file_name in csvs.items():
        if not csv_id.startswith("NewParams"):
            continue
        depth = int(csv_id.split("_")[-1][:-2])
        df = pd.read_csv(vis_stim_ds.path_full / file_name)
        df["logger_fname"] = file_name
        dfs_by_depth[depth] = df
    param_log = pd.concat(dfs_by_depth.values(), ignore_index=True)
    param_log.sort_values(by="HarpTime", inplace=True)
    param_log.reset_index(drop=True, inplace=True)
    assert (
        param_log.Frameindex.diff().min() > -3
    ), "Frame index and harptime are not aligned"
    # make sure frame index is monotonically increasing
    param_log["Frameindex"] = np.maximum.accumulate(param_log.Frameindex.values)
    param_log["Frameindex"] = param_log.Frameindex.astype(int)

    return param_log


def get_visstim_ds(flexilims_session, harp_recording=None, vis_stim_recording=None):
    """Get visual stimulation dataset.

    This is either the visstim dataset or the harp dataset if the visstim dataset is not
    available.

    Args:
        flexilims_session (flexilims.Flexilims, optional): Flexilims session.
        harp_recording (str or pandas.Series, optional): HARP recording. Defaults to None.
        vis_stim_recording (str or pandas.Series, optional): Visual stimulation recording. Defaults to None.

    Returns:
        pandas.DataSeries: visual stimulation dataset.
    """

    if harp_recording is None and vis_stim_recording is None:
        raise ValueError("Provide at least one recording.")
    vis_stim_recording = get_str_or_recording(
        vis_stim_recording, flexilims_session=flexilims_session
    )
    harp_recording = get_str_or_recording(
        harp_recording, flexilims_session=flexilims_session
    )
    vis_stim_ds = None

    # If vis_stim_recording is provided, check if there is a vistim dataset
    if vis_stim_recording is not None:
        vis_stim_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=vis_stim_recording.name,
            dataset_type="visstim",
            allow_multiple=False,
            return_dataseries=False,
        )

    if harp_recording is None:
        assert vis_stim_ds is not None, "No visstim dataset found."
        return vis_stim_ds

    if (vis_stim_recording.name == harp_recording.name) and (vis_stim_ds is not None):
        # We have a recording that contains both a harp and a vistim ds, most likely new
        # onix recording. Return the vis_stim_ds
        return vis_stim_ds

    # No vis_stim recoridng, use harp recording
    harp_recording = get_str_or_recording(
        harp_recording, flexilims_session=flexilims_session
    )
    harp_ds = get_harp_dataset(flexilims_session, harp_recording.name)
    vis_stim_ds = harp_ds
    return vis_stim_ds
