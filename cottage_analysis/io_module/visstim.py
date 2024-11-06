"""Utility functions to load visual stimulation data from the database."""

import pandas as pd

import flexiznam as flz

from cottage_analysis.utilities.misc import get_str_or_recording


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
    flexilims_session, harp_recording=None, vis_stim_recording=None, log_name=None
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
    else:
        param_log = pd.read_csv(
            vis_stim_ds.path_full / vis_stim_ds.extra_attributes["csv_files"][log_name]
        )
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

    if harp_recording is None:
        use_harp = False
    else:
        # harp exists
        if vis_stim_recording is None:
            use_harp = True
        elif vis_stim_recording.name == harp_recording.name:
            # harp is the same as vis_stim, so use harp
            use_harp = True
        else:
            use_harp = False

    if use_harp:  # use visual stimulation recording
        harp_recording = get_str_or_recording(
            harp_recording, flexilims_session=flexilims_session
        )
        harp_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=harp_recording.name,
            dataset_type="harp",
            allow_multiple=False,
            return_dataseries=False,
        )
        vis_stim_ds = harp_ds
    else:  # use harp recording, which should contain the visual stimulation info
        vis_stim_recording = get_str_or_recording(
            vis_stim_recording, flexilims_session=flexilims_session
        )
        vis_stim_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=vis_stim_recording.name,
            dataset_type="visstim",
            allow_multiple=False,
            return_dataseries=False,
        )
    return vis_stim_ds
