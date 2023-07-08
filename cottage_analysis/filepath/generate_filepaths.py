#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 14:08:41 2021

@author: hey2

Generate filepaths for importing data files or saving analysis results 
"""
import flexiznam as flz
from pathlib import Path
from warnings import warn
from flexiznam.schema import Dataset


def get_session_children(project, mouse, session, flexilims_session=None):
    """Get children of a session

    Args:
        project (str): Name of the project
        mouse (str): Name of the mouse
        session (str): Name of the session
        flexilims_session (flexilims.Session, optional): flexilims session to interact
            with database. If None, will create one. Defaults to None.

    Raises:
        IOError: If the session cannot be found online

    Returns:
        pandas.DataFrame: Dataframe of children of the session
    """
    if flexilims_session is None:
        warn(
            "flexilims_session will become mandatory", DeprecationWarning, stacklevel=2
        )
        flexilims_session = flz.get_flexilims_session(project_id=project)
    this_sess = flz.get_entity(
        datatype="session",
        name=f"{mouse}_{session}",
        flexilims_session=flexilims_session,
    )
    if this_sess is None:
        raise IOError("Could not find session")
    sess_children = flz.get_children(
        this_sess.id, project_id=project, flexilims_session=flexilims_session
    )

    return sess_children


def get_all_recording_entries(
    project, mouse, session, protocol, flexilims_session=None
):
    """
    Get all flexilims entries (children) of a recording
    :param str project:
    :param str mouse:
    :param str session:
    :param str protocol:
    :return:
    """
    if flexilims_session is None:
        warn(
            "flexilims_session will become mandatory", DeprecationWarning, stacklevel=2
        )
        flexilims_session = flz.get_flexilims_session(project_id=project)

    sess_children = get_session_children(project, mouse, session, flexilims_session)

    all_protocol_recording_entries = sess_children[
        sess_children["name"].str[-len(protocol) :].str.contains(protocol)
    ]

    return all_protocol_recording_entries


def get_recording_entries(
    project,
    mouse,
    session,
    protocol,
    all_protocol_recording_entries=None,
    recording_no=0,
    flexilims_session=None,
):
    """
    Get all flexilims entries (children) of a recording
    :param str project:
    :param str mouse:
    :param str session:
    :param str protocol:
    :return:
    """

    if flexilims_session is None:
        warn(
            "flexilims_session will become mandatory", DeprecationWarning, stacklevel=2
        )
        flexilims_session = flz.get_flexilims_session(project_id=project)

    if all_protocol_recording_entries is None:
        warn(
            f"all_protocol_recording_entries not given, assume only one recording for protocol {protocol} in this session.\n \
            The last recording that matches this protocoll will be returned."
        )
        sess_children = get_session_children(project, mouse, session, flexilims_session)
        # protocol_recording = sess_children.loc[sess_children['name'].str.contains(protocol, case=False)]
        for i in range(len(sess_children)):
            name = sess_children.iloc[i].name
            if name[-len(protocol) :] == protocol:
                protocol_recording = sess_children.iloc[i]
    else:
        protocol_recording = all_protocol_recording_entries.iloc[recording_no]
    recording_entries = flz.get_children(
        protocol_recording.id, project_id=project, flexilims_session=flexilims_session
    )
    recording_path = protocol_recording.path

    return recording_entries, recording_path


def generate_file_folders(
    project,
    mouse,
    session,
    protocol,
    rawdata_root=None,
    root=None,
    all_protocol_recording_entries=None,
    recording_no=0,
    flexilims_session=None,
    filter_attribute="anatomical_only",
    filter_value=3,
):
    """Generate folders for raw data, preprocessed data and analyzed data


    Args:
        project (str): Name of the project
        mouse (str): Name of the mouse
        session (str): Name of the session
        protocol (str): Type of protocol
        rawdata_root (pathlib.Path, optional): Path to raw data. Defaults to None.
        root (pathlib.Path, optional): Path to processed data. Defaults to None.
        flexilims_session (flexilims.Session, optional): Flexilims session to interact
            with database. Defaults to None.
        filter_attribute (str): Dataset attribute to be filtered. Defaults to 'anatomical_only'.
        filter_value (int or str): Dataset attribute value to be filtered. Defaults to 3.

    Returns:
        rawdata_root (pathlib.Path):
        protocol_folder (pathlib.Path):
        analysis_folder (pathlib.Path):
        suite2p_folder (pathlib.Path):
        trace_folder (pathlib.Path):
    """
    if rawdata_root is None:
        rawdata_root = Path(flz.PARAMETERS["data_root"]["raw"])
    else:
        warn(
            "rawdata_root will be read from flexiznam config. Remove parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        rawdata_root = Path(rawdata_root)
    if root is None:
        root = Path(flz.PARAMETERS["data_root"]["processed"])
    else:
        warn(
            "root will be read from flexiznam config. Remove parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        root = Path(root)

    sess_children = get_session_children(
        project, mouse, session, flexilims_session=flexilims_session
    )

    # find recording paths
    recording_entries, recording_path = get_recording_entries(
        project,
        mouse,
        session,
        protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=recording_no,
        flexilims_session=flexilims_session,
    )

    rawdata_root = rawdata_root / recording_path
    # preprocess_folder = root + protocol_path
    protocol_folder = root / recording_path
    analysis_folder = root / project / "Analysis" / recording_path[len(project) + 1 :]

    suite2p_ds = sess_children[sess_children.dataset_type == "suite2p_rois"]
    suite2p_ds = suite2p_ds[suite2p_ds[filter_attribute] == filter_value]
    if len(suite2p_ds) != 1:
        print(
            f"WARNING: {len(suite2p_ds)} suite2p folders detected. Return the first path found."
        )
    suite2p_ds = suite2p_ds.iloc[0]
    suite2p_folder = root / suite2p_ds.path / "plane0"

    trace_ds = recording_entries[recording_entries.dataset_type == "suite2p_traces"]
    trace_ds = trace_ds[trace_ds[filter_attribute] == filter_value]
    if len(trace_ds) != 1:
        print(
            f"WARNING: {len(suite2p_ds)} traces detected. Return the first path found."
        )
    trace_ds = trace_ds.iloc[0]
    trace_folder = root / trace_ds.path

    return (
        rawdata_root,
        protocol_folder,
        analysis_folder,
        suite2p_folder,
        trace_folder,
    )


def generate_logger_path(
    project,
    mouse,
    session,
    protocol,
    logger_name,
    rawdata_root=None,
    root=None,
    all_protocol_recording_entries=None,
    recording_no=None,
    flexilims_session=None,
):
    """
    Generate paths for param loggers
    :param str project:
    :param str mouse:
    :param str session:
    :param str protocol:
    :param str rawdata_root:
    :param str root:
    :param str logger_name:
    :return:
    """

    if rawdata_root is None:
        rawdata_root = Path(flz.PARAMETERS["data_root"]["raw"])
    else:
        warn(
            "rawdata_root will be read from flexiznam config. Remove parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        rawdata_root = Path(rawdata_root)
    if root is None:
        root = Path(flz.PARAMETERS["data_root"]["processed"])
    else:
        warn(
            "root will be read from flexiznam config. Remove parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        root = Path(root)

    if flexilims_session is None:
        warn(
            "flexilims_session will become mandatory", DeprecationWarning, stacklevel=2
        )
        assert project is not None
        flexilims_session = flz.get_flexilims_session(project_id=project)
    elif project is not None:
        warn("Project will be read from flexilims_session. Ignore project argument")

    recording_entries, recording_path = get_recording_entries(
        project,
        mouse,
        session,
        protocol,
        all_protocol_recording_entries=all_protocol_recording_entries,
        recording_no=recording_no,
        flexilims_session=flexilims_session,
    )

    # Find logger entries
    harp = recording_entries[recording_entries.dataset_type == "harp"]
    if logger_name == "harp_message":
        logger_name = flz.get_entity(id=harp.id, flexilims_session=flexilims_session)[
            "binary_file"
        ].replace("bin", "csv")
    else:
        logger_name = flz.get_entity(id=harp.id, flexilims_session=flexilims_session)[
            "csv_files"
        ][logger_name]

    logger_path = Path(rawdata_root) / recording_path / logger_name

    return logger_path


def generate_analysis_session_folder(
    project, mouse, session, root=None, flexilims_session=None
):
    if flexilims_session is None:
        warn(
            "flexilims_session will become mandatory", DeprecationWarning, stacklevel=2
        )
        flexilims_session = flz.get_flexilims_session(project_id=project)

    if root is None:
        root = Path(flz.PARAMETERS["data_root"]["processed"])
    else:
        warn(
            "root will be read from flexiznam config. Remove parameter",
            DeprecationWarning,
            stacklevel=2,
        )
        root = Path(root)

    project_sess = flz.get_experimental_sessions(
        project_id=project, flexilims_session=flexilims_session
    )
    this_sess = project_sess[project_sess.name == mouse + "_" + session]
    sess_path = str(this_sess.path.values[0])
    analysis_sess_folder = (
        Path(root) / project / "Analysis" / sess_path[len(project) + 1 :]
    )

    return analysis_sess_folder
