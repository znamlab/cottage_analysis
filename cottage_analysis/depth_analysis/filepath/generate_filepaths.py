#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 14:08:41 2021

@author: hey2

Generate filepaths for importing data files or saving analysis results 
"""
import flexiznam as flz
from flexiznam.schema import Dataset


def get_session_children(project, mouse, session):
    '''
    Get children of a session
    :param str project: project name
    :param str mouse: mouse name
    :param str session: session name
    :return:
    '''
    project_sess = flz.get_experimental_sessions(project_id=project)
    this_sess = project_sess[project_sess.name == mouse + '_' + session]
    sess_children = flz.get_children(this_sess.id, project_id=project)

    return sess_children


def get_recording_entries(project, mouse, session, protocol):
    '''
    Get all flexilims entries (children) of a recording
    :param str project:
    :param str mouse:
    :param str session:
    :param str protocol:
    :return:
    '''
    sess_children = get_session_children(project, mouse, session)
    # protocol_recording = sess_children.loc[sess_children['name'].str.contains(protocol, case=False)]
    for i in range(len(sess_children)):
        name = sess_children.iloc[i].name
        if name[-len(protocol):] == protocol:
            protocol_recording = sess_children.iloc[i]
    recording_entries = flz.get_children(protocol_recording.id, project_id=project)
    recording_path = str(protocol_recording.path)

    return recording_entries, recording_path


def generate_file_folders(project, mouse, session, protocol, rawdata_root, root):
    '''
    Generate folders for raw data, preprocessed data and analyzed data
    :param str project:
    :param str mouse:
    :param str session:
    :param str protocol:
    :param str rawdata_root:
    :param str root:
    :return:
    '''
    sess_children = get_session_children(project, mouse, session)

    # find recording paths
    recording_entries, recording_path = get_recording_entries(project, mouse, session, protocol)

    rawdata_folder = rawdata_root + recording_path + '/'
    # preprocess_folder = root + protocol_path
    protocol_folder = root + recording_path + '/'
    first_slash = str(recording_path).find('/')
    analysis_folder = root + recording_path[:first_slash + 1] + 'Analysis/' + recording_path[first_slash + 1:] + '/'
    suite2p_folder = root + sess_children.loc[sess_children['name'].str.contains('suite2p', case=False)].path.values[
        0] + '/suite2p/plane0/'

    trace_path = recording_entries.loc[recording_entries['name'].str.contains('suite2p_trace', case=False)].path.values[
        0]
    trace_folder = root + trace_path + '/'

    return rawdata_folder, protocol_folder, analysis_folder, suite2p_folder, trace_folder


def generate_logger_path(project, mouse, session, protocol, rawdata_root, root, logger_name):
    '''
    Generate paths for param loggers
    :param ste project:
    :param ste mouse:
    :param ste session:
    :param ste protocol:
    :param ste rawdata_root:
    :param str root:
    :param str logger_name:
    :return:
    '''
    recording_entries, recording_path = get_recording_entries(project, mouse, session, protocol)

    # Find logger entries
    harp = recording_entries[recording_entries.dataset_type == 'harp']
    if logger_name == 'harp_message':
        logger_name = flz.get_entity(id=harp.id, project_id=project)['binary_file'].replace('bin','csv')
    else:
        logger_name = flz.get_entity(id=harp.id, project_id=project)['csv_files'][logger_name]

    logger_path = rawdata_root + recording_path + '/' + logger_name

    return logger_path


def generate_analysis_session_folder(root, project, mouse, session):
    project_sess = flz.get_experimental_sessions(project_id=project)
    this_sess = project_sess[project_sess.name == mouse + '_' + session]
    sess_path = str(this_sess.path.values[0])
    first_slash = sess_path.find('/')
    analysis_sess_folder = root + sess_path[:first_slash + 1] + 'Analysis/' + sess_path[first_slash + 1:] + '/'
    return analysis_sess_folder

