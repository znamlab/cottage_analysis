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
from cottage_analysis.depth_analysis.depth_preprocess import synchronisation


def load_harpmessage(project, mouse, session, protocol, all_protocol_recording_entries=None, irecording=0, redo=False):
    '''Save harpmessage into a npz file, or load existing npz file. Then load harpmessage file as a np arrray.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        all_protocol_recording_entries (DataFrame, optional): All recording entries matching the protocol name in a given session. Defaults to None.
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        redo (bool, optional): re-transform harp bin file into npz file or not. Defaults to False.

    Returns:
        np.array: loaded harpmessages as numpy array
    '''
    flexilims_session = flz.get_flexilims_session(project_id=project)
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
        flexilims_session=flexilims_session
    )
            
    #save harp message into npz, or load existing npz file 
    msg = Path(str(harpmessage_file).replace('csv','bin'))
    p_msg = protocol_folder/'sync'
    if not os.path.exists(p_msg):
        os.makedirs(p_msg)
    p_msg = p_msg / (msg.stem + '.npz')
    if (not p_msg.is_file()) or redo==True:
        print('Saving harp messages into npz...')
        harp_messages = harp.load_harp(msg, di_names=('frame_triggers','lick_detection','di2_encoder_initial_state'))
        p_msg.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p_msg, **harp_messages)
        print('Harp messages saved.')
    elif (p_msg.is_file()) and (redo==False): 
        print('harpmessage.npz already exists. Loading harpmessage.npz...')
    harp_messages = np.load(p_msg)
    print('harpmessage loaded.')
    
    return harp_messages


def find_monitor_frames(project, mouse, session, protocol, all_protocol_recording_entries=None, irecording=0, redo=True, redo_harpnpz=False):
    '''Synchronise monitor frame using the find_frames.sync_by_correlation, and save them into monitor_frames_df.pickle and monitor_db_dict.pickle under the path {trace_folder/'sync/monitor_frames/'}

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        all_protocol_recording_entries (DataFrame, optional): All recording entries matching the protocol name in a given session. Defaults to None.
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        redo (bool, optional): re-sync the monitor frames or not. Defaults to True.
        redo_harpnpz (bool, optional): re-transform harp bin file into npz file or not. Defaults to False.
    '''
    # Find paths
    flexilims_session = flz.get_flexilims_session(project_id=project)
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
        flexilims_session=flexilims_session
    )
    
    if redo:
        # Load files
        harp_messages = load_harpmessage(project=project, 
                                    mouse=mouse, 
                                    session=session, 
                                    protocol=protocol, 
                                    all_protocol_recording_entries=all_protocol_recording_entries, 
                                    irecording=irecording, 
                                    flexilims_session=flexilims_session,
                                    redo=redo_harpnpz)
        
        frame_log = pd.read_csv(rawdata_folder / "FrameLog.csv")
        expected_sequence = (
            pd.read_csv(rawdata_folder / 'random_sequence_5values_alternate.csv', header=None).loc[:, 0].values
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
        save_folder = protocol_folder/'sync/monitor_frames/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        frames_df.to_pickle(save_folder/'monitor_frames_df.pickle')  
        with open(save_folder/'monitor_db_dict.pickle', 'wb') as handle:
            pickle.dump(db_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



def generate_vs_df(project, mouse, session, protocol, all_protocol_recording_entries=None, irecording=0):
    '''Generate a dataframe that contains information for each monitor frame. This requires monitor frames to be synced first.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        all_protocol_recording_entries (DataFrame, optional): All recording entries matching the protocol name in a given session. Defaults to None.
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to 0.

    Returns:
        DataFrame: contains information for each monitor frame.
    '''
    flexilims_session = flz.get_flexilims_session(project_id=project)
    sess_children = generate_filepaths.get_session_children(project=project, mouse=mouse, session=session, flexilims_session=flexilims_session)
    sess_children_protocols = sess_children[sess_children['name'].str.contains('(SpheresPermTubeReward|Fourier|Retinotopy)')]
    folder_no = sess_children_protocols.index.get_loc(sess_children_protocols[sess_children_protocols.id == all_protocol_recording_entries.iloc[irecording].id].index[0])
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(project=project, 
                                                                                mouse=mouse, 
                                                                                session=session, 
                                                                                protocol=protocol, 
                                                                                flexilims_session=flexilims_session)
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
    save_folder = protocol_folder/'sync/monitor_frames/'
    with open(save_folder/'monitor_db_dict.pickle', 'rb') as handle:
        monitor_db_dict = pickle.load(handle)
    with open(save_folder/'monitor_frames_df.pickle', 'rb') as handle:
        monitor_frames_df = pickle.load(handle)
        

    # Find frames that are not skipped  
    monitor_frame_valid = monitor_frames_df[monitor_frames_df.closest_frame.notnull()][['closest_frame','onset_time','offset_time','peak_time']]
    monitor_frame_valid['closest_frame'] = monitor_frame_valid['closest_frame'] .astype('int')
    monitor_frame_valid = monitor_frame_valid.sort_values('closest_frame')

    # Merge MouseZ and EyeZ from FrameLog.csv to frame_df according to FrameIndex
    frame_log = pd.read_csv(rawdata_folder / "FrameLog.csv")
    frame_log_z = frame_log[['FrameIndex','HarpTime','MouseZ','EyeZ']]
    frame_log_z = frame_log_z.rename(columns={'FrameIndex':'closest_frame','HarpTime':'harptime_framelog','MouseZ':'mouse_z','EyeZ':'eye_z'})
    frame_log_z['mouse_z'] = frame_log_z['mouse_z']/100  #convert cm to m
    frame_log_z['eye_z'] = frame_log_z['eye_z']/100  #convert cm to m
    vs_df = pd.merge_asof(left=monitor_frame_valid, right=frame_log_z, on='closest_frame', direction='nearest',allow_exact_matches=True)

    # Align sphere parameter with the frame (harptime later than the logged sphere time)
    vs_df = vs_df.sort_values('onset_time')
    param_log = pd.read_csv(rawdata_folder / "NewParams.csv")
    param_log_simple = param_log[['HarpTime','Radius']]
    param_log_simple = param_log_simple.rename(columns={'HarpTime':'harptime_sphere','Radius':'depth'})
    param_log_simple['onset_time'] = param_log_simple['harptime_sphere']
    vs_df = pd.merge_asof(left=vs_df, right=param_log_simple, on='onset_time', direction='backward',allow_exact_matches=False) # Does not allow exact match of sphere rendering time and frame onset time?

    # Align imaging frame time with monitor frame onset time (imaging frame time later than monitor frame onset time)
    ops = np.load(suite2p_folder/"ops.npy", allow_pickle=True)
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
    p_msg = protocol_folder/'sync/harpmessage.npz'
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

    img_frame_logger = img_frame_logger[['HarpTime','ImagingFrame']]
    img_frame_logger = img_frame_logger.rename(columns={'HarpTime':'harptime_imaging_frame','ImagingFrame':'imaging_frame'})
    img_frame_logger['onset_time'] = img_frame_logger['harptime_imaging_frame']
    vs_df = pd.merge_asof(left=vs_df, right=img_frame_logger, on='onset_time', direction='forward',allow_exact_matches=True)

    # Indicate whether it's a closed loop or open loop session
    if 'Playback' in protocol:
        vs_df['closed_loop'] = 0
    else: 
        vs_df['closed_loop'] = 1
    
    # Rename
    vs_df = vs_df.rename(columns={'closest_frame':'monitor_frame'})
    return vs_df