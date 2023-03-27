import functools
print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

import flexiznam as flm
from cottage_analysis.io_module import harp
from cottage_analysis.preprocessing import find_frames
from cottage_analysis.depth_analysis.filepath import generate_filepaths


def load_harpmessage(project, mouse, session, protocol, all_protocol_recording_entries=None, irecording=None, flexilims_session=None, redo=False):
    '''Save harpmessage into a npz file, or load existing npz file. Then load harpmessage file as a np arrray.

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        all_protocol_recording_entries (DataFrame, optional): All recording entries matching the protocol name in a given session. Defaults to None.
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to None.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        redo (bool, optional): re-transform harp bin file into npz file or not. Defaults to False.

    Returns:
        np.array: loaded harpmessages as numpy array
    '''
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


def find_monitor_frames(project, mouse, session, protocol, all_protocol_recording_entries=None, irecording=None, flexilims_session=None, redo=True, redo_harpnpz=False):
    '''Synchronise monitor frame using the find_frames.sync_by_correlation, and save them into monitor_frames_df.pickle and monitor_db_dict.pickle under the path {trace_folder/'sync/monitor_frames/'}

    Args:
        project (str): project name
        mouse (str): mouse name
        session (str): session name (Sdate)
        protocol (str): protocol name
        all_protocol_recording_entries (DataFrame, optional): All recording entries matching the protocol name in a given session. Defaults to None.
        irecording (int, optional): which recording is the current recording out of all entries in all_protocol_recording_entries. Defaults to None.
        flexilims_session (flexilims_session, optional): flexilims session. Defaults to None.
        redo (bool, optional): re-sync the monitor frames or not. Defaults to True.
        redo_harpnpz (bool, optional): re-transform harp bin file into npz file or not. Defaults to False.
    '''
    # Find paths
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
