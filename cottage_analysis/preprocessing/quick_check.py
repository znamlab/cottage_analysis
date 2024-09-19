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
from cottage_analysis.preprocessing import synchronisation

project = "hey2_3d-vision_foodres_20220101"
mouse = "PZAH10.2d"
session = "S20230613"
protocols = ["SpheresPermTubeReward", "SpheresPermTubeRewardPlayback"]

flexilims_session = flz.get_flexilims_session(project_id=project)
rawdata_root = Path(flz.PARAMETERS["data_root"]["raw"])
root = Path(flz.PARAMETERS["data_root"]["processed"])

for protocol in protocols:
    print(f"Process protocol {protocol}/{len(protocols)}")
    all_protocol_recording_entries = generate_filepaths.get_all_recording_entries(
        project=project,
        mouse=mouse,
        session=session,
        protocol=protocol,
        flexilims_session=flexilims_session,
    )
    nrecordings = len(all_protocol_recording_entries)
    for irecording in range(nrecordings):
        print(f"Process recording {irecording+1}/{nrecordings}")
        # Load files
        harp_messages = synchronisation.load_harpmessage(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            irecording=irecording,
            redo=True,
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
        # ops = np.load(suite2p_folder / "ops.npy", allow_pickle=True).item()
        p_msg = protocol_folder / "sync/harpmessage.npz"
        img_frame_logger = format_loggers.format_img_frame_logger(
            harpmessage_file=p_msg, register_address=32
        )
        # frame_number = ops["frames_per_folder"][0]
        img_frame_logger = find_img_frames.find_imaging_frames(
            harp_message=img_frame_logger,
            frame_number=60000,
            exposure_time=0.0324 * 2,
            register_address=32,
            exposure_time_tolerance=0.001,
        )
        print(
            f"found frames {len(img_frame_logger)}, \
              \n last frame timestamp {img_frame_logger.HarpTime.iloc[-1]}"
        )
