import subprocess
import flexiznam as flz
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

MODE = "MANUAL"

PROJECT = "colasa_3d-vision_revisions"
flz_session = flz.get_flexilims_session(PROJECT)

# micelist = ['PZAG16.3b', 'PZAG16.3c', 'PZAG17.3a', 'PZAH17.1e']

micelist = ["PZAG16.3c"]

# roicat_sessions = {
#    'PZAG16.3c': ['S20250219', 'S20250313'],
#    'PZAG17.3a': ['S20250303', 'S20250305', 'S20250306'],
#    'PZAG16.3b': ['S20250224', 'S20250225', 'S20250226', 'S20250310'],
#    'PZAH17.1e': ['S20250305', 'S20250306', 'S20250307', 'S202503011', 'S20250304']

# }


roicat_sessions = {"PZAG16.3c": ["S20250313"]}

processed_root = flz.get_data_root(
    "processed", project=PROJECT, flexilims_session=flz_session
)
raw_root = flz.get_data_root("raw", project=PROJECT, flexilims_session=flz_session)


def get_remaining_sessions(
    mouse, protocol="SpheresPermTubeReward", drop_processed=True
):
    # Check all sessions
    sessions = flz.get_children(
        parent_name=mouse,
        children_datatype="session",
        project_id=PROJECT,
        flexilims_session=flz_session,
    )
    # print(sessions.name)
    # List the sessions that are SphereTube
    for i in sessions.name:
        SphereTube_recordings = flz.get_children(
            parent_name=i,
            children_datatype="recording",
            project_id=PROJECT,
            flexilims_session=flz_session,
        )
        SphereTube_recordings = SphereTube_recordings[
            SphereTube_recordings["protocol"] == "SpheresPermTubeReward"
        ]
        if len(SphereTube_recordings) == 0:
            sessions = sessions[sessions["name"] != i]

    # print(sessions.name)
    # list the sessions that have suite2p data
    for i, session in sessions.iterrows():
        # print(session)
        suite2p_path = processed_root / session.path / "suite2p"
        if not os.path.isdir(suite2p_path):
            name_to_drop = session.name
            sessions = sessions[sessions["name"] != name_to_drop]
    # print(sessions.name)

    if drop_processed:
        # drop the sessions that are processed
        for i, session in sessions.iterrows():
            # print(session)
            neurons_path = processed_root / session.path / "neurons_df.pickle"
            if os.path.isfile(neurons_path):
                name_to_drop = session.name
                sessions = sessions[sessions["name"] != name_to_drop]

    # print(sessions.name)
    return sessions["name"].tolist()


def get_manual_session_names(mouse, session_dict):
    mouse_sessions = session_dict[mouse]

    session_names = []
    for session in mouse_sessions:
        session_names.append(f"{mouse}_{session}")

    return session_names


for mouse in micelist:
    # Define the base command template
    base_command = "sbatch --export=PROJECT={PROJECT},SESSION_NAME={session},CONFLICTS=overwrite,PHOTODIODE_PROTOCOL=5 run_analysis_pipeline.sh"

    # List of session names
    if MODE == "MANUAL":
        session_names = get_manual_session_names(mouse, roicat_sessions)
    if MODE == "AUTO":
        session_names = get_remaining_sessions(
            mouse, drop_processed=False
        )  # Add your actual sessions

    if not session_names:  # Skip if no sessions found
        print(f"No remaining sessions for {mouse}, skipping...")
        continue

    # Loop over session names and execute the sbatch command
    for session in session_names:
        command = base_command.format(
            session=session, PROJECT=PROJECT
        )  # Insert session dynamically
        print(f"Running: {command}")  # Print the command for debugging (optional)

        # Execute the command
        subprocess.run(command, shell=True, check=True)
