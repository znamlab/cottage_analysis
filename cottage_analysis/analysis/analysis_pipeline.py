import numpy as np
import pandas as pd
import defopt

import flexiznam as flz
from cottage_analysis.filepath import generate_filepaths
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import (
    find_depth_neurons,
    fit_gaussian_blob,
    common_utils,
)
from cottage_analysis.analysis import spheres
from functools import partial

print = partial(print, flush=True)

redo_dict = {
    "monitor_frames": 0,
    "sync": 0,
    "find_depth_neurons": 0,
    "fit_gaussian_blob": 0,
    "regenerate_spheres": 0,
    "sta": 0,
}


def main(
    project,
    mouse,
    session,
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        mouse(str): mouse name
        session(str): session name
    """
    protocol = "SpheresPermTubeReward"
    redo = redo_dict

    # For each separate recording, find monitor frames from photodiode signals
    print("---Start finding monitor frames...---")

    # Filepath, loop through each recording for each protocol
    flz_session = flz.get_flexilims_session(project)
    session_name = f"{mouse}_{session}"
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flz_session
    )
    recording_type = "two_photon"
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value=recording_type,
        flexilims_session=flz_session,
    )
    if np.sum(recordings.name.str.contains("Playback")):
        protocols = [protocol, f"{protocol}Playback"]
    else:
        protocols = [protocol]
    for iprotocol, this_protocol in enumerate(protocols):
        print(f"Start processing protocol {iprotocol+1}/{len(protocols)}")
        if redo["monitor_frames"]:
            if "Playback" not in this_protocol:
                all_recordings = recordings[
                    recordings.name.str.contains(protocols[0])
                    & (~recordings.name.str.contains("Playback"))
                ]
            else:
                all_recordings = recordings[recordings.name.str.contains("Playback")]
            nrecordings = len(all_recordings)
            for irecording in range(nrecordings):
                print(f"Processing {irecording+1}/{nrecordings}")
                synchronisation.find_monitor_frames(
                    project=project,
                    mouse=mouse,
                    session=session,
                    protocol=this_protocol,
                    irecording=irecording,
                    redo=redo["monitor_frames"],
                    redo_harpnpz=True,
                )
        else:
            print("Already analyzed, not redoing.")
    print("Monitor frame finding finished.")

    # Generate synchronisation dataframes for each recording in each protocol
    print("---Start synchronisation...---")
    for iprotocol, this_protocol in enumerate(protocols):
        print(f"Start processing protocol {iprotocol+1}/{len(protocols)}")
        if redo["sync"]:
            if "Playback" not in this_protocol:
                all_recordings = recordings[
                    recordings.name.str.contains(protocols[0])
                    & (~recordings.name.str.contains("Playback"))
                ]
            else:
                all_recordings = recordings[recordings.name.str.contains("Playback")]
            nrecordings = len(all_recordings)
            for irecording in range(nrecordings):
                print(f"Processing {irecording+1}/{nrecordings}")
                vs_df = synchronisation.generate_vs_df(
                    project=project,
                    mouse=mouse,
                    session=session,
                    protocol=this_protocol,
                    irecording=irecording,
                )
                trials_df, imaging_df = synchronisation.generate_trials_df(
                    project=project,
                    mouse=mouse,
                    session=session,
                    protocol=this_protocol,
                    vs_df=vs_df,
                    irecording=irecording,
                )
        else:
            print("Already analyzed, not redoing.")
    print("Synchronisation finished.")

    # Concatenate vs_df, trials_df from all recordings for each protocol
    if redo["sync"]:
        print("---Start concatenating all recordings...---")
        common_utils.concatenate_recordings(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
        )
        print("Concatenation finished.")

    # Find depth neurons and fit preferred depth
    print("---Start finding depth neurons...---")
    if redo["find_depth_neurons"]:
        neurons_df = find_depth_neurons.find_depth_neurons(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            rs_thr=0.2,
        )

        neurons_df = find_depth_neurons.fit_preferred_depth(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            depth_min=0.02,
            depth_max=20,
            niter=5,
        )
    else:
        print("Already analyzed, not redoing.")
    print("Depth neurons found.")

    # Fit gaussian blob to neuronal activity
    print("---Start fitting 2D gaussian blob...---")
    if redo["fit_gaussian_blob"]:
        neurons_df = fit_gaussian_blob.analyze_rs_of_tuning(
            project=project,
            mouse=mouse,
            session=session,
            protocol="SpheresPermTubeReward",
            rs_thr=0.01,
            param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
            niter=5,
        )
    else:
        print("Already analyzed, not redoing.")
    print("Gaussian blot fitting finished.")

    # Regenerate sphere stimuli
    print("---Start regenerating sphere stimuli...---")
    if redo["regenerate_spheres"]:
        (
            rawdata_folder,
            _,
            _,
            _,
            _,
        ) = generate_filepaths.generate_file_folders(
            project=project,
            mouse=mouse,
            session=session,
            protocol=protocol,
            all_protocol_recording_entries=None,
            recording_no=0,
        )

        param_log = pd.read_csv(rawdata_folder / "NewParams.csv")
        param_log = param_log.rename(columns={"Radius": "Depth"})

        output = spheres.regenerate_frames(
            frame_times=imaging_df["harptime_imaging_trigger"].values,
            trials_df=trials_df,
            vs_df=vs_df,
            param_logger=param_log,
            time_column="HarpTime",
            resolution=1,
            sphere_size=10,
            azimuth_limits=(-120, 120),
            elevation_limits=(-40, 40),
            verbose=True,
            output_datatype="int16",
            output=None,
        )
        print("Visual stimuli regeneration finished.")


if __name__ == "__main__":
    defopt.run(main)
