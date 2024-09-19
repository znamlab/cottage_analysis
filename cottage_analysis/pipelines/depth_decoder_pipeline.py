import os
import numpy as np
import pandas as pd
import defopt
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings

import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    population_depth_decoder,
)
from cottage_analysis.plotting import depth_decoder_plots

from cottage_analysis.pipelines import pipeline_utils

def main(
    project, session_name, conflicts="skip", photodiode_protocol=5, use_slurm=False
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        use_slurm(bool): whether to use slurm to run the fit in the pipeline. Default False.
    """
    print(
        f"------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------"
    )
    params = {
        "trial_average": False,
        "rolling_window": 0.5,
        "downsample_window": 0.5,
        "Cs": np.logspace(-1,1,3),
        # "Cs": np.logspace(-3, 3, 7),
        "continuous_still": 1,
        "still_time": 1,
        "still_thr": 0.05,
        "speed_bins": np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]),
        "special_sfx": "",
    }
    
    if use_slurm:
        slurm_folder = Path(os.path.expanduser(f"~/slurm_logs"))
        slurm_folder.mkdir(exist_ok=True)
        slurm_folder = Path(slurm_folder / f"{session_name}")
        slurm_folder.mkdir(exist_ok=True)
    else:
        slurm_folder = None

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )
    suite2p_datasets = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        project_id=project,
        flexilims_session=flexilims_session,
        return_dataseries=False,
        filter_datasets={"anatomical_only": 3},
    )
    suite2p_dataset = suite2p_datasets[0]
    frame_rate = suite2p_dataset.extra_attributes["fs"]

    # Synchronisation
    print("---Start synchronisation...---")
    vs_df_all, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )

    # Run depth decoder
    assert len(trials_df_all.closed_loop.unique()) in [1, 2]
    if len(trials_df_all.closed_loop.unique()) == 1:
        print("This is a closed-loop only session")
    elif len(trials_df_all.closed_loop.unique()) == 2:
        print("This is a session with open loop")
    
    outputs_all = []
    for closed_loop in np.sort(trials_df_all.closed_loop.unique()):
        outputs = []
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        print(f"---Start depth decoder{sfx}...---")
        out = population_depth_decoder.depth_decoder(
            trials_df_all[trials_df_all.closed_loop == closed_loop],
            flexilims_session=flexilims_session,
            session_name=session_name,
            closed_loop=closed_loop,
            trial_average=params["trial_average"],
            rolling_window=params["rolling_window"],
            frame_rate=frame_rate,
            downsample_window=params["downsample_window"],
            random_state=42,
            kernel="linear",
            Cs=params["Cs"],
            k_folds=5,
            use_slurm=use_slurm,
            neurons_ds=neurons_ds,
            decoder_dict_path=neurons_ds.path_full.parent / f"decoder_results{params['special_sfx']}.pickle",
            special_sfx=params["special_sfx"],
        )
        outputs.append(out)
        outputs_all.append(out)
        
        # Get accuracy for different speed bins
        print(f"Calculating accuracy for depth decoder at different speed bins{sfx}")
        job_dependency = outputs if use_slurm else None
        out = population_depth_decoder.find_acc_speed_bins(decoder_dict_path=neurons_ds.path_full.parent / f"decoder_results{params['special_sfx']}.pickle",
                                                                                         recording_type=sfx,
                                                                                         speed_bins=params["speed_bins"],
                                                                                         continuous_still=params["continuous_still"], 
                                                                                         still_thr=params["still_thr"], 
                                                                                         still_time=params["still_time"],
                                                                                         frame_rate=frame_rate,
                                                                                         use_slurm=use_slurm,
                                                                                         slurm_folder=slurm_folder,
                                                                                         scripts_name=f"decoder_speedbins{sfx}{params['special_sfx']}",
                                                                                         job_dependency=job_dependency,)
        outputs_all.append(out)

    job_dependency = outputs_all if use_slurm else None
    depth_decoder_plots.plot_decoder_session(decoder_dict_path=neurons_ds.path_full.parent / f"decoder_results{params['special_sfx']}.pickle",
                                             neurons_ds=neurons_ds,
                                             session_name=session_name,
                                             flexilims_session=flexilims_session,
                                             photodiode_protocol=photodiode_protocol,
                                             params=params,
                                             use_slurm=use_slurm,
                                             slurm_folder=slurm_folder,
                                             scripts_name=f"decoder_plots_{params['special_sfx']}",
                                             job_dependency=job_dependency,)
                         

if __name__ == "__main__":
    defopt.run(main)
