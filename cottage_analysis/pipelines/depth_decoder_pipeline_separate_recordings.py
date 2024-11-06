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
        "Cs": np.logspace(-3, 3, 7),
        "continuous_still": 1,
        "still_time": 1,
        "still_thr": 0.05,
        "speed_bins": np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]),
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
    decoder_dict = {}
    assert len(trials_df_all.closed_loop.unique()) in [1, 2]
    if len(trials_df_all.closed_loop.unique()) == 1:
        print("This is a closed-loop only session")
    elif len(trials_df_all.closed_loop.unique()) == 2:
        print("This is a session with open loop")

    unique_recordings = np.sort(trials_df_all["recording"].unique())
    openloop_positions = pd.Series(unique_recordings).str.contains("Playback")
    closedloop_positions = ~pd.Series(unique_recordings).str.contains("Playback")
    n_openloop = np.sum(openloop_positions)
    n_closedloop = len(unique_recordings) - n_openloop

    if (n_closedloop == 1) and (n_openloop <= 0):
        print(
            "This session doesn't contain multiple closedloop / openloop session. skip."
        )
        return

    else:
        for i, recording in enumerate(unique_recordings):
            select_trials = trials_df_all["recording"] == recording
            closed_loop = trials_df_all[select_trials].closed_loop.values[0]
            if closed_loop:
                sfx = "_closedloop"
            else:
                sfx = "_openloop"
            print(f"Fitting decoder for recording: {sfx}{i}")
            print(f"---Start depth decoder{sfx}...---")
            out = population_depth_decoder.depth_decoder(
                trials_df_all[select_trials],
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
                decoder_dict_path=neurons_ds.path_full.parent
                / f"decoder_results{sfx}{i}.pickle",
                special_sfx=f"{sfx}{i}",
            )


if __name__ == "__main__":
    defopt.run(main)
