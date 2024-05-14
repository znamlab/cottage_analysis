import os
import numpy as np
import pandas as pd
import defopt
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    fit_gaussian_blob,
    common_utils,
    openloop,
)
from cottage_analysis.plotting import basic_vis_plots, sta_plots

from cottage_analysis.pipelines import pipeline_utils


def main(
    project, session_name, conflicts="skip", photodiode_protocol=5, use_slurm=False,
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        use_slurm(bool): whether to use slurm to run the fit in the pipeline. Default False.
        run_depth_fit(bool): whether to run the depth fit. Default True.
        run_rf(bool): whether to run the rf fit. Default True.
        run_rsof_fit(bool): whether to run the rsof fit. Default True.
        run_plot(bool): whether to run the plot. Default True.
    """
    print(
        f"------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------"
    )
    if use_slurm:
        slurm_folder = Path(os.path.expanduser(f"~/slurm_logs"))
        slurm_folder.mkdir(exist_ok=True)
        slurm_folder = Path(slurm_folder / f"{session_name}")
        slurm_folder.mkdir(exist_ok=True)
    else:
        slurm_folder = None

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    flexilims_session = flz.get_flexilims_session(project)
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

    # print("---Start fitting 2D gaussian blob...---")
    # outputs = []
    # common_params = dict(
    #     rs_thr=0.01,
    #     param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
    #     niter=10,
    #     min_sigma=0.25,
    # )

    # to_do = [
    #     ("gaussian_2d", None, 1),
    #     # ("gaussian_2d", "even", 1),
    #     # ("gaussian_additive", None, 1),
    #     # ("gaussian_OF", None, 1),
    #     # ("gaussian_2d", None, 5),
    #     # ("gaussian_additive", None, 5),
    #     # ("gaussian_OF", None, 5),
    #     # ("gaussian_ratio", None, 1),
    #     # ("gaussian_ratio", None, 5),
    # ]

    arr = trials_df_all['closed_loop'].values
    zeros, ones = openloop.find_zeros_before_ones(arr)
    if len(zeros) == 0:
        print("No open loop before closed loop trials found.")
    else:
    #     for model, trials, k_folds in to_do:
    #         name = f"{session_name}_{model}"
    #         if trials is not None:
    #             name += "_crossval"
    #         name += f"_k{k_folds}"
    #         print(f"Fitting {model}")
    #         i=0
    #         for openloop_trials, closedloop_trials in zip(zeros, ones):
    #             new_name = name + f"_openclosed{i}"
    #             openloop_trials = openloop_trials.tolist()
    #             closedloop_trials = closedloop_trials.tolist()
    #             out = pipeline_utils.load_and_fit(
    #                 project,
    #                 session_name,
    #                 photodiode_protocol,
    #                 model=model,
    #                 choose_trials=trials,
    #                 use_slurm=use_slurm,
    #                 slurm_folder=slurm_folder,
    #                 scripts_name=new_name,
    #                 k_folds=k_folds,
    #                 closedloop_trials=closedloop_trials,
    #                 openloop_trials=openloop_trials,
    #                 special_sfx=f"_openclosed{i}",
    #                 **common_params,
    #             )
    #             outputs.append(out)
    #             i+=1
    #             print("---RS OF fit finished. Neurons_df saved.---")
        
        # Merge fit dataframes
        # job_dependency = outputs if use_slurm else None
        out = pipeline_utils.merge_fit_dataframes(
            project,
            session_name,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            job_dependency=None,
            scripts_name=f"{session_name}_merge_fit_dataframes_openclosed",
            conflicts=conflicts,
            prefix="fit_rs_of_tuning_gaussian_2d_k1_openclosed", 
            suffix="",
            column_suffix=-12,
            filetype=".pickle",
            target_filename="neurons_df_openclosed.pickle"
        )

        print("---Analysis finished. Neurons_df saved.---")

if __name__ == "__main__":
    
    defopt.run(main)
