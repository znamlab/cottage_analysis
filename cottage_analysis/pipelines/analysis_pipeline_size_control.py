import os
import numpy as np
import pandas as pd
import defopt
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    fit_gaussian_blob,
    common_utils,
    size_control,
)
from cottage_analysis.plotting import basic_vis_plots, sta_plots

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

    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )
    if (neurons_ds.get_flexilims_entry() is not None) and conflicts == "skip":
        print(f"Session {session_name} already processed... reading saved data...")
    else:
        # Synchronisation
        print("---Start synchronisation...---")
        vs_df_all, trials_df_all = size_control.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets={"anatomical_only": 3},
            recording_type="two_photon",
            protocol_base="SizeControl",
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )

        # Find depth neurons and fit preferred depth
        print("---Start finding depth neurons...---")
        print("Find depth neurons...")
        neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
            trials_df=trials_df_all,
            neurons_ds=neurons_ds,
            rs_thr=0.2,
            alpha=0.05,
        )

        # Fit preferred depth 
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=1,
            param="depth",
        )
        
        # Fit cross-validated preferred depth with half the trials
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials="odd",
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=1,
            param="depth",
        )
        
        # Fit cross-validated preferred depth with 5-fold cross-validation 
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=5,
            param="depth",
        )
        
        # Fit preferred depth for different sizes
        for isize, size in enumerate(np.sort(trials_df_all["size"].unique())):
            neurons_df_temp, neurons_ds = find_depth_neurons.fit_preferred_depth(
                trials_df=trials_df_all[trials_df_all["size"] == size],
                neurons_df=neurons_df,
                neurons_ds=neurons_ds,
                closed_loop=1,
                choose_trials=None,
                depth_min=0.02,
                depth_max=20,
                niter=10,
                min_sigma=0.5,
                k_folds=1,
                param="depth",
            )
            neurons_df[f"preferred_depth_size{int(size)}"] = neurons_df_temp["preferred_depth_closedloop"]
            neurons_df[f"depth_tuning_popt_size{int(size)}"] = neurons_df_temp["depth_tuning_popt_closedloop"]
        
        # Fit preferred physical size
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=1,
            param="size",
        )
        
        # Fit cross-validated preferred physical size with half the trials
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials="odd",
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=1,
            param="size",
        )
        
        # Fit cross-validated preferred physical size with 5-fold cross-validation 
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=5,
            param="size",
        )
        
        
        # Save neurons_df
        neurons_df.to_pickle(neurons_ds.path_full)
        neurons_ds.update_flexilims(mode="update")
        
    return neurons_df
        


if __name__ == "__main__":
    defopt.run(main)
