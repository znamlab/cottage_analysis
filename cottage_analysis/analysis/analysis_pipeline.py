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
    pipeline_utils,
    common_utils,
)
from cottage_analysis.plotting import basic_vis_plots


def main(project, session_name, conflicts="skip", photodiode_protocol=2):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
    """
    print(
        f"------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------"
    )
    flexilims_session = flz.get_flexilims_session(project)
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

    # Find depth neurons and fit preferred depth
    print("---Start finding depth neurons...---")
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )
    print("Find depth neurons...")
    neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
        trials_df=trials_df_all,
        neurons_ds=neurons_ds,
        rs_thr=0.2,
        alpha=0.05,
    )

    print("Fit preferred depth...")
    # Find preferred depth of closed loop with all data
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
    )

    # Find preferred depth of closed loop with half the data for plotting purposes
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
    )

    # Find r-squared of k-fold cross validation
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
    )

    # Fit gaussian blob to neuronal activity
    print("---Start fitting 2D gaussian blob...---")
    neurons_df, neurons_ds = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df_all,
        neurons_df=neurons_df,
        neurons_ds=neurons_ds,
        choose_trials=None,
        rs_thr=0.01,
        param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
        niter=10,
        min_sigma=0.25,
    )

    neurons_df.to_pickle(neurons_ds.path_full)
    neurons_ds.update_flexilims(mode=conflicts)

    # Plot basic plots
    print("---Start plotting...---")
    basic_vis_plots.basic_vis_session(
        neurons_df=neurons_df, trials_df=trials_df_all, neurons_ds=neurons_ds
    )

    # # Regenerate sphere stimuli
    # print("---Start regenerating sphere stimuli...---")
    # harp_ds = flz.get_datasets(
    #     flexilims_session=flexilims_session,
    #     origin_name=recording["name"],
    #     dataset_type="harp",
    #     allow_multiple=False,
    #     return_dataseries=False,
    # )
    # paramlog_path = harp_ds.path_full / harp_ds.csv_files["NewParams"]
    # param_log = pd.read_csv(paramlog_path)

    # output = spheres.regenerate_frames(
    #     frame_times=imaging_df["harptime_imaging_trigger"].values,
    #     trials_df=trials_df,
    #     vs_df=vs_df,
    #     param_logger=param_log,
    #     time_column="HarpTime",
    #     resolution=5,
    #     sphere_size=10,
    #     azimuth_limits=(-120, 120),
    #     elevation_limits=(-40, 40),
    #     verbose=True,
    #     output_datatype="int16",
    #     output=None,
    # )
    # print("Visual stimuli regeneration finished.")


if __name__ == "__main__":
    defopt.run(main)
