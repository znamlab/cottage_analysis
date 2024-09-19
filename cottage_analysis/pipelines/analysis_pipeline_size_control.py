import numpy as np
import defopt
import pandas as pd

import flexiznam as flz
from cottage_analysis.analysis import (
    find_depth_neurons,
    size_control,
)
from cottage_analysis.plotting import basic_vis_plots

from cottage_analysis.pipelines import pipeline_utils

PROTOCOL_BASE = "SizeControl" # "SpherePermTubeReward"

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
    # Synchronisation
    print("---Start synchronisation...---")
    vs_df_all, trials_df_all = size_control.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base=PROTOCOL_BASE,
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    
    # Add trial number to flexilims
    trial_no_closedloop = len(trials_df_all[trials_df_all["closed_loop"] == 1])
    trial_no_openloop = len(trials_df_all[trials_df_all["closed_loop"] == 0])
    ndepths = len(trials_df_all["depth"].unique())
    flz.update_entity(
        "session",
        name=session_name,
        mode="update",
        attributes={"closedloop_trials": trial_no_closedloop,
                    "openloop_trials": trial_no_openloop,
                    "ndepths": ndepths},
        flexilims_session=flexilims_session,
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


    # Find depth neurons and fit preferred depth
    print("---Start finding depth neurons...---")
    depth_fit_params = {
        "depth_min": 0.02,
        "depth_max": 20,
        "niter": 10,
        "min_sigma": 0.5,
    }
    print("Find depth neurons...")
    neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
        trials_df=trials_df_all,
        neurons_ds=neurons_ds,
        rs_thr=0.2,
        alpha=0.05,
    )

    # Find preferred depth for all data & running (current frame > 5cm/s) & not-running (previous 14+current frame < 5cm/s)
    for rs_thr, rs_thr_max, still_only, still_time, special_sfx in zip(
        [None, 0.05, None],
        [None, None, 0.05],
        [False, False, True],
        [0, 0, 1],
        ["", "_running", "_notrunning"],
    ):
        print(f"Fit preferred depth{special_sfx}...")
        # Find preferred depth of closed loop with all trials
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            rs_thr=rs_thr,
            rs_thr_max=rs_thr_max,
            still_only=still_only,
            still_time=still_time,
            frame_rate=frame_rate,
            depth_min=depth_fit_params["depth_min"],
            depth_max=depth_fit_params["depth_max"],
            niter=depth_fit_params["niter"],
            min_sigma=depth_fit_params["min_sigma"],
            k_folds=1,
            special_sfx=special_sfx,
        )

        # Find preferred depth of closed loop with half the data for plotting purposes
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials="odd",
            rs_thr=rs_thr,
            rs_thr_max=rs_thr_max,
            still_only=still_only,
            still_time=still_time,
            frame_rate=frame_rate,
            depth_min=depth_fit_params["depth_min"],
            depth_max=depth_fit_params["depth_max"],
            niter=depth_fit_params["niter"],
            min_sigma=depth_fit_params["min_sigma"],
            k_folds=1,
            special_sfx=special_sfx,
        )

        # Find r-squared of k-fold cross validation
        neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            rs_thr=rs_thr,
            rs_thr_max=rs_thr_max,
            still_only=still_only,
            still_time=still_time,
            frame_rate=frame_rate,
            depth_min=depth_fit_params["depth_min"],
            depth_max=depth_fit_params["depth_max"],
            niter=depth_fit_params["niter"],
            min_sigma=depth_fit_params["min_sigma"],
            k_folds=5,
            special_sfx=special_sfx,
        )
                
    # Fit preferred depth for different sizes
    for isize, size in enumerate(np.sort(trials_df_all["size"].unique())):
        print(f"Fit preferred depth for size {size}...")
        neurons_df_temp, neurons_ds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_all[trials_df_all["size"] == size],
            neurons_df=neurons_df.copy(),
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            rs_thr=None,
            rs_thr_max=None,
            still_only=False,
            still_time=0,
            frame_rate=frame_rate,
            depth_min=depth_fit_params["depth_min"],
            depth_max=depth_fit_params["depth_max"],
            niter=depth_fit_params["niter"],
            min_sigma=depth_fit_params["min_sigma"],
            k_folds=1,
            param="depth",
        )
        neurons_df[f"preferred_depth_size{int(size)}"] = neurons_df_temp[
            "preferred_depth_closedloop"
        ]
        neurons_df[f"depth_tuning_popt_size{int(size)}"] = neurons_df_temp[
            "depth_tuning_popt_closedloop"
        ]

    print(f"Fit preferred physical size...")
    # Fit preferred physical size
    neurons_df, neurons_ds = find_depth_neurons.fit_preferred_depth(
        trials_df=trials_df_all,
        neurons_df=neurons_df,
        neurons_ds=neurons_ds,
        closed_loop=1,
        choose_trials=None,
        rs_thr=None,
        rs_thr_max=None,
        still_only=False,
        still_time=0,
        frame_rate=frame_rate,
        depth_min=depth_fit_params["depth_min"],
        depth_max=depth_fit_params["depth_max"],
        niter=depth_fit_params["niter"],
        min_sigma=depth_fit_params["min_sigma"],
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
        rs_thr=None,
        rs_thr_max=None,
        still_only=False,
        still_time=0,
        frame_rate=frame_rate,
        depth_min=depth_fit_params["depth_min"],
        depth_max=depth_fit_params["depth_max"],
        niter=depth_fit_params["niter"],
        min_sigma=depth_fit_params["min_sigma"],
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
        rs_thr=None,
        rs_thr_max=None,
        still_only=False,
        still_time=0,
        frame_rate=frame_rate,
        depth_min=depth_fit_params["depth_min"],
        depth_max=depth_fit_params["depth_max"],
        niter=depth_fit_params["niter"],
        min_sigma=depth_fit_params["min_sigma"],
        k_folds=5,
        param="size",
    )

    # Save neurons_df
    neurons_df.to_pickle(neurons_ds.path_full.parent/"neurons_df_size_control.pickle")
    
    # Merge fit dataframes
    out = pipeline_utils.merge_fit_dataframes(
        project,
        session_name,
        use_slurm=use_slurm,
        slurm_folder="",
        job_dependency=None,
        scripts_name="",
        conflicts=conflicts,
        prefix="neurons_df",
        suffix="_size_control",
        exclude_keywords=[], 
        include_keywords=[],
        target_column_suffix=None,
        target_column_prefix="",
        filetype=".pickle",
        target_filename="neurons_df.pickle",
    )
    
    # Plotting
    basic_vis_plots.size_control_session(
        neurons_df=neurons_df, trials_df=trials_df_all, neurons_ds=neurons_ds
    )

    return neurons_df


if __name__ == "__main__":
    defopt.run(main)
