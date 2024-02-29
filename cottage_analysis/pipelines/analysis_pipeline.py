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
)
from cottage_analysis.plotting import basic_vis_plots, sta_plots

from cottage_analysis.pipelines import pipeline_utils

# TODO: add decoder
# TODO: separate steps


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
    if use_slurm:
        slurm_folder = Path(os.path.expanduser(f"~/slurm_logs"))
        slurm_folder.mkdir(exist_ok=True)
        slurm_folder = Path(slurm_folder/f"{session_name}")
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
    if (neurons_ds.get_flexilims_entry() is not None) and conflicts == "skip":
        print(f"Session {session_name} already processed... reading saved data...")
    else:
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
        
        # Save neurons_df
        neurons_df.to_pickle(neurons_ds.path_full)

        # Regenerate sphere stimuli
        print("---RF analysis...---")
        print("Generating sphere stimuli...")
        for is_closedloop in trials_df_all["closed_loop"].unique():
            if is_closedloop:
                sfx = "_closedloop"
            else:
                sfx = "_openloop"   
            frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets={"anatomical_only": 3},
                recording_type="two_photon",
                is_closedloop=is_closedloop,
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                resolution=5,
            )

            print(f"Fitting RF{sfx}...")
            coef, r2, best_reg_xys, best_reg_depths = spheres.fit_3d_rfs_hyperparam_tuning(
                imaging_df_all,
                frames_all[:, :, int(frames_all.shape[2] // 2) :],
                reg_xys=np.geomspace(2.5,10240,13),
                reg_depths=np.geomspace(2.5,10240,13),
                shift_stim=2,
                use_col="dffs",
                k_folds=5,
                tune_separately=True,
                validation=False,
            )

            print("Fitting ipsi RF...")
            coef_ipsi, r2_ipsi = spheres.fit_3d_rfs_ipsi(
                imaging_df_all,
                frames_all[:, :, : int(frames_all.shape[2] // 2)],
                best_reg_xys,
                best_reg_depths,
                shift_stim=2,
                use_col="dffs",
                k_folds=5,
                validation=False,
            )

            for col in [f"rf_coef{sfx}", 
                        f"rf_rsq{sfx}", 
                        f"rf_coef_ipsi{sfx}", 
                        f"rf_rsq_ipsi{sfx}"]:
                neurons_df[col] = [[np.nan]] * len(neurons_df)

            for i, _ in neurons_df.iterrows():
                neurons_df.at[i, f"rf_coef{sfx}"] = coef[:, :, i]
                neurons_df.at[i, f"rf_coef_ipsi{sfx}"] = coef_ipsi[:, :, i]
                neurons_df.at[i, f"rf_rsq{sfx}"] = r2[i, :]
                neurons_df.at[i, f"rf_rsq_ipsi{sfx}"] = r2_ipsi[i, :]
                neurons_df.at[i, f"rf_reg_xy{sfx}"] = best_reg_xys[i]
                neurons_df.at[i, f"rf_reg_depth{sfx}"] = best_reg_depths[i]

        # Save neurons_df
        neurons_df.to_pickle(neurons_ds.path_full)

        # # Update neurons_ds on flexilims
        # neurons_ds.update_flexilims(mode="update")
        
        # Merge fit dataframes
        out = pipeline_utils.merge_fit_dataframes(
            project,
            session_name,
            use_slurm=0,
            slurm_folder=slurm_folder,
            job_dependency=None,
            scripts_name=f"{session_name}_merge_fit_dataframes",
        )
        
        print("---Analysis finished. Neurons_df saved.---")

        # # Fit gaussian blob to neuronal activity
        # print("---Start fitting 2D gaussian blob...---")
        # outputs = []
        # common_params = dict(
        #     rs_thr=0.01,
        #     param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
        #     niter=5,
        #     min_sigma=0.25,
        # )
                
        # to_do = [
        #     ("gaussian_2d", None, 1),
        #     ("gaussian_2d", "even", 1),
        #     ("gaussian_additive", None, 1),
        #     ("gaussian_OF", None, 1),
        #     ("gaussian_2d", None, 5),
        #     ("gaussian_additive", None, 5),
        #     ("gaussian_OF", None, 5),
        # ]

        # for model, trials, k_folds in to_do:
        #     name = f"{session_name}_{model}"
        #     if trials is not None:
        #         name += "_crossval"
        #     name += f"_k{k_folds}"  
        #     print(f"Fitting {model}")
        #     out = pipeline_utils.load_and_fit(
        #         project,
        #         session_name,
        #         photodiode_protocol,
        #         model=model,
        #         choose_trials=trials,
        #         use_slurm=use_slurm,
        #         slurm_folder=slurm_folder,
        #         scripts_name=name,
        #         k_folds=k_folds,
        #         **common_params,
        #     )
        #     outputs.append(out)

        # # Merge fit dataframes
        # job_dependency = outputs if use_slurm else None
        # out = pipeline_utils.merge_fit_dataframes(
        #     project,
        #     session_name,
        #     use_slurm=use_slurm,
        #     slurm_folder=slurm_folder,
        #     job_dependency=job_dependency,
        #     scripts_name=f"{session_name}_merge_fit_dataframes",
        # )
        
        # print("---Analysis finished. Neurons_df saved.---")

        # # Plot basic plots
        # print("---Start basic vis plotting...---")
        # pipeline_utils.run_basic_plots(
        #     project,
        #     session_name,
        #     photodiode_protocol,
        #     use_slurm=use_slurm,
        #     slurm_folder=slurm_folder,
        #     job_dependency=job_dependency,
        #     scripts_name=f"{session_name}_basic_vis_plots",
        # )
        # print("---Plotting finished. ---")


if __name__ == "__main__":
    defopt.run(main)
