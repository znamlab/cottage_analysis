import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
import defopt
import pickle

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import spheres, sta, find_depth_neurons
from cottage_analysis.plotting import basic_vis_plots, grating_plots, plotting_utils, sta_plots
from cottage_analysis.pipelines import pipeline_utils

def main(project, session_name, conflicts="skip", photodiode_protocol=5):
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
    
    # Regenerating frames for all recordings
    frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=None,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
        resolution=5,
    )
    
    # Create neurons_df
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )

    # Analyze depth responses
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
        niter=5,
        min_sigma=0.5,
        k_folds=5,
    )
    
    neurons_df.to_pickle(neurons_ds.path_full)
    neurons_df = pd.read_pickle(neurons_ds.path_full)
    
    # Fit RFs of all ROIs
    print("---Start fitting RFs for all ROIs...---")
    coef, r2, best_reg_xy, best_reg_depth  = spheres.fit_3d_rfs_hyperparam_tuning(imaging_df_all, 
                                                   frames_all[:,:,int(frames_all.shape[2]//2):], 
                                                   reg_xys=[20, 40, 80, 160, 320], 
                                                   reg_depths=[20, 40, 80, 160, 320], 
                                                   shift_stims=2, 
                                                   use_col="dffs", 
                                                   k_folds=5, 
                                                   tune_separately=False, 
                                                   use_validation_set=False)
    
    coef_ipsi, r2_ipsi = spheres.fit_3d_rfs(imaging_df_all, 
                                            frames_all[:,:,:int(frames_all.shape[2]//2)], 
                                            reg_xy=best_reg_xy, 
                                            reg_depth=best_reg_depth, 
                                            shift_stim=2, 
                                            use_col="dffs", 
                                            k_folds=5, 
                                            choose_rois=[],
                                            mode="test")
    
    sig, sig_ipsi = spheres.find_sig_rfs(coef, coef_ipsi, n_std=5)
    print(f"Number of significant RFs at contra side: {np.mean(sig)}")
    print(f"Number of significant RFs at ipsi side: {np.mean(sig_ipsi)}")
    
    rf_dict = {"coef":coef,
            "r2":r2,
            "best_reg_xy":best_reg_xy,
            "best_reg_depth":best_reg_depth,
            "coef_ipsi":coef_ipsi,
            "r2_ipsi":r2_ipsi,
            "sig":sig,
            "sig_ipsi":sig_ipsi}
    with open(neurons_ds.path_full.parent/'rf_fit.pickle', 'wb') as handle:
        pickle.dump(rf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(neurons_ds.path_full.parent/'rf_fit.pickle', 'rb') as handle:
        rf_dict = pickle.load(handle)
    coef = rf_dict["coef"]
        
    # Plot all ROIs
    print("---Start plotting...---")
    depth_list = find_depth_neurons.find_depth_list(trials_df_all)
    sta_plots.basic_vis_SFTF_session(coef=coef, 
                           neurons_df=neurons_df, 
                           trials_df=trials_df_all, 
                           depth_list=depth_list, 
                           frames=frames_all, 
                           save_dir=neurons_ds.path_full.parent, 
                           fontsize_dict={"title": 10, "tick": 10, "label": 10})

if __name__ == "__main__":
    defopt.run(main)
