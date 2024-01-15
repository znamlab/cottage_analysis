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
)
from cottage_analysis.plotting import basic_vis_plots, sta_plots

from cottage_analysis.pipelines import pipeline_utils

# TODO: add decoder


def main(
    project,
    session_name,
    conflicts="skip",
    photodiode_protocol=5,
    sync_kwargs=None,
    use_onix=True,
    return_multiunit=False,
    harp_is_in_recording=False,
    exp_sd=0.1,
    rate_bin=0.01,
    unit_list=None,
    rs_thr=0.0002,
    do_rf=True,
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
    """
    print(
        f"""
        -------------------------------------
        Start analysing {session_name}   
        -------------------------------------
        """
    )
    ephys_kwargs = dict(
        return_multiunit=return_multiunit,
        exp_sd=exp_sd,
        rate_bin=rate_bin,
        unit_list=unit_list,
    )
    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )
    if (neurons_ds.get_flexilims_entry() is not None) and conflicts == "skip":
        print(
            f"Session {session_name} already processed... reading saved neurons_df..."
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)

        print("Regenerating vis-stim dataframes...")
        vs_df_all, trials_df_all = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=None,
            recording_type="behaviour",
            protocol_base="SpheresPermTubeReward",
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            harp_is_in_recording=harp_is_in_recording,
            use_onix=use_onix,
            conflicts="skip",
            sync_kwargs=sync_kwargs,
            ephys_kwargs=ephys_kwargs,
        )

        if do_rf:
            frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets=None,
                recording_type="behaviour",
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                resolution=5,
                sync_kwargs=sync_kwargs,
                use_onix=use_onix,
                harp_is_in_recording=harp_is_in_recording,
                ephys_kwargs=ephys_kwargs,
            )

        print("Redoing plotting...")

    else:
        # Synchronisation
        print("---Start synchronisation...---")
        vs_df_all, trials_df_all = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=None,
            recording_type="behaviour",
            protocol_base="SpheresPermTubeReward",
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            harp_is_in_recording=harp_is_in_recording,
            use_onix=use_onix,
            conflicts=conflicts,
            sync_kwargs=sync_kwargs,
            ephys_kwargs=ephys_kwargs,
        )

        # Find depth neurons and fit preferred depth
        print("---Start finding depth neurons...---")
        print("Find depth neurons...")
        neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
            trials_df=trials_df_all,
            neurons_ds=neurons_ds,
            rs_thr=0.01,
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
            rs_thr=rs_thr,
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
            rs_thr=rs_thr,
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
            rs_thr=rs_thr,
            depth_min=0.02,
            depth_max=20,
            niter=10,
            min_sigma=0.5,
            k_folds=5,
        )
        if do_rf:
            # Fit gaussian blob to neuronal activity
            print("---Start fitting 2D gaussian blob...---")
            neurons_df, neurons_ds = fit_gaussian_blob.fit_rs_of_tuning(
                trials_df=trials_df_all,
                neurons_df=neurons_df,
                neurons_ds=neurons_ds,
                model="gaussian_2d",
                choose_trials=None,
                rs_thr=rs_thr,
                param_range={
                    "rs_min": 0.005,
                    "rs_max": 5,
                    "of_min": 0.03,
                    "of_max": 3000,
                },
                niter=10,
                min_sigma=0.25,
            )

            # Fit gaussian blob cross validation for closed_loop only
            neurons_df, neurons_ds = fit_gaussian_blob.fit_rs_of_tuning(
                trials_df=trials_df_all,
                neurons_df=neurons_df,
                neurons_ds=neurons_ds,
                model="gaussian_2d",
                choose_trials="even",
                closedloop_only=True,
                rs_thr=rs_thr,
                param_range={
                    "rs_min": 0.005,
                    "rs_max": 5,
                    "of_min": 0.03,
                    "of_max": 3000,
                },
                niter=10,
                min_sigma=0.25,
            )
            # Save neurons_df
            neurons_df.to_pickle(neurons_ds.path_full)

            # Fit with additive RS-OF model
            print("---Start fitting additive RS-OF model...---")
            neurons_df, neurons_ds = fit_gaussian_blob.fit_rs_of_tuning(
                trials_df=trials_df_all,
                neurons_df=neurons_df,
                neurons_ds=neurons_ds,
                model="gaussian_additive",
                choose_trials=None,
                rs_thr=rs_thr,
                param_range={
                    "rs_min": 0.005,
                    "rs_max": 5,
                    "of_min": 0.03,
                    "of_max": 3000,
                },
                niter=10,
                min_sigma=0.25,
            )
            # Save neurons_df
            neurons_df.to_pickle(neurons_ds.path_full)

            # Fit with OF-only model
            print("---Start fitting OF-only model...---")
            neurons_df, neurons_ds = fit_gaussian_blob.fit_rs_of_tuning(
                trials_df=trials_df_all,
                neurons_df=neurons_df,
                neurons_ds=neurons_ds,
                model="gaussian_OF",
                choose_trials=None,
                rs_thr=rs_thr,
                param_range={
                    "rs_min": 0.005,
                    "rs_max": 5,
                    "of_min": 0.03,
                    "of_max": 3000,
                },
                niter=10,
                min_sigma=0.25,
            )
            # Save neurons_df
            neurons_df.to_pickle(neurons_ds.path_full)

            # Regenerate sphere stimuli
            print("---RF analysis...---")
            print("Generating sphere stimuli...")
            frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets=None,
                recording_type="behaviour",
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                resolution=5,
                sync_kwargs=sync_kwargs,
                use_onix=use_onix,
                harp_is_in_recording=harp_is_in_recording,
                ephys_kwargs=ephys_kwargs,
            )

            print("Fitting RF...")
            (
                coef,
                r2,
                best_reg_xys,
                best_reg_depths,
            ) = spheres.fit_3d_rfs_hyperparam_tuning(
                imaging_df_all,
                frames_all[:, :, int(frames_all.shape[2] // 2) :],
                reg_xys=[10, 20, 40, 80, 160, 320, 640],
                reg_depths=[10, 20, 40, 80, 160, 320, 640],
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
                shift_stims=2,
                use_col="dffs",
                k_folds=5,
                validation=False,
            )

            for col in ["rf_coef", "rf_rsq", "rf_coef_ipsi", "rf_rsq_ipsi"]:
                neurons_df[col] = [[np.nan]] * len(neurons_df)

            for i, _ in neurons_df.iterrows():
                neurons_df.at[i, "rf_coef"] = coef[:, :, i]
                neurons_df.at[i, "rf_coef_ipsi"] = coef_ipsi[:, :, i]
                neurons_df.at[i, "rf_rsq"] = r2[i, :]
                neurons_df.at[i, "rf_rsq_ipsi"] = r2_ipsi[i, :]

            # Save neurons_df
            neurons_df.to_pickle(neurons_ds.path_full)

            # Update neurons_ds on flexilims
            neurons_ds.update_flexilims(mode="update")
            print("---Analysis finished. Neurons_df saved.---")

    # Plot basic plots
    print("---Start basic vis plotting...---")
    print("Plotting Depth responses...")
    kwargs = dict(
        rs_thr=0.01,
        rs_curve=dict(speed_min=0.001, speed_max=1, speed_thr=0.001),
        RS_OF_matrix_log_range={
            "rs_bin_log_min": -2.5,
            "rs_bin_log_max": 2.5,
            "rs_bin_num": 11,
            "of_bin_log_min": -1.5,
            "of_bin_log_max": 3.5,
            "of_bin_num": 11,
            "log_base": 10,
        },
    )
    basic_vis_plots.basic_vis_session(
        neurons_df=neurons_df, trials_df=trials_df_all, neurons_ds=neurons_ds, **kwargs
    )
    if do_rf:
        # Plot all ROI RFs
        print("Plotting RFs...")
        depth_list = find_depth_neurons.find_depth_list(trials_df_all)
        coef = np.stack(neurons_df["rf_coef"], axis=2)
        sta_plots.basic_vis_sta_session(
            coef=coef,
            neurons_df=neurons_df,
            trials_df=trials_df_all,
            depth_list=depth_list,
            frames=frames_all,
            save_dir=neurons_ds.path_full.parent,
            fontsize_dict={"title": 10, "tick": 10, "label": 10},
        )
    print("---Plotting finished. ---")


if __name__ == "__main__":
    sess_new = dict(
        session_name="BRYA142.5d_S20231005",
        harp_is_in_recording=False,
        return_multiunit=True,
        exp_sd=None,
        rate_bin=50e-3,
        rs_thr=0.0001,
        conflicts="overwrite",
        # unit_list=[12, 14, 16],
    )

    sess_old = dict(
        session_name="BRAC6692.4a_S20220831",
        harp_is_in_recording=True,
        return_multiunit=False,
        exp_sd=150e-3,
        rate_bin=100e-3,
        rs_thr=0.001,
        conflicts="overwrite",
        # unit_list=[23, 62, 63, 73, 117, 206, 242, 258, 270, 282],
        unit_list=[63, 73, 117],
        do_rf=False,
    )
    main(
        project="blota_onix_pilote",
        sync_kwargs=dict(frame_detection_height=0.05, detect_only=False),
        use_onix=False,
        **sess_new,
        # unit_list=[0, 14],
    )
