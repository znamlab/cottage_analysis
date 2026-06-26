import os
import numpy as np
import pandas as pd
import defopt
from pathlib import Path
import warnings
import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    treadmill,
)
from cottage_analysis.analysis.spheres import rf_fitting
from cottage_analysis.pipelines import pipeline_utils


def main(
    project,
    session_name,
    conflicts="skip",
    photodiode_protocol=5,
    use_slurm=False,
    run_depth_fit=True,
    run_rf=True,
    run_rsof_fit=True,
    run_plot=True,
    protocol_base: str = "SpheresPermTubeReward",
    anatomical_only=True,
    ast_neuropil=False,
    use_annotated=False,
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        use_slurm(bool): whether to use slurm to run the fit in the pipeline. Default
             False.
        run_depth_fit(bool): whether to run the depth fit. Default True.
        run_rf(bool): whether to run the rf fit. Default True.
        run_rsof_fit(bool): whether to run the rsof fit. Default True.
        run_plot(bool): whether to run the plot. Default True.
        protocol_base(str): protocol base name. Default "SpheresPermTubeReward".
        anatomical_only(bool): whether to only use anatomical datasets. Default True.
        ast_neuropil(bool): whether to use ASt neuropil correction. Default False.
        use_annotated(bool): Filter s2p dataset by "annotated=True", default False
    """
    print(
        f"   ------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------"
    )
    print(f"Using {protocol_base}")
    if use_slurm:
        slurm_folder = Path(os.path.expanduser(f"~/slurm_logs"))
        slurm_folder.mkdir(exist_ok=True)
        slurm_folder = Path(slurm_folder / f"{session_name}")
        slurm_folder.mkdir(exist_ok=True)
    else:
        slurm_folder = None
    filter_rois = {}
    if anatomical_only:
        print("Only using anatomical datasets...")
        filter_rois["anatomical_only"] = 3
    if use_annotated:
        filter_rois["annotated"] = True
        exclude_datasets = None
    else:
        exclude_datasets = {"annotated": True}
    # Traces can be filtered by the same attributes as rois but have ASt too
    filter_traces = dict(**filter_rois)
    if ast_neuropil:
        print("Using ASt neuropil correction...")
        filter_traces["ast_neuropil"] = True
    else:
        filter_traces["ast_neuropil"] = False

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
        return
    if neurons_ds.path_full.exists():
        # If there is a neurons_df, load it to overwrite only the parts that we run in
        # this instance of the pipeline
        neurons_df = pd.read_pickle(neurons_ds.path_full)
    else:
        neurons_df = None
    # Synchronisation
    print("---Start synchronisation...---")
    if protocol_base == "SpheresTubeMotor":
        run_rf = False
        _, trials_df_all = treadmill.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_traces,
            exclude_datasets=exclude_datasets,
            conflicts=conflicts,
            recording_type="two_photon",
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )
    else:
        _, trials_df_all = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_traces,
            exclude_datasets=exclude_datasets,
            conflicts=conflicts,
            recording_type="two_photon",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )

    # Add trial number to flexilims
    if protocol_base == "SpheresTubeMotor":
        trial_no = len(trials_df_all)
        flz.update_entity(
            "session",
            name=session_name,
            mode="update",
            attributes={
                "treadmill_trials": trial_no,
            },
            flexilims_session=flexilims_session,
        )
    else:
        trial_no_closedloop = len(trials_df_all[trials_df_all["closed_loop"] == 1])
        trial_no_openloop = len(trials_df_all[trials_df_all["closed_loop"] == 0])
        ndepths = len(trials_df_all["depth"].unique())
        flz.update_entity(
            "session",
            name=session_name,
            mode="update",
            attributes={
                "closedloop_trials": trial_no_closedloop,
                "openloop_trials": trial_no_openloop,
                "ndepths": ndepths,
            },
            flexilims_session=flexilims_session,
        )

    # Check that neurons_df matches the number of ROIs in traces
    if len(trials_df_all) > 0 and "dff_stim" in trials_df_all.columns:
        nrois = trials_df_all.dff_stim.iloc[0].shape[1]
        if neurons_df is not None and len(neurons_df) != nrois:
            print(
                f"   WARNING: neurons_df has {len(neurons_df)} ROIs, but traces have {nrois} ROIs."
            )
            print(
                f"   Re-initializing neurons_df to match traces (overwriting previous state)."
            )
            neurons_df = None

        if neurons_df is None:
            print(f"   Initializing neurons_df with {nrois} ROIs.")
            neurons_df = pd.DataFrame({"roi": np.arange(nrois)})

        # Enforce that index and ROI array positions match perfectly
        neurons_df.index = np.arange(len(neurons_df), dtype=int)
        print(
            f"   Saving neurons_df (ensuring 0-based contiguous index) to {neurons_ds.path_full}"
        )
        assert all(~np.isnan(neurons_df["roi"].values)), "ROIs in neurons_df are NaN."
        neurons_df.to_pickle(neurons_ds.path_full)
    else:
        print("   WARNING: No trials or traces found to verify ROI count.")

    suite2p_datasets = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        project_id=project,
        flexilims_session=flexilims_session,
        return_dataseries=False,
        filter_datasets=filter_rois,
        exclude_datasets=exclude_datasets,
    )
    suite2p_dataset = suite2p_datasets[0]
    frame_rate = suite2p_dataset.extra_attributes["fs"]

    is_multidepth = "multidepth" in protocol_base
    if is_multidepth:
        run_depth_fit = False
        run_rsof_fit = False

    # Treadmill only parameter
    max_rs2motor_diff = 0.3 if protocol_base == "SpheresTubeMotor" else None
    if protocol_base == "SpheresTubeMotor":
        special_sfx_base = "_treadmill"
    else:
        special_sfx_base = ""
    if run_depth_fit:
        # finished = pipeline_utils.save_finish_time(finished,
        # col="depth_fit_started")
        depth_fit_params = {
            "depth_min": 0.02,
            "depth_max": 20,
            "niter": 10,
            "min_sigma": 0.5,
        }
        if protocol_base == "SpheresTubeMotor":
            # With treadmill, depth min and max can be lot smaller/larger
            depth_fit_params["depth_max"] = np.ceil(trials_df_all.depth.max())
            depth_fit_params["depth_min"] = np.round(trials_df_all.depth.min(), 4)

        # Find depth neurons and fit preferred depth
        print("---Start finding depth neurons...---")
        print("Find depth neurons...")
        neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
            trials_df=trials_df_all,
            neurons_ds=neurons_ds,
            neurons_df=neurons_df,
            rs_thr=None,
            alpha=0.05,
            special_sfx=special_sfx_base,
            max_rs2motor_diff=max_rs2motor_diff,
        )

        print("Fit preferred depth...")
        # Find preferred depth for all data & running (current frame > 5cm/s) &
        # not-running (previous 14+current frame < 5cm/s)
        for rs_thr, rs_thr_max, still_only, still_time, special_sfx in zip(
            [None, 0.05, None],
            [None, None, 0.05],
            [False, False, True],
            [0, 0, 1],
            ["", "_running", "_notrunning"],
        ):
            print(f"Fit preferred depth{special_sfx}{special_sfx_base}...")
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
                special_sfx=special_sfx + special_sfx_base,
                max_rs2motor_diff=max_rs2motor_diff,
            )

            # Find preferred depth of closed loop with half the data for plotting
            # purposes
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
                special_sfx=special_sfx + special_sfx_base,
                max_rs2motor_diff=max_rs2motor_diff,
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
                special_sfx=special_sfx + special_sfx_base,
                max_rs2motor_diff=max_rs2motor_diff,
            )

        # Save neurons_df
        assert all(~np.isnan(neurons_df["roi"].values)), "ROIs in neurons_df are NaN."
        neurons_df.to_pickle(neurons_ds.path_full)
        # Save a copy with special_sfx_base in the name
        target_file = neurons_ds.path_full.with_name(
            f"neurons_df_for_depthfit{special_sfx_base}.pickle"
        )
        print(f"Saving separate depth tuning fitting files in {target_file}...")
        neurons_df.to_pickle(target_file)

        # Update neurons_ds on flexilims
        neurons_ds.update_flexilims(mode="update")
        print("Depth tuning fitting finished. Neurons_df saved.")

    # Regenerate sphere stimuli
    if run_rf:
        print("---RF analysis...---")
        # finished = pipeline_utils.save_finish_time(finished, col="rf_started")
        print("Generating sphere stimuli...")
        for is_closedloop in trials_df_all["closed_loop"].unique():
            if is_closedloop:
                sfx = "_closedloop"
            else:
                sfx = "_openloop"
            if is_multidepth:
                sfx += "_multidepth"

            frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets=filter_traces,
                exclude_datasets=exclude_datasets,
                recording_type="two_photon",
                is_closedloop=is_closedloop,
                is_multidepth=is_multidepth,
                protocol_base=protocol_base,
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                verbose=False,
                resolution=5,
            )

            print(f"Fitting RF{sfx}...")
            (
                coef,
                r2,
                best_reg_xys,
                best_reg_depths,
            ) = rf_fitting.fit_3d_rfs_hyperparam_tuning(
                imaging_df_all,
                frames_all[..., int(frames_all.shape[-1] // 2) :],
                reg_xys=np.geomspace(2.5, 10240, 13),
                reg_depths=np.geomspace(2.5, 10240, 13),
                shift_stim=2,
                use_col="dffs",
                k_folds=5,
                tune_separately=True,
                validation=False,
            )

            print("Fitting ipsi RF...")
            (
                coef_ipsi,
                r2_ipsi,
            ) = rf_fitting.fit_3d_rfs_ipsi(
                imaging_df_all,
                frames_all[..., : int(frames_all.shape[-1] // 2)],
                best_reg_xys,
                best_reg_depths,
                shift_stim=2,
                use_col="dffs",
                k_folds=5,
                validation=False,
            )

            if not run_depth_fit:
                assert (
                    len(neurons_df) == coef.shape[2]
                ), f"neurons_df count {len(neurons_df)} does not match coef count {coef.shape[2]}"
            for col in [
                f"rf_coef{sfx}",
                f"rf_rsq{sfx}",
                f"rf_coef_ipsi{sfx}",
                f"rf_rsq_ipsi{sfx}",
            ]:
                neurons_df[col] = [[np.nan]] * len(neurons_df)

            # Enforce that index and ROI array positions match perfectly
            assert np.all(np.diff(neurons_df.index) == 1), "Index is not contiguous"
            assert neurons_df.index[0] == 0, "Index does not start at 0"

            for i, _ in neurons_df.iterrows():
                neurons_df.at[i, f"rf_coef{sfx}"] = coef[:, :, i].copy()
                neurons_df.at[i, f"rf_coef_ipsi{sfx}"] = coef_ipsi[:, :, i].copy()
                neurons_df.at[i, f"rf_rsq{sfx}"] = r2[i, :].copy()
                neurons_df.at[i, f"rf_rsq_ipsi{sfx}"] = r2_ipsi[i, :].copy()
                neurons_df.at[i, f"rf_reg_xy{sfx}"] = best_reg_xys[i]
                neurons_df.at[i, f"rf_reg_depth{sfx}"] = best_reg_depths[i]

        # Fit RF preferred depth using Gaussian fit across depths
        from cottage_analysis.analysis.spheres.rf_analysis import fit_rf_preferred_depth

        depth_list = find_depth_neurons.find_depth_list(trials_df_all)
        print(f"Fitting RF preferred depth{sfx} (Gaussian across depths)...")
        fit_rf_preferred_depth(
            neurons_df,
            depths=depth_list,
            is_closed_loop=1,
            use_multidepth=is_multidepth,
        )

        # Save neurons_df
        assert all(~np.isnan(neurons_df["roi"].values)), "ROIs in neurons_df are NaN."
        neurons_df.to_pickle(neurons_ds.path_full)
        # Also save a copy with special_sfx_base in the name
        target_file = neurons_ds.path_full.with_name(
            f"neurons_df_for_rf{special_sfx_base}.pickle"
        )
        print(f"Saving separate RF tuning fitting files in {target_file}...")
        neurons_df.to_pickle(target_file)

        # Update neurons_ds on flexilims
        # neurons_ds.update_flexilims(mode="update")
        print("RF fitting finished. Neurons_df saved.")

    # Fit gaussian blob to neuronal activity
    if run_rsof_fit:
        print("---Start fitting 2D gaussian blob...---")
        outputs = []
        special_sfx_base = "_treadmill" if protocol_base == "SpheresTubeMotor" else ""
        common_params = dict(
            rs_thr=0.01,
            param_range={
                "rs_min": 0.005,
                "rs_max": 5,
                "of_min": 0.03,
                "of_max": 3000,
            },
            niter=10,
            min_sigma=0.25,
            run_openloop_only=False,
            file_special_sfx=special_sfx_base,
            max_rs2motor_diff=max_rs2motor_diff,
        )

        to_do = [
            ("gaussian_2d", None, 1),
            ("gaussian_2d", "even", 1),
            ("gaussian_additive", None, 1),
            ("gaussian_OF", None, 1),
            ("gaussian_2d", None, 5),
            ("gaussian_additive", None, 5),
            ("gaussian_OF", None, 5),
            ("gaussian_ratio", None, 1),
            ("gaussian_ratio", None, 5),
            ("gaussian_RS", None, 1),
            ("gaussian_RS", None, 5),
            ("gaussian_multiplicative", None, 1),
            ("gaussian_multiplicative", None, 5),
            ("gaussian_product", None, 1),
            ("gaussian_product", None, 5),
        ]

        for model, trials, k_folds in to_do:
            name = f"{session_name}_{model}"
            if trials is not None:
                name += "_crossval"
            name += f"_k{k_folds}"
            print(f"Fitting {model}...")
            out = pipeline_utils.load_and_fit(
                project,
                session_name,
                photodiode_protocol,
                model=model,
                choose_trials=trials,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=name,
                k_folds=k_folds,
                filter_datasets=filter_traces,
                protocol_base=protocol_base,
                **common_params,
            )
            outputs.append(out)
            print("---RS OF fit finished. Neurons_df saved.---")

    # After the run_rsof_fit and run_depth_fit/run_rf blocks
    if run_rsof_fit or run_depth_fit or run_rf:
        print("---Merging all fit dataframes...---")
        # Only use SLURM and dependencies if we just ran new fits
        use_slurm_merge = use_slurm if run_rsof_fit else 0
        job_dependency = outputs if (run_rsof_fit and use_slurm) else None
        exclude_keywords = ["recording", "openclosed", "openloop"] + (
            ["treadmill"] if special_sfx_base == "" else []
        )

        out = pipeline_utils.merge_fit_dataframes(
            project,
            session_name,
            use_slurm=use_slurm_merge,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
            scripts_name=f"{session_name}{special_sfx_base}_merge_fit_dataframes",
            conflicts=conflicts,
            prefix="fit_rs_of_tuning_",
            suffix=special_sfx_base,
            exclude_keywords=exclude_keywords,
            include_keywords=[],
            target_column_suffix=special_sfx_base,
            filetype=".pickle",
            target_filename="neurons_df.pickle",
        )
        if use_slurm_merge:
            print("Job started")
        else:
            print("---Analysis finished. Neurons_df saved.---")

    # Plot basic plots
    if run_plot:
        print("---Start basic vis plotting...---")
        if run_rsof_fit:
            job_dependency = outputs if use_slurm else None
        else:
            job_dependency = None
        pipeline_utils.run_basic_plots(
            project,
            session_name,
            photodiode_protocol,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
            filter_datasets=filter_traces,
            scripts_name=f"{session_name}_basic_vis_plots",
        )
        print("---Plotting finished. ---")


if __name__ == "__main__":
    defopt.run(main)
