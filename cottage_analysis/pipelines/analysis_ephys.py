import os
import numpy as np
import pandas as pd
import defopt
from pathlib import Path
import warnings
import json
import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    treadmill,
)
from cottage_analysis.analysis.spheres import rf_fitting
from cottage_analysis.pipelines import pipeline_utils
from znamutils import slurm_it


@slurm_it(
    conda_env="onix-3dvision",
    slurm_options={"time": "48:00:00", "cpus-per-task": 16, "mem": "128G"},
)
def main(
    project: str,
    session_name: str,
    *,
    conflicts: str = "skip",
    photodiode_protocol: int = 5,
    run_rsof_fit_on_separate_slurm_jobs: bool = False,
    run_depth_fit: bool = True,
    run_rf: bool = False,
    run_rsof_fit: bool = True,
    run_plot: bool = True,
    sync_kwargs: str = None,
    use_onix: bool = False,
    return_multiunit: bool = False,
    harp_is_in_recording: bool = True,
    exp_sd: float = None,
    rate_bin: float = 0.03,
    rs_thr: float = 0.0002,
    ephys_dataset_type: str = "aind_pipeline",
    filter_datasets: str = None,
    protocol_base: str = "SphereTube",
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        run_rsof_fit_on_separate_slurm_jobs(bool): whether to use slurm to run the fit in the pipeline. Default False.
        run_depth_fit(bool): whether to run the depth fit. Default True.
        run_rf(bool): whether to run the rf fit. Default True.
        run_rsof_fit(bool): whether to run the rsof fit. Default True.
        run_plot(bool): whether to run the plot. Default True.
        sync_kwargs(str): json string of kwargs for synchronization.
        use_onix(bool): whether to use onix. Default False.
        return_multiunit(bool): whether to return multiunit. Default False.
        harp_is_in_recording(bool): whether harp is in recording. Default True.
        exp_sd(float): expected standard deviation. Default 0.1.
        rate_bin(float): rate bin. Default 0.03.
        rs_thr(float): rs threshold. Default 0.0002.
        ephys_dataset_type(str): datatype for spikes. Default 'aind_pipeline'
        filter_datasets(str): json string of datasets to filter.
        protocol_base(str): protocol base name. Default "SphereTube".
    """
    unit_list = None  # removed option to select specific units
    if isinstance(sync_kwargs, str):
        sync_kwargs = json.loads(sync_kwargs)
    if isinstance(filter_datasets, str):
        filter_datasets = json.loads(filter_datasets)
    print(
        f"""
        -------------------------------------
        Start analysing {session_name}
        -------------------------------------
        """
    )
    # Print all arguements to make debugging easier
    print("Arguments:")
    for arg, value in locals().items():
        print(f"{arg}: {value}")
    print("")
    if filter_datasets is None:
        filter_datasets = {}
    frame_rate = 1 / rate_bin

    ephys_kwargs = dict(
        return_multiunit=return_multiunit,
        exp_sd=exp_sd,
        rate_bin=rate_bin,
        unit_list=unit_list,
        dataset_type=ephys_dataset_type,
    )
    if run_rsof_fit_on_separate_slurm_jobs:
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
    if (neurons_ds.get_flexilims_entry() is not None) and conflicts == "skip":
        print(f"Session {session_name} already processed... reading saved data...")
        return

    if neurons_ds.path_full.exists() and (not run_depth_fit):
        # If there is a neurons_df, load it to overwrite only the parts that we run
        # in this instance of the pipeline
        print("Reloading neurons_df")
        neurons_df = pd.read_pickle(neurons_ds.path_full)
    else:
        neurons_df = None
    # Synchronisation
    print("")
    print("---Start synchronisation...---")
    if protocol_base in ["SpheresTubeMotor", "SphereTubeTreadmill"]:
        run_rf = False
        _, trials_df_all = treadmill.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_datasets,
            conflicts="skip",
            recording_type="behaviour",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            harp_is_in_recording=harp_is_in_recording,
            use_onix=use_onix,
            sync_kwargs=sync_kwargs,
            ephys_kwargs=ephys_kwargs,
        )
    else:
        _, trials_df_all = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_datasets,
            conflicts="skip",
            recording_type="behaviour",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            harp_is_in_recording=harp_is_in_recording,
            use_onix=use_onix,
            sync_kwargs=sync_kwargs,
            ephys_kwargs=ephys_kwargs,
        )

    # Add trial number to flexilims
    if protocol_base in ["SpheresTubeMotor", "SphereTubeTreadmill"]:
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

    is_multidepth = "multidepth" in protocol_base
    if is_multidepth:
        run_depth_fit = False
        run_rsof_fit = False

    if run_depth_fit:
        # finished = pipeline_utils.save_finish_time(finished,
        # col="depth_fit_started")
        depth_fit_params = {
            "depth_min": 0.02,
            "depth_max": 20,
            "niter": 10,
            "min_sigma": 0.5,
        }
        if protocol_base in ["SpheresTubeMotor", "SphereTubeTreadmill"]:
            special_sfx_base = "_treadmill"
            # With treadmill, depth min and max can be lot smaller/larger
            depth_fit_params["depth_max"] = np.ceil(trials_df_all.depth.max())
            depth_fit_params["depth_min"] = np.round(trials_df_all.depth.min(), 4)
        else:
            special_sfx_base = ""

        # Find depth neurons and fit preferred depth
        print("")
        print("---Start finding depth neurons...---")
        print("Find depth neurons...")
        neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
            trials_df=trials_df_all,
            neurons_ds=neurons_ds,
            neurons_df=neurons_df,
            rs_thr=None,
            alpha=0.05,
            special_sfx=special_sfx_base,
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
            )

        # Save neurons_df
        neurons_df.to_pickle(neurons_ds.path_full)

        # Update neurons_ds on flexilims
        neurons_ds.update_flexilims(mode="update")
        print("Depth tuning fitting finished. Neurons_df saved.")

    # Fit gaussian blob to neuronal activity
    if run_rsof_fit:
        print("---Start fitting 2D gaussian blob...---")
        outputs = []
        if protocol_base in ["SpheresTubeMotor", "SphereTubeTreadmill"]:
            special_sfx_base = "_treadmill"
        else:
            special_sfx_base = ""
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
            recording_type="behaviour",
            protocol_base=protocol_base,
            filter_datasets=filter_datasets,
            use_slurm=run_rsof_fit_on_separate_slurm_jobs,
            slurm_folder=slurm_folder,
            ephys_kwargs=ephys_kwargs,
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
                scripts_name=name,
                k_folds=k_folds,
                **common_params,
            )
            outputs.append(out)
            if run_rsof_fit_on_separate_slurm_jobs:
                print(f"Started job {out}")
            else:
                print("---RS OF fit finished. Neurons_df saved.---")

        # Merge fit dataframes
        job_dependency = outputs if run_rsof_fit_on_separate_slurm_jobs else None
        out = pipeline_utils.merge_fit_dataframes(
            project,
            session_name,
            use_slurm=run_rsof_fit_on_separate_slurm_jobs,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
            scripts_name=f"{session_name}_merge_fit_dataframes",
            conflicts=conflicts,
            prefix="fit_rs_of_tuning_",
            suffix=special_sfx_base,
            exclude_keywords=["recording", "openclosed", "openloop"],
            include_keywords=[],
            target_column_suffix=special_sfx_base,
            filetype=".pickle",
            target_filename="neurons_df.pickle",
        )
        print("---Analysis finished. Neurons_df saved.---")

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
                filter_datasets=filter_datasets,
                recording_type="behaviour",
                is_closedloop=is_closedloop,
                is_multidepth=is_multidepth,
                protocol_base=protocol_base,
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                verbose=False,
                resolution=5,
                sync_kwargs=sync_kwargs,
                use_onix=use_onix,
                harp_is_in_recording=harp_is_in_recording,
                ephys_kwargs=ephys_kwargs,
            )

            print(f"Fitting RF{sfx}...")
            (
                coef,
                r2,
                best_reg_xys,
                best_reg_depths,
            ) = rf_fitting.fit_3d_rfs_hyperparam_tuning(
                imaging_df_all,
                frames_all[..., int(frames_all.shape[2] // 2) :],
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
                frames_all[..., : int(frames_all.shape[2] // 2)],
                best_reg_xys,
                best_reg_depths,
                shift_stim=2,
                use_col="dffs",
                k_folds=5,
                validation=False,
            )

            if not run_depth_fit:
                neurons_df = pd.read_pickle(neurons_ds.path_full)
            for col in [
                f"rf_coef{sfx}",
                f"rf_rsq{sfx}",
                f"rf_coef_ipsi{sfx}",
                f"rf_rsq_ipsi{sfx}",
            ]:
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

        # Update neurons_ds on flexilims
        # neurons_ds.update_flexilims(mode="update")
        print("RF fitting finished. Neurons_df saved.")

    if (run_depth_fit or run_rf) and not run_rsof_fit:
        special_sfx_base = "_treadmill" if protocol_base == "SpheresTubeMotor" else ""
        try:
            # Merge fit dataframes
            out = pipeline_utils.merge_fit_dataframes(
                project,
                session_name,
                use_slurm=0,
                slurm_folder=slurm_folder,
                job_dependency=None,
                scripts_name=f"{session_name}_merge_fit_dataframes",
                conflicts=conflicts,
                prefix="fit_rs_of_tuning_",
                suffix=special_sfx_base,
                exclude_keywords=["recording", "openclosed", "openloop"],
                include_keywords=[],
                target_column_suffix=special_sfx_base,
                filetype=".pickle",
                target_filename="neurons_df.pickle",
            )
        except TypeError:
            print("No rsof dataframe to merge. Skipping")
        print("---Analysis finished. Neurons_df saved.---")

    # Plot basic plots
    if run_plot:
        print("---Start basic vis plotting...---")
        if run_rsof_fit:
            job_dependency = outputs if run_rsof_fit_on_separate_slurm_jobs else None
        else:
            job_dependency = None
        out = pipeline_utils.run_basic_plots(
            project,
            session_name,
            photodiode_protocol,
            use_slurm=run_rsof_fit_on_separate_slurm_jobs,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
            filter_datasets=filter_datasets,
            scripts_name=f"{session_name}_basic_vis_plots",
            protocol_base=protocol_base,
            recording_type="behaviour",
        )
        if run_rsof_fit_on_separate_slurm_jobs:
            print(f"Started plottingjob {out}")
        else:
            print("---Plotting finished. ---")


if __name__ == "__main__":
    defopt.run(main)


if False:
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
            protocol_base=protocol_base,
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
                protocol_base=protocol_base,
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
            protocol_base=protocol_base,
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
        if do_rs_of:
            # Fit gaussian blob to neuronal activity
            print("---Start fitting 2D gaussian blob...---")
            neurons_df, neurons_ds = fit_gaussian_blob.fit_rs_of_tuning(
                trials_df=trials_df_all,
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
        if do_rf:
            sfx = "_closedloop"
            # Regenerate sphere stimuli
            print("---RF analysis...---")
            print("Generating sphere stimuli...")
            frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets=None,
                recording_type="behaviour",
                protocol_base=protocol_base,
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                resolution=5,
                sync_kwargs=sync_kwargs,
                use_onix=use_onix,
                harp_is_in_recording=harp_is_in_recording,
                ephys_kwargs=ephys_kwargs,
            )

            print(f"Fitting RF{sfx}...")
            (
                coef,
                r2,
                best_reg_xys,
                best_reg_depths,
            ) = cottage_analysis.analysis.spheres.rf_fitting.fit_3d_rfs_hyperparam_tuning(
                imaging_df_all,
                frames_all[:, :, int(frames_all.shape[2] // 2) :],
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
            ) = cottage_analysis.analysis.spheres.rf_fitting.fit_3d_rfs_ipsi(
                imaging_df_all,
                frames_all[:, :, : int(frames_all.shape[2] // 2)],
                best_reg_xys,
                best_reg_depths,
                shift_stim=2,
                use_col="dffs",
                k_folds=5,
                validation=False,
            )

            for col in [
                f"rf_coef{sfx}",
                f"rf_rsq{sfx}",
                f"rf_coef_ipsi{sfx}",
                f"rf_rsq_ipsi{sfx}",
            ]:
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
                slurm_folder=None,
                job_dependency=None,
                scripts_name=f"{session_name}_merge_fit_dataframes",
            )
            neurons_ds.extra_attributes.update(neu_attr)
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
        coef = np.stack(neurons_df["rf_coef_closedloop"], axis=2)
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
