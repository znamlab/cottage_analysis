import numpy as np
import pandas as pd
import defopt
import warnings
import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    treadmill,
)
from cottage_analysis.analysis.spheres import rf_fitting
from cottage_analysis.analysis.spheres.rf_analysis import fit_rf_preferred_depth
from cottage_analysis.pipelines import pipeline_utils
from znamutils import slurm_it

CONDA_ENV = "v1_depth_map"


def main(
    project: str,
    session_name: str,
    *,
    protocol_base: str = "SpheresPermTubeReward",
    photodiode_protocol: int = 5,
    anatomical_only: bool = True,
    use_annotated: bool = True,
):
    """
    Run RF analysis only for a given session and protocol, saving results to a separate file.

    Args:
        project (str): project name (e.g. 'ccyp_l5_3d_vision')
        session_name (str): session name (e.g. 'PZAG22.1b_S20260303b')
        protocol_base (str): protocol base name. Default "SpheresPermTubeReward".
        photodiode_protocol (int): 2 or 5.
        anatomical_only (bool): whether to only use anatomical datasets. Default True.
        use_annotated (bool): Filter s2p dataset by "annotated=True", default True
    """
    print(
        f"--------------------------------------------------\n"
        f"Starting RF-only analysis for {session_name} ({protocol_base})\n"
        f"--------------------------------------------------"
    )

    filter_rois = {}
    if anatomical_only:
        print("Only using anatomical datasets...")
        filter_rois["anatomical_only"] = 3
    if use_annotated:
        filter_rois["annotated"] = True
        exclude_datasets = None
    else:
        exclude_datasets = {"annotated": True}

    filter_traces = dict(**filter_rois)
    filter_traces[
        "ast_neuropil"
    ] = False  # Default to False as in standard pipeline config

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts="skip",
    )

    # Load existing neurons_df to avoid losing other info, but we WILL NOT overwrite the main neurons_df.pickle
    if neurons_ds.path_full.exists():
        print(f"Loading existing neurons_df from {neurons_ds.path_full}")
        neurons_df = pd.read_pickle(neurons_ds.path_full).copy()
    else:
        neurons_df = None

    # Synchronisation (needed for trials_df_all and to verify ROI count)
    print("---Start synchronisation...---")
    if protocol_base == "SpheresTubeMotor":
        _, trials_df_all = treadmill.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_traces,
            exclude_datasets=exclude_datasets,
            conflicts="skip",
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
            conflicts="skip",
            recording_type="two_photon",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )

    if len(trials_df_all) > 0 and "dff_stim" in trials_df_all.columns:
        nrois = trials_df_all.dff_stim.iloc[0].shape[1]
        if neurons_df is not None and len(neurons_df) != nrois:
            print(
                f"WARNING: neurons_df has {len(neurons_df)} ROIs, but traces have {nrois} ROIs. Re-initializing."
            )
            neurons_df = None
        if neurons_df is None:
            neurons_df = pd.DataFrame({"roi": np.arange(nrois)})
        neurons_df.index = np.arange(len(neurons_df), dtype=int)
    else:
        raise RuntimeError("No trials/traces found. Cannot run RF analysis.")

    # Get frame rate
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

    if protocol_base == "SpheresTubeMotor":
        special_sfx_base = "_treadmill"
    else:
        special_sfx_base = ""

    print("---RF analysis...---")
    print("Generating sphere stimuli...")
    if "closed_loop" in trials_df_all.columns:
        closed_loops = trials_df_all["closed_loop"].unique()
    else:
        closed_loops = [True]

    for is_closedloop in closed_loops:
        if is_closedloop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        if is_multidepth:
            sfx += "_multidepth"
        if protocol_base == "SpheresTubeMotor":
            sfx += "_treadmill"

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
        # Same calls as `analysis_pipeline.fit_session_rfs`. The grid is searched
        # in-process by the eigendecomposition solver of `rf_fitting`.
        (
            coef,
            r2,
            best_reg_xys,
            best_reg_depths,
        ) = rf_fitting.fit_3d_rfs_hyperparam_tuning(
            imaging_df_all,
            frames_all[..., int(frames_all.shape[-1] // 2) :],
            reg_xys=rf_fitting.DEFAULT_REG_GRID,
            reg_depths=rf_fitting.DEFAULT_REG_GRID,
            shift_stim=2,
            use_col="dffs",
            k_folds=5,
        )

        print(f"Fitting ipsi RF{sfx}...")
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
        )

        for col in [
            f"rf_coef{sfx}",
            f"rf_rsq{sfx}",
            f"rf_coef_ipsi{sfx}",
            f"rf_rsq_ipsi{sfx}",
        ]:
            neurons_df[col] = [[np.nan]] * len(neurons_df)

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

    depth_list = find_depth_neurons.find_depth_list(trials_df_all)
    print(f"Fitting RF preferred depth{sfx} (Gaussian across depths)...")
    # `fit_rf_preferred_depth` reads and writes the unsuffixed pipeline columns
    # (`rf_coef_closedloop[_multidepth]`). Run it on a scratch dataframe holding the
    # coefficients of this protocol under that name, then copy the results back with
    # the protocol suffix, so closed-loop columns of neurons_df are never touched.
    base_sfx = "_closedloop_multidepth" if is_multidepth else "_closedloop"
    closedloop_sfx = base_sfx + special_sfx_base
    scratch = pd.DataFrame(
        {f"rf_coef{base_sfx}": neurons_df[f"rf_coef{closedloop_sfx}"]}
    )
    # Default bounds (2 cm to 20 m) are set for the closed-loop depths; the treadmill
    # spans 2.2 mm to 36.7 m. Use the session depths, as the depth-tuning fit does.
    depth_bounds = (
        np.log(np.round(np.min(depth_list), 4)),
        np.log(np.ceil(np.max(depth_list))),
    )
    fit_rf_preferred_depth(
        scratch,
        depths=depth_list,
        is_closed_loop=1,
        use_multidepth=is_multidepth,
        depth_bounds=depth_bounds,
    )
    for col in ["rf_preferred_depth", "rf_depth_popt", "rf_depth_rsq"]:
        neurons_df[f"{col}{closedloop_sfx}"] = scratch[f"{col}{base_sfx}"].values

    # Save to a separate pickle file to avoid overwriting existing ones
    target_file = (
        neurons_ds.path_full.parent / f"neurons_df_rf_only{special_sfx_base}.pickle"
    )
    print(f"Saving separate RF tuning fitting results to {target_file}...")
    neurons_df.to_pickle(target_file)
    print("RF-only analysis finished successfully.")


@slurm_it(
    conda_env=CONDA_ENV,
    # Sync, frame regeneration and the RF fit all run in this job.
    slurm_options={
        "mem": "128G",
        "time": "24:00:00",
        "partition": "ncpu",
        "cpus-per-task": 16,
    },
    print_job_id=True,
)
def run_rf_only_session(
    project,
    session_name,
    protocol_base="SpheresTubeMotor",
    photodiode_protocol=5,
    anatomical_only=True,
    use_annotated=True,
):
    """Slurm-submittable wrapper around :func:`main`.

    Submit one job per session (see ``submit_rf_only.py``) by calling this with
    ``use_slurm=True, slurm_folder=..., scripts_name=...``.
    """
    main(
        project=project,
        session_name=session_name,
        protocol_base=protocol_base,
        photodiode_protocol=photodiode_protocol,
        anatomical_only=anatomical_only,
        use_annotated=use_annotated,
    )


if __name__ == "__main__":
    defopt.run(main)
