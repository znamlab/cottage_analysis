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
        protocol_base(str): protocol base name. Default "SpheresPermTubeReward".
        anatomical_only(bool): whether to only use anatomical datasets. Default True.
        ast_neuropil(bool): whether to use ASt neuropil correction. Default False.
        use_annotated(bool): Filter s2p dataset by "annotated=True", default False
    """
    print(f"   ------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------")
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
    if neurons_ds.path_full.exists():
        # If there is a neurons_df, load it to overwrite only the parts that we run in
        # this instance of the pipeline
        neurons_df = pd.read_pickle(neurons_ds.path_full)
    else:
        neurons_df = None
    # Synchronisation
    print("---Start synchronisation...---")
    if protocol_base == "SpheresTubeMotor":
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

    # Fit gaussian blob to neuronal activity

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
            "log_amplitude_max": 10.0,
        },
        n_starts=5,
        # top_k=3,
        min_sigma=0.25,
        run_openloop_only=False,
        file_special_sfx=special_sfx_base,
        max_rs2motor_diff=max_rs2motor_diff,
    )

    to_do = [
        ("gaussian_2d", None, 1),
        ("gaussian_2d", "even", 1),
        ("gaussian_2d", None, 5),
        ("gaussian_additive", None, 1),
        ("gaussian_additive", None, 5),
        ("gaussian_OF", None, 1),
        ("gaussian_OF", None, 5),
        ("gaussian_ratio", None, 1),
        ("gaussian_ratio", None, 5),
        ("gaussian_RS", None, 1),
        ("gaussian_RS", None, 5),
        ("gaussian_multiplicative", None, 1),
        ("gaussian_multiplicative", None, 5),
    ]

    for model, trials, k_folds in to_do:
        name = f"{session_name}_{model}"
        if trials is not None:
            name += "_crossval"
        name += f"_k{k_folds}"
        name += "_torch"
        print(f"Fitting {model}...")
        out = pipeline_utils.load_and_fit_torch(
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
        print("---RS/OF fit finished...---")


if __name__ == "__main__":
    defopt.run(main)
