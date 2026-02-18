import os
import numpy as np
import pandas as pd
import defopt
from pathlib import Path
import warnings
import flexiznam as flz
from cottage_analysis.analysis import spheres
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.analysis.fit_gaussian_blob import fit_rs_of_tuning


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
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        use_slurm(bool): whether to use slurm to run the fit in the pipeline. Default False.
        run_depth_fit(bool): whether to run the depth fit. Default True.
        run_rf(bool): whether to run the rf fit. Default True.
        run_rsof_fit(bool): whether to run the rsof fit. Default True.
        run_plot(bool): whether to run the plot. Default True.
        protocol_base(str): protocol base name. Default "SpheresPermTubeReward".
        anatomical_only(bool): whether to only use anatomical datasets. Default True.
        ast_neuropil(bool): whether to use ASt neuropil correction. Default False.

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
    filter_datasets = {}
    if anatomical_only:
        print("Only using anatomical datasets...")
        filter_datasets["anatomical_only"] = 3
    if ast_neuropil:
        print("Using ASt neuropil correction...")
        filter_datasets["ast_neuropil"] = True
    else:
        filter_datasets["ast_neuropil"] = False
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    flexilims_session = flz.get_flexilims_session(project)

    # Synchronisation
    print("---Start synchronisation...---")
    _, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets=filter_datasets,
        conflicts=conflicts,
        recording_type="two_photon",
        protocol_base=protocol_base,
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )

    # Fit gaussian blob to neuronal activity
    print("---Start fitting 2D gaussian blob...---")
    outputs = []
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
        file_special_sfx="no_acc",
    )

    to_do = [
        ("gaussian_2d", None, 1),
        ("gaussian_2d", None, 5),
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
            filter_datasets=filter_datasets,
            max_acc=1,
            **common_params,
        )
        outputs.append(out)
        print("---RS OF fit finished. Neurons_df saved.---")


if __name__ == "__main__":
    defopt.run(main)
