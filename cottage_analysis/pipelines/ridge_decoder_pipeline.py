import os
import defopt
import warnings
from pathlib import Path
import flexiznam as flz
from cottage_analysis.summary_analysis import get_session_list
from cottage_analysis.pipelines.ridge_decoder_utils import run_session


def main(
    project: str = "ccyp_l5_3d_vision",
    session_name: str = None,
    conflicts: str = "skip",
    photodiode_protocol: int = 5,
    anatomical_only: bool = True,
    ast_neuropil: bool = False,
    use_slurm: bool = False,
    rolling_window: float = None,
    downsample_window: float = None,
    log_transform: bool = True,
    rs_thr: float = None,
    alphas: list[float] = None,
    k_folds: int = 5,
    random_state: int = 42,
    run_neuron_subsets: bool = True,
    subset_sizes: list[int] = None,
):
    """
    Run Ridge decoder (RS/OF/depth) for sessions of a project and save results as parquet files.

    Args:
        project (str): Project name. Defaults to "ccyp_l5_3d_vision".
        session_name (str): Name of the session to run. If None, runs on all sessions in the project.
        conflicts (str): How to handle conflicts in flexilims. Default "skip".
        photodiode_protocol (int): 2 or 5. Default 5.
        anatomical_only (bool): Whether to only use anatomical datasets. Default True.
        ast_neuropil (bool): Whether to use ASt neuropil correction. Default False.
        use_slurm (bool): Whether to run each session as a separate slurm job. Default False.
        rolling_window (float): Rolling average window in seconds. None = no smoothing.
        downsample_window (float): Downsampling window in seconds. None = no downsampling.
        log_transform (bool): Whether to log-transform the target variable. Default True.
        rs_thr (float): Running speed threshold (m/s). Frames below excluded. None = no filter.
        alphas (list[float]): Ridge regularisation strengths. Default [0.01, 0.1, 1, 10, 100, 1000].
        k_folds (int): Number of cross-validation folds. Default 5.
        random_state (int): Random seed for reproducibility. Default 42.
        run_neuron_subsets (bool): Run decoder with neuron subsets and save results. Default True.
        subset_sizes (list[int]): Neuron subset sizes to test. None = auto.
    """
    flexilims_session = flz.get_flexilims_session(project_id=project)

    runs = []
    if session_name is not None:
        exp_session = flz.get_entity(
            datatype="session", name=session_name, flexilims_session=flexilims_session
        )
        recordings = flz.get_entities(
            datatype="recording",
            origin_id=exp_session["id"],
            flexilims_session=flexilims_session,
        )
        has_motor = False
        if "protocol" in recordings.columns:
            has_motor = (
                recordings["protocol"]
                .str.contains("SpheresTubeMotor|Treadmill", na=False)
                .any()
            )
        if not has_motor:
            has_motor = (
                recordings["name"]
                .str.contains("SpheresTubeMotor|Treadmill", na=False)
                .any()
            )

        has_closedloop = False
        if "protocol" in recordings.columns:
            has_closedloop = (
                recordings["protocol"]
                .str.contains("SpheresPermTubeReward", na=False)
                .any()
            )
        if not has_closedloop:
            has_closedloop = (
                recordings["name"].str.contains("SpheresPermTubeReward", na=False).any()
            )

        if has_closedloop:
            runs.append((session_name, False))
        if has_motor:
            runs.append((session_name, True))
        if not runs:
            # fallback
            runs.append((session_name, False))
    else:
        print(f"Querying all sessions for project {project}...", flush=True)
        closedloop_sessions = get_session_list.get_sessions(
            flexilims_session, exclude_openloop=False
        )
        motor_sessions = get_session_list.get_motor_session_list(flexilims_session)

        for sess in closedloop_sessions:
            runs.append((sess, False))
        for sess in motor_sessions:
            runs.append((sess, True))

    print(f"Total runs to process: {len(runs)}", flush=True)

    filter_datasets = {}
    if anatomical_only:
        filter_datasets["anatomical_only"] = 3
    if ast_neuropil:
        filter_datasets["ast_neuropil"] = True
    else:
        filter_datasets["ast_neuropil"] = False

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for i, (sess, is_treadmill) in enumerate(runs):
        sess_type = "motor" if is_treadmill else "closed-loop"
        suffix = "_motor" if is_treadmill else "_closedloop"
        print(
            f"\n========================================\n"
            f"Submitting/Processing run {i+1}/{len(runs)}: {sess} ({sess_type})\n"
            f"========================================",
            flush=True,
        )

        if use_slurm:
            slurm_folder = Path(os.path.expanduser(f"~/slurm_logs/{sess}"))
            slurm_folder.mkdir(parents=True, exist_ok=True)
        else:
            slurm_folder = None

        try:
            # Call the slurm_it decorated run_session
            run_session(
                sess=sess,
                project=project,
                conflicts=conflicts,
                photodiode_protocol=photodiode_protocol,
                filter_datasets=filter_datasets,
                rolling_window=rolling_window,
                downsample_window=downsample_window,
                log_transform=log_transform,
                rs_thr=rs_thr,
                alphas=alphas,
                k_folds=k_folds,
                random_state=random_state,
                run_neuron_subsets=run_neuron_subsets,
                subset_sizes=subset_sizes,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=f"ridge_decoder_{sess}{suffix}",
                is_treadmill=is_treadmill,
            )
        except Exception as e:
            print(
                f"Error submitting/processing session {sess} ({sess_type}): {e}",
                flush=True,
            )


if __name__ == "__main__":
    defopt.run(main, cli_options="all")
