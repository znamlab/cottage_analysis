import os
import json
import subprocess
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import flexiznam as flz
from cottage_analysis.summary_analysis.summary_utils import concatenate_all_neurons_df
from cottage_analysis.analysis import fit_gaussian_blob
from znamutils.decorators import slurm_it
from cottage_analysis.pipelines import pipeline_utils

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECTS = ["ccyp_l5_3d_vision", "colasa_3d-vision_revisions"]
SESSIONS_TO_EXCLUDE = {
    "PZAG22.1b_S20260220": "1000 more frames than triggers in the treadmill recording"
}
MODELS = [
    "gaussian_OF",
    "gaussian_RS",
    "gaussian_additive",
    "gaussian_2d",
    "gaussian_ratio",
    "gaussian_multiplicative",
    "gaussian_product",
]

# (model, choose_trials, k_folds)
MODEL_CONFIGS = [
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


def _config_filename(model_name, choose_trials, k_folds):
    trials_sfx = f"_{choose_trials}" if choose_trials is not None else ""
    return f"fit_rs_of_tuning_{model_name}{trials_sfx}_k{k_folds}_treadmill_trial_average.pickle"


def _fit_session_one_model(
    project, session_name, model_name, choose_trials=None, k_folds=1
):
    """Fit one (model, choose_trials, k_folds) config for a single session and save the pickle file."""
    print(
        f"Fitting {model_name} (choose_trials={choose_trials}, k_folds={k_folds}) for {session_name} ({project})..."
    )

    if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
        photodiode_protocol = 2
    else:
        photodiode_protocol = 5

    neurons_ds, _, _, trials_df_all = pipeline_utils.load_session(
        project=project,
        session_name=session_name,
        photodiode_protocol=photodiode_protocol,
        regenerate_frames=False,
        filter_datasets=dict(annotated=True),
        protocol_base="SpheresTubeMotor",
    )

    is_multidepth = trials_df_all.recording_name.str.contains("multidepth")
    trials_df_all = trials_df_all[~is_multidepth]

    fit_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df_all,
        model=model_name,
        choose_trials=choose_trials,
        trial_sfx="",
        rs_thr=0.01,
        niter=5,
        k_folds=k_folds,
        max_rs2motor_diff=0.3,
        trial_average=True,
    )

    target_path = neurons_ds.path_full.with_name(
        _config_filename(model_name, choose_trials, k_folds)
    )
    fit_df.to_pickle(target_path)
    print(f"Saved to {target_path}")


def submit_fitting_array(
    sessions_by_project,
    configs=None,
    tasks_dir=None,
    conda_env="v1_depth_map",
    mem="16G",
    time="2:00:00",
    partition="ncpu",
):
    """Submit a slurm array job where each task fits one (session, config) pair.

    Args:
        sessions_by_project (dict): {project: [session_name, ...]}
        configs (list, optional): List of (model, choose_trials, k_folds) tuples.
            Defaults to MODEL_CONFIGS.
        tasks_dir (str or Path, optional): Directory to write tasks JSON and sbatch
            script. Defaults to ~/.cache/fit_rsof_array.
        conda_env (str): Conda environment to activate in each array task.
        mem (str): Memory per task.
        time (str): Wall time per task.
        partition (str): Slurm partition.

    Returns:
        str: Submitted slurm job ID.
    """
    if configs is None:
        configs = MODEL_CONFIGS

    tasks = [
        {
            "project": project,
            "session": session_name,
            "model": model_name,
            "choose_trials": choose_trials,
            "k_folds": k_folds,
        }
        for project, sessions in sessions_by_project.items()
        for session_name in sessions
        if session_name not in SESSIONS_TO_EXCLUDE
        for model_name, choose_trials, k_folds in configs
    ]

    if not tasks:
        raise ValueError("No tasks to submit after filtering excluded sessions.")

    tasks_dir = (
        Path(tasks_dir) if tasks_dir else Path.home() / ".cache" / "fit_rsof_array"
    )
    tasks_dir.mkdir(parents=True, exist_ok=True)

    tasks_file = tasks_dir / "tasks.json"
    tasks_file.write_text(json.dumps(tasks, indent=2))

    script_path = Path(__file__).resolve()
    log_dir = tasks_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=fit_rsof
#SBATCH --array=0-{len(tasks) - 1}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={log_dir}/fit_rsof_%A_%a.log

conda run -n {conda_env} python -u {script_path} --tasks-file {tasks_file}
"""
    sbatch_file = tasks_dir / "submit.sh"
    sbatch_file.write_text(sbatch_script)

    result = subprocess.run(
        ["sbatch", str(sbatch_file)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted array job {job_id} ({len(tasks)} tasks)")
    print(f"Tasks file: {tasks_file}")
    print(f"Logs: {log_dir}/fit_rsof_{job_id}_*.log")
    return job_id


@slurm_it(
    conda_env="v1_depth_map",
    slurm_options={"mem": "16G", "time": "1:00:00", "partition": "ncpu"},
    from_imports={
        "cottage_analysis.pipelines.fit_rsof_trial_average": "merge_and_concatenate_results"
    },
    print_job_id=True,
)
def merge_and_concatenate_results(sessions_by_project, configs=None):
    """Load all session-level fit pickles, rename columns, merge, and save."""
    if configs is None:
        configs = MODEL_CONFIGS

    print("Starting merge and concatenation of all sessions...")

    all_dfs = []
    for project, sessions in sessions_by_project.items():
        if not sessions:
            continue
        flexilims_session = flz.get_flexilims_session(project_id=project)

        print(f"Loading base neurons_df for project: {project}...")
        project_df = concatenate_all_neurons_df(
            flexilims_session=flexilims_session,
            session_list=sessions,
            filename="neurons_df.pickle",
            read_iscell=False,
            filter_datasets=dict(annotated=True),
        )
        project_df["project"] = project

        session_fit_dfs = []
        for session_name in sessions:
            neurons_ds = pipeline_utils.create_neurons_ds(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=project,
                conflicts="skip",
            )

            session_merged = None
            for model_name, choose_trials, k_folds in configs:
                fit_path = neurons_ds.path_full.with_name(
                    _config_filename(model_name, choose_trials, k_folds)
                )
                if not fit_path.exists():
                    print(f"Warning: {fit_path} not found.")
                    continue

                fit_df = pd.read_pickle(fit_path)
                fit_df_renamed = fit_df.rename(
                    columns={
                        col: f"{col}_treadmill_trial_average"
                        for col in fit_df.columns
                        if col != "roi"
                    }
                )

                if session_merged is None:
                    session_merged = fit_df_renamed
                else:
                    session_merged = pd.merge(
                        session_merged, fit_df_renamed, on="roi", how="inner"
                    )

            if session_merged is not None:
                session_merged["session"] = session_name
                session_fit_dfs.append(session_merged)

        if session_fit_dfs:
            project_fits = pd.concat(session_fit_dfs, ignore_index=True)
            project_df_merged = pd.merge(
                project_df, project_fits, on=["session", "roi"], how="left"
            )
            all_dfs.append(project_df_merged)
        else:
            all_dfs.append(project_df)

    if all_dfs:
        neurons_df_merged = pd.concat(all_dfs, ignore_index=True)
        output_path = Path(
            "/camp/home/blota/code/v1_depth_map/neurons_df_trial_average.pickle"
        )
        neurons_df_merged.to_pickle(output_path)
        print(f"Saved to {output_path}")
    else:
        print("No dataframes to concatenate.")


if __name__ == "__main__":
    import argparse
    import time
    import random

    parser = argparse.ArgumentParser(
        description="Run one array task: fit one (session, model) pair."
    )
    parser.add_argument("--tasks-file", required=True, help="Path to tasks JSON file.")
    args = parser.parse_args()

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # Stagger startup to avoid portalocker contention on the flexiznam token file
    print("Waiting a random time before starting task...")
    time.sleep(random.uniform(0, min(task_id * 0.1, 30)))
    with open(args.tasks_file) as f:
        tasks = json.load(f)
    print("Starting task", task_id, "of", len(tasks))
    task = tasks[task_id]
    _fit_session_one_model(
        task["project"],
        task["session"],
        task["model"],
        choose_trials=task["choose_trials"],
        k_folds=task["k_folds"],
    )
