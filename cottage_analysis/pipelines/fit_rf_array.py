"""Slurm array parallelization for RF fitting.

Provides a worker entry point (__main__) for individual Slurm array tasks, plus
orchestration functions to submit, wait for, and collect results from the array.

Two stages are supported:
  - ``hyperparam``: grid search over (reg_xy, reg_depth) combinations.
  - ``ipsi``: fit the ipsilateral side using one task per unique best
    (reg_xy, reg_depth) combination.

Usage from sbatch::

    python fit_rf_array.py --stage-dir /path/to/stage --mode hyperparam
    python fit_rf_array.py --stage-dir /path/to/stage --mode ipsi
"""

import gc
import json
import os
import pickle
import subprocess
import time as time_module
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

print = partial(print, flush=True)

CONDA_ENV = "v1_depth_map"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_inputs(stage_dir, imaging_df, frames, config):
    """Save shared inputs for array tasks to *stage_dir*."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "results").mkdir(exist_ok=True)
    (stage_dir / "logs").mkdir(exist_ok=True)

    imaging_df.to_pickle(stage_dir / "imaging_df.pkl")
    np.save(stage_dir / "frames.npy", frames)
    with open(stage_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _submit_array(
    stage_dir,
    tasks,
    conda_env=CONDA_ENV,
    mem="32G",
    time_limit="4:00:00",
    partition="ncpu",
):
    """Write *tasks.json*, generate an sbatch script, and submit the job array.

    Returns:
        str: The submitted Slurm job ID.
    """
    stage_dir = Path(stage_dir)
    tasks_file = stage_dir / "tasks.json"
    tasks_file.write_text(json.dumps(tasks, indent=2))

    script_path = Path(__file__).resolve()
    log_dir = stage_dir / "logs"
    mode = stage_dir.name  # "hyperparam" or "ipsi"

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=rf_{mode}
#SBATCH --array=0-{len(tasks) - 1}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --output={log_dir}/rf_{mode}_%A_%a.log

conda run -n {conda_env} python -u {script_path} --stage-dir {stage_dir} --mode {mode}
"""
    sbatch_file = stage_dir / "submit.sh"
    sbatch_file.write_text(sbatch_script)

    result = subprocess.run(
        ["sbatch", str(sbatch_file)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted {mode} array job {job_id} ({len(tasks)} tasks)")
    return job_id


# ---------------------------------------------------------------------------
# Public API – submit / wait / collect
# ---------------------------------------------------------------------------


def wait_for_results(stage_dir, n_tasks, poll_interval=30, timeout=86400):
    """Block until all *n_tasks* result files exist in *stage_dir/results/*.

    Args:
        stage_dir (str or Path): Directory for this stage.
        n_tasks (int): Expected number of result files.
        poll_interval (int): Seconds between checks. Default 30.
        timeout (int): Maximum seconds to wait. Default 86400 (24 h).

    Raises:
        TimeoutError: If not all results appear within *timeout*.
    """
    results_dir = Path(stage_dir) / "results"
    start = time_module.time()

    while True:
        existing = list(results_dir.glob("result_*.pkl"))
        if len(existing) >= n_tasks:
            print(f"All {n_tasks} results collected.")
            return

        elapsed = time_module.time() - start
        if elapsed > timeout:
            found_ids = {int(p.stem.split("_")[1]) for p in existing}
            missing = sorted(set(range(n_tasks)) - found_ids)
            raise TimeoutError(
                f"Timeout after {elapsed:.0f}s waiting for results. "
                f"Missing tasks: {missing}"
            )

        print(
            f"  {len(existing)}/{n_tasks} results, "
            f"elapsed {elapsed:.0f}s, waiting {poll_interval}s..."
        )
        time_module.sleep(poll_interval)


def submit_hyperparam_array(
    work_dir,
    imaging_df,
    frames,
    reg_xys,
    reg_depths,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    validation=False,
    conda_env=CONDA_ENV,
    mem="32G",
    time_limit="4:00:00",
    partition="ncpu",
):
    """Serialize inputs and submit a Slurm array for hyperparameter grid search.

    Each array task fits one ``(reg_xy, reg_depth)`` combination for all ROIs.

    Args:
        work_dir (str or Path): Top-level working directory for this RF fit run.
            A ``hyperparam`` subdirectory will be created.
        imaging_df (pd.DataFrame): Imaging dataframe.
        frames (np.ndarray): Stimulus frames (contra half).
        reg_xys (array-like): Spatial regularization values.
        reg_depths (array-like): Depth regularization values.
        shift_stim (int): Stimulus shift in frames.
        use_col (str): Column in imaging_df to use.
        k_folds (int): Number of cross-validation folds.
        validation (bool): Whether to use a validation split.
        conda_env (str): Conda environment name.
        mem (str): Memory per task.
        time_limit (str): Wall time per task.
        partition (str): Slurm partition.

    Returns:
        tuple: ``(job_id, n_tasks)``
    """
    stage_dir = Path(work_dir) / "hyperparam"

    config = dict(
        shift_stim=shift_stim,
        use_col=use_col,
        k_folds=k_folds,
        validation=validation,
    )
    _save_inputs(stage_dir, imaging_df, frames, config)

    tasks = []
    for reg_xy in reg_xys:
        for reg_depth in reg_depths:
            tasks.append({"reg_xy": float(reg_xy), "reg_depth": float(reg_depth)})

    job_id = _submit_array(stage_dir, tasks, conda_env, mem, time_limit, partition)
    return job_id, len(tasks)


def collect_hyperparam_results(
    work_dir, tune_separately=True, r2_threshold=0.01
):
    """Load results from the hyperparam array and select best hyperparameters.

    Args:
        work_dir (str or Path): Top-level working directory (same as passed to
            :func:`submit_hyperparam_array`).
        tune_separately (bool): Select best hyperparams per ROI (True) or
            globally (False).
        r2_threshold (float): R² threshold for "good neuron" counting (used when
            ``tune_separately=False``).

    Returns:
        tuple: ``(coef, r2, best_reg_xys, best_reg_depths)`` – same shapes as
        :func:`cottage_analysis.analysis.spheres.rf_fitting.fit_3d_rfs_hyperparam_tuning`.
    """
    stage_dir = Path(work_dir) / "hyperparam"
    results_dir = stage_dir / "results"

    with open(stage_dir / "tasks.json") as f:
        tasks = json.load(f)

    n_tasks = len(tasks)

    # Load first result to determine array shapes
    with open(results_dir / "result_0.pkl", "rb") as f:
        first = pickle.load(f)

    k_folds, n_features, nrois = first["coef"].shape
    n_splits = first["r2"].shape[1]

    all_coef = np.zeros((n_tasks, k_folds, n_features, nrois))
    all_r2s = np.zeros((n_tasks, nrois, n_splits))
    hyperparams = np.zeros((n_tasks, 2))

    for i in range(n_tasks):
        with open(results_dir / f"result_{i}.pkl", "rb") as f:
            result = pickle.load(f)
        all_coef[i] = result["coef"]
        all_r2s[i] = result["r2"]
        hyperparams[i] = [result["reg_xy"], result["reg_depth"]]

    if not tune_separately:
        good_neuron_percs = np.array(
            [np.mean(all_r2s[i, :, 1] > r2_threshold) for i in range(n_tasks)]
        )
        max_idx = np.argmax(good_neuron_percs)
        best_reg_xy, best_reg_depth = hyperparams[max_idx]
        print(
            f"Best param for all ROIs: "
            f"reg_xy: {best_reg_xy}, "
            f"reg_depth: {best_reg_depth}, "
            f"R2>{r2_threshold}: {good_neuron_percs[max_idx]:.4f}"
        )
        coef = all_coef[max_idx]
        r2 = all_r2s[max_idx]
        best_reg_xys = np.ones(nrois) * best_reg_xy
        best_reg_depths = np.ones(nrois) * best_reg_depth
    else:
        best_hyperparam_idxs = np.argmax(all_r2s[:, :, 1], axis=0)
        coef = np.zeros_like(all_coef[0])
        r2 = np.zeros((nrois, n_splits))
        best_reg_xys = np.zeros(nrois)
        best_reg_depths = np.zeros(nrois)
        for iroi in range(nrois):
            idx = best_hyperparam_idxs[iroi]
            best_reg_xys[iroi], best_reg_depths[iroi] = hyperparams[idx]
            coef[:, :, iroi] = all_coef[idx, :, :, iroi]
            r2[iroi, :] = all_r2s[idx, iroi, :]
            print(
                f"Best param found for ROI {iroi}: "
                f"reg_xy: {best_reg_xys[iroi]}, "
                f"reg_depth: {best_reg_depths[iroi]}"
            )

        unique_combos = np.unique(hyperparams[best_hyperparam_idxs], axis=0)
        print(
            f"Found {len(unique_combos)} unique best hyperparam combos "
            f"across {nrois} ROIs"
        )

    return coef, r2, best_reg_xys, best_reg_depths


def submit_ipsi_array(
    work_dir,
    imaging_df,
    frames,
    best_reg_xys,
    best_reg_depths,
    shift_stim=2,
    use_col="dffs",
    k_folds=5,
    validation=False,
    conda_env=CONDA_ENV,
    mem="32G",
    time_limit="4:00:00",
    partition="ncpu",
):
    """Submit a Slurm array for ipsilateral RF fitting.

    One array task is created per unique ``(reg_xy, reg_depth)`` combination
    found in *best_reg_xys* / *best_reg_depths*.

    Args:
        work_dir (str or Path): Top-level working directory.  An ``ipsi``
            subdirectory will be created.
        imaging_df (pd.DataFrame): Imaging dataframe.
        frames (np.ndarray): Stimulus frames (ipsi half).
        best_reg_xys (np.ndarray): Best spatial reg per ROI from hyperparam step.
        best_reg_depths (np.ndarray): Best depth reg per ROI from hyperparam step.
        shift_stim, use_col, k_folds, validation: Fitting parameters.
        conda_env, mem, time_limit, partition: Slurm parameters.

    Returns:
        tuple: ``(job_id, n_tasks)``
    """
    stage_dir = Path(work_dir) / "ipsi"

    config = dict(
        shift_stim=shift_stim,
        use_col=use_col,
        k_folds=k_folds,
        validation=validation,
        nrois=len(best_reg_xys),
    )
    _save_inputs(stage_dir, imaging_df, frames, config)

    # One task per unique (reg_xy, reg_depth) combination
    best_regs = np.stack([best_reg_xys, best_reg_depths], axis=1)
    unique_regs = np.unique(best_regs, axis=0)

    tasks = []
    for reg in unique_regs:
        roi_indices = np.where(np.all(best_regs == reg, axis=1))[0]
        tasks.append(
            {
                "reg_xy": float(reg[0]),
                "reg_depth": float(reg[1]),
                "roi_indices": roi_indices.tolist(),
            }
        )

    print(
        f"Ipsi fitting: {len(tasks)} unique (reg_xy, reg_depth) combos "
        f"for {len(best_reg_xys)} ROIs"
    )
    job_id = _submit_array(stage_dir, tasks, conda_env, mem, time_limit, partition)
    return job_id, len(tasks)


def collect_ipsi_results(work_dir):
    """Load ipsi array results and assemble final coefficient / R² arrays.

    Args:
        work_dir (str or Path): Top-level working directory.

    Returns:
        tuple: ``(coef_ipsi, r2_ipsi)`` with shapes
        ``(k_folds, n_features, nrois)`` and ``(nrois, n_splits)``.
    """
    stage_dir = Path(work_dir) / "ipsi"
    results_dir = stage_dir / "results"

    with open(stage_dir / "tasks.json") as f:
        tasks = json.load(f)
    with open(stage_dir / "config.json") as f:
        config = json.load(f)

    nrois = config["nrois"]

    # Load first result to determine shapes
    with open(results_dir / "result_0.pkl", "rb") as f:
        first = pickle.load(f)

    k_folds, n_features, _ = first["coef"].shape
    n_splits = first["r2"].shape[1]

    coef = np.zeros((k_folds, n_features, nrois))
    r2 = np.zeros((nrois, n_splits))

    for i in range(len(tasks)):
        with open(results_dir / f"result_{i}.pkl", "rb") as f:
            result = pickle.load(f)
        roi_indices = np.array(result["roi_indices"])
        coef[:, :, roi_indices] = result["coef"]
        r2[roi_indices, :] = result["r2"]

    return coef, r2


# ---------------------------------------------------------------------------
# Worker entry point (called by each Slurm array task)
# ---------------------------------------------------------------------------


def _run_worker(stage_dir, mode):
    """Execute one array task based on ``SLURM_ARRAY_TASK_ID``."""
    from cottage_analysis.analysis.spheres.rf_fitting import (
        fit_3d_rfs,
        fit_3d_rfs_multidepth,
    )

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    stage_dir = Path(stage_dir)

    with open(stage_dir / "tasks.json") as f:
        tasks = json.load(f)
    with open(stage_dir / "config.json") as f:
        config = json.load(f)

    task = tasks[task_id]
    print(f"Task {task_id}/{len(tasks)}: {task}")

    imaging_df = pd.read_pickle(stage_dir / "imaging_df.pkl")
    frames = np.load(stage_dir / "frames.npy")

    if frames.ndim == 4:
        fit_func = fit_3d_rfs_multidepth
    elif frames.ndim == 3:
        fit_func = fit_3d_rfs
    else:
        raise ValueError("frames must be 3D or 4D")

    fit_kwargs = dict(
        shift_stim=config["shift_stim"],
        use_col=config["use_col"],
        k_folds=config["k_folds"],
        validation=config["validation"],
    )

    if mode == "hyperparam":
        coef_list, r2 = fit_func(
            imaging_df,
            frames,
            reg_xy=task["reg_xy"],
            reg_depth=task["reg_depth"],
            **fit_kwargs,
        )
        gc.collect()
        result = {
            "coef": np.stack(coef_list),
            "r2": r2,
            "reg_xy": task["reg_xy"],
            "reg_depth": task["reg_depth"],
        }

    elif mode == "ipsi":
        roi_indices = np.array(task["roi_indices"])
        coef_list, r2 = fit_func(
            imaging_df,
            frames,
            reg_xy=task["reg_xy"],
            reg_depth=task["reg_depth"],
            choose_rois=roi_indices,
            **fit_kwargs,
        )
        gc.collect()
        result = {
            "coef": np.stack(coef_list),
            "r2": r2,
            "roi_indices": roi_indices.tolist(),
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")

    result_path = stage_dir / "results" / f"result_{task_id}.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Task {task_id} completed. Saved to {result_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RF fitting Slurm array worker.")
    parser.add_argument(
        "--stage-dir", required=True, help="Path to the stage directory."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["hyperparam", "ipsi"],
        help="Fitting stage to run.",
    )
    args = parser.parse_args()

    _run_worker(args.stage_dir, args.mode)
