"""Slurm array parallelization for RS/OF tuning fits.

The session is loaded once and saved to a work directory: ``trials_df`` without its
response columns (``dff_*`` / ``spks_*``, one array of frames x ROIs per trial), plus one
file per batch of ROIs with the responses of those ROIs only, so that each task reads a
small fraction of the data. A single Slurm array then
runs every ``(model, choose_trials, k_folds)`` fit on every ROI batch, each task calling
``fit_gaussian_blob.fit_rs_of_tuning`` on one CPU. Each ROI is fitted independently (``iterate_fit`` reseeds for every ROI), so the
results are identical to fitting all ROIs in one job. A dependent merge job assembles
the batches of each fit into the usual ``fit_rs_of_tuning_{...}.pickle`` next to
``neurons_df`` (as :func:`pipeline_utils.load_and_fit` does), or in ``out_dir``, and
deletes the work directory unless ``--keep-work-dir`` is given. If any task fails, the
merge does not run and the work directory is kept, with the task logs in
``work_dir/logs``; the failed array indices can then be resubmitted with
``work_dir/submit_fit.sh``.

Usage from sbatch::

    python fit_rsof_array.py --work-dir /path/to/work_dir --mode fit
    python fit_rsof_array.py --work-dir /path/to/work_dir --mode merge [--keep-work-dir]
"""

import argparse
import json
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

print = partial(print, flush=True)

CONDA_ENV = "v1_depth_map"
# ~30 min per task for the slowest fit (gaussian_2d, k_folds=5, ~40 s per ROI) on a
# 1-hour session, ~2-5 min for the fastest ones
ROI_BATCH_SIZE = 50
# Default limit of simultaneously running tasks per array, to leave room for other users
MAX_CONCURRENT = 200


def _result_file(work_dir, i_fit, roi_start, roi_stop):
    return (
        Path(work_dir)
        / "results"
        / f"fit{i_fit:02d}_rois{roi_start:05d}-{roi_stop:05d}.pkl"
    )


def _batch_file(work_dir, roi_start, roi_stop):
    return Path(work_dir) / "dff_batches" / f"rois{roi_start:05d}-{roi_stop:05d}.pkl"


def _roi_columns(trials_df, n_rois):
    """Columns of trials_df holding one (frames x ROIs) response array per trial."""
    cols = []
    for col in trials_df.columns:
        if not str(col).startswith(("dff_", "spks_")):
            continue
        first = trials_df[col].iloc[0]
        if (
            isinstance(first, np.ndarray)
            and first.ndim == 2
            and first.shape[1] == n_rois
        ):
            cols.append(col)
    return cols


def prepare_work_dir(
    work_dir, trials_df, fits, fit_params, roi_batch_size=ROI_BATCH_SIZE
):
    """Save the inputs of an RS/OF array run and define its tasks.

    Args:
        work_dir (str or Path): Working directory. Emptied if it already exists.
        trials_df (pd.DataFrame): Trials dataframe passed to `fit_rs_of_tuning`.
        fits (list of dict): One dict per fit with keys ``model``, ``choose_trials``,
            ``k_folds`` and ``target`` (path of the merged output pickle).
        fit_params (dict): Keyword arguments of `fit_rs_of_tuning` shared by all fits
            (e.g. ``rs_thr``, ``param_range``, ``niter``, ``min_sigma``,
            ``max_rs2motor_diff``, ``use_col``). Must be JSON serialisable.
        roi_batch_size (int): Number of ROIs per array task.

    Returns:
        int: Number of array tasks.
    """
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    for sub_dir in ("results", "logs", "dff_batches"):
        (work_dir / sub_dir).mkdir(parents=True)

    resp_col = {"dffs": "dff_stim", "spks": "spks_stim"}[
        fit_params.get("use_col", "dffs")
    ]
    n_rois = trials_df[resp_col].iloc[0].shape[1]
    roi_columns = _roi_columns(trials_df, n_rois)
    trials_df.drop(columns=roi_columns).to_pickle(work_dir / "trials_df.pkl")
    batches = [
        (start, min(start + roi_batch_size, n_rois))
        for start in range(0, n_rois, roi_batch_size)
    ]
    for start, stop in batches:
        pd.to_pickle(
            {
                col: [
                    np.ascontiguousarray(resp[:, start:stop]) for resp in trials_df[col]
                ]
                for col in roi_columns
            },
            _batch_file(work_dir, start, stop),
        )
    tasks = [
        {"fit": i_fit, "roi_start": start, "roi_stop": stop}
        for i_fit in range(len(fits))
        for start, stop in batches
    ]
    config = dict(
        fits=[dict(fit, target=str(fit["target"])) for fit in fits],
        fit_params=fit_params,
        n_rois=int(n_rois),
        roi_columns=roi_columns,
    )
    (work_dir / "config.json").write_text(json.dumps(config, indent=2))
    (work_dir / "tasks.json").write_text(json.dumps(tasks, indent=2))
    return len(tasks)


def _sbatch(script, script_file, dependency=None):
    """Write *script* to *script_file*, submit it and return the job ID."""
    script_file = Path(script_file)
    script_file.write_text(script)
    cmd = ["sbatch", "--parsable"]
    if dependency is not None:
        cmd.append(f"--dependency=afterok:{dependency}")
    result = subprocess.run(cmd + [str(script_file)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")
    return result.stdout.strip().split(";")[0]


def submit_array_and_merge(
    work_dir,
    n_tasks,
    conda_env=CONDA_ENV,
    mem="2G",
    time_limit="4:00:00",
    partition="ncpu",
    max_concurrent=MAX_CONCURRENT,
    merge_log_dir=None,
    cleanup=True,
    job_name="rsof_fit",
):
    """Submit the fit array and the merge job depending on it.

    Args:
        work_dir (str or Path): Working directory prepared by `prepare_work_dir`.
        n_tasks (int): Number of array tasks.
        conda_env (str): Conda environment name.
        mem (str): Memory per array task.
        time_limit (str): Wall time per array task.
        partition (str): Slurm partition.
        max_concurrent (int, optional): Maximum number of simultaneously running
            array tasks. None for no limit. Defaults to `MAX_CONCURRENT`.
        merge_log_dir (str or Path, optional): Folder for the merge job log.
            Defaults to ``work_dir/logs`` (deleted with the work directory on success).
        cleanup (bool): Delete the work directory after a successful merge.
        job_name (str): Name of the array job; the merge job gets ``_merge`` appended.

    Returns:
        tuple: ``(array_job_id, merge_job_id)``
    """
    work_dir = Path(work_dir)
    script_path = Path(__file__).resolve()
    log_dir = work_dir / "logs"
    merge_log_dir = Path(merge_log_dir) if merge_log_dir is not None else log_dir
    array = f"0-{n_tasks - 1}" + (f"%{max_concurrent}" if max_concurrent else "")
    # scipy's curve_fit is single-threaded; letting numpy/scipy start one BLAS thread
    # per core on a 1-CPU allocation slows the fits down many-fold.
    env = "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
    # Tasks must import the same code as the submitting process (e.g. a source snapshot)
    if os.environ.get("PYTHONPATH"):
        env += f"\nexport PYTHONPATH={os.environ['PYTHONPATH']}"
    keep = "" if cleanup else " --keep-work-dir"

    array_job = _sbatch(
        f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array={array}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --output={log_dir}/rsof_fit_%A_%a.log
{env}
conda run -n {conda_env} python -u {script_path} --work-dir {work_dir} --mode fit
""",
        work_dir / "submit_fit.sh",
    )
    merge_job = _sbatch(
        f"""#!/bin/bash
#SBATCH --job-name={job_name}_merge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition={partition}
#SBATCH --output={merge_log_dir}/rsof_merge_%j.log
{env}
conda run -n {conda_env} python -u {script_path} --work-dir {work_dir} --mode merge{keep}
""",
        work_dir / "submit_merge.sh",
        dependency=array_job,
    )
    print(
        f"Submitted RS/OF fit array {array_job} ({n_tasks} tasks) "
        f"and merge job {merge_job}"
    )
    return array_job, merge_job


def submit_rsof_fits(
    work_dir,
    project,
    session_name,
    photodiode_protocol,
    fits,
    fit_params,
    file_special_sfx="",
    roi_batch_size=ROI_BATCH_SIZE,
    base_name=None,
    filter_datasets=None,
    exclude_datasets=None,
    protocol_base="SpheresPermTubeReward",
    recording_type="two_photon",
    ephys_kwargs=None,
    tread_kwargs=None,
    lag=0,
    out_dir=None,
    merge_log_dir=None,
    **slurm_kwargs,
):
    """Load a session once and fit several RS/OF models on it with a Slurm array.

    Produces the same output files as calling :func:`pipeline_utils.load_and_fit`
    for each fit, next to ``neurons_df`` or in ``out_dir``.

    Args:
        work_dir (str or Path): Temporary working directory.
        project (str): Project name in flexilims.
        session_name (str): Session name in the format {Mouse}_{Session}.
        photodiode_protocol (int): Photodiode protocol used for syncing.
        fits (list of tuple): ``(model, choose_trials, k_folds)`` for each fit.
        fit_params (dict): Keyword arguments of `fit_rs_of_tuning` shared by all fits.
            Its ``use_col`` ("dffs" or "spks"), if any, also selects the traces loaded.
        file_special_sfx (str, optional): Suffix added to the output filenames.
        roi_batch_size (int, optional): Number of ROIs per array task.
        base_name, filter_datasets, exclude_datasets, protocol_base, recording_type,
            ephys_kwargs, tread_kwargs, lag: Passed to
            `pipeline_utils.load_trials_for_rsof_fit`.
        out_dir (str or Path, optional): Folder for the merged fit files. Defaults to
            the folder of ``neurons_df``.
        merge_log_dir (str or Path, optional): Folder for the merge job log.
        **slurm_kwargs: Passed to `submit_array_and_merge`.

    Returns:
        str: Job ID of the merge job, to use as dependency for downstream steps.
    """
    from cottage_analysis.pipelines import pipeline_utils

    neurons_ds, trials_df = pipeline_utils.load_trials_for_rsof_fit(
        project,
        session_name,
        photodiode_protocol,
        base_name=base_name,
        filter_datasets=filter_datasets,
        exclude_datasets=exclude_datasets,
        protocol_base=protocol_base,
        recording_type=recording_type,
        ephys_kwargs=ephys_kwargs,
        tread_kwargs=tread_kwargs,
        use_col=fit_params.get("use_col", "dffs"),
        lag=lag,
    )
    out_dir = neurons_ds.path_full.parent if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_list = [
        dict(
            model=model,
            choose_trials=choose_trials,
            k_folds=k_folds,
            target=out_dir
            / pipeline_utils.rsof_fit_filename(
                model, choose_trials, k_folds, file_special_sfx
            ),
        )
        for model, choose_trials, k_folds in fits
    ]
    n_tasks = prepare_work_dir(
        work_dir, trials_df, fit_list, fit_params, roi_batch_size
    )
    _, merge_job = submit_array_and_merge(
        work_dir, n_tasks, merge_log_dir=merge_log_dir, **slurm_kwargs
    )
    return merge_job


def run_fit_task(work_dir, task_id):
    """Fit one batch of ROIs for one model (one array task)."""
    from cottage_analysis.analysis import fit_gaussian_blob

    work_dir = Path(work_dir)
    task = json.loads((work_dir / "tasks.json").read_text())[task_id]
    config = json.loads((work_dir / "config.json").read_text())
    fit = config["fits"][task["fit"]]
    start, stop = task["roi_start"], task["roi_stop"]
    print(
        f"Task {task_id}: {fit['model']} choose_trials={fit['choose_trials']} "
        f"k_folds={fit['k_folds']}, ROIs {start}-{stop - 1}"
    )

    trials_df = pd.read_pickle(work_dir / "trials_df.pkl")
    for col, responses in pd.read_pickle(_batch_file(work_dir, start, stop)).items():
        trials_df[col] = responses
    fit_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df,
        model=fit["model"],
        choose_trials=fit["choose_trials"],
        k_folds=fit["k_folds"],
        **config["fit_params"],
    )
    # the responses only hold this batch: ROIs are numbered from 0 in fit_df
    fit_df["roi"] = fit_df["roi"] + start
    target = _result_file(work_dir, task["fit"], start, stop)
    tmp = target.with_suffix(".tmp")
    fit_df.to_pickle(tmp)
    tmp.rename(target)
    print(f"Saved {target}")


def merge_results(work_dir, cleanup=True):
    """Assemble the ROI batches of each fit into its target file.

    Args:
        work_dir (str or Path): Working directory of the array run.
        cleanup (bool): Delete the working directory once all fits are merged.
    """
    work_dir = Path(work_dir)
    config = json.loads((work_dir / "config.json").read_text())
    tasks = json.loads((work_dir / "tasks.json").read_text())
    n_rois = config["n_rois"]

    for i_fit, fit in enumerate(config["fits"]):
        files = [
            _result_file(work_dir, i_fit, t["roi_start"], t["roi_stop"])
            for t in tasks
            if t["fit"] == i_fit
        ]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing results for {fit['target']}: {missing}")
        fit_df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
        fit_df = fit_df.sort_values("roi").reset_index(drop=True)
        if not np.array_equal(fit_df["roi"].to_numpy(), np.arange(n_rois)):
            raise ValueError(f"ROIs of merged {fit['target']} are not 0..{n_rois - 1}")
        fit_df.to_pickle(fit["target"])
        print(f"Fit results saved to {fit['target']}")

    if cleanup:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mode", choices=["fit", "merge"], required=True)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()
    if args.mode == "fit":
        run_fit_task(args.work_dir, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        merge_results(args.work_dir, cleanup=not args.keep_work_dir)
