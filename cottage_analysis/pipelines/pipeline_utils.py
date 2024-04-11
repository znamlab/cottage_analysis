import subprocess
import shlex
import scipy
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from functools import partial
import warnings
from pandas.errors import SettingWithCopyWarning

import flexiznam as flz
from znamutils import slurm_it

from cottage_analysis.analysis import spheres, fit_gaussian_blob, find_depth_neurons
from cottage_analysis.plotting import basic_vis_plots, sta_plots

print = partial(print, flush=True)

CONDA_ENV = "2p_analysis_cottage2"


def create_neurons_ds(
    session_name, flexilims_session=None, project=None, conflicts="skip"
):
    """Create a neurons_df dataset from flexilims.

    Args:
        session_name (str): session name. {Mouse}_{Session}.
        flexilims_session (Series, optional): flexilims session object. Defaults to None.
        project (str, optional): project name. Defaults to None. Must be provided if flexilims_session is None.
        conflicts (str, optional): how to handle conflicts. Defaults to "skip".
    """
    assert flexilims_session is not None or project is not None
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=project)
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )

    # Create a neurons_df dataset from flexilism
    neurons_ds = flz.Dataset.from_origin(
        origin_id=exp_session.id,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        conflicts=conflicts,
    )
    neurons_ds.path = neurons_ds.path.parent / f"neurons_df.pickle"

    return neurons_ds


def sbatch_session(
    project,
    session_name,
    pipeline_filename,
    conflicts,
    photodiode_protocol,
    use_slurm=False,
    **kwargs,
):
    """Start sbatch script to run analysis_pipeline on a single session.

    Args:

    """

    script_path = str(
        Path(__file__).parent.parent.parent / "sbatch" / pipeline_filename
    )

    log_fname = f"{session_name}_%j.out"

    log_path = str(Path(__file__).parent.parent.parent / "logs" / f"{log_fname}")

    args = f"--export=PROJECT={project},SESSION_NAME={session_name},CONFLICTS={conflicts},PHOTODIODE_PROTOCOL={photodiode_protocol},USE_SLURM={int(use_slurm)}"
    for key, value in kwargs.items():
        args+=f",{key.upper()}={int(value)}"

    args = args + f" --output={log_path}"

    command = f"sbatch {args} {script_path}"
    print(command)
    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def load_session(
    project, session_name, photodiode_protocol=None, regenerate_frames=False
):
    """Load data from a single session.

    This function is used to load data from a single session.

    Args:
        project (str): project name.
        session_name (str): session name. {Mouse}_{Session}.
        photodiode_protocol (str, optional): photodiode protocol. Defaults to None.
        regenerate_frames (bool, optional): whether to regenerate frames. Defaults to False.

    Returns:
        neurons_df (pd.DataFrame): neurons_df dataframe.
        vs_df_all (pd.DataFrame): vs_df_all dataframe.
        trials_df_all (pd.DataFrame): trials_df_all dataframe.
        frames_all (pd.DataFrame): frames_all dataframe. Only returned if regenerate_frames is True.
        imaging_df_all (pd.DataFrame): imaging_df_all dataframe. Only returned if regenerate_frames is True.
    """

    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts="skip",
    )
    if neurons_ds.get_flexilims_entry() is None:
        raise flz.FlexilimsError(f"Session {session_name} not processed...")

    neurons_df = pd.read_pickle(neurons_ds.path_full)
    vs_df_all, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    out = [neurons_ds, neurons_df, vs_df_all, trials_df_all]
    if regenerate_frames:
        frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            filter_datasets={"anatomical_only": 3},
            recording_type="two_photon",
            protocol_base="SpheresPermTubeReward",
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            resolution=5,
        )
        out = out + [frames_all, imaging_df_all]
    return tuple(out)


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={"mem": "32G", "time": "47:00:00", "cpus-per-task": 8, "partition": "ncpu"},
    print_job_id=True,
)
def load_and_fit(
    project,
    session_name,
    photodiode_protocol,
    model,
    choose_trials,
    rs_thr,
    param_range,
    niter,
    min_sigma,
    k_folds=1,
):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    (neurons_ds, neurons_df, vs_df_all, trials_df_all,) = load_session(
        project, session_name, photodiode_protocol, regenerate_frames=False
    )

    # do the fit
    fit_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df_all,
        model=model,
        choose_trials=choose_trials,
        rs_thr=rs_thr,
        param_range=param_range,
        niter=niter,
        min_sigma=min_sigma,
        k_folds=k_folds,
    )
    # save fit_df
    # create name from model and choose_trials
    suffix = f"{model}"
    if choose_trials is not None:
        suffix = suffix + f"_crossval"
    suffix = suffix + f"_k{k_folds}"
    target = neurons_ds.path_full.with_name(f"fit_rs_of_tuning_{suffix}.pickle")
    fit_df.to_pickle(target)
    return fit_df


@slurm_it(conda_env=CONDA_ENV, slurm_options={"mem": "16G", "time": "2:00:00", "partition": "ncpu"})
def merge_fit_dataframes(project, session_name, conflicts="skip"):
    """Merge fit dataframe from all fits

    Args:
        project (str): project name.
        session_name (str): session name. {Mouse}_{Session}.
        conflicts (str, optional): how to handle conflicts. Defaults to "skip".
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts="skip",
    )

    # load the main neurons_df
    neurons_df = pd.read_pickle(neurons_ds.path_full)
    to_remove = []

    for df in neurons_ds.path_full.parent.glob("fit_rs_of_tuning_*.pickle"):
        print(f"Merging {df}...")
        to_remove.append(df)
        df = pd.read_pickle(df)
        assert (df.roi == neurons_df.roi).all(), "ROI mismatch"
        for col in df.columns:
            if col == "roi":
                continue
            if col in neurons_df.columns:
                if conflicts == "skip":
                    print(f"WARNING: Skipping column {col} - already present in neurons_df")
                elif conflicts == "overwrite":
                    neurons_df[col] = df[col]
                    print(f"WARNING: Overwriting column {col}")
            else:
                neurons_df[col] = df[col]

    # save the new neurons_df
    neurons_df.to_pickle(neurons_ds.path_full)
    print("All dataframes merged. Neurons_df saved.")
    return neurons_df


@slurm_it(conda_env=CONDA_ENV, slurm_options={"mem": "16G", "time": "9:00:00", "partition": "ncpu"})
def run_basic_plots(project, session_name, photodiode_protocol):
    """Run basic plots on a session."""

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    (
        neurons_ds,
        neurons_df,
        vs_df_all,
        trials_df_all,
        frames_all,
        imaging_df_all,
    ) = load_session(project, session_name, photodiode_protocol, regenerate_frames=True)

    kwargs = {
        "RS_OF_matrix_log_range": {
            "rs_bin_log_min": 0,
            "rs_bin_log_max": 2.5,
            "rs_bin_num": 6,
            "of_bin_log_min": -1.5,
            "of_bin_log_max": 3.5,
            "of_bin_num": 11,
            "log_base": 10,
        }
    }

    basic_vis_plots.basic_vis_session(
        neurons_df=neurons_df, trials_df=trials_df_all, neurons_ds=neurons_ds, **kwargs
    )

    # Plot all ROI RFs
    print("Plotting RFs...")
    depth_list = find_depth_neurons.find_depth_list(trials_df_all)
    for is_closedloop in trials_df_all.closed_loop.unique():
        if is_closedloop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        coef = np.stack(neurons_df[f"rf_coef{sfx}"], axis=2)
        sta_plots.basic_vis_sta_session(
            coef=coef,
            neurons_df=neurons_df,
            trials_df=trials_df_all,
            depth_list=depth_list,
            frames=frames_all,
            is_closedloop=is_closedloop,
            save_dir=neurons_ds.path_full.parent,
            fontsize_dict={"title": 10, "tick": 10, "label": 10},
        )
