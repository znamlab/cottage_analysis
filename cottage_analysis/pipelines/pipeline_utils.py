import subprocess
import shlex
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial, reduce
import warnings
from pandas.errors import SettingWithCopyWarning
import flexiznam as flz
from znamutils.decorators import slurm_it
from cottage_analysis.analysis import (
    spheres,
    fit_gaussian_blob,
    find_depth_neurons,
    treadmill,
)
from cottage_analysis.plotting import basic_vis_plots, sta_plots

print = partial(print, flush=True)

CONDA_ENV = "v1_depth_map"


def get_current_time():
    return time.strftime("%Y%m%d"), time.strftime("%H%M%S")


def save_finish_time(finished, col):
    finished[f"{col}_day"] = get_current_time()[0]
    finished[f"{col}_time"] = get_current_time()[1]
    return finished


def create_neurons_ds(
    session_name,
    flexilims_session=None,
    project=None,
    conflicts="skip",
    base_name=None,
):
    """Create a neurons_df dataset from flexilims.

    Args:
        session_name (str): session name. {Mouse}_{Session}.
        flexilims_session (Series, optional): flexilims session object. Defaults to
            None.
        project (str, optional): project name. Defaults to None. Must be provided if
            flexilims_session is None.
        conflicts (str, optional): how to handle conflicts. Defaults to "skip".
        base_name (str, optional): base name for the dataset. Defaults to None.

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
        base_name=base_name,
        conflicts=conflicts,
    )
    fname = base_name if base_name else "neurons_df"
    neurons_ds.path = neurons_ds.path.parent / f"{fname}.pickle"

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

    if "log_fname" in kwargs.keys():
        log_fname = f"{session_name}_{kwargs['log_fname']}_%j.out"
    else:
        log_fname = f"{session_name}_%j.out"

    if "log_path" in kwargs.keys():
        print(f"Using custom log path {kwargs['log_path']}")
        log_path = str(
            Path(__file__).parent.parent.parent
            / "logs"
            / f"{kwargs['log_path']}"
            / f"{log_fname}"
        )
    else:
        log_path = str(Path(__file__).parent.parent.parent / "logs" / f"{log_fname}")

    args = f"--export=PROJECT={project},SESSION_NAME={session_name},CONFLICTS={conflicts},PHOTODIODE_PROTOCOL={photodiode_protocol},USE_SLURM={int(use_slurm)}"
    # Handle other kwargs for export
    for key, value in kwargs.items():
        if key == "protocol_base":
            args += f",PROTOCOL_BASE={value}"
        elif key == "use_annotated":
            args += f",USE_ANNOTATED={value}"
        elif key not in ["log_fname", "log_path"]:
            args += f",{key.upper()}={int(value)}"

    args = args + f" --output={log_path}"
    args = args + f" --job-name={session_name}_{pipeline_filename.split('.')[0]}"
    command = f"sbatch {args} {script_path}"
    print(command)
    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def load_session(
    project,
    session_name,
    photodiode_protocol=None,
    regenerate_frames=False,
    base_name=None,
    filter_datasets=None,
    exclude_datasets=None,
    protocol_base="SpheresPermTubeReward",
    recording_type="two_photon",
    ephys_kwargs=None,
):
    """Load data from a single session.

    This function is used to load data from a single session.

    Args:
        project (str): project name.
        session_name (str): session name. {Mouse}_{Session}.
        photodiode_protocol (int, optional): photodiode protocol. Defaults to None.
        regenerate_frames (bool, optional): whether to regenerate frames. Defaults to
            False.
        base_name (str, optional): base name for the dataset. Defaults to None.
        filter_datasets (dict, optional): filter datasets. Defaults to
            {"anatomical_only": 3}.
        exclude_datasets (dict, optional): exclude datasets. Defaults to None.
        protocol_base (str, optional): protocol base name. Defaults to
            "SpheresPermTubeReward".
        recording_type (str, optional): recording type. Defaults to "two_photon".
        ephys_kwargs (dict, optional): ephys kwargs for spike rate generation.
            Defaults to None.

    Returns:
        neurons_df (pd.DataFrame): neurons_df dataframe.
        vs_df_all (pd.DataFrame): vs_df_all dataframe.
        trials_df_all (pd.DataFrame): trials_df_all dataframe.
        frames_all (pd.DataFrame): frames_all dataframe. Only returned if
            regenerate_frames is True.
        imaging_df_all (pd.DataFrame): imaging_df_all dataframe. Only returned if
            regenerate_frames is True.
    """
    if filter_datasets is None:
        filter_datasets = {"anatomical_only": 3}

    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts="skip",
        base_name=base_name,
    )
    if neurons_ds.get_flexilims_entry() is None:
        raise flz.FlexilimsError(f"Session {session_name} not processed...")

    neurons_df = pd.read_pickle(neurons_ds.path_full)
    if protocol_base == "SpheresTubeMotor":
        vs_df_all, trials_df_all = treadmill.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_datasets,
            exclude_datasets=exclude_datasets,
            recording_type=recording_type,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            ephys_kwargs=ephys_kwargs,
        )
    else:
        vs_df_all, trials_df_all = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            filter_datasets=filter_datasets,
            exclude_datasets=exclude_datasets,
            recording_type=recording_type,
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            ephys_kwargs=ephys_kwargs,
        )
    out = [neurons_ds, neurons_df, vs_df_all, trials_df_all]
    if regenerate_frames:
        frames_all, imaging_df_all = spheres.regenerate_frames_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            filter_datasets=filter_datasets,
            exclude_datasets=exclude_datasets,
            recording_type=recording_type,
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
            resolution=5,
            verbose=False,
            ephys_kwargs=ephys_kwargs,
        )
        out = out + [frames_all, imaging_df_all]
    return tuple(out)


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={
        "mem": "32G",
        "time": "7-00:00:00",
        "partition": "ncpu",
        "cpus-per-task": 8,
    },
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
    trial_sfx="",
    file_special_sfx="",
    run_closedloop_only=False,
    run_openloop_only=False,
    base_name=None,
    filter_datasets=None,
    exclude_datasets=None,
    protocol_base="SpheresPermTubeReward",
    recording_type="two_photon",
    ephys_kwargs=None,
    max_rs2motor_diff=None,
    max_acc=None,
):
    """Load data for a session and fit a running speed and optic flow tuning model.

    Note:
        The results are saved as pickle files with names following the pattern:
        `fit_rs_of_tuning_{model}[_crossval]_k{k_folds}{file_special_sfx}.pickle`.
        Crossval is added if k_folds > 1. These files are later merged into the main
        `neurons_df` by `merge_fit_dataframes`.
        To avoid column name conflicts during merging, ensure that `trial_sfx` or
        `model` (which determines `model_sfx`) is unique for each fit performed on the
        same session.

    Args:
        project (str): Project name in flexilims.
        session_name (str): Session name in the format {Mouse}_{Session}.
        photodiode_protocol (int): Photodiode protocol used for syncing.
        model (str): Model to fit. One of "gaussian_2d",
            "gaussian_additive", "gaussian_OF", "gaussian_RS", "gaussian_ratio".
        choose_trials (str or list): Trials to include in the fit. Can be a list of
            trial indices or a string (e.g., "even", "odd").
        rs_thr (float): Running speed threshold (m/s) to include frames.
        param_range (dict): Range of parameters for the fit. Usually contains
            "rs_min", "rs_max", "of_min", "of_max".
        niter (int): Number of iterations for stochastic fit optimization.
        min_sigma (float): Minimum sigma value for the gaussian model.
        k_folds (int, optional): Number of folds for cross-validation. If > 1, the model
            will be evaluated using cross-validation. Defaults to 1.
        trial_sfx (str, optional): Suffix for saved column names in the output dataframe.
            Defaults to "". Example: "_crossval".
        file_special_sfx (str, optional): Suffix added to the saved pickle filename.
            Defaults to "". Example: "_openclosed0".
        run_closedloop_only (bool, optional): Whether to fit only closed-loop protocols.
            Defaults to False.
        run_openloop_only (bool, optional): Whether to fit only open-loop protocols.
            Defaults to False.
        base_name (str, optional): Base name for the neurons_df dataset in flexilims.
            Defaults to None.
        filter_datasets (dict, optional): Dictionary to filter datasets from flexilims.
            Defaults to {"anatomical_only": 3}.
        exclude_datasets (dict, optional): Dictionary to exclude datasets from flexilims.
            Defaults to None.
        protocol_base (str, optional): Base protocol name (e.g., "SpheresPermTubeReward").
            Defaults to "SpheresPermTubeReward".
        recording_type (str, optional): Type of recording (e.g., "two_photon").
            Defaults to "two_photon".
        ephys_kwargs (dict, optional): Additional arguments for ephys data processing.
            Defaults to None.
        max_rs2motor_diff (float, optional): Maximum absolute ratio of
            (rs - motor_speed)/rs for frame selection. Defaults to None.
        max_acc (float, optional): Maximum acceleration ratio threshold for frame
            selection. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe containing the fitted parameters and performance
            metrics for each ROI. The result is also saved as a pickle file.
    """
    if filter_datasets is None:
        filter_datasets = {"anatomical_only": 3}

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    (
        neurons_ds,
        _,
        _,
        trials_df_all,
    ) = load_session(
        project,
        session_name,
        photodiode_protocol,
        regenerate_frames=False,
        base_name=base_name,
        filter_datasets=filter_datasets,
        exclude_datasets=exclude_datasets,
        protocol_base=protocol_base,
        recording_type=recording_type,
        ephys_kwargs=ephys_kwargs,
    )
    # create name from model and choose_trials
    suffix = f"{model}"
    if isinstance(choose_trials, str):
        suffix = suffix + f"_crossval"
    suffix = suffix + f"_k{k_folds}"

    # remove any multidepth experiment
    is_multidepth = trials_df_all.recording_name.str.contains("multidepth")
    trials_df_all = trials_df_all[~is_multidepth]

    # do the fit
    fit_df = fit_gaussian_blob.fit_rs_of_tuning(
        trials_df=trials_df_all,
        model=model,
        choose_trials=choose_trials,
        trial_sfx=trial_sfx,
        rs_thr=rs_thr,
        param_range=param_range,
        niter=niter,
        min_sigma=min_sigma,
        k_folds=k_folds,
        run_closedloop_only=run_closedloop_only,
        run_openloop_only=run_openloop_only,
        max_rs2motor_diff=max_rs2motor_diff,
        max_acc=max_acc,
    )
    # save fit_df
    target = neurons_ds.path_full.with_name(
        f"fit_rs_of_tuning_{suffix}{file_special_sfx}.pickle"
    )
    fit_df.to_pickle(target)
    print(f"Fit results saved to {target}")

    return fit_df


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={"mem": "16G", "time": "2:00:00", "partition": "ncpu"},
)
def merge_fit_dataframes(
    project,
    session_name,
    conflicts="skip",
    prefix="fit_rs_of_tuning_",
    suffix="",
    exclude_keywords=["recording", "openclosed"],
    include_keywords=[],
    target_column_suffix=None,
    target_column_prefix="",  # "_recording"
    filetype=".pickle",
    target_filename="neurons_df.pickle",
    base_name=None,
):
    """Merge fit dataframe from all fits

    Args:
        project (str): project name.
        session_name (str): session name. {Mouse}_{Session}.
        conflicts (str, optional): how to handle conflicts. Defaults to "skip".
        prefix (str, optional): prefix of the files to merge. Defaults to
            "fit_rs_of_tuning_".
        suffix (str, optional): suffix of the files to merge. Defaults to ""
        column_suffix (int | str, optional): digits for the source filename, which
            becomes the special suffix to append to each column name of the dataframe to
            be merged. If str, will directly be used as sfx. Defaults to None.
        filetype (str, optional): filetype of the files to merge. Defaults to ".pickle".
        target_filename (str, optional): target filename. Defaults to
            "neurons_df.pickle".
        base_name (str, optional): base name for the dataset. Defaults to None.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts="skip",
        base_name=base_name,
    )
    # load the main neurons_df
    neurons_df = pd.read_pickle(neurons_ds.path_full)

    search_str = f"{prefix}*{suffix}{filetype}"
    dfs_to_merge = []
    for df_name in neurons_ds.path_full.parent.glob(search_str):
        if exclude_keywords:
            # if the name contains any keywords that needs to be excluded
            if any([keyword in str(df_name) for keyword in exclude_keywords]):
                print(f"Exclude files {df_name}")
            else:  # name doesn't contain anything that needs to be excluded
                if include_keywords:
                    # if name doesn't contain all the things that need to be included
                    if not all(
                        [keyword in str(df_name) for keyword in include_keywords]
                    ):
                        print(f"Exclude files {df_name}")
                    else:
                        tmp = pd.read_pickle(df_name)
                        dfs_to_merge.append(tmp)
                else:
                    tmp = pd.read_pickle(df_name)
                    dfs_to_merge.append(tmp)
        else:
            tmp = pd.read_pickle(df_name)
            dfs_to_merge.append(tmp)

    # Checking that the number of ROIs is the same across dataframes with fit results
    assert all(
        [df["roi"].equals(neurons_df["roi"]) for df in dfs_to_merge]
    ), "ROIs in dataframes do not match neurons_df."
    assert all(~np.isnan(neurons_df["roi"].values)), "ROIs in neurons_df are NaN."

    if target_column_suffix is not None:
        if isinstance(target_column_suffix, str):
            suffix_to_add = target_column_suffix
        else:
            suffix_to_add = f"{target_column_prefix}_" + "_".join(
                str(df_name.stem).split("_")[target_column_suffix:]
            )

        # rename all columns before merging
        for df in dfs_to_merge:
            df.columns = [
                f"{col}{suffix_to_add}" if col != "roi" else "roi" for col in df.columns
            ]
    rsof_df = reduce(lambda x, y: pd.merge(x, y, on="roi", how="inner"), dfs_to_merge)

    if conflicts == "skip":
        columns_to_add = rsof_df.columns.difference(neurons_df.columns).to_list()
        columns_to_add.append("roi")
        neurons_df = pd.merge(neurons_df, rsof_df.loc[:, columns_to_add], on="roi")
        print(
            f"New columns written to neurons_df: {[name for name in columns_to_add if name != 'roi']}"
        )

    elif conflicts == "overwrite":
        columns_to_drop = neurons_df.columns.intersection(rsof_df.columns).to_list()
        columns_to_drop.remove("roi")
        neurons_df = neurons_df.drop(columns_to_drop, axis=1)
        neurons_df = pd.merge(neurons_df, rsof_df, on="roi")
        print(f"New columns written to neurons_df: {rsof_df.columns.to_list()}")
        print(f"Data overwritten in neurons_df: {columns_to_drop}")

    # save the new neurons_df
    neurons_df.to_pickle(neurons_ds.path_full.parent / target_filename)

    return neurons_df


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={"mem": "16G", "time": "6:00:00", "partition": "ncpu"},
)
def run_basic_plots(
    project,
    session_name,
    photodiode_protocol,
    do_sta=True,
    do_basic_vis=True,
    filter_datasets=None,
    protocol_base="SpheresPermTubeReward",
    recording_type="two_photon",
    ephys_kwargs=None,
):
    """Run basic plots on a session.

    Args:
        project (str): project name.
        session_name (str): session name. {Mouse}_{Session}.
        photodiode_protocol (int): photodiode protocol.
        do_sta (bool, optional): whether to run sta plots. Defaults to True.
        do_basic_vis (bool, optional): whether to run basic visualisation plots.
            Defaults to True.
    """

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    (
        neurons_ds,
        neurons_df,
        _,
        trials_df_all,
        frames_all,
        _,
    ) = load_session(
        project,
        session_name,
        photodiode_protocol,
        regenerate_frames=True,
        filter_datasets=filter_datasets,
        protocol_base=protocol_base,
        recording_type=recording_type,
        ephys_kwargs=ephys_kwargs,
    )

    # Remove multidepth if there are any
    is_multidepth = trials_df_all.recording_name.str.contains("multidepth")
    trials_df_all = trials_df_all[~is_multidepth]

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
    if do_basic_vis:
        basic_vis_plots.basic_vis_session(
            neurons_df=neurons_df,
            trials_df=trials_df_all,
            neurons_ds=neurons_ds,
            **kwargs,
        )

    if not do_sta:
        return
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
        )


def load_treadmill_and_sphere_datasets(
    project,
    mouse,
    session,
    photodiode_protocol=5,
    filter_datasets=None,
    recording_type="two_photon",
    protocol_base_sphere="SpheresPermTubeReward",
    tread_kwargs=None,
    **kwargs,
):
    """
    Load neurons_df and trials_dfs for treadmill and sphere recordings of a session.

    Args:
        project (str): project name.
        mouse (str): mouse name.
        session (str): session date (e.g. S20250401).
        photodiode_protocol (int): photodiode protocol. Defaults to 5.
        filter_datasets (dict): filter datasets for suite2p.
        recording_type (str): recording type. Defaults to "two_photon".
        protocol_base_sphere (str): protocol base for sphere recordings.
        **kwargs: additional arguments for sync_all_recordings.

    Returns:
        tuple: (neurons_df, trials_df_tread, trials_df_sphere)
    """
    session_name = f"{mouse}_{session}"
    flexilims_session = flz.get_flexilims_session(project_id=project)

    # Load neurons_df
    neurons_ds = create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
    )
    if neurons_ds.get_flexilims_entry() is None:
        raise flz.FlexilimsError(f"Session {session_name} not processed...")

    neurons_df = pd.read_pickle(neurons_ds.path_full)

    # Load treadmill trials
    if tread_kwargs is None:
        tread_kwargs = {}

    _, trials_df_tread = treadmill.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        photodiode_protocol=photodiode_protocol,
        filter_datasets=filter_datasets,
        recording_type=recording_type,
        **dict(tread_kwargs, **kwargs),
    )

    # Load sphere (closed-loop) trials
    _, trials_df_sphere = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        photodiode_protocol=photodiode_protocol,
        filter_datasets=filter_datasets,
        recording_type=recording_type,
        protocol_base=protocol_base_sphere,
        **kwargs,
    )

    return neurons_df, trials_df_tread, trials_df_sphere
