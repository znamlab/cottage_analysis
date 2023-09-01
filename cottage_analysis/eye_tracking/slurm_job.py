import subprocess, shlex
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import flexiznam as flz
from cottage_analysis.eye_tracking import eye_model_fitting
from cottage_analysis.utilities import slurm_helper


def slurm_dlc_pupil(
    camera_ds_id,
    model_name,
    origin_id,
    project,
    crop=False,
    conflicts="abort",
    slurm_folder=None,
    job_dependency=None,
):
    """Start slurm job to track pupil

    Args:
        camera_ds_id (str): Hexadecimal code of the camera dataset on flexilims
        model_name (str): Name of the model to use. Must be in the `DLC_models` shared
            project folder
        origin_id (str, optional): Hexadecimal code of the origin on flexilims. If
            not None, a flexilims entry with be created from this origin, otherwise
            nothing is uploaded. Defaults to None.
        project (str, optional): Mandatory if `origin_id` is not None. Name of the
            project on flexilims. Defaults to None.
        crop (bool, optional): Whether to crop the video to the ROI defined in the
            uncropped dlc_tracking. Defaults to False.
        conflicts (str, optional): How to handle conflicts. Can be "abort", "skip" or
            "overwrite". Defaults to "abort".
        slurm_folder (str, optional): Path to create the slurm script, python
            script and slurms log files. If None, will make one using from_flexilims
        job_dependency (str, optional): Job id to wait for before starting this job.

    Returns:
        subprocess.Process: The process job
    """
    flm_sess = flz.get_flexilims_session(project)
    camera_ds = flz.Dataset.from_flexilims(id=camera_ds_id, flexilims_session=flm_sess)
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]

    assert video_path.exists(), f"Video {video_path} does not exist"
    suffix = "cropped" if crop else "uncropped"
    basename = f"{video_path.stem}_dlc_tracking_{suffix}"

    if slurm_folder is None:
        ds = flz.Dataset.from_origin(
            origin_id=origin_id,
            dataset_type="dlc_tracking",
            flexilims_session=flm_sess,
            base_name=basename,
            conflicts=conflicts,
        )
        slurm_folder = ds.path_full
        slurm_folder.mkdir(exist_ok=True)
        del ds

    # Format arguments
    arguments = dict(
        camera_ds_id=str(camera_ds_id),
        model_name=str(model_name),
        origin_id=str(origin_id),
        project=str(project),
        conflicts=str(conflicts),
        crop=bool(crop),
    )

    source_script = Path(__file__).parent / "slurm_scripts" / "dlc_track.py"

    slurm_helper.python_script_from_template(
        slurm_folder,
        source_script,
        target_script_name=f"{basename}.py",
        arguments=arguments,
    )
    # add a slurm script to start it
    slurm_options = dict(
        ntasks=1,
        time="12:00:00",
        mem="32G",
        gres="gpu:1",
        partition="gpu",
        output=slurm_folder / f"{basename}.out",
        error=slurm_folder / f"{basename}.err",
    )
    slurm_helper.create_slurm_sbatch(
        slurm_folder,
        script_name=f"{basename}.sh",
        python_script=slurm_folder / f"{basename}.py",
        conda_env="dlc_nogui",
        slurm_options=slurm_options,
        module_list=[
            "cuDNN/8.1.1.33-CUDA-11.2.1",
        ],
    )

    job_id = slurm_helper.run_slurm_batch(
        f"{slurm_folder / basename}.sh", job_dependency
    )
    return job_id


def fit_ellipses(
    camera_ds_id,
    project_id,
    slurm_folder,
    likelihood_threshold=None,
    job_dependency=None,
):
    """Fit DLC eye tracking output with ellipse

    This will generate a .sh and .py scripts in target_folder and use them to start a
    sbatch job.

    Args:
        camera_ds_id (str): Hexadecimal code of the camera dataset on flexilims
        likelihood_threshold (float, optional): Likelihood value to exclude points from
        fit. Defaults to None.
        slurm_folder (str, optional): Path to create the slurm script, python and
            slurms log files. If None, will use target_folder
        job_dependency (str, optional): Job id to wait for before starting this job.

    Returns:
        subprocess.process: Process running the job
    """

    python_script = Path(slurm_folder) / "fit_ellipses.py"

    # Make a python script
    if likelihood_threshold is not None:
        likelihood_threshold = float(likelihood_threshold)

    arguments = dict(
        camera_ds_id=str(camera_ds_id),
        project_id=str(project_id),
        likelihood=likelihood_threshold,
    )

    source = (
        Path(__file__).parent / "slurm_scripts" / "post_dlc_ellipse_fit.py"
    ).read_text()
    for k, v in arguments.items():
        source = source.replace(f'"XXX_{k.upper()}_XXX"', repr(v))
    with open(python_script, "w") as fhandle:
        fhandle.write(source)

    slurm_helper.create_slurm_sbatch(
        slurm_folder,
        script_name="fit_ellipses.sh",
        python_script=python_script,
        conda_env="cottage_analysis",
        slurm_options=None,
        module_list=None,
    )

    # Now run the job
    job_id = slurm_helper.run_slurm_batch(
        slurm_folder / "fit_ellipses.sh", job_dependency
    )
    return job_id


def reproject_pupils(camera_dataset_name, project, target_folder, phi0, theta0):
    """Find best eye parameters and eye rotation to reproject pupils

    There are two solutions for each ellipse fit. Only one is selected by limiting the
    search in +/- pi/2 around phi0 and theta0

    This will generate a .sh and .py scripts in target_folder and use them to start a
    sbatch job.

    Args:
        camera_dataset_name (str): Name of the camera dataset as on flexilims
        project (str): Name of the project
        target_folder (str): Full path to save data
        phi0 (float): Centre phi value for initial search
        theta0 (float): Centre theta value for initial search

    Returns:
        subprocess.process: Process running the job
    """
    target_folder = Path(target_folder)

    python_script = target_folder / "find_gaze.py"

    # Make a python script
    arguments = dict(
        camera_dataset_name=str(camera_dataset_name),
        target_folder=str(target_folder),
        project=project,
        plot=True,
        phi0=phi0,
        theta0=theta0,
    )

    source = (
        Path(__file__).parent / "slurm_scripts" / "reproject_ellipse_kerr.py"
    ).read_text()
    for k, v in arguments.items():
        source = source.replace(f'"XXX_{k.upper()}_XXX"', repr(v))
    with open(python_script, "w") as fhandle:
        fhandle.write(source)

    slurm_helper.create_slurm_sbatch(
        target_folder,
        script_name="find_gaze.sh",
        python_script=python_script,
        conda_env="cottage_analysis",
        slurm_options=dict(mem="8G", time="24:00:00"),
        module_list=None,
    )

    # Now run the job
    command = f"sbatch {target_folder / 'find_gaze.sh'}"
    proc = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc
