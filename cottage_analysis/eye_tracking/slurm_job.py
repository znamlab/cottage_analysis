import subprocess, shlex
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import flexiznam as flz
from cottage_analysis.eye_tracking import eye_model_fitting
from cottage_analysis.utilities import slurm_helper


def slurm_dlc_pupil(
    video_path,
    model_name,
    target_folder,
    crop_info=None,
    origin_id=None,
    project=None,
    filter=False,
    label=False,
):
    """Start slurm job to track pupil

    Args:
        video_path (str): Full path to video file
        model_name (str): Name of the model to use. Must be in the `DLC_models` shared
            project folder
        target_folder (str, optional): Folder to save results. Folder name will be used
            as dataset name on flexilims if origin_id is not None
        crop_info (list, optional): Cropping limit to run dlc on video subset. Must be a
            list [xmin, xmax, ymin, ymax]. Defaults to None
        origin_id (str, optional): Hexadecimal code of the origin on flexilims. If
            not None, a flexilims entry with be created from this origin, otherwise
            nothing is uploaded. Defaults to None.
        project (str, optional): Mandatory if `origin_id` is not None. Name of the
            project on flexilims. Defaults to None.
        filter (bool, optional): Filter predictions. Defaults to False.
        label (bool, optional): Generate a labeled copy of the video. Defaults to False.

    Returns:
        subprocess.Process: The process job
    """

    assert Path(video_path).exists(), f"Video {video_path} does not exist"

    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=True, parents=True)

    processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
    config_file = processed_path / "DLC_models" / model_name / "config.yaml"

    slurm_script = target_folder / "dlc_track.sh"
    python_script = target_folder / "dlc_track.py"

    # Make a python script
    video_path = Path(video_path)
    assert video_path.exists()
    arguments = dict(
        video=str(video_path),
        model=str(config_file),
        target=str(target_folder),
        filter=bool(filter),
        label=bool(label),
    )
    if crop_info is not None:
        arguments["crop_info"] = list(crop_info)

    if origin_id is not None:
        if project is None:
            raise IOError("`project` must be specified if `origin_id` is not None")
        arguments["project"] = str(project)
        arguments["origin_id"] = str(origin_id)

    source = (
        Path(__file__).parent / "slurm_scripts" / "dlc_track_filter_label.py"
    ).read_text()
    for k, v in arguments.items():
        source = source.replace(f'"XXX_{k.upper()}_XXX"', repr(v))
    with open(python_script, "w") as fhandle:
        fhandle.write(source)

    # add a slurm script to start it
    slurm_options = dict(
        ntasks=1,
        time="12:00:00",
        mem="32G",
        gres="gpu:1",
        partition="gpu",
        output=target_folder / "dlc_track.out",
        error=target_folder / "dlc_track.err",
    )
    slurm_options["job-name"] = f"dlc_{model_name}"
    with open(slurm_script, "w") as fhandle:
        fhandle.write("#!/bin/bash\n")
        options = "\n".join([f"#SBATCH --{k}={v}" for k, v in slurm_options.items()])
        fhandle.writelines(options)
        # add some boilerplate code
        boiler = "\n".join(
            [
                "",
                "ml cuDNN/8.1.1.33-CUDA-11.2.1",
                "ml Anaconda3",
                "source activate base",
                "conda activate dlc_nogui",
                "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/dlc_nogui/lib/",
                "",
            ]
        )
        fhandle.write(boiler)

        # and the real call
        fhandle.write(f"\n\npython {python_script}\n")

    # Now run the job
    command = f"sbatch {slurm_script}"
    print(command)
    proc = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc


def create_crop_file(camera_ds, dlc_ds):
    """Create a crop file for DLC tracking

    Uses the results of the uncropped tracking to find the crop area and save it in a
    crop file. This crop file can then be used to crop the video before tracking.

    Args:
        camera_ds (flexilims.Dataset): Camera dataset, must contain project information
        dlc_ds (flexilims.Dataset): dlc_tracking dataset, containing uncropped tracking

    Returns:
        dict: Crop information

    """
    if camera_ds.project is None:
        raise IOError("Camera dataset has no project information")

    metadata_path = camera_ds.path_full / camera_ds.extra_attributes["metadata_file"]
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if crop_file.exists():
        print("Crop file already exists. Delete manually to redo")
        with open(crop_file, "r") as fhandle:
            crop_info = yaml.safe_load(fhandle)
        return crop_info

    with open(metadata_path, "r") as fhandle:
        metadata = yaml.safe_load(fhandle)
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    print("Creating crop file")
    dlc_res = pd.read_hdf(dlc_file)
    # Find DLC crop area
    borders = np.zeros((4, 2))
    for iw, w in enumerate(
        (
            "left_eye_corner",
            "right_eye_corner",
            "top_eye_lid",
            "bottom_eye_lid",
        )
    ):
        vals = dlc_res.xs(w, level=1, axis=1)
        vals.columns = vals.columns.droplevel("scorer")
        v = np.nanmedian(vals[["x", "y"]].values, axis=0)
        borders[iw, :] = v

    borders = np.vstack([np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)])
    borders += ((np.diff(borders, axis=0) * 0.2).T @ np.array([[-1, 1]])).T
    for i, w in enumerate(["Width", "Height"]):
        borders[:, i] = np.clip(borders[:, i], 0, metadata[w])
    borders = borders.astype(int)
    crop_info = dict(
        xmin=int(borders[0, 0]),
        xmax=int(borders[1, 0]),
        ymin=int(borders[0, 1]),
        ymax=int(borders[1, 1]),
        dlc_source=str(dlc_ds.path),
        dlc_ds_id=dlc_ds.id,
    )
    with open(crop_file, "w") as fhandle:
        yaml.dump(crop_info, fhandle)
    print("Crop file created")
    return crop_info


def fit_ellipses(dlc_file, target_folder, likelihood_threshold=None):
    """Fit DLC eye tracking output with ellipse

    This will generate a .sh and .py scripts in target_folder and use them to start a
    sbatch job.

    Args:
        dlc_file (str): Full path to dlc .h5 file
        target_folder (str): Full path to save data
        likelihood_threshold (float, optional): Likelihood value to exclude points from
        fit. Defaults to None.

    Returns:
        subprocess.process: Process running the job
    """
    target_folder = Path(target_folder)

    python_script = target_folder / "fit_ellipses.py"

    # Make a python script
    dlc_file = Path(dlc_file)
    assert dlc_file.exists()
    arguments = dict(dlc_file=str(dlc_file), target=str(target_folder))
    if likelihood_threshold is not None:
        arguments["likelihood"] = float(likelihood_threshold)

    source = (
        Path(__file__).parent / "slurm_scripts" / "post_dlc_ellipse_fit.py"
    ).read_text()
    for k, v in arguments.items():
        source = source.replace(f'"XXX_{k.upper()}_XXX"', repr(v))
    with open(python_script, "w") as fhandle:
        fhandle.write(source)

    slurm_helper.create_slurm_sbatch(
        target_folder,
        script_name="fit_ellipses.sh",
        python_script=python_script,
        conda_env="cottage_analysis",
        slurm_options=None,
        module_list=None,
    )

    # Now run the job
    command = f"sbatch {target_folder / 'fit_ellipses.sh'}"
    print(command)
    proc = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc


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
    print(command)
    proc = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return proc
