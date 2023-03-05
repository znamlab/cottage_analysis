import subprocess, shlex
from pathlib import Path
import flexiznam as flm
from cottage_analysis.eye_tracking import eye_model_fitting
from cottage_analysis.utilities import slurm_helper


def dlc_track(
    video_path,
    model_name,
    target_folder,
    crop=False,
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
        target_folder (str): Folder to save results
        crop (bool, optional): Crop the video before tracking. If True crop info must be
            defined either in f"{video.stem}_crop_tracking.yml" or as previous dlc track
            see slurm_scripts/dlc_track_filter_label.py for more info. Defaults to False
        origin_id (str, optional): Hexadecimal code of the origin on flexilims. If
            not None, a flexilims entry with be created from this origin, otherwise
            nothing is uploaded. Defaults to None.
        project (str, optional): Mandatory is `origin_id` is not None. Name of the
            project on flexilims. Defaults to None.
        filter (bool, optional): Filter prediction. Defaults to False.
        label (bool, optional): Generate a labeled copy of the video. Defaults to False.

    Returns:
        subprocess.Process: The process job
    """
    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=True)

    processed_path = Path(flm.PARAMETERS["data_root"]["processed"])
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
        crop=bool(crop),
    )
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
                "source /camp/apps/eb/software/Anaconda/conda.env.sh",
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


def reproject_pupils(camera_dataset_name, project, target_folder):
    """Find best eye parameters and eye rotation to reproject pupils

    This will generate a .sh and .py scripts in target_folder and use them to start a
    sbatch job.

    Args:
        camera_dataset_name (str): Name of the camera dataset as on flexilims
        project (str): Name of the project
        target_folder (str): Full path to save data

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
