import subprocess, shlex
from pathlib import Path
import flexiznam as flm


def dlc_track(video_path, model_name, target_folder):
    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=True)

    processed_path = Path(flm.PARAMETERS["data_root"]["processed"])
    config_file = processed_path / "DLC_models" / model_name / "config.yaml"

    slurm_script = target_folder / "dlc_track.sh"
    python_script = target_folder / "dlc_track.py"

    # Make a python script
    video_path = Path(video_path)
    assert video_path.exists()
    arguments = dict(video=video_path, model=config_file, target=target_folder)
    source = (Path(__file__).parent / "label_video.py").read_text()
    for k, v in arguments.items():
        source = source.replace(f"XXX_{k.upper()}_XXX", str(v))
    with open(python_script, "w") as fhandle:
        fhandle.write(source)

    # add a slurm script to start it
    slurm_options = dict(
        ntasks=1,
        time="12:00:00",
        mem="32G",
        gres="gpu:1",
        partition="gpu",
        output=target_folder / "eye_track.out",
        error=target_folder / "eye_track.err",
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
