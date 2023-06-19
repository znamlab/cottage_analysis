"""Function to help to generate and run slurm scripts"""
from pathlib import Path
import subprocess
import shlex


def run_slurm_batch(script_path, job_dependency=None):
    """Run a slurm script

    Args:
        script_path (str): Full path to the script
        job_dependency (str, optional): Job ID that needs to finish before running
            sbtach. Defaults to None.

    Returns:
        str: Job ID of the sbatch job
    """
    if job_dependency is not None:
        dep = f"--dependency=afterok:{job_dependency} "
    else:
        dep = ""
    command = f"sbatch {dep}{script_path}"

    procout = subprocess.check_output(shlex.split(command))
    # get the job id
    job_id = procout.decode("utf-8").split(" ")[-1].strip()
    return job_id


def create_slurm_sbatch(
    target_folder,
    script_name,
    python_script,
    conda_env,
    slurm_options=None,
    module_list=None,
):
    """Create a slurm sh script that will call a python script

    Args:
        target_folder (str): Where to write the script?
        script_name (str): Name of the script
        python_script (str): Path to the python script
        conda_env (str): Name of the conda environment to load
        slurm_options (dict, optional): Options to give to sbatch. Defaults to None.
        module_list (list, optional): List of modules to load before calling the python
            script. Defaults to None.
    """
    target_folder = Path(target_folder)
    default_options = dict(
        ntasks=1,
        time="12:00:00",
        mem="32G",
        partition="cpu",
        output=target_folder / script_name.replace(".sh", ".out"),
        error=target_folder / script_name.replace(".sh", ".err"),
    )
    if slurm_options is None:
        slurm_options = {}

    slurm_options = dict(default_options, **slurm_options)

    with open(target_folder / script_name, "w") as fhandle:
        fhandle.write("#!/bin/bash\n")
        options = "\n".join([f"#SBATCH --{k}={v}" for k, v in slurm_options.items()])
        fhandle.writelines(options)
        # add some boilerplate code
        if module_list is not None:
            boiler = "\n" + "\n".join([f"ml {module}" for module in module_list]) + "\n"
        else:
            boiler = "\n"
        boiler += "\n".join(
            [
                "ml Anaconda3",
                "source activate base",
                f"conda activate {conda_env}",
                f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/{conda_env}/lib/",
                "",
            ]
        )
        fhandle.write(boiler)

        # and the real call
        fhandle.write(f"\n\npython {python_script}\n")


def python_script_from_template(
    target_folder, source_script, target_script_name=None, arguments=None
):
    """Create a python script from a template

    Arguments in the template should be of the form XXX_ARGUMENT_XXX. They will be
    replaced by the value of `arguments["ARGUMENT"]`

    Args:
        target_folder (str): Where to write the script?
        source_script (str): Path to the template script
        target_script_name (str, optional): Name of the target script if different from
            source_script. Defaults to None.
        arguments (dict, optional): Dictionary of arguments to replace in the template.
            Defaults to None.
    """
    source_script = Path(source_script)
    if arguments is None:
        arguments = {}
    source = source_script.read_text()
    for k, v in arguments.items():
        source = source.replace(f'"XXX_{k.upper()}_XXX"', repr(v))
    if target_script_name is None:
        target_script_name = source_script.name
    python_script = Path(target_folder) / target_script_name
    with open(python_script, "w") as fhandle:
        fhandle.write(source)
