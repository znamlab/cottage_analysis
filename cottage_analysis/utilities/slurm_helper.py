"""Function to help to generate and run slurm scripts"""
from pathlib import Path


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
            boiler = "\n".join([f"ml {module}" for module in module_list])
        else:
            boiler = "\n"
        boiler += "\n".join(
            [
                "ml Anaconda3",
                "source /camp/apps/eb/software/Anaconda/conda.env.sh",
                "source activate base",
                f"conda activate {conda_env}",
                f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/{conda_env}/lib/",
                "",
            ]
        )
        fhandle.write(boiler)

        # and the real call
        fhandle.write(f"\n\npython {python_script}\n")


def python_script_from_template(target_folder, source_script, arguments=None):
    if arguments is None:
        arguments = {}
    source = Path(source_script).read_text()
    for k, v in arguments.items():
        source = source.replace(f"XXX_{k.upper()}_XXX", str(v))
    python_script = Path(target_folder) / source_script.name
    with open(python_script, "w") as fhandle:
        fhandle.write(source)
