import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import flexiznam as flz
import subprocess
import shlex

from cottage_analysis.analysis import common_utils
from functools import partial

print = partial(print, flush=True)


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
    project, session_name, pipeline_filename, conflicts, photodiode_protocol
):
    """Start sbatch script to run analysis_pipeline on a single session.

    Args:

    """

    script_path = str(
        Path(__file__).parent.parent.parent / "sbatch" / pipeline_filename
    )

    log_fname = f"{session_name}_%j.out"

    log_path = str(Path(__file__).parent.parent.parent / "logs" / f"{log_fname}")

    args = f"--export=PROJECT={project},SESSION_NAME={session_name},CONFLICTS={conflicts},PHOTODIODE_PROTOCOL={photodiode_protocol}"

    args = args + f" --output={log_path}"

    command = f"sbatch {args} {script_path}"
    print(command)
    subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
