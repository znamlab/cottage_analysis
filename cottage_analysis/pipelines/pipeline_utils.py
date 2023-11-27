import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy
import flexiznam as flz

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



# # session paths
# assert flexilims_session is not None or project is not None
# if flexilims_session is None:
#     flexilims_session = flz.get_flexilims_session(project_id=project)
# exp_session = flz.get_entity(
#     datatype="session", name=session_name, flexilims_session=flexilims_session
# )
# root = Path(flz.PARAMETERS["data_root"]["processed"])
# session_folder = root / exp_session.path
    
# # if neurons_ds exists on flexilims and conflicts is set to skip, load the existing neurons_df
# if (neurons_ds.flexilims_status() != "not online") and (conflicts == "skip"):
#     print("Loading existing neurons_df file...")
#     return np.load(neurons_ds.path_full), neurons_df

# # save neurons_df
# neurons_ds.path_full.parent.mkdir(parents=True, exist_ok=True)
# neurons_df.to_pickle(neurons_ds.path_full)

# # update flexilims
# neurons_ds.extra_attributes["depth_neuron_criteria"] = "anova"
# neurons_ds.extra_attributes["depth_neuron_RS_threshold"] = rs_thr
# neurons_ds.update_flexilims(mode="overwrite")


