import os
import numpy as np
import pandas as pd
import flexiznam as flz
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.analysis import common_utils
from cottage_analysis.io_module import suite2p as s2p_io
from scipy import stats
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm
from cottage_analysis.summary_analysis import get_session_list
from pathlib import Path


def concatenate_all_neurons_df(
    flexilims_session,
    session_list,
    filename="neurons_df.pickle",
    cols=None,
    read_iscell=True,
    verbose=False,
    filter_datasets=None,
):
    if filter_datasets is None:
        filter_datasets = {"anatomical_only": 3}
    isess = 0
    for session in session_list:
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        if os.path.exists(neurons_ds.path_full.parent / filename):
            print(f"Concatenating {neurons_ds.path_full.parent / filename}...")
            neurons_df = pd.read_pickle(neurons_ds.path_full.parent / filename)
            if isinstance(neurons_df, dict):
                neurons_df_temp = pd.DataFrame(columns=cols, index=[0])
                neurons_df = common_utils.dict2df(neurons_df, neurons_df_temp, cols, 0)
            if (cols is None) or (set(cols).issubset(neurons_df.columns.tolist())):
                if cols is None:
                    neurons_df = neurons_df
                else:
                    neurons_df = neurons_df[cols]
                suite2p_ds = flz.get_datasets(
                    flexilims_session=flexilims_session,
                    origin_name=session,
                    dataset_type="suite2p_rois",
                    filter_datasets=filter_datasets,
                    allow_multiple=False,
                    return_dataseries=False,
                )
                if read_iscell:
                    iscell = s2p_io.load_is_cell(suite2p_ds.path_full)
                    neurons_df["iscell"] = iscell

                neurons_df["session"] = session
                if isess == 0:
                    neurons_df_all = neurons_df
                else:
                    neurons_df_all = pd.concat(
                        [neurons_df_all, neurons_df], ignore_index=True
                    )

                if verbose:
                    print(f"Finished concat {filename} from session {session}")
                isess += 1
            else:
                print(f"ERROR: SESSION {session}: specified cols not all in neurons_df")
        else:
            print(f"ERROR: SESSION {session}: {filename} not found")

    return neurons_df_all


def load_project_subsets(
    project_or_session,
    session_list=None,
    filename="ridge_decoder_neuron_subsets_motor.parquet",
    session_to_exclude=None,
):
    """Load and concatenate subset parquet files for a project.

    Args:
        project_or_session (str or flexilims.Flexilims): Project name or flexilims session.
        session_list (list of str, optional): List of sessions to load. If None, queries all V1 motor sessions.
        filename (str): Name of the subsets parquet file.
        session_to_exclude (list of str, optional): List of sessions to exclude.

    Returns:
        pd.DataFrame: Concatenated subset results.
    """

    if isinstance(project_or_session, str):
        flexilims_session = flz.get_flexilims_session(project_id=project_or_session)
    else:
        flexilims_session = project_or_session

    if session_list is None:
        session_list = get_session_list.get_motor_session_list(flexilims_session)

    project_sessions = flz.get_entities("session", flexilims_session=flexilims_session)

    all_subsets = []
    for session_name in session_list:
        if session_to_exclude is not None:
            if session_name in session_to_exclude:
                continue
        if session_name in project_sessions.index:
            nominal_depth = project_sessions.loc[session_name, "nominal_depth"]
        else:
            print(f"Session {session_name} not found in session entities. Skipping.")
            continue

        neurons_ds = flz.get_datasets(
            origin_name=session_name,
            dataset_type="neurons_df",
            flexilims_session=flexilims_session,
            filter_datasets={"annotated": True},
            allow_multiple=True,
        )
        if not neurons_ds:
            neurons_ds = flz.get_datasets(
                origin_name=session_name,
                dataset_type="neurons_df",
                flexilims_session=flexilims_session,
                allow_multiple=True,
            )

        if not neurons_ds:
            print(f"Skipping {session_name}: No neurons_df dataset found.")
            continue

        ds = neurons_ds[0]
        session_folder = Path(ds.path_full).parent
        subsets_parquet_path = session_folder / filename

        if subsets_parquet_path.exists():
            try:
                df = pd.read_parquet(subsets_parquet_path)
                df["session"] = session_name
                if isinstance(nominal_depth, (list, np.ndarray, pd.Series)):
                    resolved_depth = np.mean(nominal_depth)
                else:
                    resolved_depth = nominal_depth
                df["nominal_depth"] = resolved_depth
                all_subsets.append(df)
            except Exception as e:
                print(f"Error loading parquet for {session_name}: {e}")
        else:
            print(
                f"No subsets parquet found for {session_name} at {subsets_parquet_path}"
            )

    if all_subsets:
        return pd.concat(all_subsets, ignore_index=True)
    else:
        return pd.DataFrame()
