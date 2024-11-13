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


def concatenate_all_neurons_df(
    flexilims_session,
    session_list,
    filename="neurons_df.pickle",
    cols=None,
    read_iscell=True,
    verbose=False,
):
    isess = 0
    for session in session_list:
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        if os.path.exists(neurons_ds.path_full.parent / filename):
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
                    filter_datasets={"anatomical_only": 3},
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
