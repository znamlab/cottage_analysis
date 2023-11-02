import os
import numpy as np
import pandas as pd
import defopt
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import (
    spheres,
    gratings,
    find_depth_neurons,
    pipeline_utils,
    common_utils,
    fit_gaussian_blob,
    size_control,
)
from cottage_analysis.plotting import basic_vis_plots, grating_plots, plotting_utils


def main(
    project,
    session_name,
    filter_datasets=None,
    conflicts="skip",
    photodiode_protocol=5,
    protocol_base="SFTF",
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        filter_datasets(dict): filter datasets by suite2p keywords. Default is None. For anatomical segmentation, use {"anatomical_only": 3}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
    """
    print(
        f"------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------"
    )
    flexilims_session = flz.get_flexilims_session(project)

    # Analyze & concat all SFTF recordings from a session
    trials_df_all, dff_mean_all = gratings.analyze_grating_responses(
        project=project,
        session=session_name,
        filter_datasets=filter_datasets,
        photodiode_protocol=photodiode_protocol,
        protocol_base=protocol_base,
    )
    return trials_df_all


def plot_sftf_roi(project, session_name, trials_df_all, roi, mode="fitted"):
    """Plot grating tuning for a given ROI

    Args:
        project (str): project name
        session_name(str): {Mouse}_{Session}
        trials_df_all (pd.DataFrame): dataframe containing trials_df of all recordings
        roi (int): which ROI to visualize
        mode (str, optional): "fitted" or "raw". Defaults to "fitted".
    """
    flexilims_session = flz.get_flexilims_session(project)
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=None,
        conflicts="skip",
    )
    os.makedirs(neurons_ds.path_full.parent / f"plots/roi{roi}", exist_ok=True)

    sf_range = np.sort(np.unique(trials_df_all.SpatialFrequency))
    tf_range = np.sort(np.unique(trials_df_all.TemporalFrequency))
    angle_range = np.sort(np.unique(trials_df_all.Angle))

    if mode == "raw":
        numerical_columns = trials_df_all.loc[
            :, trials_df_all.columns.str.isnumeric().isna()
        ]
        named_columns = trials_df_all[
            ["Angle", "SpatialFrequency", "TemporalFrequency"]
        ]
        select_df = pd.concat([numerical_columns, named_columns], axis=1)

        # Plot grating tuning with the raw data
        plt.figure()
        grating_plots.plot_sftf_tuning(dff_mean=select_df, roi=roi)
        plt.savefig(
            neurons_ds.path_full.parent / f"plots/roi{roi}" / f"sftf_tuning{roi}.png",
            dpi=300,
        )

    elif mode == "fitted":
        grating_plots.plot_sftf_fit(
            neuron_series=neurons_df_sftf.iloc[roi],
            sf_range=[
                np.log(np.sort(np.unique(trials_df_sftf.SpatialFrequency))[0]),
                np.log(np.sort(np.unique(trials_df_sftf.SpatialFrequency))[-1]),
            ],
            tf_range=[
                np.log(np.sort(np.unique(trials_df_sftf.TemporalFrequency))[0]),
                np.log(np.sort(np.unique(trials_df_sftf.TemporalFrequency))[-1]),
            ],
            sf_ticks=np.sort(trials_df_sftf.SpatialFrequency.unique()),
            tf_ticks=np.sort(trials_df_sftf.TemporalFrequency.unique()),
            min_sigma=0.25,
        )
        plt.savefig(
            neurons_ds.path_full.parent / f"plots/roi{roi}" / f"sftf_fit{roi}.png",
            dpi=300,
        )
