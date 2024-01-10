from pathlib import Path
import numpy as np
import warnings
import pandas as pd


def load_kilosort_folder(kilosort_folder, return_multiunit=True):
    """Load kilosort folder and return good units and multiunits.

    Args:
        kilosort_folder (str | Path): path to kilosort folder
        return_multiunit (bool, optional): whether to return multiunits. Defaults to True.

    Returns:
        pd.DataFrame: kilosort data
        dict: good units
        dict: multiunits
    """
    kilosort_folder = Path(kilosort_folder)
    ks_data = dict()
    for w in ["times", "clusters"]:
        ks_data[w] = np.load(kilosort_folder / ("spike_%s.npy" % w)).reshape(-1)
    for w in ["group", "info"]:
        target = kilosort_folder / ("cluster_%s.tsv" % w)
        if not target.exists():
            warnings.warn("missing %s" % target)
            continue
        ks_data[w] = pd.read_csv(kilosort_folder / ("cluster_%s.tsv" % w), sep="\t")
    # get good units
    good = ks_data["info"][ks_data["info"].group == "good"]
    good_units = {}
    for cluster_id in good.cluster_id.values:
        # get unit spike index in ephys
        spike_index = ks_data["times"][ks_data["clusters"] == cluster_id]
        good_units[cluster_id] = spike_index

    if not return_multiunit:
        return ks_data, good_units
    # get mua units
    mua = ks_data["info"][ks_data["info"].group == "mua"]
    mua_units = {}
    for cluster_id in mua.cluster_id.values:
        # get unit spike index in ephys
        spike_index = ks_data["times"][ks_data["clusters"] == cluster_id]
        mua_units[cluster_id] = spike_index

    return ks_data, good_units, mua_units
