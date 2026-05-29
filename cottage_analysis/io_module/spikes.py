from pathlib import Path
import numpy as np
import warnings
import pandas as pd
from cottage_analysis.utilities import time_series_analysis as tsa
from cottage_analysis.ephys import aind_utils


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
    # all all the cluster_SOMETHING.tsv
    for target in kilosort_folder.glob("cluster_*.tsv"):
        w = target.stem[len("cluster_") :]
        ks_data[w] = pd.read_csv(target, sep="\t")
    if "info" not in ks_data:
        if "decoder_label" in ks_data:
            print("Using automatic decoder label for spike sorting")
            lab = ks_data["decoder_label"].copy()
            sua = lab["decoder_label"] == "sua"
            lab.loc[sua, "decoder_label"] = "good"
            lab.columns = ["cluster_id", "group"]
            ks_data["info"] = lab
        else:
            raise ValueError("Could not find label")
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


def load_aind_folder(aind_folder, return_multiunit=True):
    """Return aind_data, good_units and mua_units from aind sorting

    Args:
        aind_folder (str): main folder containing the output of the AIND pipeline
        return_multiunit (bool, optional): return only 'good' or 'SUA' if True,
            otherwise return everything but 'noise' in mua_units

    Returns:
        pd.DataFrame: kilosort data
        dict: good units
        dict: multiunits
    """

    analyzer = aind_utils.load_aind_analyzer(aind_folder)
    _ = aind_utils.get_aind_curation_labels(aind_folder, analyzer=analyzer)

    all_spikes = analyzer.sorting.to_spike_vector()
    unit_ids = analyzer.unit_ids

    aind_data = {
        "analyzer": analyzer,
        "times": all_spikes["sample_index"],
        "clusters": unit_ids[all_spikes["unit_index"]],
    }

    # info
    info = pd.DataFrame({"cluster_id": unit_ids})
    if "label" in analyzer.sorting.get_property_keys():
        labels = analyzer.sorting.get_property("label")
        # Map AIND labels to KS-like groups
        # aind labels are usually 'sua', 'mua', 'noise', 'unsorted'
        group_map = {"sua": "good", "mua": "mua", "noise": "noise", "good": "good"}
        info["group"] = [group_map.get(l, "unsorted") for l in labels]
    else:
        info["group"] = "unsorted"
    aind_data["info"] = info

    # good units (SUA)
    good_ids = info[info.group == "good"].cluster_id.values
    good_units = {}
    for uid in good_ids:
        good_units[uid] = analyzer.sorting.get_unit_spike_train(uid)

    if not return_multiunit:
        return aind_data, good_units

    # mua units (MUA)
    mua_ids = info[info.group == "mua"].cluster_id.values
    mua_units = {}
    for uid in mua_ids:
        mua_units[uid] = analyzer.sorting.get_unit_spike_train(uid)

    return aind_data, good_units, mua_units


def get_smoothed_spike_rate(
    units, bins, exp_sd=0.1, conflicts="skip", save_folder=None
):
    """Smooth spike rate with exponential kernel.

    If conflicts is not `overwrite`, will try to reuse last saved spike rate file with
    same exp_sd.

    Args:
        units (dict): dictionary of units
        bins (np.ndarray): time bins
        exp_sd (float, optional): standard deviation of exponential kernel. Defaults to 0.1.
        return_multiunit (bool, optional): whether to return multiunits. Defaults to False.
        conflicts (str, optional): how to deal with conflicts. Defaults to 'skip'.

    Returns:
        np.ndarray: spike rate for each unit
        list: unit ids
    """
    spks = np.zeros((len(bins) - 1, len(units)))

    if exp_sd is not None:
        resave = False
        rate_file = save_folder / f"spike_rate_exp_sd.npz"
        if rate_file.exists() and conflicts != "overwrite":
            print(f"Loading spike rate from {rate_file}...")
            loaded_spks = dict(np.load(rate_file))
            if np.any(loaded_spks["bins"] != bins) or (loaded_spks["exp_sd"] != exp_sd):
                loaded_spks = {"bins": bins, "exp_sd": exp_sd}
                warnings.warn(
                    "Loaded spike rate bins do not match current bins."
                    + " Re-calculating spike rate..."
                )
        else:
            loaded_spks = {"bins": bins, "exp_sd": exp_sd}

    unit_ids = np.array(list(sorted(units.keys())))
    for iu, unit in enumerate(unit_ids):
        spk = units[unit]
        valid_spk = spk[(spk > bins[0]) & (spk < bins[-1])]
        if exp_sd is not None:
            if f"unit{unit}" in loaded_spks:
                rate = loaded_spks[f"unit{unit}"]
            else:
                resave = True
                print(
                    f"Filtering unit {unit} with exp_sd={exp_sd}..."
                    + f" ({iu+1}/{len(units)})"
                )
                if not len(valid_spk):
                    warnings.warn(f"Unit {unit} has no valid spike.")
                    rate = np.zeros(len(bins) - 1)
                else:
                    time, filtered = tsa.half_exp_density(valid_spk, sd=exp_sd)
                    filtered /= np.diff(time[:2])
                    rate = filtered[time.searchsorted(bins[:-1])]
                loaded_spks[f"unit{unit}"] = rate
        else:
            rate = np.histogram(units[unit], bins=bins)[0] / np.diff(bins[:2])
        spks[:, iu] = rate

    if (exp_sd is not None) and (save_folder is not None) and resave:
        print(f"Saving spike rate to {rate_file}...")
        np.savez(rate_file, **loaded_spks)
    return spks, unit_ids
