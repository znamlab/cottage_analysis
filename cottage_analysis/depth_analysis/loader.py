import pickle
import numpy as np
from pathlib import Path
import flexiznam as flz


def get_recording_data(recording, flexilims_session, two_photon=True):

    processed = Path(flz.PARAMETERS["data_root"]["processed"])
    out = dict()

    # get vis stim
    with open(processed / recording.path / "img_VS.pickle", "rb") as handle:
        out["param_logger"] = pickle.load(handle)
    with open(processed / recording.path / "stim_dict.pickle", "rb") as handle:
        out["stim_dict"] = pickle.load(handle)

    # get suite2p data
    sess_ds = flz.get_children(
        parent_id=recording.origin_id,
        flexilims_session=flexilims_session,
        children_datatype="dataset",
    )
    suite2p_roi = sess_ds[sess_ds.dataset_type == "suite2p_rois"]
    assert len(suite2p_roi) == 1
    suite2p_roi = flz.Dataset.from_flexilims(
        data_series=suite2p_roi.iloc[0], flexilims_session=flexilims_session
    )
    out["ops"] = np.load(
        suite2p_roi.path_full / "suite2p" / "plane0" / "ops.npy", allow_pickle=True
    ).item()

    if not two_photon:
        # stop here with just behaviour
        return out

    # also load 2p data traces
    out["iscell"] = np.load(
        suite2p_roi.path_full / "suite2p" / "plane0" / "iscell.npy", allow_pickle=True
    )[:, 0]
    rec_ds = flz.get_children(
        parent_id=recording.id,
        flexilims_session=flexilims_session,
        children_datatype="dataset",
    )
    suite2p_traces = rec_ds[rec_ds.dataset_type == "suite2p_traces"]
    assert len(suite2p_traces) == 1
    suite2p_traces = flz.Dataset.from_flexilims(
        data_series=suite2p_traces.iloc[0], flexilims_session=flexilims_session
    )
    for datafile in suite2p_traces.path_full.glob('*.npy'):
        out[datafile.stem] = np.load(datafile, allow_pickle=True)
    
    project = [k for k,v in flz.PARAMETERS['project_ids'].items() if v==flexilims_session.project_id][0]
    analysis_folder = processed / project / 'Analysis' / Path(*recording.genealogy) / 'plane0'
    import os
    os.listdir(analysis_folder)
    return out