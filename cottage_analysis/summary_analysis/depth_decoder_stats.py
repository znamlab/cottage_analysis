import pandas as pd
import numpy as np

from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.analysis import common_utils

def concatenate_all_decoder_results(
    flexilims_session, session_list, filename="decoder_results.pickle"
):
    all_sessions = []
    for session in session_list:
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        filepath = neurons_ds.path_full.parent / filename
        if filepath.is_file():
            decoder_dict = pd.read_pickle(neurons_ds.path_full.parent / filename)
            decoder_dict["ndepths"] = len(decoder_dict["conmat_closedloop"])
            decoder_dict["session"] = session
            print(f"SESSION {session}: decoder_results concatenated")
            all_sessions.append(decoder_dict)
        else:
            print(f"ERROR: SESSION {session}: decoder_results not found")
            continue
    results = pd.DataFrame(all_sessions)
    return results


def calculate_error(conmat):
    conmat = np.array(conmat)
    ndepths = conmat.shape[0]
    if ndepths not in [5, 8]:
        mean_error = np.nan
    else:
        m = np.repeat(np.arange(ndepths)[np.newaxis, :], ndepths, axis=0)

        errs = np.abs(m - m.T).flatten()

        mean_error = np.nansum(conmat.flatten() * errs) / np.nansum(conmat.flatten())
        if ndepths == 5:
            mean_error = mean_error * np.log2(np.sqrt(10))
    return mean_error


def make_change_level_conmat(conmat):
    conmat = np.array(conmat)
    sum = np.sum(conmat)
    each = sum // (conmat.shape[0] ** 2)
    conmat_chance = np.ones_like(conmat) * each
    return conmat_chance


def calculate_error_all_sessions(decoder_results):
    for sfx in ["closedloop", "openloop"]:
        if f"conmat_{sfx}" in decoder_results.columns:
            decoder_results[f"conmat_chance_{sfx}"] = decoder_results[
                f"conmat_{sfx}"
            ].apply(make_change_level_conmat)
            decoder_results[f"error_{sfx}"] = decoder_results[f"conmat_{sfx}"].apply(
                calculate_error
            )
            decoder_results[f"error_chance_{sfx}"] = decoder_results[
                f"conmat_chance_{sfx}"
            ].apply(calculate_error)
            for i in range(len(decoder_results[f"conmat_speed_bins_{sfx}"].iloc[0])):
                decoder_results[f"conmat_speed_bins_{sfx}_{i}"] = (
                    decoder_results.conmat_speed_bins_closedloop.apply(lambda x: x[i])
                )
            conmat_speed_bins_cols = common_utils.find_columns_containing_string(
                decoder_results, f"conmat_speed_bins_{sfx}_"
            )
            for i, col in enumerate(conmat_speed_bins_cols):
                decoder_results[f"error_speed_bins_{sfx}_{i}"] = decoder_results[
                    col
                ].apply(calculate_error)
    return decoder_results