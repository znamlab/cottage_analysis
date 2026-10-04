import flexiznam as flz
import warnings
import numpy as np
import pandas as pd
from cottage_analysis.preprocessing import synchronisation
#from cottage_analysis.analysis.fit_gaussian_blob import fit_sftf_tuning


def analyze_grating_responses(
    project,
    session,
    filter_datasets=None,
    photodiode_protocol=5,
    protocol_base="SFTF",
    return_volumes=True,
):
    flexilims_session = flz.get_flexilims_session(project_id=project)

    recordings = flz.get_children(
        flexilims_session=flexilims_session,
        parent_name=session,
        children_datatype="recording",
    )
    recordings = recordings[recordings.name.str.contains(protocol_base)]
    if len(recordings) >1 :
        warnings.warn(f"Multiple recordings found for {session}, using the first one.")
    dfs = []
    dff_mean_all = []
    for i, recording in recordings.iterrows():
        vs_df = synchronisation.generate_vs_df(
            recording=recording,
            photodiode_protocol=photodiode_protocol,
            flexilims_session=flexilims_session,
            project=project,
        )
        img_df = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=recording,
            flexilims_session=flexilims_session,
            filter_datasets=filter_datasets,
            return_volumes=return_volumes,
        )

        trials_df, dff_mean = generate_trials_df(img_df)
        trials_df["irecording"] = i
        dfs.append(trials_df)
        dff_mean_all.append(dff_mean)
        continue
    return pd.concat(dfs, axis=0, ignore_index=True), dff_mean_all


def generate_trials_df(img_df, skip_first_n_volumes=2):
    # select rows of img_df where SpatialFrequency, TemporalFrequency, Angle change
    trials_df = (
        img_df.loc[
            img_df[["SpatialFrequency", "TemporalFrequency", "Angle"]]
            .diff()
            .any(axis=1)
        ]
        .copy()
        .reset_index(drop=True)
    )
    trials_df["stim_start"] = trials_df["imaging_volume"] + skip_first_n_volumes
    trials_df["stim_end"] = trials_df["stim_start"].shift(-1)
    # drop the last row
    trials_df = trials_df.iloc[:-1].copy()

    # Assign dffs array to trials_df
    trials_df["dff_stim"] = trials_df.apply(
        lambda x: np.stack(
            img_df.dffs.loc[int(x.stim_start) : int(x.stim_end)]
        ).squeeze(),
        axis=1,
    )
    dff_mean = trials_df["dff_stim"].apply(lambda x: np.mean(x, axis=0)).to_list()
    return trials_df, dff_mean


# Recorded in fit_meta.json, so fits made before and after a change to how
# responses are cleaned can be told apart
SFTF_CLEANING = "drop_nonfinite_per_roi"
# Likewise for stimulus/imaging synchronisation. Before 9f422c3 (on main), param
# logs were matched to FrameLog row numbers instead of FrameIndex, so in
# recordings whose FrameIndex does not start at 0 (Dec 2025 onwards) every
# stimulus was labelled ~200-270 monitor frames (~1.5-1.9 s) late.
SFTF_SYNC = "frameindex_9f422c3"


def sftf_fit_outdated(meta):
    """Why a saved SFTF fit must be redone, or None if it is current.

    Args:
        meta (dict or None): contents of fit_meta.json.
    """
    if meta is None:
        return "no fit_meta.json"
    if meta.get("sync") != SFTF_SYNC:
        return f"sync={meta.get('sync')!r}, not {SFTF_SYNC!r} (stimulus labels misaligned)"
    if meta.get("cleaning") != SFTF_CLEANING:
        return f"cleaning={meta.get('cleaning')!r}, not {SFTF_CLEANING!r}"
    return None


def format_sftf_trials(trials_df):
    """Turn trials_df into the layout fit_sftf_tuning expects.

    Each ROI's response on a trial is its dF/F averaged over the stimulus
    window, in an integer-named column (the ROI id). inf, which comes from a
    near-zero F0, is set to NaN; nothing else is changed, and fit_sftf_tuning
    leaves NaN trials out of that ROI's fit.

    Args:
        trials_df (pd.DataFrame): output of analyze_grating_responses.

    Returns:
        pd.DataFrame: `SpatialFrequency`, `TemporalFrequency`, `Angle` and one
            column per ROI.
    """
    response_matrix = np.stack(
        trials_df["dff_stim"].apply(lambda x: np.mean(x, axis=0)).values
    )
    response_matrix[~np.isfinite(response_matrix)] = np.nan
    responses_df = pd.DataFrame(
        response_matrix,
        columns=np.arange(response_matrix.shape[1]),
        index=trials_df.index,
    )
    return pd.concat(
        [trials_df[["SpatialFrequency", "TemporalFrequency", "Angle"]], responses_df],
        axis=1,
    )


def summarize_sftf_fit(neurons_df):
    """Count ROIs affected by missing or extreme responses in a fit_sftf_tuning output.

    Returns:
        dict: counts, suitable for printing and for fit_meta.json.
    """
    return {
        "n_rois": len(neurons_df),
        "n_rois_with_nonfinite_trials": int((neurons_df["n_nonfinite_trials"] > 0).sum()),
        "n_rois_with_extreme_trials": int((neurons_df["n_extreme_trials"] > 0).sum()),
        "n_rois_not_fit": int(neurons_df["rsq"].isna().sum()),
    }
