import flexiznam as flz
import warnings
import numpy as np
import pandas as pd
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis.fit_gaussian_blob import fit_sftf_tuning


def analyze_grating_responses(
    project,
    session,
    filter_datasets=None,
    photodiode_protocol=5,
    protocol_base="SFTF",
    return_volumes=True,
    use_col="dffs",
):
    """Single-trial grating responses of the SFTF recordings of a session.

    Args:
        use_col (str): response, "dffs" (dF/F) or "spks" (deconvolved spikes). The
            integer ROI columns of the returned trials_df hold its mean over each trial,
            and `dff_stim` (dF/F) or `spks_stim` (spikes) the trial's volumes. Defaults to
            "dffs".
    """
    flexilims_session = flz.get_flexilims_session(project_id=project)

    recordings = flz.get_children(
        flexilims_session=flexilims_session,
        parent_name=session,
        children_datatype="recording",
    )
    recordings = recordings[recordings.name.str.contains(protocol_base)]
    if len(recordings) > 1:
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
            add_spikes=use_col == "spks",
        )

        trials_df, dff_mean = generate_trials_df(img_df, use_col=use_col)
        trials_df["irecording"] = i
        dfs.append(trials_df)
        dff_mean_all.append(dff_mean)
        continue
    return pd.concat(dfs, axis=0, ignore_index=True), dff_mean_all


def generate_trials_df(img_df, skip_first_n_volumes=2, use_col="dffs"):
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
    # skip the onset volumes; the window still ends at the next trial's onset
    trials_df["stim_start"] = trials_df["imaging_volume"] + skip_first_n_volumes
    trials_df["stim_end"] = trials_df["imaging_volume"].shift(-1)
    # drop the last row
    trials_df = trials_df.iloc[:-1].copy()

    # Assign dffs (or spks) array to trials_df
    stim_col = {"dffs": "dff_stim", "spks": "spks_stim"}[use_col]
    trials_df[stim_col] = trials_df.apply(
        lambda x: np.stack(
            img_df[use_col].loc[int(x.stim_start) : int(x.stim_end)]
        ).squeeze(),
        axis=1,
    )
    # mean response over [stim_start, stim_end), one integer column per ROI (read by
    # fit_sftf_tuning)
    dff_mean = [
        np.stack(img_df[use_col].loc[int(start) : int(end) - 1])
        .reshape(int(end) - int(start), -1)
        .mean(axis=0)
        for start, end in zip(trials_df["stim_start"], trials_df["stim_end"])
    ]
    trials_df = pd.concat(
        [trials_df.reset_index(drop=True), pd.DataFrame(np.stack(dff_mean, axis=0))],
        axis=1,
    )
    return trials_df, dff_mean
