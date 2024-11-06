import pandas as pd
import defopt
import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    gratings,
    find_depth_neurons,
    fit_gaussian_blob,
)
from cottage_analysis.plotting import grating_plots
from cottage_analysis.pipelines import pipeline_utils


def main(project, session_name, conflicts="skip", photodiode_protocol=5):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
    """
    print(
        f"------------------------------- \n \
        Start analysing {session_name} SFTF   \n \
        -------------------------------"
    )
    flexilims_session = flz.get_flexilims_session(project)
    # Synchronisation
    print("---Start synchronisation...---")
    # # Sync sphere recordings
    # print("---Syncing sphere recordings...---")
    # vs_df_all, trials_df_all = spheres.sync_all_recordings(
    #     session_name=session_name,
    #     flexilims_session=flexilims_session,
    #     project=project,
    #     filter_datasets={"anatomical_only": 3},
    #     recording_type="two_photon",
    #     protocol_base="SpheresPermTubeReward",
    #     photodiode_protocol=photodiode_protocol,
    #     return_volumes=True,
    # )
    # Sync SFTF recordings
    print("---Syncing SFTF recordings...---")
    trials_df_all_sftf, dff_mean_all_sftf = gratings.analyze_grating_responses(
        project=project,
        session=session_name,
        filter_datasets={"anatomical_only": 3},
        photodiode_protocol=photodiode_protocol,
        protocol_base="SFTF",
    )

    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )

    # if (neurons_ds.get_flexilims_entry() is not None) and conflicts == "skip":
    #     print("Session already analyzed.")
    #     neurons_df = pd.read_pickle(neurons_ds.path_full)

    # Anlyze SFTF responses
    print("---Fitting SFTF responses...---")
    neurons_df_sftf = fit_gaussian_blob.fit_sftf_tuning(
        trials_df=trials_df_all_sftf, niter=10, min_sigma=0.25
    )
    neurons_df_sftf.to_pickle(neurons_ds.path_full.parent / "neurons_df_sftf.pickle")

    # # Analyze depth responses
    # # Find depth neurons and fit preferred depth
    # print("---Start finding depth neurons...---")
    # print("Find depth neurons...")
    # neurons_df, neurons_ds = find_depth_neurons.find_depth_neurons(
    #     trials_df=trials_df_all,
    #     neurons_ds=neurons_ds,
    #     rs_thr=0.2,
    #     alpha=0.05,
    # )

    # # Visualize all neurons
    # print("---Start visualisation---")
    # grating_plots.basic_vis_SFTF_session(
    #     neurons_df=neurons_df,
    #     trials_df_depth=trials_df_all,
    #     trials_df_sftf=trials_df_all_sftf,
    #     add_depth=True,
    #     save_dir=neurons_ds.path_full.parent,
    #     fontsize_dict={"title": 15, "label": 10, "tick": 10},
    # )


if __name__ == "__main__":
    defopt.run(main)
