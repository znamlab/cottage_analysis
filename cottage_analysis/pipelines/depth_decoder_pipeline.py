import os
import numpy as np
import defopt
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings

import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    population_depth_decoder,
)
from cottage_analysis.plotting import depth_decoder_plots

from cottage_analysis.pipelines import pipeline_utils


def main(
    project, session_name, conflicts="skip", photodiode_protocol=5, use_slurm=False
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        use_slurm(bool): whether to use slurm to run the fit in the pipeline. Default False.
    """
    print(
        f"------------------------------- \n \
        Start analysing {session_name}   \n \
        -------------------------------"
    )
    if use_slurm:
        slurm_folder = Path(os.path.expanduser(f"~/slurm_logs"))
        slurm_folder.mkdir(exist_ok=True)
        slurm_folder = Path(slurm_folder / f"{session_name}")
        slurm_folder.mkdir(exist_ok=True)
    else:
        slurm_folder = None

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    flexilims_session = flz.get_flexilims_session(project)

    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        conflicts=conflicts,
    )
    suite2p_datasets = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        project_id=project,
        flexilims_session=flexilims_session,
        return_dataseries=False,
        filter_datasets={"anatomical_only": 3},
    )
    suite2p_dataset = suite2p_datasets[0]
    frame_rate = suite2p_dataset.extra_attributes["fs"]

    # Synchronisation
    print("---Start synchronisation...---")
    vs_df_all, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )

    # Run depth decoder
    decoder_dict = {}
    for closed_loop in np.sort(trials_df_all.closed_loop.unique()):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        print(f"---Start depth decoder for closed loop {closed_loop}...---")
        acc, conmat, best_params = population_depth_decoder.depth_decoder(
            trials_df_all[trials_df_all.closed_loop == closed_loop],
            flexilims_session=flexilims_session,
            session_name=session_name,
            closed_loop=closed_loop,
            trial_average=False,
            rolling_window=0.5,
            frame_rate=frame_rate,
            downsample_window=0.5,
            random_state=42,
            kernel="linear",
            Cs=np.logspace(-3, 3, 7),
            k_folds=5,
        )
        print(f"Accuracy{sfx}: {acc}")
        decoder_dict[f"accuracy{sfx}"] = acc
        decoder_dict[f"conmat{sfx}"] = conmat
        decoder_dict[f"best_C{sfx}"] = best_params["C"]

    # Save decoder results
    with open(neurons_ds.path_full.parent / "decoder_results.pickle", "wb") as handle:
        pickle.dump(decoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Results saved.")

    # Plot confusion matrix
    plt.figure()
    for i, sfx in enumerate(["_closedloop", "_openloop"]):
        plt.subplot(1, 2, i + 1)
        depth_decoder_plots.plot_confusion_matrix(
            decoder_dict[f"conmat{sfx}"],
            decoder_dict[f"accuracy{sfx}"],
            normalize=True,
            fontsize_dict={"text": 10, "label": 10, "title": 10},
        )
    os.makedirs(neurons_ds.path_full.parent / "plots" / "depth_decoder", exist_ok=True)
    plt.savefig(
        neurons_ds.path_full.parent
        / "plots"
        / "depth_decoder"
        / "confusion_matrix.png",
        dpi=300,
    )
    print("Confusion matrix plotted.")


if __name__ == "__main__":
    defopt.run(main)
