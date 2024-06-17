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
    params = {
        "trial_average": False,
        "rolling_window": 0.5,
        "downsample_window": 0.5,
        "Cs": np.logspace(-3, 3, 7),
        "continuous_still": 1,
        "still_time": 1,
        "still_thr": 0.05,
        "speed_bins": np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]),
    }
    
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
    assert len(trials_df_all.closed_loop.unique()) in [1, 2]
    if len(trials_df_all.closed_loop.unique()) == 1:
        print("This is a closed-loop only session")
    elif len(trials_df_all.closed_loop.unique()) == 2:
        print("This is a session with open loop")
        
    for closed_loop in np.sort(trials_df_all.closed_loop.unique()):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        print(f"---Start depth decoder{sfx}...---")
        acc, conmat, best_params, y_test_all, y_preds_all, trials_df = population_depth_decoder.depth_decoder(
            trials_df_all[trials_df_all.closed_loop == closed_loop],
            flexilims_session=flexilims_session,
            session_name=session_name,
            closed_loop=closed_loop,
            trial_average=params["trial_average"],
            rolling_window=params["rolling_window"],
            frame_rate=frame_rate,
            downsample_window=params["downsample_window"],
            random_state=42,
            kernel="linear",
            Cs=params["Cs"],
            k_folds=5,
        )
        print(f"Accuracy{sfx}: {acc}")
        decoder_dict[f"accuracy{sfx}"] = acc
        decoder_dict[f"conmat{sfx}"] = conmat
        decoder_dict[f"best_C{sfx}"] = best_params["C"]
        decoder_dict[f"y_test_all{sfx}"] = y_test_all
        decoder_dict[f"y_preds_all{sfx}"] = y_preds_all
        decoder_dict[f"trials_df{sfx}"] = trials_df
        
        # Get accuracy for different speed bins
        print(f"Calculating accuracy for depth decoder at different speed bins{sfx}")
        acc_speed_bins, conmat_speed_bins = population_depth_decoder.find_acc_speed_bins(trials_df,
                                                                                         params["speed_bins"],
                                                                                         y_test=y_test_all, 
                                                                                         y_preds=y_preds_all, 
                                                                                         continuous_still=params["continuous_still"], 
                                                                                         still_thr=params["still_thr"], 
                                                                                         still_time=params["still_time"],
                                                                                         frame_rate=frame_rate)
        decoder_dict[f"acc_speed_bins{sfx}"] = acc_speed_bins
        decoder_dict[f"conmat_speed_bins{sfx}"] = conmat_speed_bins
    decoder_dict["params"] = params
    
    # Save decoder results
    with open(neurons_ds.path_full.parent / "decoder_results.pickle", "wb") as handle:
        pickle.dump(decoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Results saved.")

    # Plot confusion matrix
    print("Plotting confusion matrices...")
    plt.figure()
    for i, closed_loop in enumerate(np.sort(trials_df_all.closed_loop.unique())):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        plt.subplot(1, 2, i + 1)
        depth_decoder_plots.plot_confusion_matrix(
            decoder_dict[f"conmat{sfx}"],
            decoder_dict[f"accuracy{sfx}"],
            normalize=True,
            fontsize_dict={"text": 10, "label": 10, "title": 10, "tick": 5},
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
        
    plt.figure(figsize=(3*2,3*len(params["speed_bins"]+1)))
    for i, closed_loop in enumerate(np.sort(trials_df_all.closed_loop.unique())):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        for ispeed, speed_bin in enumerate(params["speed_bins"]):
            if ispeed == 0:
                title_sfx = "still"
            else:
                title_sfx = f"speed {params['speed_bins'][ispeed-1]:.1f}-{speed_bin:.1f} m/s"
            if len(decoder_dict[f"conmat_speed_bins{sfx}"][ispeed]) > 0:
                plt.subplot2grid((len(params["speed_bins"])+1, 2), (ispeed, i))
                depth_decoder_plots.plot_confusion_matrix(
                    decoder_dict[f"conmat_speed_bins{sfx}"][ispeed],
                    decoder_dict[f"acc_speed_bins{sfx}"][ispeed],
                    normalize=True,
                    fontsize_dict={"text": 5, "label": 5, "title": 10, "tick": 5},
                    title_sfx=title_sfx,
                )
        for speed_bin in [params["speed_bins"][-1]]:
            if len(decoder_dict[f"conmat_speed_bins{sfx}"][-1]) > 0:
                title_sfx = f"speed > {speed_bin:.1f} m/s"
                plt.subplot2grid((len(params["speed_bins"])+1, 2), (len(params["speed_bins"]), i))
                depth_decoder_plots.plot_confusion_matrix(
                    decoder_dict[f"conmat_speed_bins{sfx}"][-1],
                    decoder_dict[f"acc_speed_bins{sfx}"][-1],
                    normalize=True,
                    fontsize_dict={"text": 5, "label": 5, "title": 10, "tick": 5},
                    title_sfx=title_sfx,
                )
    plt.tight_layout()
    plt.savefig(
        neurons_ds.path_full.parent
        / "plots"
        / "depth_decoder"
        / f"confusion_matrix_speed_bins.png",
        dpi=300,
    )
    print("Confusion matrix for different speed bins plotted.")
    


if __name__ == "__main__":
    defopt.run(main)
