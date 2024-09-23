import matplotlib.pyplot as plt
import numpy as np
import itertools
from znamutils import slurm_it
import pickle
import os
from cottage_analysis.analysis import spheres
import flexiznam as flz
from pathlib import Path

CONDA_ENV = "2p_analysis_cottage2"


def plot_confusion_matrix(conmat, acc, normalize, fontsize_dict, title_sfx=""):
    # Normalize confusion matrix with true labels
    if normalize:
        conmat = conmat / conmat.sum(axis=1)[:, np.newaxis]

    plt.imshow(conmat, interpolation="nearest", cmap="Blues")
    fmt = "d"
    thresh = conmat.max() / 2.0
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        display = conmat[i, j]
        if display >= 1:
            display = str(int(display))
        else:
            display = f"{display:.2f}"
        plt.text(
            j,
            i,
            display,
            horizontalalignment="center",
            color="white" if conmat[i, j] > thresh else "black",
            fontsize=fontsize_dict["text"],
        )
    plt.xlabel("Predicted depth class", fontsize=fontsize_dict["label"])
    plt.ylabel("True depth class", fontsize=fontsize_dict["label"])
    plt.title(f"Accuracy {acc:.2f} {title_sfx}", fontsize=fontsize_dict["title"])
    plt.xticks(fontsize=fontsize_dict["tick"])
    plt.yticks(fontsize=fontsize_dict["tick"])


@slurm_it(
    conda_env=CONDA_ENV,
    slurm_options={
        "mem": "32G",
        "time": "18:00:00",
        "partition": "ncpu",
    },
    print_job_id=True,
)
def plot_decoder_session(decoder_dict_path,
                         save_path,
                        session_name,
                        project,
                        photodiode_protocol,
                        speed_bins,
                         ):
    flexilims_session = flz.get_flexilims_session(project)
    _, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    with open(decoder_dict_path, "rb") as f:
        decoder_dict = pickle.load(f)
    # Plot confusion matrix
    print("Plotting confusion matrices...")
    plt.figure()
    for i, closed_loop in enumerate(np.sort(trials_df_all.closed_loop.unique())):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        plt.subplot(1, 2, i + 1)
        plot_confusion_matrix(
            decoder_dict[f"conmat{sfx}"],
            decoder_dict[f"accuracy{sfx}"],
            normalize=True,
            fontsize_dict={"text": 10, "label": 10, "title": 10, "tick": 5},
        )
        plt.title(sfx[1:], fontsize=10)
    os.makedirs(Path(save_path) / "plots" / "depth_decoder", exist_ok=True)
    plt.savefig(
        Path(save_path)
        / "plots"
        / "depth_decoder"
        / "confusion_matrix.png",
        dpi=300,
    )
    print("Confusion matrix plotted.")
        
    plt.figure(figsize=(3*2,3*(len(speed_bins)+1)))
    for i, closed_loop in enumerate(np.sort(trials_df_all.closed_loop.unique())):
        if closed_loop:
            sfx = "_closedloop"
        else:
            sfx = "_openloop"
        for ispeed, speed_bin in enumerate(speed_bins):
            if ispeed == 0:
                title_sfx = f"still{sfx}"
            else:
                title_sfx = f"speed{sfx} {speed_bins[ispeed-1]:.1f}-{speed_bin:.1f} m/s"
            if len(decoder_dict[f"conmat_speed_bins{sfx}"][ispeed]) > 0:
                plt.subplot2grid((len(speed_bins)+1, 2), (ispeed, i))
                plot_confusion_matrix(
                    decoder_dict[f"conmat_speed_bins{sfx}"][ispeed],
                    decoder_dict[f"acc_speed_bins{sfx}"][ispeed],
                    normalize=True,
                    fontsize_dict={"text": 5, "label": 5, "title": 10, "tick": 5},
                    title_sfx=title_sfx,
                )
        for speed_bin in [speed_bins[-1]]:
            if len(decoder_dict[f"conmat_speed_bins{sfx}"][-1]) > 0:
                title_sfx = f"speed > {speed_bin:.1f} m/s"
                plt.subplot2grid((len(speed_bins)+1, 2), (len(speed_bins), i))
                plot_confusion_matrix(
                    decoder_dict[f"conmat_speed_bins{sfx}"][-1],
                    decoder_dict[f"acc_speed_bins{sfx}"][-1],
                    normalize=True,
                    fontsize_dict={"text": 5, "label": 5, "title": 10, "tick": 5},
                    title_sfx=title_sfx,
                )
    plt.tight_layout()
    plt.savefig(
        Path(save_path)
        / "plots"
        / "depth_decoder"
        / f"confusion_matrix_speed_bins.png",
        dpi=300,
    )
    print("Confusion matrix for different speed bins plotted.")