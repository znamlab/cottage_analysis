import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from cottage_analysis.plotting import basic_vis_plots


def plot_sta(stas, roi):
    sta = stas[roi, :, :]
    vmax = np.quantile(sta[np.isfinite(sta)], 0.999)
    plt.imshow(sta, origin="lower", vmin=-vmax, vmax=vmax, cmap="bwr")


def plot_sta_fit(
    coef, depth_list, frames, roi, plot_rows=5, plot_cols=1, plot_y=0, plot_idx=1
):
    coef_mean = np.mean(np.stack(coef, axis=2), axis=2)
    fitted_sta = coef_mean[:-1, :].reshape(
        len(depth_list), frames.shape[1], int(frames.shape[2] // 2), coef_mean.shape[1]
    )
    for i in range(len(depth_list)):
        plt.subplot2grid([plot_rows, plot_cols], [i + plot_idx, plot_y])
        vmax = np.quantile(fitted_sta[:, :, :, roi], 0.999)
        plt.imshow(
            fitted_sta[i, :, :, roi],
            origin="lower",
            vmax=vmax,
            vmin=-vmax,
            cmap="bwr",
            extent=[0, 120, -40, 40],
        )
        plt.colorbar()


def basic_vis_sta(
    coef,
    neurons_df,
    trials_df,
    depth_list,
    frames,
    roi,
    is_closedloop=1,
    plot_rows=6,
    plot_cols=1,
    plot_y=0,
    fontsize_dict={"title": 10, "tick": 10, "label": 10},
):
    plt.subplot2grid([1 + len(depth_list), plot_cols], [0, plot_y])
    basic_vis_plots.plot_depth_tuning_curve(
        neurons_df=neurons_df,
        trials_df=trials_df,
        roi=roi,
        rs_thr=0.2,
        plot_fit=False,
        linewidth=3,
        linecolor="k",
        fit_linecolor="r",
        closed_loop=is_closedloop,
        fontsize_dict=fontsize_dict,
    )
    plt.title(f"roi{roi}")

    trials_df = trials_df[trials_df.closed_loop == is_closedloop]
    plot_sta_fit(
        coef,
        depth_list,
        frames,
        roi,
        plot_rows=1 + len(depth_list),
        plot_cols=plot_cols,
        plot_y=plot_y,
    )


def basic_vis_sta_session(
    coef,
    neurons_df,
    trials_df,
    depth_list,
    frames,
    save_dir=None,
    fontsize_dict={"title": 10, "tick": 10, "label": 10},
):
    for is_closedloop in np.sort(trials_df.closed_loop.unique()):
        if is_closedloop:
            sfx = "closedloop"
        else:
            sfx = "openloop"
            
        if save_dir is not None:
            os.makedirs(save_dir / "plots" / f"sta_{sfx}", exist_ok=True)
            
        for i in tqdm(range(len(neurons_df) // 10 + 1)):
            if (i * 10) < len(neurons_df):
                iroi = 0
                plt.figure(figsize=(30, 18))
                max_roi = np.min([(i + 1) * 10, len(neurons_df) - 1])
                for roi in np.arange(len(neurons_df))[i * 10 : max_roi]:
                    basic_vis_sta(
                        coef,
                        neurons_df,
                        trials_df,
                        depth_list,
                        frames,
                        roi,
                        is_closedloop=is_closedloop,
                        plot_rows=len(depth_list) + 1,
                        plot_cols=10,
                        plot_y=iroi,
                        fontsize_dict=fontsize_dict,
                    )
                    iroi += 1
                plt.savefig(
                    save_dir
                    / "plots"
                    / f"sta_{sfx}"
                    / f"roi{np.arange(len(neurons_df))[i*10]}-{np.arange(len(neurons_df))[max_roi]}.png",
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close()
