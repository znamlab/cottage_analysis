import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import flexiznam as flz
from znamutils.decorators import slurm_it
from cottage_analysis.analysis import spheres, treadmill, population_ridge_decoder


# Define the Slurm-wrapped function for processing a single session
@slurm_it(
    conda_env="v1_depth_map",
    slurm_options={
        "mem": "32G",
        "time": "4:00:00",
        "partition": "ncpu",
    },
    print_job_id=True,
)
def run_session(
    sess: str,
    project: str,
    conflicts: str = "skip",
    photodiode_protocol: int = 5,
    filter_datasets: dict = None,
    rolling_window: float = None,
    downsample_window: float = None,
    log_transform: bool = True,
    rs_thr: float = None,
    alphas: list = None,
    k_folds: int = 5,
    random_state: int = 42,
    run_neuron_subsets: bool = False,
    subset_sizes: list = None,
    is_treadmill: bool = None,
):
    """
    Run Ridge decoder for a single session and save results to parquet.
    This function is wrapped by slurm_it to support execution on Slurm.
    """
    if filter_datasets is None:
        filter_datasets = {"anatomical_only": 3, "ast_neuropil": False}

    flexilims_session = flz.get_flexilims_session(project_id=project)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print(
        f"\n========================================\n"
        f"Processing session: {sess}\n"
        f"========================================",
        flush=True,
    )

    # Prefer annotated neurons_df if available, otherwise fall back to single match
    neurons_ds = flz.get_datasets(
        origin_name=sess,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        filter_datasets={"annotated": True},
        allow_multiple=True,
    )
    if not neurons_ds or len(neurons_ds) == 0:
        neurons_ds = flz.get_datasets(
            origin_name=sess,
            dataset_type="neurons_df",
            flexilims_session=flexilims_session,
            allow_multiple=True,
        )
        if neurons_ds and len(neurons_ds) > 1:
            ds_names = [ds.full_name for ds in neurons_ds]
            raise AssertionError(
                f"No annotated neurons_df found and got multiple datasets: {ds_names}"
            )

    if not neurons_ds or len(neurons_ds) == 0:
        print(
            f"Warning: No neurons_df dataset found in flexilims for session {sess}. Skipping.",
            flush=True,
        )
        return

    ds = neurons_ds[0]
    session_folder = Path(ds.path_full).parent
    neurons_df_path = session_folder / "neurons_df.pickle"

    if not neurons_df_path.exists():
        print(
            f"Warning: Local neurons_df file {neurons_df_path} does not exist. Skipping.",
            flush=True,
        )
        return

    # 2. Synchronize recordings
    # Determine if it's a treadmill session or spheres session
    exp_session = flz.get_entity(
        datatype="session", name=sess, flexilims_session=flexilims_session
    )
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        flexilims_session=flexilims_session,
    )
    if is_treadmill is None:
        is_treadmill = False
        if "protocol" in recordings.columns:
            is_treadmill = (
                recordings["protocol"]
                .str.contains("SpheresTubeMotor|Treadmill", na=False)
                .any()
            )
        if not is_treadmill:
            is_treadmill = (
                recordings["name"]
                .str.contains("SpheresTubeMotor|Treadmill", na=False)
                .any()
            )

    # Try with annotated datasets first, fall back to original filter
    annotated_filter = dict(filter_datasets) if filter_datasets else {}
    annotated_filter["annotated"] = True

    def _do_sync(filt):
        if is_treadmill:
            print(
                f"Detected treadmill session. Synchronizing with no-cut version...",
                flush=True,
            )
            return treadmill.sync_all_recordings(
                session_name=sess,
                flexilims_session=flexilims_session,
                project=project,
                filter_datasets=filt,
                recording_type="two_photon",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                acceleration_time=None,
                cut_trial_end=None,
                trial_duration=None,
            )
        else:
            print(f"Detected spheres session. Synchronizing...", flush=True)
            return spheres.sync_all_recordings(
                session_name=sess,
                flexilims_session=flexilims_session,
                project=project,
                filter_datasets=filt,
                recording_type="two_photon",
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
            )

    try:
        vs_df_all, trials_df_all = _do_sync(annotated_filter)
    except FileNotFoundError:
        print(
            "No annotated suite2p dataset found, retrying without annotated filter...",
            flush=True,
        )
        vs_df_all, trials_df_all = _do_sync(filter_datasets)

    # Get frame rate
    suite2p_datasets = flz.get_datasets(
        origin_name=sess,
        dataset_type="suite2p_rois",
        project_id=project,
        flexilims_session=flexilims_session,
        return_dataseries=False,
        filter_datasets={"anatomical_only": 3},
    )
    suite2p_dataset = suite2p_datasets[0]
    frame_rate = suite2p_dataset.extra_attributes["fs"]
    print(f"Imaging frame rate: {frame_rate:.4f} Hz", flush=True)

    if len(trials_df_all) == 0:
        print(
            f"Warning: Synced trials_df is empty for session {sess}. Skipping.",
            flush=True,
        )
        return

    n_rois = trials_df_all["dff_stim"].iloc[0].shape[1]
    print(
        f"Number of ROIs: {n_rois}, Number of trials: {len(trials_df_all)}", flush=True
    )

    # 3. Initialize output DataFrames
    neurons_df_ridge = pd.DataFrame({"roi": np.arange(n_rois)})
    predictions_df_ridge = pd.DataFrame(
        {"trial_no": trials_df_all["trial_no"].values},
        index=trials_df_all.index,
    )

    targets = ["OF_stim", "RS_stim", "depth"]
    conditions = [1, 0]  # 1: closedloop, 0: openloop
    has_results = False

    for cond in conditions:
        cond_str = "closedloop" if cond == 1 else "openloop"
        # Check if we have trials for this condition
        cond_trials = trials_df_all[trials_df_all.closed_loop == cond]
        if len(cond_trials) == 0:
            print(f"  No trials found for {cond_str}.", flush=True)
            continue

        for target in targets:
            print(f"  Decoding {target} under {cond_str}...", flush=True)
            try:
                decoder_kwargs = dict(
                    target_col=target,
                    closed_loop=cond,
                    frame_rate=frame_rate,
                    rolling_window=rolling_window,
                    downsample_window=downsample_window,
                    log_transform=log_transform,
                    rs_thr=rs_thr,
                    k_folds=k_folds,
                    random_state=random_state,
                    verbose=False,
                )
                if alphas is not None:
                    decoder_kwargs["alphas"] = alphas
                res = population_ridge_decoder.continuous_decoder(
                    trials_df_all,
                    **decoder_kwargs,
                )

                # Store session metrics
                neurons_df_ridge[f"ridge_r2_{target}_{cond_str}"] = res["r2"]
                neurons_df_ridge[f"ridge_pearson_r_{target}_{cond_str}"] = res[
                    "pearson_r"
                ]
                neurons_df_ridge[f"ridge_mse_{target}_{cond_str}"] = res["mse"]
                neurons_df_ridge[f"ridge_mae_{target}_{cond_str}"] = res["mae"]

                # Store coefficients across folds
                fold_coefs = [fr["model"].coef_ for fr in res["fold_results"]]
                mean_coefs = np.mean(fold_coefs, axis=0)
                std_coefs = np.std(fold_coefs, axis=0)

                neurons_df_ridge[f"ridge_weight_mean_{target}_{cond_str}"] = mean_coefs
                neurons_df_ridge[f"ridge_weight_std_{target}_{cond_str}"] = std_coefs
                for f_idx, coef in enumerate(fold_coefs):
                    neurons_df_ridge[
                        f"ridge_weight_fold{f_idx}_{target}_{cond_str}"
                    ] = coef

                # Store predictions and true values
                predictions_df_ridge[f"ridge_pred_{target}_{cond_str}"] = res[
                    "y_pred_trials"
                ]
                predictions_df_ridge[f"ridge_true_{target}_{cond_str}"] = res[
                    "y_test_trials"
                ]

                has_results = True

            except Exception as e:
                print(
                    f"  Warning: Failed to decode {target} ({cond_str}): {e}",
                    flush=True,
                )

    # 4. Save results to parquet if we have any successful decodes
    suffix = "_motor" if is_treadmill else "_closedloop"
    if has_results:
        neurons_parquet_path = session_folder / f"ridge_decoder_neurons{suffix}.parquet"
        predictions_parquet_path = (
            session_folder / f"ridge_decoder_predictions{suffix}.parquet"
        )

        neurons_df_ridge.to_parquet(neurons_parquet_path, index=False)
        predictions_df_ridge.to_parquet(predictions_parquet_path, index=False)

        print(f"Saved: {neurons_parquet_path}", flush=True)
        print(f"Saved: {predictions_parquet_path}", flush=True)
    else:
        print(f"No Ridge decoder results obtained for session {sess}.", flush=True)

    # 5. Optionally run neuron subset analysis
    if run_neuron_subsets:
        print(f"\nRunning neuron subset analysis...", flush=True)
        subset_rows = []
        decoder_funcs = {
            "OF_stim": population_ridge_decoder.of_decoder,
            "RS_stim": population_ridge_decoder.rs_decoder,
            "depth": population_ridge_decoder.depth_decoder,
        }

        for cond in conditions:
            cond_str = "closedloop" if cond == 1 else "openloop"
            cond_trials = trials_df_all[trials_df_all.closed_loop == cond]
            if len(cond_trials) == 0:
                continue

            for target in targets:
                print(f"  Subset analysis: {target} ({cond_str})...", flush=True)
                try:
                    subset_kwargs = dict(
                        closed_loop=cond,
                        frame_rate=frame_rate,
                        rolling_window=rolling_window,
                        downsample_window=downsample_window,
                        log_transform=log_transform,
                        rs_thr=rs_thr,
                        k_folds=k_folds,
                    )
                    if alphas is not None:
                        subset_kwargs["alphas"] = alphas
                    sub_res = population_ridge_decoder.decode_with_neuron_subsets(
                        trials_df_all,
                        decoder_func=decoder_funcs[target],
                        subset_sizes=subset_sizes,
                        random_state=random_state,
                        **subset_kwargs,
                    )
                    for i, size in enumerate(sub_res["subset_sizes"]):
                        subset_rows.append(
                            {
                                "target": target,
                                "condition": cond_str,
                                "subset_size": size,
                                "n_resamples": sub_res["n_resamples"][i],
                                "r2_mean": sub_res["r2_mean"][i],
                                "r2_std": sub_res["r2_std"][i],
                                "pearson_r_mean": sub_res["pearson_r_mean"][i],
                                "pearson_r_std": sub_res["pearson_r_std"][i],
                                "mse_mean": sub_res["mse_mean"][i],
                                "mse_std": sub_res["mse_std"][i],
                                "mae_mean": sub_res["mae_mean"][i],
                                "mae_std": sub_res["mae_std"][i],
                            }
                        )
                except Exception as e:
                    print(
                        f"  Warning: Subset analysis failed for {target} ({cond_str}): {e}",
                        flush=True,
                    )

        if subset_rows:
            subsets_df = pd.DataFrame(subset_rows)
            subsets_parquet_path = (
                session_folder / f"ridge_decoder_neuron_subsets{suffix}.parquet"
            )
            subsets_df.to_parquet(subsets_parquet_path, index=False)
            print(f"Saved: {subsets_parquet_path}", flush=True)
        else:
            print(f"No neuron subset results obtained for session {sess}.", flush=True)
