import os
import shutil
import time
import numpy as np
import pandas as pd
import defopt
from pathlib import Path
import warnings
import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
)
from cottage_analysis.pipelines import pipeline_utils


def copy_fit_df(neurons_ds, model, choose_trials, k_folds, recording_type='closedloop', file_special_sfx=''):

    # create source and target file name
    suffix = f"{model}"
    if isinstance(choose_trials, str):
        suffix = suffix + f"_crossval"
    suffix = suffix + f"_k{k_folds}"
    
    # copy fit_df from source to target
    source_path = neurons_ds.path_full.with_name(
        f"fit_rs_of_tuning_{suffix}.pickle"
    )
    target_path = neurons_ds.path_full.with_name(
        f"fit_rs_of_tuning_{suffix}{file_special_sfx}.pickle"
    )
    fit_df = pd.read_pickle(source_path)
    fit_df = fit_df.loc[:,fit_df.columns.str.contains(f"{recording_type}|roi")]
    print(f"Copying {source_path.name} to {target_path.name}")
    fit_df.to_pickle(target_path)
    
    return fit_df
    

def main(
    project,
    session_name,
    conflicts="skip",
    photodiode_protocol=5,
    use_slurm=False,
):
    """
    Main function to analyze a session.

    Args:
        project(str): project name
        session_name(str): {Mouse}_{Session}
        conflicts(str): "skip", "append", or "overwrite"
        photodiode_protocol(int): 2 or 5.
        use_slurm(bool): whether to use slurm to run the fit in the pipeline. Default False.
        run_depth_fit(bool): whether to run the depth fit. Default True.
        run_rf(bool): whether to run the rf fit. Default True.
        run_rsof_fit(bool): whether to run the rsof fit. Default True.
        run_plot(bool): whether to run the plot. Default True.
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
    if (neurons_ds.get_flexilims_entry() is not None) and conflicts == "skip":
        print(f"Session {session_name} already processed... reading saved data...")
    else:
        # # read finished time
        # if os.path.exists(neurons_ds.path_full.parent / "finished.pickle"):
        #     finished = pd.read_pickle(neurons_ds.path_full.parent / "finished.pickle")
        #     for key, item in zip(["run_depth_fit", "run_rf", "run_rsof_fit", "run_plot"], 
        #                          [run_depth_fit, run_rf, run_rsof_fit, run_plot]):
        #         finished[key] = item
        # else:
        #     finished = {
        #         "run_depth_fit": run_depth_fit,
        #         "run_rf": run_rf,
        #         "run_rsof_fit": run_rsof_fit,
        #         "run_plot": run_plot,
        #     }
        #     finished = pd.DataFrame(finished, index=[0])
        # finished.to_pickle(neurons_ds.path_full.parent / "finished.pickle")
        
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

        # Fit gaussian blob to neuronal activity
        print("---Start fitting 2D gaussian blob...---")
        outputs = []
        common_params = dict(
            rs_thr=0.01,
            param_range={
                "rs_min": 0.005,
                "rs_max": 5,
                "of_min": 0.03,
                "of_max": 3000,
            },
            niter=10,
            min_sigma=0.25,
        )

        to_do = [
            # ("gaussian_2d", None, 1),
            ("gaussian_2d", None, 5),
        ]

        # Count the number of closedloop / openloop recordings in the session. no need to redo the analysis if there was only 1 closedloop or openloop recordings
        unique_recordings = np.sort(trials_df_all["recording"].unique())
        openloop_positions = pd.Series(unique_recordings).str.contains("Playback")
        closedloop_positions = (~pd.Series(unique_recordings).str.contains("Playback"))
        n_openloop = np.sum(openloop_positions)
        n_closedloop = len(unique_recordings) - n_openloop
        assert(n_closedloop > 0)
        recordings_todo = np.zeros(len(unique_recordings))
        if (n_closedloop == 1):
            if (n_openloop == 0):
                print(f"Session {session_name} has one closedloop recording only. Copying existing fit_df for closedloop...")
                recordings_todo[:] = 0
            elif (n_openloop == 1):
                print(f"Session {session_name} has only one closedloop and one openloop recording. Copying existing fit_df for closedloop and openloop...")
                recordings_todo[:] = 0
            elif (n_openloop > 1):
                print(f"Session {session_name} has only one closedloop recording. Copying existing fit_df for closedloop...")
                print("Fitting for separate openloop recordings...")
                recordings_todo[openloop_positions] = 1
        elif (n_closedloop > 1):
            if (n_openloop == 0):
                print(f"Session {session_name} has only closedloop recordings but multiple ones.")
                print(f"Fitting for separate closedloop recordings...")
                recordings_todo[:] = 1
            if (n_openloop == 1):
                print(f"Session {session_name} has only one openloop recording. Copying existing fit_df for openloop...")
                print("Fitting for separate closedloop recordings...")
                recordings_todo[closedloop_positions] = 1
            if (n_openloop > 1):
                print(f"Session {session_name} has multiple closedloop and openloop recordings.")
                print("Fitting for separate closedloop and openloop recordings...")
                recordings_todo[:] = 1
        
        for model, trials, k_folds in to_do:
            name = f"{session_name}_{model}"
            if trials is not None:
                name += "_crossval"
            name += f"_k{k_folds}"
            print(f"Fitting {model}...")
            for i, (recording, fit_recording) in enumerate(zip(unique_recordings, recordings_todo)):
                choose_trials = trials_df_all[trials_df_all.recording == recording].index.tolist()
                new_name = name + f"_recording_{recording}"
                if "Playback" in recording:
                    recording_type = "openloop"
                else:
                    recording_type = "closedloop"
                if fit_recording:
                    print(f"Fitting rsof model {model}_k{k_folds} for recording {recording}")
                    out = pipeline_utils.load_and_fit(
                        project,
                        session_name,
                        photodiode_protocol,
                        model=model,
                        choose_trials=choose_trials,
                        use_slurm=use_slurm,
                        slurm_folder=slurm_folder,
                        scripts_name=new_name,
                        k_folds=k_folds,
                        file_special_sfx=f"_recording_{'_'.join(recording.split('_')[-2:])}_{recording_type}_{i}",
                        **common_params,
                    )
                    outputs.append(out)
                else:
                    copy_fit_df(neurons_ds, 
                                model, 
                                choose_trials, 
                                k_folds, 
                                recording_type=recording_type, 
                                file_special_sfx=f"_recording_{'_'.join(recording.split('_')[-2:])}_{recording_type}_{i}")
            print("---RS OF fit finished. Neurons_df saved.---")

        # Merge fit dataframes
        job_dependency = outputs if use_slurm else None
        
        if np.sum(recordings_todo) == 0:
            job_dependency = None
        out = pipeline_utils.merge_fit_dataframes(
            project,
            session_name,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
            scripts_name=f"{session_name}_merge_fit_dataframes_separate_recordings",
            conflicts=conflicts,
            prefix="fit_rs_of_tuning_",
            suffix="",
            exclude_keywords=["openclosed"], 
            include_keywords=["recording","loop"],
            target_column_suffix=-2,
            target_column_prefix="_recording",
            filetype=".pickle",
            target_filename="neurons_df.pickle",
        )

        print("---Analysis finished. Neurons_df saved.---")   

if __name__ == "__main__":

    defopt.run(main)
