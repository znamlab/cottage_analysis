import os
import defopt
from pathlib import Path
import warnings
import flexiznam as flz
from cottage_analysis.analysis import (
    spheres,
    openloop,
)

from cottage_analysis.pipelines import pipeline_utils


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
    _, trials_df_all = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=project,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )

    arr = trials_df_all["closed_loop"].values
    zeros, _ = openloop.find_zeros_before_ones(arr)
    if len(zeros) == 0:
        print("No open loop before closed loop trials found.")
    else:
        # Merge fit dataframes
        # job_dependency = outputs if use_slurm else None
        pipeline_utils.merge_fit_dataframes(
            project,
            session_name,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            job_dependency=None,
            scripts_name=f"{session_name}_merge_fit_dataframes_openclosed",
            conflicts=conflicts,
            prefix="fit_rs_of_tuning_gaussian_2d_k1_openclosed",
            suffix="",
            column_suffix=-12,
            filetype=".pickle",
            target_filename="neurons_df_openclosed.pickle",
        )

        print("---Analysis finished. Neurons_df saved.---")


if __name__ == "__main__":

    defopt.run(main)
