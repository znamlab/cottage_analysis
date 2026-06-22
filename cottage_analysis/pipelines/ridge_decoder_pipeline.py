import os
import defopt
import warnings
from pathlib import Path
import flexiznam as flz
from cottage_analysis.summary_analysis import get_session_list
from cottage_analysis.pipelines.ridge_decoder_utils import run_session


def main(
    project: str = "ccyp_l5_3d_vision",
    session_name: str = None,
    conflicts: str = "skip",
    photodiode_protocol: int = 5,
    anatomical_only: bool = True,
    ast_neuropil: bool = False,
    use_slurm: bool = False,
):
    """
    Run Ridge decoder (RS/OF/depth) for sessions of a project and save results as parquet files.

    Args:
        project (str): Project name. Defaults to "ccyp_l5_3d_vision".
        session_name (str): Name of the session to run. If None, runs on all sessions in the project.
        conflicts (str): How to handle conflicts in flexilims. Default "skip".
        photodiode_protocol (int): 2 or 5. Default 5.
        anatomical_only (bool): Whether to only use anatomical datasets. Default True.
        ast_neuropil (bool): Whether to use ASt neuropil correction. Default False.
        use_slurm (bool): Whether to run each session as a separate slurm job. Default False.
    """
    flexilims_session = flz.get_flexilims_session(project_id=project)

    if session_name is not None:
        sessions = [session_name]
    else:
        print(f"Querying all sessions for project {project}...", flush=True)
        sessions = get_session_list.get_sessions(flexilims_session, exclude_openloop=False)

    print(f"Total sessions to process: {len(sessions)}", flush=True)

    filter_datasets = {}
    if anatomical_only:
        filter_datasets["anatomical_only"] = 3
    if ast_neuropil:
        filter_datasets["ast_neuropil"] = True
    else:
        filter_datasets["ast_neuropil"] = False

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for i, sess in enumerate(sessions):
        print(
            f"\n========================================\n"
            f"Submitting/Processing session {i+1}/{len(sessions)}: {sess}\n"
            f"========================================",
            flush=True,
        )

        if use_slurm:
            slurm_folder = Path(os.path.expanduser(f"~/slurm_logs/{sess}"))
            slurm_folder.mkdir(parents=True, exist_ok=True)
        else:
            slurm_folder = None

        try:
            # Call the slurm_it decorated run_session
            run_session(
                sess=sess,
                project=project,
                conflicts=conflicts,
                photodiode_protocol=photodiode_protocol,
                filter_datasets=filter_datasets,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=f"ridge_decoder_{sess}",
            )
        except Exception as e:
            print(f"Error submitting/processing session {sess}: {e}", flush=True)


if __name__ == "__main__":
    defopt.run(main)
