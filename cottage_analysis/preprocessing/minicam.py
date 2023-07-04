"""Function to preprocess minicam data."""


from functools import partial
from pathlib import Path
import flexiznam as flz
from flexiznam.schema import CameraData
from cottage_analysis.io_module import video
from cottage_analysis.utilities import slurm_helper


def run_deinterleave(camera_ds, redo=False, use_slurm=True, dependency=None):
    """Run deinterleave on a camera dataset.

    Args:
        camera_ds (flexiznam.Dataset): camera dataset
        redo (bool, optional): whether to redo the deinterleave. Defaults to False.
        use_slurm (bool, optional): whether to use slurm. Defaults to True.
        dependency (str, optional): dependency for slurm. Defaults to None.

    Returns:
        str: job id if use_slurm is True
        str: path to the deinterleaved video
    """
    flm_sess = flz.get_flexilims_session(project_id=camera_ds.project_id)
    target_name = f"{camera_ds.dataset_name}_deinterleaved"

    target_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type=CameraData.DATASET_TYPE,
        base_name=target_name,
        conflicts="skip",
        flexilims_session=flm_sess,
    )

    if target_ds.flexilims_status() != "not online" and not redo:
        return None, target_ds.path_full

    print("Deinterleaving %s" % camera_ds.full_name)
    if use_slurm:
        func = partial(
            slurm_deinterleave,
            slurm_folder=target_ds.path_full,
            dependency=dependency,
        )
    else:
        func = deinterleave
    job_id, slurm_folder = func(camera_ds.id, project_id=camera_ds.project_id)

    return job_id, slurm_folder


def deinterleave(camera_ds_id, project_id):
    """Deinterleave a camera dataset.

    Will deinterleave the video file and update the dataset in flexilims.

    Args:
        camera_ds_id (str): id of the camera dataset
        project_id (str): id of the project
    """
    flm_sess = flz.get_flexilims_session(project_id=project_id)
    camera_ds = flz.Dataset.from_flexilims(id=camera_ds_id, flexilims_session=flm_sess)
    target_name = f"{camera_ds.dataset_name}_deinterleaved"
    target_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type=CameraData.DATASET_TYPE,
        base_name=target_name,
        conflicts="skip",
        flexilims_session=flm_sess,
    )
    target_ds.extra_attributes = dict(
        metadata_file=camera_ds.extra_attributes["metadata_file"],
        video_file=target_name + ".mp4",
    )
    target_ds.path_full.mkdir(parents=True, exist_ok=True)
    video.io_func.deinterleave_camera(
        camera_file=camera_ds.path_full / camera_ds.extra_attributes["video_file"],
        target_file=target_ds.path_full / target_ds.extra_attributes["video_file"],
        make_grey=False,
        verbose=True,
        intrinsic_calibration=None,
    )
    camera_ds.path_full / camera_ds.extra_attributes["metadata_file"]
    target_ds.update_flexilims(conflicts="overwrite")
    return None, target_ds.path_full


def slurm_deinterleave(camera_ds_id, project_id, slurm_folder, dependency=None):
    slurm_folder = Path(slurm_folder)
    python_script = slurm_folder / "deinterleave.py"
    flm_sess = flz.get_flexilims_session(project_id=project_id)
    camera_ds = flz.Dataset.from_flexilims(id=camera_ds_id, flexilims_session=flm_sess)

    slurm_helper.python_script_single_func(
        target_file=python_script,
        function_name="cottage_analysis.preprocessing.minicam.deinterleave",
        arguments=dict(camera_ds_id=camera_ds_id, project_id=camera_ds.project_id),
        imports="cottage_analysis.preprocessing.minicam",
    )
    slurm_helper.create_slurm_sbatch(
        target_folder=slurm_folder,
        script_name="deinterleave.sh",
        python_script=python_script,
        conda_env="cottage_analysis",
        slurm_options=dict(mem="8G", time="24:00:00"),
        module_list=["FFmpeg"],
    )

    job_id = slurm_helper.run_slurm_batch(
        slurm_folder / "deinterleave.sh", job_dependency=dependency
    )
    return job_id, slurm_folder
