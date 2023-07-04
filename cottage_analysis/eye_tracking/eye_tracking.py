import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from functools import partial
import flexiznam as flz
from znamutils import slurm_it
from cottage_analysis.eye_tracking import slurm_job, diagnostic, eye_model_fitting

envs = flz.PARAMETERS["conda_envs"]

def run_all(
    flexilims_session,
    dlc_model,
    camera_ds_id=None,
    camera_ds_name=None,
    redo=False,
    use_slurm=True,
    dependency=None,
):
    """Run all preprocessing steps for a session

    Args:
        flexilims_session (flexilims.Session): Flexilims session
        dlc_model (str): Name of the dlc model to use, must be in the `DLC_MODELS`
            project
        camera_ds_id (flexiznam.Schema.CameraDataset or str): Hexadecimal id of camera
            dataset on flexilims or camera dataset object. Defaults to None.
        camera_ds_name (str): Name of the camera dataset on flexilims. Ignored if
            camera_ds_id is not None. Defaults to None.
        redo (bool, optional): Redo step if data already exists. Defaults to False.
        use_slurm (bool, optional): Start slurm jobs. Defaults to True.
        dependency (str, optional): Dependency for slurm. Defaults to None.

    Returns:
        pandas.DataFrame: Log of job id of each step
    """

    if camera_ds_id is None:
        assert (
            camera_ds_name is not None
        ), "Must provide camera_ds_name if camera_ds_id is None"
        camera_ds = flz.Dataset.from_flexilims(
            name=camera_ds_name, flexilims_session=flexilims_session
        )
    elif isinstance(camera_ds_id, str):
        camera_ds = flz.Dataset.from_flexilims(
            id=camera_ds_id, flexilims_session=flexilims_session
        )
    else:
        camera_ds = camera_ds_id

    log = dict(dataset_name=camera_ds.full_name)

    # Run uncropped DLC
    job_id, slurm_folder = run_dlc(
        camera_ds,
        flexilims_session,
        dlc_model=dlc_model,
        crop=False,
        redo=redo,
        use_slurm=use_slurm,
        job_dependency=dependency,
    )
    log["dlc_uncropped"] = job_id if job_id is not None else "Done"
    if not use_slurm and (job_id is not None):
        print("Cannot chain jobs without slurm, skipping cropping and ellipse fit")
        return pd.DataFrame(log)

    # Run cropped DLC
    job_id, slurm_folder = run_dlc(
        camera_ds,
        flexilims_session,
        dlc_model=dlc_model,
        crop=True,
        redo=redo,
        job_dependency=job_id,
        use_slurm=use_slurm,
    )
    log["dlc_cropped"] = job_id if job_id is not None else "Done"
    if not use_slurm and (job_id is not None):
        print("Cannot chain jobs without slurm, skipping ellipse fit")
        return pd.DataFrame(log)

    # Run ellipse fit
    job_id, slurm_folder = run_fit_ellipse(
        camera_ds,
        flexilims_session,
        likelihood_threshold=None,
        job_dependency=job_id,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
    )
    log["ellipse"] = job_id if job_id is not None else "Done"

    # Run reprojection
    job_id, slurm_folder = run_reproject_eye(
        camera_ds=camera_ds,
        slurm_folder=slurm_folder,
        theta0=np.deg2rad(20),
        phi0=0,
        job_dependency=job_id,
        use_slurm=True,
        redo=False,
    )
    log["reprojection"] = job_id if job_id is not None else "Done"

    return pd.Series(log)


def run_dlc(
    camera_ds,
    flexilims_session,
    dlc_model,
    crop=False,
    redo=False,
    use_slurm=True,
    slurm_folder=None,
    job_dependency=None,
):
    """Run dlc tracking on a camera dataset

    Wrapper around eye_tracking.slurm_job.slurm_dlc_pupil (which does the proper tracking)

    Will find the corresponding dlc dataset and delete it if redo is True.
    If crop is True, one uncropped dataset must exist, and a crop file will be created

    Args:
        camera_ds (Dataset): The camera dataset to track
        flm_sess (flexilims.Session): The flexilims session
        dlc_model (str): Name of the dlc model to use
        crop (bool, optional): Whether to crop the video. Defaults to False.
        redo (bool, optional): Whether to redo tracking if it already exists. If True,
            the tracking data and the flexilims entry will be deleted first, before
            rerunning the analysis. Defaults to False.
        use_slurm (bool, optional): Whether to use slurm to run the job. Defaults to
            True.
        slurm_folder (str, optional): Path to the folder where to create the slurm scripts
            and slurm logs. Defaults to None, in which case the folder will be created
            using from_flexilims.
        job_dependency (str, optional): Job id to wait for before starting the job.
            Defaults to None.
    """

    # Check if analysis is already done and delete output if redo=True
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    which = "cropped" if crop else "uncropped"
    ds = ds_dict[which]
    if ds is not None:
        if not redo:
            print("  Already done. Skip")
            return None, ds.path_full
        else:
            print("  Erasing previous tracking to redo")
            # delete labeled and filtered version too. DLC would just not do anything
            # if the output files already exist
            delete_tracking_dataset(ds, flexilims_session)

    # Now start the job/function
    if use_slurm:
        func = partial(
            slurm_job.slurm_dlc_pupil,
            slurm_folder=slurm_folder,
            job_dependency=job_dependency,
        )
    else:
        func = dlc_pupil

    process, slurm_folder = func(
        camera_ds_id=camera_ds.id,
        model_name=dlc_model,
        origin_id=camera_ds.origin_id,
        project=camera_ds.project,
        crop=crop,
    )
    return process, slurm_folder


def run_fit_ellipse(
    camera_ds,
    flexilims_session,
    likelihood_threshold=None,
    job_dependency=None,
    redo=False,
    use_slurm=True,
    slurm_folder=None,
):
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    if ds_dict["cropped"] is not None:
        dlc_ds = ds_dict["cropped"]
        dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
        target = dlc_ds.path_full / f"{dlc_file.stem}_ellipse_fits.csv"
        if target.exists():
            if not redo:
                print("  Already done. Skip")
                return None, target.parent
            os.remove(target)
    if use_slurm:
        if slurm_folder is None:
            slurm_folder = dlc_ds.path_full
        func = partial(
            slurm_job.fit_ellipses,
            job_dependency=job_dependency,
            slurm_folder=slurm_folder,
        )
    else:
        if ds_dict["cropped"] is None:
            raise IOError("No cropped dataset found")
        func = fit_ellipse
    job_id = func(
        camera_ds_id=camera_ds.id,
        project_id=camera_ds.project_id,
        likelihood_threshold=likelihood_threshold,
    )
    return job_id, slurm_folder


def run_reproject_eye(
    camera_ds,
    slurm_folder,
    theta0=np.deg2rad(20),
    phi0=0,
    job_dependency=None,
    use_slurm=True,
    redo=False,
):
    """Run the reproject_eye function on a camera dataset

    DLC and ellipse fitting must have been done first

    Args:
        camera_ds (flexiznam.Dataset): The camera dataset to reproject
        slurm_folder (str): Path to the folder where to create the slurm scripts
            and slurm logs.
        theta0 (float, optional): Initial guess for the theta angle. Defaults to
            np.deg2rad(20).
        phi0 (int, optional): Initial guess for the phi angle. Defaults to 0.
        job_dependency (str, optional): Job id to wait for before starting the job.
            Defaults to None.
        use_slurm (bool, optional): Whether to use slurm to run the job. Defaults to
            True.
        redo (bool, optional): Whether to redo the reprojection if it already exists.
            Defaults to False.
    """
    if not use_slurm:
        raise NotImplementedError("Only slurm is implemented for now")
    target = Path(slurm_folder) / f"{camera_ds.dataset_name}_eye_rotation_by_frame.npy"
    if target.exists() and not redo:
        print("  Already done. Skip")
        return None, target.parent

    job_id, path = slurm_job.reproject_pupils(
        camera_ds=camera_ds,
        target_folder=slurm_folder,
        theta0=theta0,
        phi0=phi0,
        job_dependency=job_dependency,
    )
    return job_id, path


def delete_tracking_dataset(ds, flexilims_session):
    """Delete a dlc_tracking dataset

    Args:
        ds (flexiznam.Dataset): dlc_tracking dataset to delete
        flexilims_session (flexilims.Session): Flexilims session
    """
    filenames = []
    for suffix in ["", "_filtered"]:
        p = ds.path_full / ds.extra_attributes["dlc_file"]
        basename = p.with_name(p.stem + suffix)
        for ext in [".h5", ".csv"]:
            filenames.append(basename.with_suffix(ext))
        filenames.append(basename.with_name(basename.stem + "_labeled.mp4"))
    filenames.append(p.with_name(p.stem + "_meta.pickle"))
    # also delete slurm files
    for ext in ["sh", "py", "err", "out"]:
        filenames.append(ds.path_full / f"dlc_track.{ext}")
    for fname in filenames:
        if fname.exists():
            print(f"        deleting {fname}")
            os.remove(fname)
    # also remove the flexilims entry
    flexilims_session.delete(ds.id)


@slurm_it(conda_env=envs['dlc'], slurm_modules="cuDNN/8.1.1.33-CUDA-11.2.1", slurm_options=dict(ntasks=1,
        time="12:00:00",
        mem="32G",
        gres="gpu:1",
        partition="gpu"))
def dlc_pupil(
    camera_ds_id,
    model_name,
    origin_id,
    project,
    crop=False,
    conflicts="abort",
):
    """Run dlc tracking on a video

    This is the function that actually runs the tracking. It is called by run_dlc
    directly when not using slurm or by slurm_job.slurm_dlc_pupil when using slurm.

    Args:
        video_path (str): Path to the video file
        model_name (str): Name of the dlc model to use. Must be in the `DLC_models`
            project
        origin_id (str): hex code of the origin on flexilims
        project (str): Name of the project on flexilims
        crop (bool, optional): Whether to crop the video. Defaults to False.
        conflicts (str, optional): How to handle conflicts when creating the dataset on
            flexilims. Defaults to "abort".
    """
    # import dlc only in functions that need it as it takes a long time to load
    import deeplabcut

    flexilims_session = flz.get_flexilims_session(project)
    camera_ds = flz.Dataset.from_flexilims(
        flexilims_session=flexilims_session, id=camera_ds_id
    )
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]

    processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
    dlc_model_config = processed_path / "DLC_models" / model_name / "config.yaml"

    video_path = Path(video_path)

    if crop:
        uncropped_ds = ds_dict["uncropped"]
        assert uncropped_ds is not None, "No uncropped dataset found"
        crop_info = create_crop_file(camera_ds, uncropped_ds)
        crop_info = [
            crop_info["xmin"],
            crop_info["xmax"],
            crop_info["ymin"],
            crop_info["ymax"],
        ]
        suffix = "cropped"
    else:
        crop_info = None
        suffix = "uncropped"

    basename = f"{video_path.stem}_dlc_tracking_{suffix}"
    flm_sess = flz.get_flexilims_session(project)
    ds = flz.Dataset.from_origin(
        origin_id=origin_id,
        dataset_type="dlc_tracking",
        flexilims_session=flm_sess,
        base_name=basename,
        conflicts=conflicts,
    )
    target_folder = Path(ds.path_full)
    target_folder.mkdir(exist_ok=True, parents=True)

    print(f"Doing %s" % video_path, flush=True)
    analyse_kwargs = dict(
        config=dlc_model_config,
        videos=[str(video_path)],
        videotype="",
        shuffle=1,
        trainingsetindex=0,
        gputouse=None,
        save_as_csv=False,
        in_random_order=True,
        destfolder=str(target_folder),
        batchsize=None,
        cropping=crop_info,
        TFGPUinference=True,
        dynamic=(False, 0.5, 10),
        modelprefix="",
        robust_nframes=False,
        allow_growth=False,
        use_shelve=False,
        auto_track=True,
        n_tracks=None,
        calibrate=False,
        identity_only=False,
    )

    print("Analyzing", flush=True)
    out = deeplabcut.analyze_videos(**analyse_kwargs)

    dlc_output = target_folder / f"{video_path.stem}{out}.h5"
    if not dlc_output.exists():
        raise IOError(
            f"DLC ran but I cannot find the output. {dlc_output} does not exist."
        )

    # Adding to flexilims
    print("Updating flexilims", flush=True)
    ds.extra_attributes = dict(
        analyse_kwargs,
        dlc_prefix=out,
        dlc_file=f"{video_path.stem}{out}.h5",
    )
    ds.update_flexilims(mode="overwrite")

    # Save diagnostic plot
    print("Saving diagnostic plot", flush=True)
    diagnostic.check_cropping(dlc_ds=ds, camera_ds=camera_ds)
    return ds, ds.path_full


def get_tracking_datasets(camera_ds, flexilims_session):
    """Get the dlc tracking datasets corresponding to a camera dataset

    This will raise an error if more than one dataset is found for a given type

    Args:
        camera_ds (flexilims.Dataset): Camera dataset
        flexilims_session (flexilims.Session): Flexilims session

    Returns:
        dict: Dictionary with keys "cropped" and "uncropped", containing the
            corresponding datasets. If no dataset is found, the corresponding value is
            None
    """
    dlc_datasets = flz.get_children(
        parent_id=camera_ds.origin_id,
        children_datatype="dataset",
        flexilims_session=flexilims_session,
    )
    dlc_datasets = dlc_datasets[dlc_datasets["dataset_type"] == "dlc_tracking"]
    ds_dict = dict(cropped=None, uncropped=None)
    for ds_name, series in dlc_datasets.iterrows():
        ds = flz.Dataset.from_flexilims(
            data_series=series, flexilims_session=flexilims_session
        )
        vid = ds.extra_attributes["videos"]
        assert (
            len(vid) == 1
        ), f"{ds_name} tracking with more than one video, is that normal?"
        # exclude tracking for other videos
        if not vid[0].endswith(camera_ds.extra_attributes["video_file"]):
            continue
        if isinstance(ds.extra_attributes["cropping"], list):
            if ds_dict["cropped"] is not None:
                raise IOError("More than one cropped dataset")
            ds_dict["cropped"] = ds
        else:
            if ds_dict["uncropped"] is not None:
                raise IOError("More than one uncropped dataset")
            ds_dict["uncropped"] = ds
    return ds_dict


def create_crop_file(camera_ds, dlc_ds):
    """Create a crop file for DLC tracking

    Uses the results of the uncropped tracking to find the crop area and save it in a
    crop file. This crop file can then be used to crop the video before tracking.

    Args:
        camera_ds (flexilims.Dataset): Camera dataset, must contain project information
        dlc_ds (flexilims.Dataset): dlc_tracking dataset, containing uncropped tracking

    Returns:
        dict: Crop information

    """
    if dlc_ds.project is None:
        raise IOError("dlc_tracking dataset has no project information")

    metadata_path = camera_ds.path_full / camera_ds.extra_attributes["metadata_file"]
    video_path = camera_ds.path_full / camera_ds.extra_attributes["video_file"]
    crop_file = dlc_ds.path_full / f"{video_path.stem}_crop_tracking.yml"

    if crop_file.exists():
        print("Crop file already exists. Delete manually to redo")
        with open(crop_file, "r") as fhandle:
            crop_info = yaml.safe_load(fhandle)
        return crop_info

    with open(metadata_path, "r") as fhandle:
        metadata = yaml.safe_load(fhandle)
    metadata = {k.lower(): v for k, v in metadata.items()}
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    print("Creating crop file")
    dlc_res = pd.read_hdf(dlc_file)
    # Find DLC crop area
    borders = np.zeros((4, 2))
    for iw, w in enumerate(
        (
            "left_eye_corner",
            "right_eye_corner",
            "top_eye_lid",
            "bottom_eye_lid",
        )
    ):
        vals = dlc_res.xs(w, level=1, axis=1)
        vals.columns = vals.columns.droplevel("scorer")
        v = np.nanmedian(vals[["x", "y"]].values, axis=0)
        borders[iw, :] = v

    borders = np.vstack([np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)])
    borders += ((np.diff(borders, axis=0) * 0.2).T @ np.array([[-1, 1]])).T
    for i, w in enumerate(["width", "height"]):
        borders[:, i] = np.clip(borders[:, i], 0, metadata[w])
    borders = borders.astype(int)
    crop_info = dict(
        xmin=int(borders[0, 0]),
        xmax=int(borders[1, 0]),
        ymin=int(borders[0, 1]),
        ymax=int(borders[1, 1]),
        dlc_source=str(dlc_ds.path),
        dlc_ds_id=dlc_ds.id,
    )
    with open(crop_file, "w") as fhandle:
        yaml.dump(crop_info, fhandle)
    print("Crop file created")
    return crop_info


def fit_ellipse(
    camera_ds_id,
    project_id,
    likelihood_threshold=None,
):

    flexilims_session = flz.get_flexilims_session(project_id)
    camera_ds = flz.Dataset.from_flexilims(
        id=camera_ds_id, flexilims_session=flexilims_session
    )
    ds_dict = get_tracking_datasets(camera_ds, flexilims_session)
    if ds_dict["cropped"] is None:
        raise IOError("No cropped dataset found")
    dlc_ds = ds_dict["cropped"]
    dlc_file = dlc_ds.path_full / dlc_ds.extra_attributes["dlc_file"]
    target = dlc_ds.path_full / f"{dlc_file.stem}_ellipse_fits.csv"

    assert dlc_file.exists()

    print(f"Doing %s" % dlc_file)
    ellipse_fits = eye_model_fitting.fit_ellipses(
        dlc_file,
        likelihood_threshold=likelihood_threshold,
    )
    print(f"Fitted, save to {target}")
    ellipse_fits.to_csv(target, index=False)
    print("Done")
