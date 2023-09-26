import os
from pathlib import Path
import struct
import cv2
import numpy as np
import mmap
import pandas as pd

DEPTH_DICT = {8: np.uint8, 16: np.uint16}


def load_video(data_folder, camera, order="F", metadata_file=None, binary_file=None):
    """Load the video from a camera saved as raw binary file + metadata

    The metadata file is a txt file with the following format:

    ```
    width: 640
    height: 480
    depth: 8
    ```

    Args:
        data_folder (str): path to the folder containing the data
        camera (str): name of the camera
        order (str, optional): order of the frames in the binary file. Defaults to "F".
        metadata_file (str, optional): path to the metadata file. Defaults to None.
        binary_file (str, optional): path to the binary file. Defaults to None.

    Returns:
        np.ndarray: array of shape (height, width, frames)

    """
    if metadata_file is None:
        metadata_file = os.path.join(data_folder, "%s_metadata.txt" % camera)
    assert os.path.isfile(metadata_file)
    metadata = {}
    with open(metadata_file, "r") as m_raw:
        for line in m_raw:
            if line.strip():
                k, v = line.strip().split(":")
                metadata[k.strip().lower()] = int(v.strip().replace("U", ""))
                # note that CV datatype are 'U8' or 'U16', remove the U before int.
    if binary_file is None:
        binary_file = os.path.join(data_folder, "%s_data.bin" % camera)
    assert os.path.isfile(binary_file)
    data = np.memmap(binary_file, dtype=DEPTH_DICT[metadata["depth"]], mode="r")
    if order != None:
        data = data.reshape((metadata["height"], metadata["width"], -1), order=order)
    return data


def write_array_to_video(
    target_file,
    video_array,
    frame_rate,
    is_color=False,
    verbose=True,
    codec="mp4v",
    extension=".mp4",
    min_brightness=None,
    max_brightness=None,
    perc_saturation=0,
    overwrite=False,
):
    """Write an array to a mp4 file

    The array must shape must be (lines/height x columns/width x frames)

    Args:
        target_file (str): path to the target file
        video_array (np.ndarray): array to write
        frame_rate (float): frame rate of the video
        is_color (bool, optional): is the video color? Defaults to False.
        verbose (bool, optional): print progress? Defaults to True.
        codec (str, optional): codec to use. Defaults to "mp4v".
        extension (str, optional): extension of the file. Defaults to ".mp4".
        min_brightness (int, optional): minimum brightness. Defaults to None.
        max_brightness (int, optional): maximum brightness. Defaults to None.
        perc_saturation (int, optional): percentage of saturation to remove. Defaults to 0.
        overwrite (bool, optional): overwrite the file if it exists. Defaults to False.

    Returns:
        None
    """
    if not target_file.endswith(extension):
        target_file += extension
    if not overwrite:
        assert not os.path.isfile(target_file)
    if verbose:
        print("Creating writer")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # writer to `output` in mp4v at 20 fps of the correct size, NOT color
    out = cv2.VideoWriter(
        target_file,
        fourcc,
        frame_rate,
        (video_array.shape[1], video_array.shape[0]),
        is_color,
    )

    # rescale contrast
    if verbose:
        print("Finding saturation limits")
    if perc_saturation:
        sample_hist = np.percentile(
            video_array, [perc_saturation / 2, 100 - perc_saturation / 2]
        )
    else:
        sample_hist = [0, 255]
        if min_brightness is not None:
            sample_hist[0] = min_brightness
        if max_brightness is not None:
            sample_hist[1] = max_brightness
    if verbose:
        msg = ""
        n_frames = video_array.shape[2]
        frames_done = 0
    for frame in range(video_array.shape[2]):
        frame = np.array(video_array[:, :, frame])
        if min_brightness is not None:
            frame[frame < min_brightness] = min_brightness
        if max_brightness is not None:
            frame[frame > max_brightness] = max_brightness
        if sample_hist[0] != 0 or sample_hist[1] != 255:
            rescaled_frame = (
                (np.array(frame, dtype=float) - sample_hist[0]) / sample_hist[1] * 255
            )
            rescaled_frame[rescaled_frame > 255] = 255
            rescaled_frame[rescaled_frame < 0] = 0
            rescaled_frame = np.array(rescaled_frame, dtype=np.uint8)
        else:
            rescaled_frame = frame
        out.write(rescaled_frame)
        frames_done += 1
        if verbose:
            erase_line = "\b" * len(msg)
            msg = "%d/%d frames done (%d%%)" % (
                frames_done,
                n_frames,
                frames_done / n_frames * 100,
            )
            print(erase_line + msg, flush=True, end="")
    if verbose:
        print("Done!")
    out.release()
    return


def deinterleave_camera(
    camera_file,
    target_file,
    make_grey=False,
    verbose=True,
    intrinsic_calibration=None,
    frame_rate=60.0,
):
    """Deinterleave a video file

    This function is intended for Wehrcam, which save NTSC video as interlaced. The even
    pixels correspond to the first acquired field, the odd pixels to the second field.
    This function deinterleave the video and save it as a new file with the same
    resolution by interpolating the missing pixels.

    Args:
        camera_file (str): path to the video file
        target_file (str): path to the target file
        make_grey (bool, optional): convert to grey? Defaults to False.
        verbose (bool, optional): print progress? Defaults to True.
        intrinsic_calibration (dict, optional): intrinsic calibration parameters.
            Defaults to None.
        frame_rate (float, optional): frame rate of the video. Defaults to 60.

    Returns:
        None
    """
    camera_file = Path(camera_file)
    if not camera_file.exists():
        raise IOError("%s is not a file" % camera_file)
    target_file = Path(target_file)
    if not target_file.parent.is_dir():
        raise IOError("Folder %s does not exists" % target_file.parent)
    cap = cv2.VideoCapture(str(camera_file))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if intrinsic_calibration is not None:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            intrinsic_calibration["mtx"],
            intrinsic_calibration["dist"],
            (frame_width, frame_height),
            1,
            (frame_width, frame_height),
        )
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = (
        chr(fcc & 0xFF)
        + chr((fcc >> 8) & 0xFF)
        + chr((fcc >> 16) & 0xFF)
        + chr((fcc >> 24) & 0xFF)
    )
    output = cv2.VideoWriter(
        str(target_file),
        cv2.VideoWriter_fourcc(*fcc),
        frame_rate,
        (frame_width, frame_height),
    )
    ret, frame = cap.read()

    if make_grey:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    deint_frame = np.zeros_like(frame)
    iframe = 0
    while ret:
        if verbose:
            iframe += 1
            if iframe % 1000 == 0:
                print("... frame %d" % iframe)
        if intrinsic_calibration is not None:

            dst = cv2.undistort(
                frame,
                intrinsic_calibration["mtx"],
                intrinsic_calibration["dist"],
                None,
                newcameramtx,
            )
        for ilines in range(2):
            deint_frame[::2, ...] = frame[ilines::2, ...]
            deint_frame[1::2, ...] = deint_frame[::2, ...]
            output.write(deint_frame)
        ret, frame = cap.read()
    if verbose:
        print("Done (total %d frames)" % iframe)
    cap.release()
    output.release()
