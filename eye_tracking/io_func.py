import os
import numpy as np
import cv2

DEPTH_DICT = {8: np.uint8,
              16: np.uint16}


def load_video(data_folder, camera='right_eye_camera'):
    """Load the video from an eye cam"""
    metadata_file = os.path.join(data_folder, '%s_metadata.txt' % camera)
    assert os.path.isfile(metadata_file)
    metadata = {}
    with open(metadata_file, 'r') as m_raw:
        for line in m_raw:
            if line.strip():
                k, v = line.strip().split(":")
                metadata[k.strip()] = int(v.strip())

    binary_file = os.path.join(data_folder, '%s_data.bin' % camera)
    assert os.path.isfile(binary_file)
    data = np.memmap(binary_file, dtype=DEPTH_DICT[metadata['Depth']], mode='r')
    data = data.reshape((metadata['Height'], metadata['Width'],  -1), order='F')
    return data


def write_mp4(target_file, video_array, frame_rate, is_color=False, codec='mp4v', extension='.mp4', overwrite=False):
    """Write an array to a mp4 file

    The array must shape must be (lines/height x columns/width x frames)
    """
    if not target_file.endswith(extension):
        target_file += extension
    if not overwrite:
        assert not os.path.isfile(target_file)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    # writer to `output` in mp4v at 20 fps of the correct size, NOT color
    out = cv2.VideoWriter(target_file, fourcc, frame_rate, (video_array.shape[1], video_array.shape[0]), is_color)
    for frame in range(video_array.shape[2]):
        out.write(video_array[:, :, frame])
    return out


if __name__ == "__main__":
    ROOT_DIR = "../resources/test_data"
    ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision/"
    MOUSE = "PZAH4.1c"
    SESSION = "S20210406"
    RECORDING = "R184923"
    camera = 'right_eye_camera'
    data_folder = os.path.join(ROOT_DIR, MOUSE, SESSION, RECORDING)
    vid = load_video(data_folder=data_folder, camera=camera)
    print('Video loaded')
    codec = 'RGBA'
    extension = '.avi'
    target_file = '_'.join([MOUSE, SESSION, RECORDING, camera]) + extension
    write_mp4(target_file=os.path.join(data_folder, target_file), video_array=vid, frame_rate=120,
              codec=codec, extension=extension)
    print('done')
