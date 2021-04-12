import os
import numpy as np

DEPTH_DICT = {8: np.uint8,
              16: np.uint16}


def load_video(mouse, session, recording, root_dir, camera='right_eye_camera'):
    """Load the video from an eye cam"""
    data_folder = os.path.join(root_dir, mouse, session, recording)
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

if __name__ == "__main__":
    ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision/"
    MOUSE = "PZAH3.1c"
    SESSION = "S20210406"
    RECORDING = "R174548"
    load_video(mouse=MOUSE, session=SESSION, recording=RECORDING, root_dir=ROOT_DIR)