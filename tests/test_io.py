import pytest
import os
from eye_tracking import io_func, create_training_dataset

ROOT_DIR = "./resources/test_data"
OUT_DIR = os.path.join(ROOT_DIR, 'tests_outputs')
MOUSE = "PZAH3.1c"
SESSION = "S20210406"
RECORDING = "R174548"
EXAMPLE_HARP = "harp_messages_example.bin"

for file_name in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, file_name))


def test_load_video():
    data = io_func.load_video(ROOT_DIR, camera='right_eye_camera')
    assert data.shape == (1080, 1440, 2)


def test_harp():
    fpath = os.path.join(ROOT_DIR, EXAMPLE_HARP)

    msg_df = io_func.read_message(fpath, verbose=False)
    assert msg_df.shape == (5000, 11)
    msg_df = io_func.read_message(fpath, verbose=False, valid_addresses=32)
    assert msg_df.shape == (429, 11)
    msg_df = io_func.read_message(fpath, verbose=False, valid_addresses=(12, 0))
    assert msg_df.shape == (2, 11)
    msg_df = io_func.read_message(fpath, verbose=False, valid_addresses=(12, 44), valid_msg_type=2)
    assert msg_df.shape == (0, 0)
    msg_df = io_func.read_message(fpath, verbose=False, valid_msg_type='event')
    assert msg_df.shape == (4895, 11)
    msg_df = io_func.read_message(fpath, verbose=False, valid_addresses=(12, 44), valid_msg_type=['write', 'Read'])
    assert msg_df.shape == (2, 11)
    msg_df = io_func.read_message(fpath, verbose=False, valid_addresses=(12, 44), valid_msg_type=[1, 3])
    assert msg_df.shape == (4469, 11)

