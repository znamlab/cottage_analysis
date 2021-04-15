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
