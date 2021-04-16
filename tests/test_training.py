import pytest
import os
from eye_tracking import io_func, create_training_dataset

ROOT_DIR = "./resources/test_data"
OUT_DIR = os.path.join(ROOT_DIR, 'tests_outputs')
MOUSE = "PZAH3.1c"
SESSION = "S20210406"
RECORDING = "R174548"
EXAMPLE_HARP = "harp_messages_example.bin"


def test_subsample():
    create_training_dataset.generate_subset(input_dir=ROOT_DIR, output_dir=OUT_DIR, camera='right_eye_camera')

