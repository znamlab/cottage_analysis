import pytest
from eye_tracking import io_func

ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision/"
MOUSE = "PZAH3.1c"
SESSION = "S20210406"
RECORDING = "R174548"


def test_load_video():
    data = io_func.load_video(mouse=MOUSE, session=SESSION, recording=RECORDING, root_dir=ROOT_DIR)
    assert data.shape == (1080, 1440, 291)
