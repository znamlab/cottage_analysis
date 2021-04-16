from cottage_analysis.io_module.video import io_func

ROOT_DIR = "./resources/test_data"

def test_load_video():
    data = io_func.load_video(ROOT_DIR, camera='right_eye_camera')
    assert data.shape == (1080, 1440, 2)
