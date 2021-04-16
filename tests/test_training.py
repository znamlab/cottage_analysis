from pathlib import Path
import os
from cottage_analysis.eye_tracking import create_training_dataset

ROOT_DIR = Path("tests", "test_data")
OUT_DIR = os.path.join(ROOT_DIR, 'tests_outputs')
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)


for file_name in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, file_name))


def test_subsample():
    create_training_dataset.generate_subset(input_dir=ROOT_DIR, output_dir=OUT_DIR, camera='right_eye_camera')

