from pathlib import Path
import os

ROOT_DIR = Path("tests", "test_data")
from cottage_analysis.eye_tracking import create_training_dataset


OUT_DIR = os.path.join(ROOT_DIR, 'tests_outputs')
MOUSE = "PZAH3.1c"
SESSION = "S20210406"
RECORDING = "R174548"
EXAMPLE_HARP = "harp_messages_example.bin"

for file_name in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, file_name))

def test_subsample():
    create_training_dataset.generate_subset(input_dir=ROOT_DIR, output_dir=OUT_DIR, camera='right_eye_camera')

