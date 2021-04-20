"""Holds function to generate training videos out of binary files"""
import os
import numpy as np
from cottage_analysis.io_module.video import io_func


def generate_subset(input_dir, camera, output_dir, num_frame=100, **video_kwargs):
    """Generate a small video for each experiment found in input_dir"""
    assert os.path.isdir(output_dir)
    output_names = []
    for root, dirs, files in os.walk(input_dir):
        if not any(f.startswith(camera) for f in files):
            continue
        recording = '_'.join(root.split('/')[-3:])
        print('Extracting video for %s' % recording)
        try:
            video_array = io_func.load_video(data_folder=root, camera=camera)
        except AssertionError:
            print('Cannot load video! Skipping')
            continue
        output_name = os.path.join(output_dir, '%s_%s_sample' % (recording, camera))
        vid = make_mini_movie(video_array, num_frame=num_frame, output=output_name, **video_kwargs)
        if vid is not None:
            output_names.append(output_name)
    return output_names


def make_mini_movie(video_array, output, num_frame=100, overwrite=False, **video_kwargs):
    """Take one binary file and subsample a few random frame to generate a small video

    This also rescales the video to the full 8 bits with prec_saturation pixels behind
    saturated
    """
    kwargs = dict(frame_rate=60, extension='.mp4', codec='mp4v')
    kwargs.update(video_kwargs)
    target_file = output
    if not target_file.endswith(kwargs['extension']):
        target_file += kwargs['extension']
    if os.path.isfile(target_file) and not overwrite:
        print('This video was already extracted. Skipping')
        return
    if (num_frame is None) or (video_array.shape[2] < num_frame):
        sample = video_array
    else:
        index = np.random.randint(0, video_array.shape[2], num_frame)
        sample = video_array[:, :, index]

    io_func.write_array_to_video(output, sample, overwrite=overwrite, **kwargs)
    return 1


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    print('Running on %s' % hostname)
    if hostname == 'C02Z85AULVDC':
        # that's my laptop
        ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision/"
    else:
        # should be on camp
        ROOT_DIR = "/camp/lab/znamenskiyp/home/shared/projects/3d_vision/"

    OUTPUT_DIR = os.path.join(ROOT_DIR, "EyeCamCalibration/RightEyeCam/TrainingData")
    ROOT_DIR = os.path.join(ROOT_DIR, 'PZAH4.1c', 'S20210406', 'R184923')
    print('Saving in %s' % OUTPUT_DIR)
    video = generate_subset(input_dir=ROOT_DIR, camera='right_eye_camera', num_frame=None,
                            output_dir=OUTPUT_DIR, codec='mp4v', extension='.mp4',
                            max_brightness=30, overwrite=True, perc_saturation=0)
