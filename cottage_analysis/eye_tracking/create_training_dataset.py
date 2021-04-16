"""Holds function to generate training videos out of binary files"""
import os
import numpy as np
from cottage_analysis.io_module.video import io_func


def generate_subset(input_dir, camera, output_dir, **video_kwargs):
    """Generate a small video for each experiment found in input_dir"""
    assert os.path.isdir(output_dir)
    output_names = []
    for root, dirs, files in os.walk(input_dir):
        if not any(f.startswith(camera) for f in files):
            continue
        recording = '_'.join(root.split('/')[-3:])
        print('Extracting video for %s' % recording)
        video_array = io_func.load_video(data_folder=root, camera=camera)
        output_name = os.path.join(output_dir, '%s_%s_sample' % (recording, camera))
        make_mini_movie(video_array, output=output_name, **video_kwargs)
        output_names.append(output_name)
    return output_names


def make_mini_movie(video_array, output, num_frame=100, perc_saturation=1, **video_kwargs):
    """Take one binary file and subsample a few random frame to generate a small video

    This also rescales the video to the full 8 bits with prec_saturation pixels behind
    saturated
    """

    if video_array.shape[2] < num_frame:
        sample = video_array
    else:
        index = np.random.randint(0, video_array.shape[2], num_frame)
        sample = video_array[:, :, index]
    kwargs = dict(frame_rate=60)
    kwargs.update(video_kwargs)

    # rescale contrast
    sample_hist = np.percentile(sample, [perc_saturation/2, 100-perc_saturation/2])
    rescale_sample = (np.array(sample, dtype=float) - sample_hist[0]) / sample_hist[1] * 255
    rescale_sample[rescale_sample > 255] = 255
    rescale_sample[rescale_sample < 0] = 0
    rescale_sample = np.array(rescale_sample, dtype=np.uint8)
    io_func.write_array_to_video(output, rescale_sample, **kwargs)
    return rescale_sample


# if __name__ == "__main__":
#     ROOT_DIR = "/Volumes/lab-znamenskiyp/home/shared/projects/3d_vision"
#     OUTPUT_DIR = os.path.join(ROOT_DIR, "EyeCamCalibration/RightEyeCam/TrainingData")
#     video = generate_subset(input_dir=ROOT_DIR, camera='right_eye_camera',
#                             output_dir=OUTPUT_DIR, codec='RGBA', extension='.avi')
