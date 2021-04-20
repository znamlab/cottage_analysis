"""
To decide which video codec we should use, it would be helpful to create test frames
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from cottage_analysis.io_module import video
import cv2


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

    OUTPUT_DIR = os.path.join(ROOT_DIR, "Test", "video_codecs")
    PLOT_EXAMPLE = True

    # first get the data
    raw_data_folder = os.path.join(ROOT_DIR, 'PZAH4.1c', 'S20210406', 'R184923')
    camera = 'right_eye_camera'
    video_array = video.load_video(raw_data_folder, camera=camera)

    if PLOT_EXAMPLE:
        fig = plt.figure(figsize=(14, 7))
        fig2 = plt.figure(figsize=(14, 7))
        example_frame = 0
        img_kwargs = dict(cmap='Greys_r', interpolation='none')

        def plot_ex(data, plot_index, fig=fig, img_kwargs=img_kwargs):
            ax = fig.add_subplot(3, 6, plot_index)
            ax.imshow(data, **img_kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            return ax

    # I will always use the first few frames as example
    video_array = np.array(video_array[:, :, :5])
    if PLOT_EXAMPLE:

        ax = plot_ex(video_array[:, :, example_frame], 1)
        ax.set_title('Full frame')
    # crop the eye part
    video_array = video_array[150:300, 950:1100]
    if PLOT_EXAMPLE:
        ax = plot_ex(video_array[:, :, example_frame], 2)
        ax.set_title('Cropped')

    # remove the end of the range
    cutoff = 30
    video_array[video_array > cutoff] = cutoff
    # video_array = np.array(np.array(video_array, dtype=float) * 255/cutoff, dtype=np.uint8)

    if PLOT_EXAMPLE:
        ax = plot_ex(video_array[:, :, example_frame], 3)
        ax.set_title('Display changed')

    # now save with various codec
    codec_list = dict(DIVX='.avi',
                      MJPG='.avi',
                      WMV1='.avi',
                      WMV2='.avi',
                      mp4v='.mp4',
                      vp09='.mp4',
                      RGBA='.avi',
                      XVID='.avi',
                      # LCW2='.avi', not found
                      # LJ2K='.mp4',not found
                      # MJ2C='.avi', raise plenty of warnings on re-read
                      # PVW2='.avi', not found
                      FFV1='.avi',
                      HFYU='.avi',
                      PIM1='.avi',
                      MP42='.avi',
                      DIV3='.avi',
                      x264='.avi',
                      # U263='.avi',
                      # I263='.avi',  # valid size: 128x96, 176x144, 352x288, 704x576, and 1408x1152.
                      # FLV1='.avi',valide size: 128x96, 176x144, 352x288, 704x576, and 1408x1152.
                      )
    codec_list['png '] = '.avi'

    print('Saving in %s' % OUTPUT_DIR)
    for codec, extension in codec_list.items():
        print("doing %s" % codec, flush=True)
        target_file = os.path.join(OUTPUT_DIR, 'example_%s%s' % (codec, extension))
        is_color = False
        frame_rate = 30
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(target_file, fourcc, frame_rate, (video_array.shape[1], video_array.shape[0]), is_color)
        for frame in range(video_array.shape[2]):
            out.write(video_array[:, :, frame])
        out.release()
    if PLOT_EXAMPLE:
        fig.subplots_adjust(top=0.99, bottom=0, left=0.01, right=1, hspace=0.01, wspace=0.2)
        fig2.subplots_adjust(top=0.99, bottom=0, left=0.01, right=1, hspace=0.01, wspace=0.2)

    print('Re-reading')
    # now re-read and plot
    i_plot = 4
    size_dict = dict()
    for fname in os.listdir(OUTPUT_DIR):
        if fname.startswith('example_'):
            fsize = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
            size_dict[fname] = fsize

    fsizes = np.array(list(size_dict.values()))
    files = list(size_dict.keys())
    order = fsizes.argsort()
    max_size = video_array.size
    rdbu = plt.get_cmap('RdBu_r', 7)
    for fname, fsize in [(files[o], fsizes[o]) for o in order]:
        if fname.startswith('example_'):
            print('Doing %s' % fname, flush=True)

            reader = cv2.VideoCapture(os.path.join(OUTPUT_DIR, fname))
            retval, image = reader.read()
            ax = plot_ex(image[:, :, 0], i_plot, fig=fig)
            ax.set_title(fname.split('_')[1].split('.')[0] + " %.1f kb (%d%%)" % (fsize/1024, fsize/max_size*100))
            ax = plot_ex(np.array(image[:, :, 0], dtype=float) - video_array[:, :, 0], i_plot, fig=fig2,
                         img_kwargs=dict(cmap=rdbu, interpolation='none', vmin=-3, vmax=3))
            plt.colorbar(ax.get_images()[0])
            ax.set_title(fname.split('_')[1].split('.')[0] + " %.1f kb (%d%%)" % (fsize / 1024, fsize / max_size * 100))
            i_plot += 1
    fig.savefig(os.path.join(OUTPUT_DIR, 'summary_figure_display_only.png'), dpi=600)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'summary_figure_compression_change.png'), dpi=600)
    print('done')
