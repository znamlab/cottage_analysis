# Mini camera
from pathlib import Path
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cottage_analysis.io_module.video.io_func import deinterleave_camera


cam_folder = Path('/Users/blota/Data/eye_camera/')
metadata_file = cam_folder / 'eye_camera_metadata_2022-08-18T15_48_50.yml'
camera_file = cam_folder / 'eye_camera_2022-08-24T17_41_39.mp4'
target = camera_file.with_name(camera_file.name.replace('.mp4', '_deinterleaved.mp4'))
deinterleave_camera(str(camera_file), str(target))
cam_folder = Path('/Users/blota/Data/world_camera/')
metadata_file = cam_folder / 'world_camera_metadata_2022-08-18T15_48_50.yml'
camera_file = cam_folder / 'world_camera_2022-08-18T15_48_50.mp4'
deinterleave_camera(str(camera_file), str(camera_file.with_name(
    camera_file.name.replace('.mp4', '_deinterleaved.mp4'))))

cap = cv2.VideoCapture(str(target))
nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fcc = (chr(fcc & 0xff) + chr((fcc >> 8) & 0xff) +
       chr((fcc >> 16) & 0xff) + chr((fcc >> 24) & 0xff))
out = None
ret, frame = cap.read()
fig, axes = plt.subplots(2, 3)
fig.set_size_inches([7.5, 5])
eye_movie = target.with_name('eye_movie.mp4')
fig.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
frame_index = 0
txt = ''
while ret:
    for x in axes.flatten():
        x.clear()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    axes[0, 0].imshow(frame[::-1, ::-1])
    # cut the part we want
    frame = frame[200:400:, 300:650, :]
    limits = [(None, None), (0, 200), (0, 150)]
    cmap = ['Greys', 'viridis', 'plasma']
    for icol, (vmin, vmax) in enumerate(limits):
        axes[1][icol].imshow(frame[::-1, ::-1, icol], cmap=cmap[icol], vmin=vmin,
                             vmax=vmax)

    axes[0, 1].imshow(frame[::-1, ::-1])
    axes[0, 2].imshow(gray[::-1, ::-1], cmap='Greys_r', vmin=5, vmax=200)
    for i, x in enumerate(axes.flatten()):
        x.axis('off')
        if i:
            x.set_xlim([0, 250])
            x.set_ylim([300, 50])

    fig.canvas.draw()
    buf, shape = fig.canvas.print_to_buffer()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = img.reshape(shape[::-1] + (4,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    if out is None:
        out = cv2.VideoWriter(str(eye_movie), cv2.VideoWriter_fourcc(*fcc), 60,
                              shape)
    out.write(img)
    frame_index += 1
    nchar = len(txt)
    txt = 'Frame %d, %05.2f %%' % (frame_index, (frame_index/nframe) * 100)
    print('\b' * nchar + txt)
    ret, frame = cap.read()
cap.release()
out.release()
