"""
Utilities function for plotting

Functions that help handling figures but not specific to any analysis
"""
import numpy as np
import cv2


def get_img_from_fig(fig):
    """Get the array from a matplotlib figure

    This is particularly useful to generate videos from matplotlib videos

    Args:
        fig (plt.Figure): figure handle

    Returns:
        image_from_plot (np.array): RGB image from figure

    """
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot


def write_fig_to_video(fig, video_capture):
    """Save the figure as last frame of an opened video capture

    Use cv2.VideoCapture to create

    Args:
        fig (plt.figure): Matplotlib figure to save
        video_capture (cv2.VideoCapture): video capture object, should be created with 
            relevant parameters (fps, codecs, etc...)
    """
    img_array = get_img_from_fig(fig)
    # convert RGB to BGR for cv2
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    video_capture.write(img_array)


