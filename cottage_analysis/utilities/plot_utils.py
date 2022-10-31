"""
Utilities function for plotting

Functions that help handling figures but not specific to any analysis
"""
import numpy as np


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


