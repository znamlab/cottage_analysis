import numpy as np


def get_heading(qs, w=0, x=1, y=2, z=3):
    """Compute heading from quaternion

    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Rotation_matrices

    Args:
        qs (np.ndarray): N by 4 quaternion array
        w (int, optional): w index. Defaults to 0.
        x (int, optional): x index. Defaults to 1.
        y (int, optional): y index. Defaults to 2.
        z (int, optional): z index. Defaults to 3.

    Returns:
        np.ndarray: heading array in radians
    """

    heading = np.arctan2(
        2.0 * (qs[:, z] * qs[:, w] + qs[:, x] * qs[:, y]),
        -1.0 + 2.0 * (qs[:, w] * qs[:, w] + qs[:, x] * qs[:, x]),
    )
    return heading
