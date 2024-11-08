"""Helper functions for loading Suite2p data."""
import numpy as np
from pathlib import Path


def load_is_cell(suite2p_path: Path) -> np.ndarray:
    """Load the iscell array from Suite2p.

    Args:
        suite2p_path: Path to the Suite2p output directory.

    Returns:
        iscell: Array of booleans indicating whether each ROI is a cell.
    """
    suite2p_path = Path(suite2p_path)
    plane_idx = 0
    folder = suite2p_path / f"plane{plane_idx}"
    is_cell = []
    while folder.exists():
        is_cell.append(np.load(folder / "iscell.npy"))
        plane_idx += 1
        folder = suite2p_path / f"plane{plane_idx}"

    return np.vstack(is_cell)[:, 0].astype(bool)
