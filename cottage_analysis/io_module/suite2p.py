"""Helper functions for loading Suite2p data."""
import numpy as np
from pathlib import Path


def load_is_cell(suite2p_path: Path) -> np.ndarray:
    """Load the iscell array from Suite2p.

    If a `combined` directory exists, the function loads the iscell array from it.
    Otherwise, it loads the iscell array from each plane directory and concatenates them
    
    Args:
        suite2p_path: Path to the Suite2p output directory.

    Returns:
        iscell: Array of booleans indicating whether each ROI is a cell.
    """
    suite2p_path = Path(suite2p_path)
    is_cell = []

    if suite2p_path.joinpath("combined").exists():
        folder = suite2p_path.joinpath("combined")
        is_cell.append(np.load(folder / "iscell.npy"))
    else:
        plane_idx = 0
        folder = suite2p_path / f"plane{plane_idx}"
        while folder.exists():
            is_cell.append(np.load(folder / "iscell.npy"))
            plane_idx += 1
            folder = suite2p_path / f"plane{plane_idx}"

    return np.vstack(is_cell)[:, 0].astype(bool)