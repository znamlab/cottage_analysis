from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cottage_analysis")
except PackageNotFoundError:
    # package is not installed
    pass

from cottage_analysis import (
    analysis,
    dlc,
    ephys,
    eye_tracking,
    imaging,
    io_module,
    pipelines,
    plotting,
    preprocessing,
    summary_analysis,
)
