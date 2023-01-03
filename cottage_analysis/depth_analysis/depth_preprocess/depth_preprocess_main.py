'''
main script for basic preprocessing of 2p depth experimental data

For now, use env: 2p_analysis_clone
'''

#IMPORTS
from typing import Any

import os
import sys
import defopt
import pickle
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit
from suite2p.extraction import masks
import matplotlib
from matplotlib import cm

import cottage_analysis as cott
from cottage_analysis.depth_analysis.filepath import *
from cottage_analysis.imaging.common import *
from cottage_analysis.imaging.common import imaging_loggers_formatting as format_loggers
from cottage_analysis.stimulus_structure import sphere_structure as vis_stim_structure
from cottage_analysis.depth_analysis.plotting.plotting_utils import *
from cottage_analysis.depth_analysis.depth_preprocess.process_params import *
