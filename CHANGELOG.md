# Change log

## v2.1.0

### Features
- **Ephys Pipeline**: Improved ephys pipeline and added `export_onix_to_si` for easier integration with SpikeInterface.
- **RF Analysis**: Refactored RF analysis into separate fitting and analysis modules. Added support for multidepth fits and masking coefficients.
- **Plotting Enhancements**:
    - Added many new options to `RS_OF_matrix` plots (synchronized `vmax`, custom ticks, grey background, toggleable R2 text labels).
    - Added multidepth options for stimulus reconstruction plots.
    - Updated `plot_treadmill_vs_closedloop_matrix` for better comparison.
- **Simulation**: Added functionality to simulate treadmill data as a 2D Gaussian plus an exponential kernel.
- **Utilities**: Added utility functions to interpret Gaussian fit parameters and added progress bars for long-running tasks like widefield video transformation.

### Bugfixes
- **Data Alignment**: Fixed a critical mismatch between dataframe index and ROI locations in `neurons_df`.
- **Fitting**: Excluded negative Optic Flow (OF) in RS/OF fits and fixed `fit_preferred_depth` for open-loop sessions.
- **Robustness**: Improved NaN handling across RF analysis functions and `neurons_df` generation.
- **Compatibility**: Fixed several Pandas warnings and adapted the codebase for Pandas 3.0.

### Dependencies
- Added `seaborn` to `install_requires` in `setup.py`.

## v2.0.4

### Main changes

- Overwrite neurons_df if it has the wrong number of ROI

### Bugfix

- Fix synchronization issue if FrameLog start after frame 0

### Minor

- Clearer error message in sync when failing to find datasets
- Option to load_session without returning the neurons_df (useful before it's created)

## v2.0.3
### Major changes
- Added analysis code for the mismatch experiments.
- Changed `preprocessing/synchronisation.py` to enable preprocessing of the mismatch stimulus protocol.
- Remove hardcoded `plane0` for reading the suite2p `iscell.py` file. Read `iscell.py` from all folders automatically instead.
### Minor changes
- Reformat SFTF analysis pipeline to remove all depth analysis parts.
- Minor format changes to SFTF plots.

## v2.0.2
### Major changes
- Move core analysis code from v1_depth_map to cottage_analysis:
    - Added plotting functions and plotting utils to the `plotting` package.
    - Added the `summary_analysis` package to store functions to do summary analysis and plot summary plots across sessions.

### Notes
- We need to retire `basic_vis_plots.py` from the `plotting` package!

## v2.0
### Major changes
- This is the working version for figure plotting utils for the repo v1_depth_map, written for the biorxiv submission of the manuscript "A depth map of visual space in the primary visual cortex" on Sept 27th, 2024.
