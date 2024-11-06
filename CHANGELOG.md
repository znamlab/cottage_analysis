# Change log
## v2.0
### Major changes
- This is the working version for figure plotting utils for the repo v1_depth_map, written for the biorxiv submission of the manuscript "A depth map of visual space in the primary visual cortex" on Sept 27th, 2024.

## v2.0.2
### Major changes
- Move core analysis code from v1_depth_map to cottage_analysis:
    - Added plotting functions and plotting utils to the `plotting` package.
    - Added the `summary_analysis` package to store functions to do summary analysis and plot summary plots across sessions.

### Notes
- We need to retire `basic_vis_plots.py` from the `plotting` package