# Change log

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
    - Added plotting functions and plotting utils to the `plotting` package.
    - Added the `summary_analysis` package to store functions to do summary analysis and plot summary plots across sessions.

### Notes
- We need to retire `basic_vis_plots.py` from the `plotting` package!
  
## v2.0
### Major changes
