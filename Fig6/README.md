# Figure 6, Connectivity Analyses

This folder contains the MATLAB code used to reproduce the analyses and
visualizations shown in **Figure 6**.

The analyses quantify the dimensionality of neural population activity
and functional coupling within and between the bilateral 6d arrays
across recording sessions.

## Main script

-   `Fig6.m` --- reproduces the analyses and visualizations for Fig.
    6A--F.

## Data

The script uses preprocessed firing-rate data from seven recording
sessions:

`45, 85, 86, 92, 93, 106, 107`

The input files are loaded from the Fig. 5 data directory:

``` text
../Fig5/data/day45.mat
../Fig5/data/day85.mat
../Fig5/data/day86.mat
../Fig5/data/day92.mat
../Fig5/data/day93.mat
../Fig5/data/day106.mat
../Fig5/data/day107.mat
```

-   **Left 6d:** channels 1--256
-   **Right 6d:** channels 257--384

## Output files

Intermediate analysis results are saved in the `results/` directory:

The `results/` directory is created automatically if it does not already
exist.

## MATLAB requirements

The code requires MATLAB and the following toolboxes:

-   **Statistics and Machine Learning Toolbox**

-   **Parallel Computing Toolbox**


## Running the analysis

Set the MATLAB working directory to the folder containing `Fig6.m` 

The script assumes that the corresponding Fig. 5 data files are
available at `../Fig5/data/`.

Because several analyses use parallel processing and bootstrap or
permutation-style resampling, execution time may vary depending on the
available hardware.

## Notes

-   The script contains all analysis-specific helper functions as local
    functions at the end of `Fig6.m`.

