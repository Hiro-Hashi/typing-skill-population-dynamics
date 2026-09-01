# Figure 1 analysis and visualization

This directory contains the MATLAB code and data required to reproduce the analyses and visualizations associated with Figure 1 of the manuscript:

**Learning-related population dynamics in right and left dorsal premotor cortex during typing skill acquisition**

## Contents

```text
project_root/
├── Fig1.m
├── data/
│   ├── t18FingerSweepData.mat
│   └── SpatiotemporalClusterResults_Orig.mat
├── external/
│   └── fieldtrip/
└── results/
```

`Fig1.m` contains the analysis code and all custom helper functions required for Figure 1.

The script performs the following major steps:

1. Loads the finger-sweep neural dataset.
2. Converts threshold-crossing counts (ncTX) to firing rates using Gaussian smoothing.
3. Performs spatiotemporal cluster-based permutation analyses with FieldTrip.
4. Generates the visualizations corresponding to Figure 1D.
5. Generates representative peristimulus time histograms (PSTHs) for Figure 1E.
6. Calculates and plots normalized mean t-values for Figure 1F.
7. Computes task-wise Hamming distances and hierarchical clustering for Figure 1G.

## Software requirements

### MATLAB

MATLAB **R2023b or later** is recommended.

### Required MATLAB toolboxes

The following MathWorks products are required:

- **Statistics and Machine Learning Toolbox**

- **Image Processing Toolbox**

### External dependency

- **FieldTrip**

Please download FieldTrip from:
https://www.fieldtriptoolbox.org/

FieldTrip is used for the spatiotemporal cluster-based permutation analysis, including:

```matlab
ft_prepare_neighbours
ft_timelockstatistics
```

Place a FieldTrip installation in:

```text
external/fieldtrip/
```

relative to the project root. The script resets the MATLAB path and initializes FieldTrip automatically:

```matlab
restoredefaultpath
rehash toolboxcache
addpath(projectDir + "/external/fieldtrip");
ft_defaults
```


## Data files

The script expects the following files in the `data/` directory.

### `t18FingerSweepData.mat`

Primary preprocessed finger-movement dataset used to generate the firing-rate data and perform the spatiotemporal cluster analysis.


### `SpatiotemporalClusterResults_Orig.mat`

Original spatiotemporal cluster-analysis results used to reproduce the Hamming-distance and dendrogram analyses in Figure 1G and H.

This file is intentionally loaded for Figure 1G and H to reproduce the results used in the manuscript.

## Running the analysis

1. Install MATLAB and the required MathWorks toolboxes.
2. Download and install FieldTrip under `external/fieldtrip/`.
3. Place the required `.mat` files in the `data/` directory.
4. Set the project root (the directory containing `Fig1.m`) as the MATLAB current working directory.
5. Run:

```matlab
Fig1
```

The script assumes that the current working directory is the project root:

```matlab
projectDir = string(pwd);
```


### Channel numbering

The code converts between the post-mapping channel order used during feature extraction and the original Blackrock channel numbering.

The three array groups are:

```text
v6d : channels   1-128
d6d : channels 129-256
r6d : channels 257-384
```


### Figure 1D

Figure 1D visualizes electrodes belonging to significant spatiotemporal clusters during the selected analysis interval.

### Figure 1E

Figure 1E shows representative PSTHs.


### Figure 1F

Figure 1F summarizes the temporal evolution of t-statistics across significantly modulated channels.


### Figure 1G

Figure 1G quantifies similarity between movement-related spatial patterns.

### Figure 1H

Hierarchical clustering is performed using Ward linkage.


## Platform-specific note for macOS

FieldTrip contains compiled MEX files. On some macOS systems, downloaded FieldTrip MEX files may be blocked by macOS security/quarantine settings.

If you encounter an error such as

    "CalcMD5.mexmaca64" cannot be opened

or

    "ft_getopt.mexmaca64" cannot be opened,

run the following command in Terminal:

```bash
sudo xattr -r -d com.apple.quarantine /path/to/Fig1D_SpatiotemporalCluster/external
```

Then restart MATLAB and run the script again.

If MATLAB reports that a FieldTrip MEX file cannot be opened or verified, ensure that FieldTrip was installed according to the official FieldTrip installation instructions and that the downloaded files are trusted before modifying macOS security attributes.



## License



## Contact

For questions regarding the analysis or code, please contact the corresponding authors of the manuscript.
