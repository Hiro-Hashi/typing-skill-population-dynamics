# Figure 4: dPCA Analysis

This directory contains the MATLAB code used to reproduce the demixed
principal component analysis (dPCA) shown in Figure 4.

## Requirements

-   MATLAB R2023b or later recommended
-   dPCA MATLAB package by Kobak et al. (2016):
    https://github.com/machenslab/dPCA

Place the dPCA package in `external/dPCA/`.

The analysis uses the finger-sweep dataset from Figure 1:

`../Fig1/data/t18FingerSweepData.mat`

## Usage

Set the directory containing `Fig4.m` as the MATLAB current working
directory. In `Fig4.m`, select the population:

``` matlab
usedArray = "r6d";   % Right 6d
```

or

``` matlab
usedArray = "l6d";   % Left 6d
```

Then run `Fig4`.

The script generates: - `r6d`: Figure 4A, C, E, and G - `l6d`: Figure
4B, D, F, and H

Optimized regularization parameters are cached in the `results/`
directory.

## Analysis

Threshold-crossing counts are converted to firing rates using a Gaussian
kernel (SD = 50 ms) and z-scored relative to the `None` condition. dPCA
separates population activity into **Hand**, **Finger**, **Gesture**,
**Common**, and **Interactions** marginalizations.

Regularization is optimized using the dPCA cross-validation procedure
with five repetitions.

## Reference

Kobak D, Brendel W, Constantinidis C, et al. Demixed principal component
analysis of neural population data. *eLife*. 2016;5:e10989.
doi:10.7554/eLife.10989
