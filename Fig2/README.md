# Figure 2 analysis

This MATLAB script reproduces the analyses and visualizations related to
**Figure 2**.

## Overview

`Fig2.m` performs HMM-based forced alignment of neural activity recorded
during closed-loop typing and compares the resulting token timing with
the online RNN decoder output. The HMM procedure includes Blank states
and refines token onset/offset times using template stretching and
temporal shifting.

The script also summarizes token occurrence, character error rate (CER),
HMM--RNN token matching, typing speed, and within-session changes across
recording days.

## Input data

The script expects the following directories relative to the project
root:

``` text
data/
├── closedLoopData/
├── templateData/
└── timeData/

results/
```

Session days analyzed are:

``` text
45, 85, 86, 92, 93, 106, 107
```

The script assumes that `projectDir` is the current MATLAB working
directory.

### Refined templates

The HMM-based forced alignment uses refined token templates derived from
the Finger Sweep data. These refined templates were generated in advance
and are provided as input files in `data/templateData/`; they are not
generated within `Fig2.m`.

For each recording session, the corresponding refined templates are
loaded from `dayXXRefinedTemplate.mat`.

## MATLAB requirements

The analyses use standard MATLAB functions together with functions from
the **Statistics and Machine Learning Toolbox**, including `fitlm`,
`corr`, `gscatter`, `normfit`, `kruskalwallis`, and `friedman`.

All custom helper functions required for the HMM alignment, token
matching, and plotting are included as local functions in `Fig2.m`.

## Reference

The HMM-based forced-alignment procedure was adapted from:

Willett FR, Avansino DT, Hochberg LR, Henderson JM, Shenoy KV.\
*High-performance brain-to-text communication via handwriting.*\
Nature. 2021;593:249--254.
