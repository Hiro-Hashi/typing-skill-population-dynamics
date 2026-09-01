# Figure 5 Analysis Code

This directory contains the MATLAB code used to generate the analyses
shown in **Figure 5**, which examines session-by-session changes in
neural representations during closed-loop finger typing.

## Main script

-   `Fig5.m`

The script performs the following analyses in Figure 5.

-   **Fig. 5A:** Session-wise normalized Mahalanobis distance matrices
    for finger-movement representations in right and left area 6d.
-   **Fig. 5B:** Hierarchical clustering of the normalized Mahalanobis
    distance matrices.
-   **Fig. 5C:** Session-wise changes in normalized Mahalanobis distance
    and Spearman correlation of representational structure between
    adjacent sessions.
-   **Fig. 5D:** Session-wise SVM classification of finger movements
    using 10-fold cross-validation.
-   **Fig. 5E:** Comparison of representational structure between right
    and left area 6d.
-   **Fig. 5F-G:** Session-wise activation and suppression derived from
    FieldTrip-based spatiotemporal cluster analysis.

## Data preprocessing

The analyses in this directory use preprocessed neural activity recorded
during closed-loop typing sessions.

The preprocessing pipeline was as follows:

1. Neural activity was recorded during closed-loop typing.
2. The continuous closed-loop data were segmented into individual typing
   tokens using forced labeling with a hidden Markov model (HMM), which
   estimated the onset and offset of each token.
3. Neural activity corresponding to each labeled token was extracted.
4. Because token duration varied across trials, the extracted activity was
   temporally aligned using within-token Time Warping method (https://github.com/ahwillia/affinewarp/).
5. The resulting neural activity was further preprocessed and z-scored for
   the analyses presented in Figure 5.

This repository provides the preprocessed data used as
input to the Figure 5 analyses rather than the raw closed-loop recordings.

Providing these processed data allows the analyses and statistical procedures
used to generate Figure 5 to be reproduced directly from the shared inputs.


## Data

The script expects session-specific `.mat` files in:

``` text
data/
```

with filenames:

``` text
day45.mat
day85.mat
day86.mat
day92.mat
day93.mat
day106.mat
day107.mat
```

The analyses use preprocessed neural activity stored in variables
including:

-   `refined_Warping_zFR`: HMM-segmented, temporally aligned, z-scored
    firing-rate activity.
-   `base_zFR`: baseline z-scored firing-rate activity used for the
    spatiotemporal cluster analysis.

The HMM-segmented activity used here was generated during preprocessing
prior to the analyses in this script.

Intermediate analysis results are saved to:

``` text
results/
```

## Requirements

### MATLAB

The code was written in MATLAB and requires functions from the following
toolboxes:

-   **Statistics and Machine Learning Toolbox**


### FieldTrip

**FieldTrip** is required for the spatial cluster analysis used in Fig.
5F-G.
https://www.fieldtriptoolbox.org

The script assumes that FieldTrip is available at:

``` text
../Fig1/external/fieldtrip/
```

relative to the Figure 5 directory. Modify the path in `Fig5.m` if
FieldTrip is installed elsewhere.

The relevant FieldTrip functions include:

-   `ft_defaults`
-   `ft_prepare_neighbours`
-   `ft_timelockstatistics`

## Notes

-   Mahalanobis distances are normalized by the square root of the
    number of channels.
-   Finger-movement classes with fewer than 10 trials are excluded from
    the relevant analyses.
-   SVM classification uses a linear SVM with stratified 10-fold
    cross-validation.
-   Channel selection for SVM classification is based on a
    Kruskal-Wallis test followed by false-discovery-rate correction.
-   Statistical comparisons across sessions use the Friedman test or
    Kruskal-Wallis test as specified in the script.
-   Spearman correlations are used to quantify similarity of
    representational structure across adjacent sessions or cortical
    areas.

## Output

Running the script generates the individual analyses and plots
corresponding to the panels of Figure 5 and saves intermediate results
in the `results/` directory.

The final multi-panel figure shown in the manuscript was assembled from
these outputs.
