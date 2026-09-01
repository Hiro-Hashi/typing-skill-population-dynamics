# Figure 3 Analysis

This directory contains the MATLAB analysis code used to reproduce the analyses shown in Figure 3.

## Software requirements

### MATLAB
- MATLAB **R2023b or later** is recommended.

### Required MathWorks toolboxes
- **Statistics and Machine Learning Toolbox**
- **Deep Learning Toolbox**

## Directory setup

Before running the script, set this Figure 3 directory as the MATLAB current working directory. The script defines

```matlab
projectDir = string(pwd);
```

and uses paths relative to `projectDir`.

The expected layout is approximately:

```text
<repository root>/
├── Fig1D_SpatiotemporalCluster/
│   └── data/
│       └── t18FingerSweepData.mat
└── <Figure 3 directory>/
    ├── <Figure 3 analysis script>.m
    ├── README.md
    ├── data/
    │   ├── LatentSpace_Autoencoder_result.mat
    │   └── SVMresults.mat
    └── results/
```

The Figure 3 script loads the preprocessed finger-sweep dataset from the sibling `Fig1D_SpatiotemporalCluster/data/` directory.

## Running the analysis

1. Open MATLAB.
2. Set the Figure 3 directory as the current working directory.
3. Confirm that the required data files are present.
4. Run the Figure 3 analysis script from the beginning.

The script contains the analyses for Figure 3A-H and local helper functions in the same file.

## Figure panels

### Figure 3A
Calculates the temporal dynamics of class-selective channels using Kruskal-Wallis tests with FDR correction for:

### Figure 3B
Performs linear multiclass SVM decoding using stratified 10-fold cross-validation and plots decoding accuracy over time for different array combinations.

### Figure 3C
Performs multi-finger movement classification using a multilayer perceptron (MLP) with 5-fold cross-validation and displays the resulting confusion matrix.

### Figure 3D
Projects neural activity into a two-dimensional latent space using an autoencoder with both reconstruction and classification objectives.

Autoencoder training is stochastic, so the exact two-dimensional latent representation can vary across runs. To reproduce the latent-space visualization shown in the manuscript, the script loads the saved result:

```text
data/LatentSpace_Autoencoder_result.mat
```

The autoencoder can also be retrained directly from the input data.

### Figure 3E
Computes pairwise Mahalanobis distances between movement classes and normalizes each distance matrix by the square root of the number of included channels.

### Figure 3F
Performs hierarchical clustering of the normalized Mahalanobis-distance matrices and visualizes the resulting dendrograms.

### Figure 3G
Plots the temporal dynamics of the normalized Mahalanobis distance.

### Figure 3H
Evaluates the relationship between SVM decoding accuracy and normalized Mahalanobis distance using:
- Pearson correlation
- linear regression

## Notes

- All custom helper functions required by this Figure 3 analysis are included as local functions in the analysis script.
- The code assumes the channel allocation and array definitions used in the accompanying dataset.
- Figure appearance can differ slightly across MATLAB releases because built-in plotting palettes and graphics defaults can change.
