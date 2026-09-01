# typing-skill-population-dynamics
Code associated with "Learning-related population dynamics in right and left dorsal premotor cortex during typing skill acquisition".
https://www.biorxiv.org/content/10.64898/2026.07.02.736059v1


# Code accompanying the manuscript

This repository contains the MATLAB code used to reproduce the analyses and main figures associated with our manuscript.

The code is organized by figure. Each figure directory contains the scripts, processed data, dependencies, and a figure-specific `README.md` describing the corresponding analysis in more detail.

## Figure-specific analyses

| Directory | Description |
|---|---|
| `Fig1/` | Code and data used for Figure 1 |
| `Fig2/` | Code and data used for Figure 2 |
| `Fig3/` | Code and data used for Figure 3 |
| `Fig4/` | Code and data used for Figure 4 |
| `Fig5/` | Code and data used for Figure 5 |
| `Fig6/` | Code and data used for Figure 6 |
| `Fig7/` | Code and data used for Figure 7 |

Please refer to the `README.md` within each figure directory for details regarding input data, analysis steps, dependencies, and output files.

## Requirements

The analyses were implemented primarily in MATLAB.

Required MATLAB toolboxes depend on the individual analysis and are documented in the corresponding figure-specific README files. 

Some analyses also rely on external MATLAB packages. These dependencies are included in, or referenced from, the corresponding figure directories where applicable.

> **MATLAB version:** Please specify the MATLAB release used for the final analyses here (for example, `MATLAB R2025a`).

## Usage

Each figure can generally be reproduced by navigating to the corresponding figure directory in MATLAB and running the main analysis script.

Because some analyses depend on intermediate processed data generated or stored in other figure directories, we recommend preserving the repository structure when running the code.

Detailed instructions are provided in the README file associated with each figure.

## Data

The repository contains processed data required to reproduce the analyses where sharing is permitted.

## Reproducibility

The code is organized to closely follow the analysis workflow used to generate the manuscript figures. Figure-specific scripts contain local helper functions where practical, and intermediate results are saved within the relevant figure directories.

Randomized analyses may produce small numerical differences across runs unless the random-number generator state is explicitly fixed.

## External dependencies

External software used by specific analyses is documented in the corresponding figure-specific README files.


## Citation

If you use this code, please cite the associated manuscript:

```text
Hashimoto H, Jude JJ, Levi-Aharoni H, Williams ZM, Simeral JD, Hochberg LR, Rubin DB.
Learning-related population dynamics in right and left dorsal premotor cortex during typing skill acquisition.
bioRxiv. 2026; 2026.07.02.736059.
doi:10.64898/2026.07.02.736059
```


