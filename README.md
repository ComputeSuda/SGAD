## Contents for SGAD

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Dependencies](#dependencies)
  - [Download](#download)
- [Citation](#citation)

# Overview

Protein-protein interactions (PPIs) represent a delicate but universal mechanism for a wide range of biological processes in living cells. Many aberrant PPIs have been  identified to be involved in various diseases, which greatly enlarge the therapeutic targets for cancers, infectious disease, neurodegenerative diseases, and so on. Nowadays, multiple descriptors based on sequence information have been employed in many deep learning predictors. However, these models are mostly lacking the sufficient description of the original structure of protein networks and incapable of capturing the highly nonlinear structure of protein networks. As a result, the predictive models suffer from generalizability and are not applicable to the unseen datasets. On the other word, most of the predictive models are not robust in dealing with the noise in the dataset, leading to the fragility in predictive models. Herein, we proposed a novel computational framework, Structural Gated Attention Deep (SGAD) Model, for PPIs network reconstruction.

# Repo Contents

- [1-3Datasets](./1-3Datasets): PPI data sets (1:3 ratio).
- [1-5Datasets](./1-5Datasets): PPI data sets (1:5 ratio).
- [Datasets](./Datasets): PPI data sets and fasta files of proteins.
- [evaluation_indicators](./evaluation_indicators): evaluation index functions.
- [fig](./fig): plotting ROC curves.
- [src](./src): model, data loading and feature extraction.


# System Requirements

The `SGAD` package requires only a standard computer with enough RAM to support the operations defined by a user.  All the experiments were run on CentOS 8, CUDA 10.1.243, CuDnn 7.0, Python 3.7, Keras 2.0, and PyTorch 1.3.0.

# Installation Guide
## Dependencies
The following dependencies are required to run SGAD properly:

- scikit-learn==0.23.0
- numpy==1.16.2
- torch==1.3.1
- keras==2.0.8
- networkx==2.4.0

## Download

```
git clone https://github.com/ComputeSuda/SGAD.git
```

# Citation

For usage of the package, please cite
[the following paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00982).
```md
Fei Zhu, et al, Protein Interaction Network Reconstruction with a Structural Gated Attention Deep
Model by Incorporating Network Structure Information, J. Chem. Inf. Model., 2022, 62, 2, 258–273.
```

This repository is distributed under [GNU General Public License v3.0](LICENSE).
