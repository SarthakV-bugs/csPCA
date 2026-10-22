# csPCA

A modular Python project for prostate cancer segmentation focused on identifying clinically significant prostate cancer (csPCA) lesions from MRI data.

This repository combines:
- preprocessing utilities for image/label mapping
- model definitions for segmentation pipelines
- training code with custom loss functions
- configuration for dataset paths and training hyperparameters
- a classical ML exploratory workflow under `csPCA_classical_ml/`

## Overview

The project is designed around a segmentation workflow for MRI-based lesion detection. It uses a configuration-driven setup to:
- load MRI and annotation data
- align imaging volumes with label masks
- prepare 2D slices or processed patches for training
- train a segmentation model with class imbalance handling
- evaluate using thresholds and post-processing

The codebase is organized to support experimentation with different model and training strategies while keeping dataset and pipeline configuration centralized in `config.py`.

## Repository structure

```text
csPCA/
├── config.py                     # Global project configuration and training settings
├── Documentation                 # High-level project notes and dataset description
├── preprocessing/               # File loading and label mapping utilities
│   ├── loading_files.py
│   └── mapping_labels.py
├── models/                      # Deep learning segmentation models
│   ├── boundarylossmodel.py
│   ├── boundarylossmodel_2p5d.py
│   ├── boundarylossmodel_multimodal.py
│   └── __init__.py
├── training/                     # Training utilities and loss functions
│   ├── losses.py
│   └── __init__.py
├── csPCA_classical_ml/          # Classical ML / image processing experiments
│   ├── Imageprocessing.py
│   ├── imgpro.py
│   └── main.py
├── .gitignore
└── README.md
```

## Dataset assumptions

The project expects MRI and label data structured around the paths declared in `config.py`:

```python
project_root = Path(__file__).parent
checkpoint_dir = project_root / 'checkpoints'
labels_root = project_root / 'data' / 'picai_labels'
mri_root = project_root / 'data' / 'mri_images'
```

This means the repository is intended to be used with local data folders such as:

```text
data/
├── mri_images/
│   └── ...
└── picai_labels/
    └── ...
```

The configuration also sets training split ratios and class imbalance handling for positive and negative cases, making this workflow suitable for lesion detection datasets with imbalanced labels.

## Key features

- MRI-based segmentation pipeline for prostate cancer
- class-imbalance-aware training configuration
- custom combined loss setup with dice, focal, and boundary terms
- augmentation and weighted sampling support
- checkpoint and early-stopping-oriented training workflow
- modular preprocessing and model components

## Setup

1. Clone the repository.
2. Create a Python environment and install the required dependencies (for example: PyTorch, NumPy, scikit-image, and any image-processing libraries used by the project).
3. Prepare your local dataset and update the paths in `config.py` to match your environment.
4. Run the relevant training or inference entrypoint for your workflow.

## Configuration

The main settings are defined in `config.py` and include:
- dataset paths
- image target size
- batch size and workers
- training epochs and learning rate
- augmentation options
- loss weights and class imbalance parameters
- threshold optimization and post-processing settings

Before training, review the values in `config.py` and adjust them for your dataset size, hardware availability, and experiment requirements.

## Usage notes

This repository is organized as a research/development codebase rather than a single packaged CLI. The workflow is intended to be adapted to the local dataset and model experiments you want to run.

Typical workflow:
1. verify dataset paths in `config.py`
2. run preprocessing to align images and labels
3. execute training code using the model and loss definitions
4. inspect checkpoints and evaluate segmentation performance


