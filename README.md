
# Ordinal Diffusion Models for Fundus Images

This repository contains the code for  "Ordinal Diffusion Models for Fundus Images". 

## Installation
Set up a python environment with a python version 3.11. Then, download the repository, activate the environment and install all other dependencies with

```bash
cd OrdinalDiffusion
pip install -e .
```

## Model Weights
Model weights can be downloaded using the download_model_weights.py script. 

```bash
python download_model_weights.py
```

## Quick Start for Generating New Images
For generating new images, download the model weights and run:
```bash
python generate_images.py --model xx
```
where xx is the model variant:
- baseline w/o structure:     xx = base
- baseline w/ structure:      xx = basestruct
- equidistant w/o structure:  xx = equi
- equidistant w structure:    xx = equistruct
- learned w/o structure:      xx = learn
- learned w/ structure:       xx = learnstruct

## Project Structure

```
configs/
└──contains config.yaml for all model training.
src/ori/
├── models/          # Model architectures
├── data/            # Data loading and preprocessing
├── metrics/         # Evaluation metrics
├── xx_training.py   # Training wrapper for different models
├── criterion.py     # Loss functions
├── optimizers.py    # Optimizer
└── utils.py         # Helper functions
evaluation.py       # For classifying generated images
generate_images.py  # Generate images
calculate_fid.py    # Calculate FID
sample.py           # Sample Class
xx_train.py         # Training script for each model
```
