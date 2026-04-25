# Dual-Scale Feature Tokenization Transformer (DS-FTT) for HSI Classification

This project presents a deep learning approach for hyperspectral image classification using a Dual-Scale Feature Module with a Vision Transformer. It extends the SSFTT model by introducing a dual-branch design to improve spectral–spatial feature extraction.

---

## Novelty: Dual-Scale Feature Module

Instead of a single feature extraction pipeline, this model uses two parallel branches to capture both local and global information.

The small-scale branch focuses on fine details, while the large-scale branch captures broader contextual features using dilated convolutions.

The combined features provide a richer representation before being passed to the transformer.

---

## Architecture Overview

### Preprocessing

The input hyperspectral image is reduced using PCA, and spatial patches are extracted around each pixel.

### Dual-Scale Feature Extraction

Two parallel branches process the patches:

* Small-scale branch for localized features
* Large-scale branch for contextual features

### Fusion and Transformer

The features are fused and converted into tokens.
These tokens are processed by a transformer encoder to learn spectral–spatial relationships.

### Classification

A classification layer predicts pixel-wise class labels.

---

## Repository Structure

The project is organized by datasets:

IndianPines
Pavia
Houston

Each folder contains training scripts, model definition, and utilities for generating classification maps.

---

## How to Run

Install dependencies:

pip install torch torchvision numpy scipy matplotlib scikit-learn

Run training:

cd Pavia
python ip_train_pavia.py

---

## Output

* Classification maps
* Metrics: Overall Accuracy (OA), Average Accuracy (AA), Kappa

---

## Acknowledgements

This project builds upon the open-source repository for the standard SSFTT method:

L. Sun, G. Zhao, Y. Zheng and Z. Wu, "Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, 2022.

We extend our gratitude to the original authors. This repository significantly modifies their baseline by introducing the Dual-Scale Feature Module to achieve superior representation capabilities.
