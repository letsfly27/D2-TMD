# D²-TMD: Dynamic Decoupling and Topology-Morphology Joint Distillation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper: **"A dynamic decoupling and topology-morphology distillation method for 3D seismic fault identification"** (Submitted to *Computers & Geosciences*).

## 📖 Overview

Accurate fault identification is of great significance for 3D seismic interpretation. While Knowledge Distillation (KD) offers an effective means to balance accuracy and efficiency, conventional KD methods suffer from topological loss, multi-granularity learning deficits, and edge diffusion when applied to dense 3D fault segmentation tasks.

To address these issues, we propose **D²-TMD**, a three-stage joint distillation framework integrating macro-level topology alignment, meso-level semantic decoupling, and micro-level geometric correction. With only **1.243M parameters** and **31 GFLOPs** (using MobileNetV4-3D as the student), D²-TMD achieves exceptional boundary depiction accuracy and topological integrity comparable to heavy teacher networks.

### ✨ Key Innovations
- **NSA (Numerical-Structural Joint Alignment)**: Exploits implicit space correlations via cross-channel co-occurrence constraints to enhance global structural priors.
- **POD (Process-Aware Omni-Granularity Decoupling)**: Guides coarse-to-fine semantic fitting through multi-scale spatial distillation with progressive dynamic weighting.
- **MBC (Morphology-Adaptive Boundary Correction)**: Suppresses edge diffusion by amplifying penalties on prediction errors in high-frequency boundary regions.

---

## 🛠️ Requirements & Installation

Create a virtual environment and install the required dependencies:

```bash
conda create -n d2tmd python=3.9
conda activate d2tmd
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm scipy
