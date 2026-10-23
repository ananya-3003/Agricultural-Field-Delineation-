# Multimodal Edge-Aware Agricultural Field Boundary Delineation Using Residual Networks and Variance-Guided Spatial-Channel Attention

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Remote Sensing](https://img.shields.io/badge/Domain-Remote%20Sensing%20%7C%20EO-green.svg)]()

This repository provides the official implementation of **EA_Delta**, an enhanced deep learning architecture for high-resolution agricultural field boundary delineation from complementary **Sentinel-1 SAR** (microwave backscatter) and **Sentinel-2 Optical** satellite imagery.

The codebase focuses on two core architectural contributions:
1. **15 Residual Shortcut Highways** across all encoder, bottleneck, and decoder stages to preserve thin boundary gradients.
2. **Standard-Deviation Squeeze-and-Excitation (SD-scSE) Attention** to prioritize high-frequency edge contrast over homogeneous crop canopy reflectance.

---

## 🌟 Architectural Enhancements

```
                      DUAL-STREAM RESIDUAL U-NET WITH SD-scSE ATTENTION
                      
   Sentinel-1 (SAR) ────► [ ResEncoder E0-E4 ] ──┐
                          (5 Residual Highways)  │
                                                 ├──► [ Central Bottleneck ] ──► [ ResDecoder D4-D1 ] ──► Boundary Mask
   Sentinel-2 (MSI) ────► [ ResEncoder E0-E4 ] ──┘    (1 Residual Highway)       (4 Residual Highways)
                          (5 Residual Highways)                │                          │
                                                               ▼                          ▼
                                                        [ SD-scSE Attn ]           [ SD-scSE Attn ]
                                                     (Mean + StdDev Channel)    (Mean + StdDev Channel)
```

### 1. 15 Residual Shortcut Highways (`dl_model/blocks.py`)
In standard convolutional U-Nets, deep downsampling operations attenuate thin boundary signals ($1\text{--}2\text{ pixels}$). To combat gradient dissipation, every encoder and decoder block incorporates a residual shortcut:

$$\mathbf{y} = \text{scSE}\Big(\text{ReLU}\big(\mathcal{F}(\mathbf{x}) + \mathcal{S}(\mathbf{x})\big)\Big)$$

Where:
* $\mathcal{F}(\mathbf{x})$: Standard convolutional branch ($\text{Conv}_{3 \times 3} \rightarrow \text{BN} \rightarrow \text{ReLU} \rightarrow \text{Conv}_{3 \times 3} \rightarrow \text{BN}$).
* $\mathcal{S}(\mathbf{x}) = \mathbf{x}$ (identity mapping) when input and output channel dimensions match.
* $\mathcal{S}(\mathbf{x}) = \text{BN}(\text{Conv}_{1 \times 1}(\mathbf{x}))$ (linear projection) when feature channels change.

---

### 2. Standard-Deviation Spatial-Channel SE (SD-scSE) Attention (`dl_model/scse.py`)
Standard Squeeze-and-Excitation (SE) mechanisms rely solely on spatial Global Average Pooling (spatial mean $\mu_c$):

$$\mu_c = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W x_c(i,j)$$

In agricultural remote sensing, boundary pixels occupy only a tiny fraction ($\sim 3\text{--}5\%$) of the scene. Consequently, $\mu_c$ is heavily dominated by interior crop canopy reflectance.

To address this, we introduce **spatial standard deviation ($\sigma_c$)**, which explicitly quantifies spatial contrast and high-frequency edge variation:

$$\sigma_c = \sqrt{\frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W (x_c(i,j) - \mu_c)^2 + \epsilon}$$

The joint descriptor vector $\mathbf{s} = [\mu_c \,\|\, \sigma_c] \in \mathbb{R}^{2C \times 1 \times 1}$ is recalibrated through a shared bottleneck multi-layer perceptron:

$$\mathbf{z}_{\text{SD-cSE}} = \mathbf{x} \cdot \sigma\left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot [\mu_c \,\|\, \sigma_c]) \right)$$

Concurrent with Spatial Attention ($\mathbf{z}_{\text{sSE}} = \mathbf{x} \cdot \sigma(\text{Conv}_{1 \times 1}(\mathbf{x}))$):

$$\mathbf{z}_{\text{SD-scSE}} = \mathbf{z}_{\text{SD-cSE}} + \mathbf{z}_{\text{sSE}}$$

This allows the network to selectively amplify edge-sensitive feature maps while suppressing interior crop texture noise.

---

## 📁 Repository Structure

```
Git repo/
├── dl_model/                   # Neural network architectures
│   ├── models.py               # FeatureLevelSCSEUNet (Baseline, Residual, SD-scSE)
│   ├── blocks.py               # ResEncoderBlock, ResDecoderBlock (15 Residual Highways)
│   ├── scse.py                 # SDChannelSE (Variance Attention), SpatialSE, SCSEBlock
│   └── __init__.py
├── losses/                     # Compound boundary loss formulations
│   ├── losses.py               # BCETverskySobelLoss, Distance-Weighted BCE, Dice Loss
│   ├── loss_builder.py         # Loss constructor
│   └── __init__.py
├── utils/                      # Data pipeline and builders
│   ├── data_loaders.py         # PyTorch Dataset & DataLoader
│   ├── data_generator.py       # Patch processing and normalizations
│   ├── builders.py             # Optimizer and scheduler configurations
│   └── __init__.py
├── config.py                   # Model hyperparameters
├── train.py                    # Modular training script
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git exclusions
```

---

## 🚀 Quick Start

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Verify Architecture & Tensor Flow (No Dataset Required)
You can immediately verify model instantiation, forward pass across S1 (SAR) and S2 (Optical) streams, and compound loss backpropagation:
```bash
python test_architecture.py
```

### 3. Training the Model
Train any architectural stage using the unified training entry point:
```bash
# 1. Author Baseline Architecture
python train.py --stage baseline --epochs 50 --batch_size 4

# 2. + 15 Residual Shortcut Highways
python train.py --stage residual --epochs 50 --batch_size 4

# 3. + Residual + SD-scSE Attention (Full Proposed Model)
python train.py --stage sd_scse --epochs 50 --batch_size 4
```

---

## 📜 Citation
```bibtex
@article{saxena2026multimodal,
  title={Multimodal Edge-Aware Agricultural Field Boundary Delineation Using Residual Networks and Variance-Guided Spatial-Channel Attention},
  author={Saxena, Ananya},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026}
}
```
