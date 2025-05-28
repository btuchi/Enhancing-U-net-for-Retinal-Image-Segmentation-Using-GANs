# Enhanced U-Net with GANs for Retinal Image Segmentation

A PyTorch implementation of an enhanced U-Net architecture combined with Generative Adversarial Networks (GANs) for improved retinal blood vessel segmentation. This project demonstrates how adversarial training can enhance traditional segmentation performance on medical imaging tasks.

## Overview

This project implements and compares two approaches for retinal vessel segmentation:
- **Standard U-Net**: Traditional encoder-decoder architecture
- **Enhanced U-Net + GAN**: U-Net generator with discriminator network for adversarial training

The enhanced approach uses adversarial loss to improve segmentation quality, achieving better boundary definition and reducing false positives in retinal vessel detection.

## Medical Application

Retinal vessel segmentation is crucial for:
- **Diabetic Retinopathy Screening**: Early detection of vascular changes
- **Glaucoma Assessment**: Monitoring optic nerve damage
- **Cardiovascular Risk Assessment**: Retinal vessels reflect systemic health
- **Automated Medical Diagnosis**: Computer-aided detection systems

## Architecture

### U-Net Generator
```
Input (3x640x640) → Encoder (Down-sampling) → Bottleneck → Decoder (Up-sampling) → Output (1x640x640)
├── Down1: 3→32 filters, MaxPool
├── Down2: 32→64 filters, MaxPool  
├── Down3: 64→128 filters, MaxPool
├── Down4: 128→256 filters, MaxPool
├── Down5: 256→512 filters (bottleneck)
├── Up1: 512+256→256 filters, Upsample
├── Up2: 256+128→128 filters, Upsample
├── Up3: 128+64→64 filters, Upsample
└── Up4: 64+32→32 filters, Upsample → 1x640x640
```

### Discriminator Network
```
Input (4x640x640) → Encoder → Global Average Pool → Binary Classification
├── Concatenated: RGB Image (3) + Segmentation Mask (1)
├── 5-layer encoder with progressive downsampling
└── Binary output: Real/Fake pair classification
```

## Getting Started

### Prerequisites
```bash
pip install torch torchvision pillow numpy scikit-learn tqdm
```

### Dataset Structure
```
data/
└── DRIVE/
    ├── training/
    │   ├── images/          # Input retinal images (.tif)
    │   ├── 1st_manual/      # Ground truth vessels (.gif)
    │   └── mask/            # Field of view masks (.gif)
    └── test/
        ├── images/
        ├── 1st_manual/
        └── mask/
```

### Training

**Standard U-Net:**
```bash
python train.py --add_gan_loss False
```

**Enhanced U-Net + GAN:**
```bash
python train.py --add_gan_loss True
```

### Testing & Evaluation
```bash
# Test standard U-Net
python test.py --model unet

# Test enhanced U-Net + GAN
python test.py --model enhanced_unet
```

## Technical Implementation

### Data Augmentation Pipeline
- **Color Augmentation**: Brightness, contrast, saturation adjustments
- **Geometric Transforms**: Horizontal/vertical flips, random crops
- **Gaussian Noise**: Robust training with noise injection
- **Normalization**: Channel-wise mean/std normalization

### Loss Functions
- **Segmentation Loss**: Binary Cross-Entropy for pixel classification
- **Adversarial Loss**: GAN loss for realistic segmentation quality
- **Combined Loss**: `L_total = L_seg + α * L_adv` (α = 0.03)

### Training Strategy
1. **Alternating Training**: Discriminator → Generator optimization
2. **Learning Rate**: Adam optimizer with lr=1e-4, β=(0.5, 0.9)
3. **Batch Size**: 2 (memory-efficient for 640x640 images)
4. **Epochs**: 1000 with checkpoints every 50 epochs

## Project Structure

```
├── model.py              # U-Net generator & discriminator architectures
├── dataset.py            # DRIVE dataset loader with augmentations
├── train.py              # Training loop with GAN/standard modes
├── test.py               # Evaluation and metrics calculation
├── transform.py          # Image preprocessing utilities
├── misc/
│   ├── data_aug.py       # Data augmentation functions
│   └── utils.py          # Training utilities
├── output/
│   ├── unet/             # Standard U-Net checkpoints & results
│   └── enhanced_unet/    # GAN-enhanced model checkpoints & results
├── unet.log              # Standard U-Net training metrics
└── enhanced_unet.log     # Enhanced U-Net training metrics
```

## Visualization & Results

The model generates:
- **Segmentation masks**: Binary vessel predictions
- **ROC curves**: True/False positive rate analysis  
- **Precision-Recall curves**: Precision vs recall trade-offs
- **Side-by-side comparisons**: Original images, ground truth, predictions

## Research Contributions

### Novel Aspects
- **Medical GAN Application**: Adversarial training for retinal segmentation
- **Architecture Optimization**: Balanced generator-discriminator design
- **Loss Function Design**: Optimal weighting of segmentation vs adversarial loss
- **Evaluation Methodology**: Comprehensive metrics for medical image segmentation

### Clinical Relevance
- **Automated Screening**: Enables large-scale retinal disease screening
- **Diagnostic Assistance**: Supports ophthalmologists in vessel analysis
- **Quantitative Analysis**: Provides objective vessel measurement tools
- **Research Platform**: Foundation for further retinal imaging research

## Contributing

Contributions welcome! Areas for improvement:
- **Multi-scale architectures**: Pyramid-based feature extraction
- **Attention mechanisms**: Focus on relevant vessel regions
- **3D extensions**: Volumetric retinal imaging
- **Real-time inference**: Mobile deployment optimization
- **Additional datasets**: Cross-dataset generalization studies

## Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **PIL (Pillow)**: Image processing
- **scikit-learn**: Evaluation metrics
- **NumPy**: Numerical operations
- **tqdm**: Progress bars


---
