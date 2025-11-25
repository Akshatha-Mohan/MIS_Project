# MIS Project

MIS project for computed tomography reconstruction.

## Project Overview

This project implements a complete pipeline for:
- Generating synthetic medical images (signal-present and signal-absent objects)
- Building a system matrix for forward projection
- Adding Poisson noise at multiple levels
- Reconstructing images from noisy projections using SVD-based pseudoinverse

## Project Structure

```
MIS-Project/
├── data/
│   ├── signal_present/     # Original signal-present images (500 images)
│   └── signal_absent/       # Original signal-absent images (500 images)
├── signal_present/          # Reconstructed signal-present images
│   ├── low/                 # Low noise level (50,000 target sum)
│   ├── medium/              # Medium noise level (25,000 target sum)
│   ├── high/                # High noise level (10,000 target sum)
│   └── very_high/           # Very high noise level (2,500 target sum)
├── signal_absent/           # Reconstructed signal-absent images
│   ├── low/
│   ├── medium/
│   ├── high/
│   └── very_high/
├── Image_generation.py       # Generate synthetic images
├── system_matrix.py          # Build system matrix H
├── validate_system_matrix.py # Validate system matrix with test objects
└── main.py                   # Main pipeline: projection, noise, reconstruction
```

## Files Description

- **Image_generation.py**: Generates clean signal-present or signal-absent objects using a lumpy background model with optional circular signal.
- **system_matrix.py**: Builds the forward projection system matrix H that maps image space to projection space.
- **validate_system_matrix.py**: Validates the system matrix by projecting simple test objects (uniform disk, point source) and verifying the geometry.
- **main.py**: Complete pipeline that:
  1. Builds system matrix and computes pseudoinverse via SVD
  2. Loads images from data directories
  3. Computes forward projections
  4. Adds Poisson noise at 4 different levels
  5. Reconstructs images using pseudoinverse
  6. Saves reconstructed images in organized folders

## Usage

### Generate Images
```bash
python Image_generation.py
```

### Validate System Matrix
```bash
python validate_system_matrix.py
```

### Run Complete Pipeline
```bash
python main.py
```

This will process all images in `data/signal_present/` and `data/signal_absent/` and save reconstructed images at different noise levels.

## System Matrix Parameters

- **Image size**: 32×32 pixels
- **Detector strips**: 32
- **Projection angles**: 16 (default)
- **Detector size (D)**: 3.0
- **Sampling**: S×S points per pixel (S=20 default)

## Noise Levels

The pipeline uses 4 noise levels based on target projection sum:
- **Low**: 50,000
- **Medium**: 25,000
- **High**: 10,000
- **Very High**: 2,500

## Requirements

- Python 3.x
- NumPy
- PyTorch
- PIL/Pillow
- Matplotlib

## Author

Akshatha Mohan

