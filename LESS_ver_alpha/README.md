# LESS Denoising Algorithm (Edited Version by Katie Brown)

> This repository contains **my edited version** of the LESS denoising code, originally written by **Jizhou Li**.  
> The core algorithm and implementation are his; I have made a small number of targeted changes and additions to adapt the code for my own use and to make it easier for others in the group to run and validate.  
>
> In particular:
> - I don't use the temporal (1D) denoising step.
> - I don't use the original `main.py` script for my workflow.
> - I have add a short demo notebook (**`LESS_demo.ipynb`**) with mock data, and a demo script (**`denoise_voltage.py`**) showing how I'd run the denoiser on a real voltage movie.

---

## Overview

LESS (Linear Expansion of Subspace Thresholding) is a lightweight, extensible framework for denoising image sequences and videos.  
It is based on the idea that real signals have structured spatial and temporal patterns, while noise is comparatively unstructured.

In practice, LESS:
- decomposes a movie into spatial and temporal components using SVD,
- denoises the **spatial components** using patch-based block matching and collaborative filtering,
- reconstructs the movie from the cleaned components, and
- uses a self-supervised loss with early stopping to avoid fitting noise.

This repository reflects how I currently use LESS in my own analyses.

---

## Dependencies

- Python 3.8+
- NumPy
- PyTorch
- tifffile
- psutil
- pynvml
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Processing Logic

LESS performs denoising on TIFF image sequences through SVD-based low-rank estimation combined with structural smoothness regularization. The framework processes temporal image data by decomposing it into spatial and temporal components, then applies specialized denoising algorithms:

1. **SVD Decomposition**: Decomposes input data into spatial (U) and temporal (Vh) components
2. **1D Temporal Denoising**: Applies median filtering with adaptive window size selection (I removed this step)
3. **2D Spatial Denoising**: Uses block-matching and collaborative filtering
4. **Adaptive Early Stopping**: Monitors reconstruction loss to prevent overfitting

## Parameters

Key parameters include:

- `--data`: Path to input TIFF file (required)
- `--verbose`, `-v`: Enable verbose output
- `--debug`: Enable debug mode with detailed information
- `--cpu`: Force CPU-only processing (default: uses CUDA if available)
- `--patch_size`: Patch size for spatial denoising (auto-estimated if not specified)
- `--top_k`: Number of similar patches for block matching (default: 18)
- `--window_size`: Search window size for block matching (default: 37)
- `--stride`: Stride for patch extraction (default: 4)
- `--pat`: Patience for early stopping (default: 5)
- `--save_dir`: Directory to save denoised results
- `--seed`: Random seed for reproducibility (default: 42)

## Example Usage

In my workflow, I call the denoiser directly from Python rather than using main.py.

Minimal example:

movie_tensor = torch.from_numpy(movie_np).float() 
denoised_tensor = denoise(data=movie_tensor).numpy()

## Code

- `LESS_demo.ipynb`: notebook with examples of running and testing the denoiser on mock data
- `denoise_voltage.py`: demo script showing how I'd run the code for a real voltage movie in HDF5 format

- `less.py`: Core LESS denoising algorithms (SVD, 1D/2D denoising)
- `utils.py`: Utility functions for block matching, aggregation, and data processing
- `main.py`: Main execution script with argument parsing and data handling (included for completeness,; I don't use this)