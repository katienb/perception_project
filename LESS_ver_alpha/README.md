# LESS Framework

A lightweight, extensible framework for efficient image/video denoising using Linear Expansion of Subspace Thresholding (LESS).

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
2. **1D Temporal Denoising**: Applies median filtering with adaptive window size selection
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

## Example

```bash
python main.py --data demoMovie.tif --verbose --debug
```

## Code

Core modules:

- `main.py`: Main execution script with argument parsing and data handling
- `less.py`: Core LESS denoising algorithms (SVD, 1D/2D denoising)
- `utils.py`: Utility functions for block matching, aggregation, and data processing

## Script

To run the main script:

```bash
python main.py --help
```
