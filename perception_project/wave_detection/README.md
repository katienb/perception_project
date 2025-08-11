# Band Power Maps Analysis

Analysis pipeline for detecting travelling waves in cortical GEVI movies by computing band-limited power maps.

## What it does

1. **Loads GEVI movie data** from HDF5 files (fps from original, data from denoised)
2. **Filters into frequency bands**: Delta (1-4 Hz), Theta (4-8 Hz), Beta (12-30 Hz), Low-Gamma (30-60 Hz)
3. **Computes instantaneous power** using Hilbert transform on filtered signals
4. **Creates side-by-side animations** showing filtered movie (left) and power map (right)
5. **Includes stimulus timing** with red/green indicator squares

## Core functions

- `load_movie()` - Load and combine original + denoised data
- `bandpass_filter()` - Butterworth filter for specific frequency bands
- `compute_power()` - Hilbert transform → instantaneous power
- `analyze_band_power()` - Main pipeline: filter → power → animate
- `create_side_by_side_animation()` - Generate visualization with stimulus timing

## Files

- **`band_power_maps.py`** - All processing functions
- **`band_power_analysis.ipynb`** - Simple notebook interface (5 cells: imports, params, load data, single band, all bands)

## Output

Interactive animations showing spatiotemporal power patterns across frequency bands, useful for identifying travelling wave dynamics in response to stimuli.