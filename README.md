# Perception Project: Dynamic Spectra Analysis

This repository contains code for preprocessing behavioral and voltage data, and computing trial-aligned dynamic spectra.

---

## Key Notebooks and Scripts

- `trial_info.ipynb`  
  Generates and verifies trial metadata for each mouse. This includes behavior-based trial categorization, validation, mapping to recording files, delay correction, and alignment to imaging frames.  
  ➤ **Outputs:**  
  - Saves trial info CSV files to the `trial_info/` folder.

- `dynamic_spectra.py`  
  Loads the trial info and associated imaging movies for a given mouse. Extracts trials for each trial type, applies regional masks (e.g., V1), computes dynamic spectra using wavelet transforms, and aggregates results.  
  ➤ **Outputs:**  
  - Saves figures of the **mean dynamic spectrum** for each trial type to the `dynamic_spectra/` folder.  
  - Saves full results (including individual trial spectra, trial metadata, and mean spectra) to an HDF5 file folder (not uploaded to GitHub).

---
