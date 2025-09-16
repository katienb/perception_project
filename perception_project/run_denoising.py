import numpy as np
import h5py
import os
import cv2
import sys
from pathlib import Path
import pandas as pd
import torch
import traceback
less_path = os.path.abspath(os.path.join("..", "LESS_ver_alpha"))
if less_path not in sys.path:
    sys.path.insert(0, less_path)
from less import denoise

def mask_movie(movie, raw_mask, binning):
    mask = cv2.resize(raw_mask, (0, 0), fx=1/binning, fy=1/binning, interpolation=cv2.INTER_LINEAR)
    movie_size = movie.shape[1:3]
    mask = mask[:movie_size[0], :movie_size[1]].astype(bool)
    return movie * mask

def run_less_denoising(movie_np, **kwargs):
    assert movie_np.ndim == 3
    movie_tensor = torch.from_numpy(movie_np).float()
    denoised_tensor = denoise(data=movie_tensor, cuda=True, **kwargs)
    return denoised_tensor.cpu().numpy()

def extract_unique_movies(mouse: str):
    """
    Parses the trial info CSV for a given mouse and extracts unique (date, measure) pairs.

    Parameters:
        mouse (str): Mouse ID (e.g. 'cfm002mjr')

    Returns:
        List of tuples: [(date1, meas1), (date2, meas2), ...]
    """
    trial_info_path = Path(f"C:/Users/Katie/Documents/Katie/Code/perception_project/perception_project/trial_info/TrialInfo_{mouse}.csv")
    df = pd.read_csv(trial_info_path)

    # Extract date and measure from TrialID
    def extract_date_meas(trial_id):
        parts = trial_id.split("/")
        date = int(parts[2])         # e.g. '20240510'
        meas = parts[3]              # e.g. 'meas00'
        return (date, meas)

    # Apply to all TrialIDs and get unique entries
    unique_movies = df['TrialID'].apply(extract_date_meas).drop_duplicates().tolist()
    return unique_movies

def denoise_full_movie(mouse, date, file, segment_length=2000, denoise_kwargs={}, movie_index=None, total_movies=None):
    input_path = f"N:/GEVI_Wave/Analysis/Visual/{mouse}/{date}/{file}/cG_unmixed_dFF.h5"
    output_path = Path(input_path).with_name("cG_unmixed_dFF_denoised.h5")
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(input_path, 'r') as f_in:
        mov_dset = f_in['mov']
        specs_grp_in = f_in['specs']
        fps = specs_grp_in['fps'][()].squeeze() 
        binning = specs_grp_in['binning'][()].squeeze() 
        raw_mask = specs_grp_in['extra_specs']['mask'][()].squeeze() 

        specs_attrs = dict(specs_grp_in.attrs.items())

        # Copy all datasets in specs (except the 'extra_specs' group)
        specs_datasets = {k: v[()] for k, v in specs_grp_in.items() if not isinstance(v, h5py.Group)}

        # Copy extra_specs datasets
        extra_specs = {k: specs_grp_in["extra_specs"][k][()] for k in specs_grp_in["extra_specs"]}

        T, H, W = mov_dset.shape

        with h5py.File(output_path, 'w') as f_out:
            dset_out = f_out.create_dataset('mov', shape=(T, H, W), dtype='float32', compression="gzip")
            specs_grp_out = f_out.create_group("specs")
            for k, v in specs_attrs.items():
                specs_grp_out.attrs[k] = v
            for k, v in specs_datasets.items():
                specs_grp_out.create_dataset(k, data=v)
            extra_grp_out = specs_grp_out.create_group("extra_specs")
            for k, v in extra_specs.items():
                extra_grp_out.create_dataset(k, data=v)
            
            for start in range(0, T, segment_length):
                end = min(start + segment_length, T)
                segment = mov_dset[start:end]
                segment = np.nan_to_num(segment)
                segment_masked = mask_movie(segment, raw_mask, binning)

                segment_denoised = run_less_denoising(segment_masked, **denoise_kwargs)
                segment_denoised = np.flip(segment_denoised, axis=1)
                dset_out[start:end] = segment_denoised

    print(f"Denoised movie saved to:\n{output_path}")

def main(mouse):
    unique_movies = extract_unique_movies(mouse)
    total = len(unique_movies)

    for i, (date, meas) in enumerate(unique_movies, start=1):
        print(f"Starting Movie {date}/{meas} ({i}/{total})", flush=True)
        try:
            denoise_full_movie(
                mouse=mouse,
                date=date,
                file=meas,
                segment_length=5000,
                denoise_kwargs={
                    "patch_size": 16,
                    "verbose": False},
                movie_index=i,
                total_movies=total)
        except Exception as e:
            print(f"Error processing {date}/{meas}: {e}", flush=True)
            #traceback.print_exc()  # Print full traceback for debugging
            continue  # Move to the next movie
    print("Complete!")

main('rfm003mjr')