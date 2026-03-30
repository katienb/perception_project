import numpy as np
import h5py
import cv2
from pathlib import Path
import pandas as pd
import torch
import traceback
import os, sys

# Get the directory where this script lives
here = os.path.dirname(os.path.abspath(__file__))
less_path = os.path.abspath(os.path.join(here, "../..", "LESS_ver_alpha"))

if less_path not in sys.path:
    sys.path.insert(0, less_path)

from less import denoise

def run_less_denoising(movie_np, **kwargs):
    movie_np = np.nan_to_num(movie_np)
    movie_tensor = torch.from_numpy(movie_np).float()
    denoised_tensor = denoise(data=movie_tensor, cuda=False, **kwargs)
    return denoised_tensor.cpu().numpy()

def denoise_full_movie(input_path, segment_length=5000, denoise_kwargs={}):
    output_path = Path(input_path).with_name("denoised.h5")
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(input_path, 'r') as f_in:
        mov_dset = f_in['mov']

        T, H, W = mov_dset.shape

        with h5py.File(output_path, 'w') as f_out:
            dset_out = f_out.create_dataset('mov', shape=(T, H, W), dtype='float32', compression="gzip")
            
            for start in range(0, T, segment_length):
                end = min(start + segment_length, T)
                segment = mov_dset[start:end]
                segment = np.nan_to_num(segment)

                segment_denoised = run_less_denoising(segment, patch_size=16, **denoise_kwargs)
                dset_out[start:end] = segment_denoised

    print(f"Denoised movie saved to:\n{output_path}")

path_movie1 = "N:/GEVI_Wave/Analysis/Visual/cfm001mjr/20231208/meas00/cG_unmixed_dFF.h5"
path_movie2 = "N:/GEVI_Wave/Analysis/Visual/cmm001mjr/20231208/meas00/cG_unmixed_dFF.h5"

denoise_full_movie(path_movie1)
denoise_full_movie(path_movie2)