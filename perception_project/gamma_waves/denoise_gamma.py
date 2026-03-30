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
    output_path = Path(input_path).with_name("Figure4_denoised.h5")
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

path1 = "C:/Users/Katie/Documents/Katie/Code/perception_project/perception_project/gamma_waves/Figure_4_Panels_D_R_m45_20211118_meas01_m45_d211118_s01KX_fast_300um-fps300-cG_umcnv.h5"
path2 = "C:/Users/Katie/Documents/Katie/Code/perception_project/perception_project/gamma_waves/Figure_5_Panels_B_T_m45_20210824_meas00_m45_d210824_s00_vis315_fast-fps294-cG_umcnv.h5"

denoise_full_movie(path1)