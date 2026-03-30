import numpy as np
import scipy as sp
import pandas as pd
import h5py
import cv2
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import os, sys

# Get the directory where this script lives
here = os.path.dirname(os.path.abspath(__file__))
less_path = os.path.abspath(os.path.join(here, "..", "LESS_ver_alpha"))

if less_path not in sys.path:
    sys.path.insert(0, less_path)

from less import denoise
from utils import set_seed

def load_movie(mouse, date, file):
    if mouse=='cfm001mjr' or mouse=='cfm002mjr':
        path = "N:/GEVI_Wave/Analysis/Visual/" + mouse + "/20" + str(date) + "/" + file + '/cG_unmixed_dFF_denoised_2.h5'
    else:
        path = "N:/GEVI_Wave/Analysis/Visual/" + mouse + "/20" + str(date) + "/" + file + '/cG_unmixed_dFF_denoised.h5'

    with h5py.File(path, 'r') as mov_file:
        specs = mov_file["specs"]
        fps = specs['fps'][()].squeeze()
        raw_mask = specs["extra_specs"]["mask"][()].squeeze()
        binning = specs["binning"][()].squeeze()

        movie = mov_file['mov']
        
    return movie, fps, raw_mask, binning

def mask_movie(movie, raw_mask, binning, flip=False):
    mask = cv2.resize(raw_mask, (0, 0), fx=1/binning, fy=1/binning,
                      interpolation=cv2.INTER_LINEAR)
    movie_size = movie.shape[1:3]
    mask = mask[:movie_size[0], :movie_size[1]].astype(bool)
    if flip:
        mask = np.flipud(mask)
    
    # Broadcast mask to movie shape
    mask = mask[None, :, :]  # if movie is (t, y, x)
    
    out = movie.astype(float).copy()
    out[:, ~mask[0]] = np.nan
    return out

def find_trials(df, **filters):
    """
    Finds trials from df that match the specified properties.

    Args:
    - df (pd.DataFrame): DataFrame containing trial information.
    - filters (dict): Key-value pairs of column names and values to filter by.
      - Supported keys include 'Date', 'Recording', 'PerceptualCat', 'Rewarded', etc.
      - If a parameter is not specified, it is ignored in filtering.

    Returns:
    - pd.DataFrame: Filtered dataframe containing trials that meet the specified criteria
        - Includes columns: ['AnimalCode', 'TrialID', 'Date', 'BFMTime', 'Duration', 'File', 'ValidTrial?']
    """

    # Start with full DataFrame and filter only valid trials
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['ValidTrial?'] == True] 

    # Apply dynamic filtering based on provided arguments
    for key, value in filters.items():
        if key in df.columns and value is not None:
            if isinstance(value, list):  # Check if the filter is a list
                filtered_df = filtered_df[filtered_df[key].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[key] == value]

    return filtered_df[["AnimalCode", "TrialID", "Date", "BFMTime", "Duration", "File", 'ValidTrial?']]

def play_movie(movie, fps, slowdown=1, title=None):
    """
    Display a (t, y, x) movie in a Jupyter notebook.

    Parameters
    ----------
    movie : np.ndarray
        3D array of shape (t, y, x)
    fps : float
        Real frame rate of the recording.
    slowdown : float, optional
        Factor by which to slow down playback (default = 1, no slowdown).
    title : str, optional
        Title displayed above the movie.
    cmap : str, optional
        Matplotlib colormap for visualization (default = 'seismic').
    """
    assert movie.ndim == 3, "movie must have shape (t, y, x)"
    t, y, x = movie.shape

    fig, ax = plt.subplots()
    im = ax.imshow(movie[0], cmap='seismic', animated=True)
    if title:
        ax.set_title(title)
    ax.axis('off')

    vmin, vmax = movie.min(), movie.max()
    im.set_clim(vmin, vmax)

    def update(frame):
        im.set_array(movie[frame])
        return [im]

    interval = 1000 / (fps / slowdown)
    anim = FuncAnimation(fig, update, frames=t, interval=interval, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())

def run_less_denoising(movie_np: np.ndarray,
                           patch_size: int = 16,
                           top_k: int = 18,
                           window_size: int = 37,
                           stride: int = 4,
                           pat: int = 5,
                           seed: int = 42,
                           verbose: bool = False) -> np.ndarray:
    """
    Run LESS denoising on a NumPy movie array.

    Parameters:
        movie_np (np.ndarray): Input movie of shape (T, H, W)
        patch_size (int): Patch size
        top_k (int): Number of best-matching patches
        window_size (int): Search window size
        stride (int): Stride for patch selection
        pat (int): Temporal patch size (number of adjacent frames)
        seed (int): Random seed
        verbose (bool): Whether to print progress

    Returns:
        np.ndarray: Denoised movie of shape (T, H, W)
    """
    assert movie_np.ndim == 3, "Input movie must have shape (T, H, W)"
    set_seed(seed)

    movie_np = np.nan_to_num(movie_np)
    movie_tensor = torch.from_numpy(movie_np).float()

    denoised_tensor = denoise(
        data=movie_tensor,
        cuda=False,               # Force CPU
        patch_size=patch_size,
        top_k=top_k,
        window_size=window_size,
        stride=stride,
        pat=pat,
        verbose=verbose
    )

    return denoised_tensor.numpy()