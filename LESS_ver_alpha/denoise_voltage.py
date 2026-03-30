"""
Demo script: running LESS denoising on a full HDF5 voltage movie.

This script shows how I apply the LESS denoiser to real voltage-imaging data
stored in an HDF5 file. The key points are:

- The denoiser is applied directly to NumPy arrays via the `denoise` function
  (i.e. I do not use the original `main.py` CLI interface).
- The full recording is split into fixed-length temporal segments
  (default: 5000 frames) to keep memory usage reasonable and make the
  computation tractable for long recordings.
- The output is a new HDF5 file containing the denoised `mov` dataset.
  Metadata (e.g. specs) are not copied to the new file.

"""

from pathlib import Path
import h5py
import numpy as np
import torch
from less import denoise


def run_less_denoising(movie_np, **kwargs):
    """
    Run LESS denoising on a single movie segment.

    Parameters
    ----------
    movie_np : np.ndarray
        Movie segment with shape (T, H, W).
    **kwargs :
        Additional keyword arguments passed directly to `denoise`
        (e.g. patch_size, top_k, window_size, etc.).

    Returns
    -------
    np.ndarray
        Denoised movie segment with the same shape as the input.
    """
    # Replace NaNs with zeros to avoid numerical issues
    movie_np = np.nan_to_num(movie_np)

    # Convert to torch tensor (LESS expects torch tensors)
    movie_tensor = torch.from_numpy(movie_np).float()

    # Run the LESS denoiser
    # Note: cuda=False here; GPU can be enabled if desired
    denoised_tensor = denoise(data=movie_tensor, cuda=False, **kwargs)

    # Convert back to NumPy for writing to disk
    return denoised_tensor.cpu().numpy()


def denoise_full_movie(input_path, segment_length=5000, denoise_kwargs={}):
    """
    Apply LESS denoising to a full movie stored in an HDF5 file.

    The movie is processed in temporal segments (default: 5000 frames)
    rather than all at once. This keeps memory usage manageable and
    makes it feasible to denoise long recordings.

    Parameters
    ----------
    input_path : str or Path
        Path to the input HDF5 file containing a dataset named 'mov'
        with shape (T, H, W).
    segment_length : int, optional
        Number of frames per segment. Default is 5000.
    denoise_kwargs : dict, optional
        Dictionary of keyword arguments passed to the LESS denoiser.
    """
    input_path = Path(input_path)
    output_path = input_path.with_name("denoised.h5")

    # Remove existing output file if it already exists
    if output_path.exists():
        output_path.unlink()

    # Open input HDF5 file
    with h5py.File(input_path, 'r') as f_in:
        mov_dset = f_in['mov']
        T, H, W = mov_dset.shape

        # Create output HDF5 file and dataset
        with h5py.File(output_path, 'w') as f_out:
            dset_out = f_out.create_dataset(
                'mov',
                shape=(T, H, W),
                dtype='float32',
                compression='gzip'
            )

            # Process the movie in temporal segments
            for start in range(0, T, segment_length):
                end = min(start + segment_length, T)

                print(f"Denoising frames {start}:{end} / {T}")

                # Load a segment from disk
                segment = mov_dset[start:end]
                segment = np.nan_to_num(segment)

                # Denoise this segment
                segment_denoised = run_less_denoising(
                    segment,
                    patch_size=16,   # explicitly set here; others can be passed via denoise_kwargs
                    **denoise_kwargs
                )

                # Write denoised segment back to output file
                dset_out[start:end] = segment_denoised

    print(f"Denoised movie saved to:\n{output_path}")


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

path_movie = (
    "N:/GEVI_Wave/Analysis/Visual/cfm001mjr/20231208/meas00/cG_unmixed_dFF.h5"
)

denoise_full_movie(path_movie)