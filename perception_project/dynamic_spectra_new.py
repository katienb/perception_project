### Computing mean dynamic spectra for all the trials in the 6 key categories for a given mouse

import pandas as pd
import numpy as np
import h5py
import pywt
import cv2
from skimage.draw import polygon2mask

def movie_path(mouse, date, file):
    """Returns the file path for a given movie recording."""
    return "N:/GEVI_Wave/Analysis/Visual/" + mouse + "/20" + str(date) + "/" + file + '/cG_unmixed_dFF.h5'

def mask_movie(movie, raw_mask, binning):
    """
    Applies a spatial mask to a movie to exclude non-brain regions.

    Args:
    - movie (numpy.ndarray): 3D array representing the movie (t, x, y).
    - raw_mask (numpy.ndarray): 2D mask array.
    - binning (float): Scaling factor for resizing the mask.

    Returns:
    - numpy.ndarray: The masked movie with non-brain regions removed.
    """
    # Resize the mask using bilinear interpolation
    mask = cv2.resize(raw_mask, (0, 0), fx=1/binning, fy=1/binning, interpolation=cv2.INTER_LINEAR)
    
    # Ensure mask has the same shape as the movie
    movie_size = movie.shape[1:3]  # Get spatial dimensions of the movie
    mask = mask[:movie_size[0], :movie_size[1]]  # Crop mask to match movie size
    mask = np.flip(mask, axis=0)
    mask = mask.astype(bool)         # Convert to boolean mask
    return movie * mask  # Apply mask

def mask_region(movie, raw_outlines, binning, spaceorigin, region, plot=False):
    """
    Applies a mask to a movie based on anatomical brain region outlines.

    Args:
    - movie (numpy.ndarray): 3D movie array (t, x, y).
    - raw_outlines (numpy.ndarray): 3D array of brain region outlines.
    - binning (float): Scaling factor for spatial binning.
    - spaceorigin (numpy.ndarray): 2D array of space origin coordinates.
    - region (str): Brain region to mask ('V1' or 'SSC').
    - plot (bool, optional): Whether to display the mask overlay on the first frame of the movie.

    Returns:
    - numpy.ndarray: The masked movie where only the selected region remains.
    """

    # Define the corresponding outline indices for the selected brain region
    if region == 'V1':
        indices=[37]
    elif region == 'SSC':
        indices=[3,11,13,15]
    else:
        print('Region must be V1 or SSC')

    spaceorigin = (spaceorigin - 1) / binning + 1  # Apply space origin transformation

    # Extract all outlines and scale them according to binning
    outlines_nums = np.arange(raw_outlines.shape[2]) 
    outlines = raw_outlines[:, :, outlines_nums] / binning

    outlines[:, 0, :] -= spaceorigin[1] - 1  # Adjust Y-coordinates
    outlines[:, 1, :] -= spaceorigin[0] - 1  # Adjust X-coordinates

    # Define the movie dimensions
    movie_shape = movie.shape[1:3]  # (height, width)

    total_mask = np.zeros(movie_shape, dtype=bool)

    for i in indices:
        # Extract the ROI outline
        outline = outlines[i, :, :]  # Shape (2, N)

        valid_indices = ~np.isnan(outline).any(axis=0)  # Find non-NaN indices
        x_coords = outline[1, valid_indices]
        y_coords = outline[0, valid_indices]

        # Create a mask using polygon2mask
        roi_mask = polygon2mask(movie_shape, np.column_stack((y_coords, x_coords)))
        roi_mask = np.flipud(roi_mask).astype(bool)
        total_mask |= roi_mask  # Any pixel belonging to at least one ROI remains unmasked

    # Apply mask to the movie
    movie_roi = movie * total_mask  # Broadcasting applies the mask to all frames

    return movie_roi

def load_and_mask(path_movie):
    """
    Loads a movie file, applies masking, and optionally extracts a specific brain region.

    Args:
    - mouse (str): Mouse identifier.
    - date (str or int): Experiment date in YYMMDD format.
    - file (str): Recording file identifier.
    - region (str, optional): Brain region to isolate ('V1' or 'SSC'). If None, the full movie is returned.

    Returns:
    - movie (numpy.ndarray): 3D array (t, x, y) representing the processed movie.
    - fps (float): Frames per second of the movie.
    """

    with h5py.File(path_movie, 'r') as mov_file:
        specs = mov_file["specs"]
        mov = mov_file['mov'][()]  
        fps = specs["fps"][()].squeeze()  
        raw_mask = specs["extra_specs"]["mask"][()].squeeze() 
        binning = specs["binning"][()].squeeze() 
        raw_outlines = specs["extra_specs"]["allenMapEdgeOutline"][()].squeeze() 
        spaceorigin = specs["spaceorigin"][()].squeeze() 

    # Replace NaN values in the movie
    movie = np.nan_to_num(mov)
    movie = np.flip(movie, axis=1)

    # Apply the brain mask to remove non-brain regions
    movie = mask_movie(movie, raw_mask, binning)

    # If a specific brain region is provided, apply the corresponding region mask
    movie = mask_region(movie, raw_outlines, binning, spaceorigin, 'V1', plot=True)

    return movie, fps

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

def spectrogram(segment, fps, f_range=(2, 60), num_freqs=100):
    """
    Compute a wavelet spectrogram of a signal segment using MATLAB-matching parameters.
    
    Parameters:
        segment: 1D numpy array (signal trace for a single trial)
        fps: frames per second (sampling rate)
        f_range: frequency range (Hz)
        num_freqs: number of frequency bins (log spaced)
        time_bandwidth: MATLAB-like 'TimeBandwidth' parameter
        voices_per_octave: MATLAB-like 'VoicesPerOctave' parameter

    Returns:
        power: normalized wavelet magnitude (freq x time)
        freqs: corresponding frequencies in Hz
    """
    dt = 1 / fps
    freqs = np.logspace(np.log10(f_range[0]), np.log10(f_range[1]), num_freqs)
    
    # Morelet wavelet
    wavelet = 'cmor2.0-1.0'
    
    # Match MATLAB scale construction
    scales = pywt.scale2frequency(wavelet, 1.0) / freqs * fps

    # Compute wavelet transform
    coeffs, actual_freqs = pywt.cwt(segment, scales, wavelet, sampling_period=dt)
    
    # Normalize as MATLAB: abs(wt) / sqrt(freq)
    amplitude = np.abs(coeffs) / np.sqrt(actual_freqs[:, np.newaxis])
    
    return amplitude, actual_freqs

def mean_dynamic_spectrum(trial_info_path, title, name, pre_window=3.0, post_window=3.0, **filters):
    """
    Computes mean dynamic spectrum and returns the data for saving.
    Args:
    - trial_info_path (str): Path to the trial info CSV file.
    - title (str): figure title
    - name (str): mouse name
    - filters (dict): Filtering conditions for trials.
    Returns:
    - dict: Dictionary containing the processed results for a single trial type.
    """
    df_all = pd.read_csv(trial_info_path)
    df = find_trials(df_all, **filters)
    unique_recordings = df[['AnimalCode', 'File', 'Date']].drop_duplicates() # Identify unique recordings
    print(f"Found {len(df)} trials in {len(unique_recordings)} recordings")
    
    all_power = []
    all_mean_signals = []
    omitted_recordings = []
    omitted_trials = []
    trial_ids = []

    i=1 # Counter for processing recordings
    # Load and process movies for each unique recording
    for _, row in unique_recordings.iterrows():
        mouse = row['AnimalCode']
        date = row['Date']
        file = row['File']

        try:
            print(f"Loading and Processing Recording {date}/{file} ({i}/{len(unique_recordings)})")
            path = movie_path(mouse, date, file)
            movie, fps = load_and_mask(path)
        except Exception as e:
            print(f"Skipping recording {date}/{file} due to error: {e}")
            omitted_recordings.append(f"{date}/{file}")
            continue  # Skip this recording and move to the next
        i+=1
        # Extract trials for this movie
        df_trials = df[(df['AnimalCode'] == mouse) & (df['Date'] == date) & (df['File'] == file)]

        for _, trial in df_trials.iterrows():
            try:
                bfm_time = trial['BFMTime']  # Trial start time in BFM coordinates

                if (bfm_time - pre_window)*fps > 0 and (bfm_time + post_window)*fps < movie.shape[0]:

                    # Compute frame range for this trial
                    start_frame = int((bfm_time - pre_window) * fps)
                    end_frame = int((bfm_time + post_window) * fps)

                    # Extract trial snippet
                    trial_clip = movie[start_frame:end_frame]
                    mean_signal = trial_clip.mean(axis=(1, 2))

                    # Compute dynamic spectrum and add to list
                    power_norm, frequencies = spectrogram(mean_signal, fps)
                    
                    all_power.append(power_norm)
                    all_mean_signals.append(mean_signal)
                    trial_ids.append(trial['TrialID'])

                else:
                    print(f"\nSkipping trial {trial['TrialID']} in {date}/{file}: index out of bounds")
                    omitted_trials.append(trial['TrialID'])
            
            except Exception as e:
                print(f"\nSkipping trial {trial['TrialID']} in {date}/{file} due to error: {e}")
                omitted_trials.append(trial['TrialID'])
                continue  # Skip this trial and move to the next

    min_t_length = min(arr.shape[1] for arr in all_power)  # Find shortest second dimension
    if max(arr.shape[1] for arr in all_power) - min_t_length > 3: print(f"Warning: min_t_length is unusually short.")
    all_power = np.stack([arr[:, :min_t_length] for arr in all_power])
    mean_spectrogram = np.mean(all_power, axis=0)
    all_mean_signals = np.stack([arr[:min_t_length] for arr in all_mean_signals])

    # Quantile normalization like MATLAB
    q01 = np.quantile(mean_spectrogram, 0.01, axis=1, keepdims=True)
    q10 = np.quantile(mean_spectrogram, 0.10, axis=1, keepdims=True)
    mean_spectrogram = (mean_spectrogram - q01) / (q10 + 1e-10) * 100

    time_arr = np.linspace(-1*pre_window, post_window, mean_spectrogram.shape[1])  # Stimulus at t=0]

    return {
        "DynamicSpectra": all_power,
        "MeanTimeSignals": all_mean_signals,
        "MeanDynamicSpectrum": mean_spectrogram,
        "TrialIDs": trial_ids,
        "OmittedRecordings": omitted_recordings,
        "OmittedTrials": omitted_trials,
        "Frequencies": frequencies,
        "Time": time_arr
    }

def save_to_h5(mouse_name, trial_type, data):
    """
    Saves the processed trial data to an HDF5 file.
    Args:
    - mouse_name (str): Mouse identifier.
    - trial_type (str): Type of trial (e.g., 'HC_Hit').
    - data (dict): Processed trial data.
    """
    filename = f"results/{mouse_name}_denoised_dynamic_spec_new.h5"

    with h5py.File(filename, "a") as h5f:
        group = h5f.require_group(trial_type)

        group.create_dataset("DynamicSpectra", data=data["DynamicSpectra"])
        group.create_dataset("TimeSignals", data=data["MeanTimeSignals"])
        group.create_dataset("MeanDynamicSpectrum", data=data["MeanDynamicSpectrum"])

        group.attrs["TrialIDs"] = np.array(data["TrialIDs"], dtype="S")
        group.attrs["OmittedRecordings"] = np.array(data["OmittedRecordings"], dtype="S")
        group.attrs["OmittedTrials"] = np.array(data["OmittedTrials"], dtype="S")
        group.attrs["Frequencies"] = data["Frequencies"]
        group.attrs["Time_Array"] = data["Time"]

    print(f"Saved {trial_type} data to {filename}")

def main(mouse_name, trial_info_file):
    # Define trial types and corresponding labels
    trial_types = ['HC Hit (3)', ['HC No Report (9)', 'Incorrect Reject (6)'], 
                   'MC Hit (2)', 'MC Miss (5)', 'MC No Report (8)',
                   'False Alarm (1)', 'Correct Rejection (4)', 'LC No Report (7)'
                   ]
    trial_type_names = ['HC Hit', 'HC Miss', 
                        'MC Hit', 'MC Miss', 'MC No Report',
                        'False Alarm', 'Correct Rejection', 'LC No Report']
    
    # Filenames for saving plots and results
    file_names = [
        f"{mouse_name}_hc_hit", f"{mouse_name}_hc_miss",
        f"{mouse_name}_mc_hit", f"{mouse_name}_mc_miss", f"{mouse_name}_mc_no_report",
        f"{mouse_name}_false_alarm", f"{mouse_name}_correct_reject", f"{mouse_name}_lc_no_report",
        ]

   # Open the HDF5 file (create if it doesn't exist)
    h5_path = f"results/{mouse_name}_dynamic_spec_new.h5"
    with h5py.File(h5_path, 'a') as h5_file:
        # Iterate through trial types and process each
        for i in range(len(trial_types)):
            if trial_type_names[i] in h5_file:    # Check if the trial type group already exists
                print(f"Skipping {trial_type_names[i]} trials - already exists in {h5_path}")
                continue  # Skip processing for this trial type

            print(f'Processing {trial_types[i]} trials')
            title = f"Mean Dynamic Spectrum: {mouse_name} {trial_type_names[i]}"
            data = mean_dynamic_spectrum(trial_info_file, title=title, name=file_names[i], TrialType=trial_types[i])
            save_to_h5(mouse_name, trial_type_names[i], data)

main('cfm002', 'trial_info/TrialInfo_cfm002mjr.csv')