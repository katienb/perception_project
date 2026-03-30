#!/usr/bin/env python3
"""
Band Power Maps Analysis

This script provides functions for analyzing cortical GEVI movies to reveal candidate travelling waves
by computing band-limited power maps. It can be imported and used in Jupyter notebooks.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import circmean
from scipy.ndimage import gaussian_filter
from pathlib import Path
import os

def movie_path(mouse, date, file):
    """Generate movie file path from components."""
    return f"N:/GEVI_Wave/Analysis/Visual/{mouse}/20{date}/{file}/cG_unmixed_dFF.h5"

def denoised_movie_path(mouse, date, file):
    """Generate denoised movie file path from components."""
    return f"N:/GEVI_Wave/Analysis/Visual/{mouse}/20{date}/{file}/cG_unmixed_dFF_denoised.h5"

def load_movie(movie_path, denoised_path):
    """
    Load movie data from HDF5 files.
    
    Args:
        movie_path (str): Path to the original movie file (for fps)
        denoised_path (str): Path to the denoised movie file (for data)
        
    Returns:
        tuple: (movie, fps) where movie is numpy array (T, Y, X) and fps is float
    """
    # Load fps from original file
    with h5py.File(movie_path, 'r') as mov_file:
        specs = mov_file["specs"]
        fps = specs['fps'][()][0][0][0]
    
    # Load movie data from denoised file
    with h5py.File(denoised_path, 'r') as mov_file:
        mov = mov_file['mov'][()]
        movie = np.nan_to_num(mov)
    
    return movie, fps

def crop_movie(movie, fps, t_start, t_stop):
    """
    Crop movie to specified time window.
    
    Args:
        movie (np.ndarray): Movie array (T, Y, X)
        fps (float): Frame rate in Hz
        t_start (float): Start time in seconds
        t_stop (float): Stop time in seconds
        
    Returns:
        np.ndarray: Cropped movie array
    """
    frame_start = int(t_start * fps)
    frame_stop = int(t_stop * fps)
    
    # Ensure bounds are valid
    frame_start = max(0, frame_start)
    frame_stop = min(movie.shape[0], frame_stop)
    
    return movie[frame_start:frame_stop].astype(np.float32)

def bandpass_filter(movie, fps, low_freq, high_freq):
    """
    Apply band-pass filter to movie along time axis.
    
    Args:
        movie (np.ndarray): Movie array (T, Y, X)
        fps (float): Frame rate in Hz
        low_freq (float): Lower frequency bound in Hz
        high_freq (float): Upper frequency bound in Hz
        
    Returns:
        np.ndarray: Band-pass filtered movie
    """
    # Design Butterworth band-pass filter
    nyquist = fps / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Create filter
    b, a = signal.butter(4, [low_norm, high_norm], btype='band')
    
    # Apply filter along time axis (axis 0)
    filtered_movie = np.zeros_like(movie)
    for i in range(movie.shape[1]):
        for j in range(movie.shape[2]):
            filtered_movie[:, i, j] = signal.filtfilt(b, a, movie[:, i, j])
    
    return filtered_movie

def compute_power(filtered_movie):
    """
    Compute instantaneous power from filtered movie.
    
    Args:
        filtered_movie (np.ndarray): Band-pass filtered movie (T, Y, X)
        
    Returns:
        np.ndarray: Power movie (T, Y, X)
    """
    # Compute analytic signal using Hilbert transform
    analytic_signal = hilbert(filtered_movie, axis=0)
    
    # Compute instantaneous power (absolute value squared)
    power = np.abs(analytic_signal) ** 2
    
    return power

def is_stimulus_on(time_sec, stimulus_onset, stimulus_duration):
    """Check if stimulus is on at given time."""
    return stimulus_onset <= time_sec <= (stimulus_onset + stimulus_duration)

def create_side_by_side_animation(filtered_movie, power_movie, fps, band_name, low_freq, high_freq, 
                                stimulus_onset, stimulus_duration, slowdown_factor, t_start, save_path=None):
    """
    Create animated visualization with raw filtered movie and power movie side by side.
    
    Args:
        filtered_movie (np.ndarray): Band-pass filtered movie (T, Y, X)
        power_movie (np.ndarray): Power movie (T, Y, X)
        fps (float): Frame rate in Hz
        band_name (str): Name of the frequency band
        low_freq (float): Lower frequency bound
        high_freq (float): Upper frequency bound
        stimulus_onset (float): Time when stimulus starts (seconds)
        stimulus_duration (float): Duration of stimulus (seconds)
        slowdown_factor (float): Factor to slow down animation
        t_start (float): Start time of the movie segment (for absolute time calculation)
        save_path (str, optional): Path to save animation
        
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    # Compute global min/max for consistent scaling
    power_vmin = np.percentile(power_movie, 1)
    power_vmax = np.percentile(power_movie, 99)
    
    filtered_vmin = np.percentile(filtered_movie, 1)
    filtered_vmax = np.percentile(filtered_movie, 99)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Initialize images
    im1 = ax1.imshow(filtered_movie[0], cmap='seismic', vmin=filtered_vmin, vmax=filtered_vmax, animated=True)
    im2 = ax2.imshow(power_movie[0], cmap='inferno', vmin=power_vmin, vmax=power_vmax, animated=True)
    
    ax1.set_title(f'{band_name} Raw ({low_freq}-{high_freq} Hz)\nSlowed Down {slowdown_factor}x - Time = {t_start:.3f} s')
    ax2.set_title(f'{band_name} Power ({low_freq}-{high_freq} Hz)\nSlowed Down {slowdown_factor}x - Time = {t_start:.3f} s')
    ax1.axis('off')
    ax2.axis('off')
    
    # Add stimulus indicator rectangles (initialized as red)
    square_size = 20
    square_x = filtered_movie.shape[2] - square_size - 10
    square_y = 10
    
    stim_rect1 = Rectangle((square_x, square_y), square_size, square_size, facecolor='red', edgecolor='white', linewidth=2)
    stim_rect2 = Rectangle((square_x, square_y), square_size, square_size, facecolor='red', edgecolor='white', linewidth=2)
    ax1.add_patch(stim_rect1)
    ax2.add_patch(stim_rect2)
    
    def animate(frame, 
                band_name=band_name, low_freq=low_freq, high_freq=high_freq, 
                slowdown_factor=slowdown_factor, t_start=t_start, fps=fps,
                stimulus_onset=stimulus_onset, stimulus_duration=stimulus_duration):
        # Update images
        im1.set_array(filtered_movie[frame])
        im2.set_array(power_movie[frame])
        
        # Calculate actual data time (relative to movie start)
        time_sec = t_start + (frame / fps)
        
        # Update titles
        ax1.set_title(f'{band_name} Raw ({low_freq}-{high_freq} Hz)\nSlowed Down {slowdown_factor}x - Time = {time_sec:.3f} s')
        ax2.set_title(f'{band_name} Power ({low_freq}-{high_freq} Hz)\nSlowed Down {slowdown_factor}x - Time = {time_sec:.3f} s')
        
        # Update stimulus indicator
        if is_stimulus_on(time_sec, stimulus_onset, stimulus_duration):
            stim_rect1.set_facecolor('green')
            stim_rect2.set_facecolor('green')
        else:
            stim_rect1.set_facecolor('red')
            stim_rect2.set_facecolor('red')
        
        return [im1, im2, stim_rect1, stim_rect2]
    
    # Calculate interval with slowdown factor
    interval = (1000 / fps) * slowdown_factor
    
    anim = animation.FuncAnimation(fig, animate, frames=filtered_movie.shape[0], 
                                 interval=interval, blit=True, repeat=True)
    plt.tight_layout()
    
    if save_path:
        save_fps = fps / slowdown_factor
        print(f"Saving {band_name}: original fps={fps}, slowdown_factor={slowdown_factor}, save_fps={save_fps}")
        
        # Try different writer and explicit duration
        duration_ms = (1000 / save_fps)  # milliseconds per frame
        print(f"Frame duration: {duration_ms:.2f} ms per frame")
        
        # Save GIF with pillow
        anim.save(save_path, writer='pillow', fps=save_fps)
        
        # Also save as MP4 for comparison (better frame rate support)
        try:
            mp4_path = save_path.replace('.gif', '.mp4')
            anim.save(mp4_path, writer='ffmpeg', fps=save_fps, bitrate=1800)
            print(f"Also saved as MP4: {mp4_path}")
        except Exception as e:
            print(f"Could not save MP4: {e}")
            
        # Try imagemagick for GIF (sometimes better than pillow)
        try:
            imagemagick_path = save_path.replace('.gif', '_imagemagick.gif')
            anim.save(imagemagick_path, writer='imagemagick', fps=save_fps)
            print(f"Also saved with imagemagick: {imagemagick_path}")
        except Exception as e:
            print(f"ImageMagick not available: {e}")
    
    return anim

def analyze_band_power(movie_segment, fps, band_name, low_freq, high_freq, 
                      stimulus_onset, stimulus_duration, slowdown_factor, t_start, 
                      save_path=None, display=True):
    """
    Analyze a movie segment for a specific frequency band.
    
    Args:
        movie_segment (np.ndarray): Cropped movie segment (T, Y, X)
        fps (float): Frame rate in Hz
        band_name (str): Name of the frequency band
        low_freq (float): Lower frequency bound in Hz
        high_freq (float): Upper frequency bound in Hz
        stimulus_onset (float): Time when stimulus starts (seconds)
        stimulus_duration (float): Duration of stimulus (seconds)
        slowdown_factor (float): Factor to slow down animation
        t_start (float): Start time of the movie segment (for absolute time calculation)
        save_path (str, optional): Path to save animation
        display (bool): Whether to display the animation
        
    Returns:
        tuple: (filtered_movie, power_movie, anim) - the processed data and animation object
    """
    
    # Apply band-pass filter
    filtered_movie = bandpass_filter(movie_segment, fps, low_freq, high_freq)
    
    # Compute power
    power_movie = compute_power(filtered_movie)
    
    # Create visualization
    anim = None
    if display:
        anim = create_side_by_side_animation(filtered_movie, power_movie, fps, band_name, low_freq, high_freq,
                                           stimulus_onset, stimulus_duration, slowdown_factor, t_start, save_path)
    
    return filtered_movie, power_movie, anim


def build_analytic_cube(movie, fps):
    """
    Construct the full analytic-signal cube for a movie using Hilbert transform.
    
    Args:
        movie (np.ndarray): Band-pass filtered movie (T, Y, X)
        fps (float): Frame rate in Hz
        
    Returns:
        dict: Dictionary containing:
            - 'Z': complex analytic signal cube (T, Y, X)
            - 'amplitude': amplitude |Z| (T, Y, X)
            - 'phase': phase angle(Z) (T, Y, X) 
            - 'inst_freq': instantaneous frequency (T, Y, X)
    """
    # Apply Hilbert transform along time axis only
    Z = hilbert(movie, axis=0)
    
    # Extract amplitude and phase
    amplitude = np.abs(Z)
    phase = np.angle(Z)
    
    # Compute instantaneous frequency (frame-to-frame phase derivative)
    dt = 1.0 / fps
    phase_diff = np.diff(phase, axis=0)
    
    # Handle phase wrapping for frequency calculation
    phase_diff = np.angle(np.exp(1j * phase_diff))  # Wrap to [-π, π]
    
    # Instantaneous frequency = phase derivative / (2π * dt)
    inst_freq = phase_diff / (2 * np.pi * dt)
    
    # Pad to match original time dimension (repeat last frame)
    inst_freq = np.concatenate([inst_freq, inst_freq[-1:]], axis=0)
    
    return {
        'Z': Z,
        'amplitude': amplitude,
        'phase': phase,
        'inst_freq': inst_freq
    }


def phase_gradient_cube(phase, fps, blur_sigma=1.0):
    """
    Compute phase-gradient vector field for all frames.
    
    Args:
        phase (np.ndarray): Phase array (T, Y, X)
        fps (float): Frame rate in Hz
        blur_sigma (float): Gaussian blur sigma for noise reduction (default: 1.0)
        
    Returns:
        dict: Dictionary containing:
            - 'grad_x': x-gradient of phase (T, Y, X)
            - 'grad_y': y-gradient of phase (T, Y, X)
            - 'speed': wave speed in pixels/second (T, Y, X)
            - 'angle': wave direction in radians (T, Y, X)
            - 'mean_speed': mean speed per frame (T,)
            - 'mean_dir': circular mean direction per frame (T,)
    """
    T, Y, X = phase.shape
    dt = 1.0 / fps
    
    # Initialize output arrays
    grad_x = np.zeros_like(phase)
    grad_y = np.zeros_like(phase)
    speed = np.zeros_like(phase)
    angle = np.zeros_like(phase)
    mean_speed = np.zeros(T)
    mean_dir = np.zeros(T)
    
    # Process each frame
    for t in range(T):
        # Optional: Apply Gaussian blur to reduce noise
        if blur_sigma > 0:
            phase_smooth = gaussian_filter(phase[t], sigma=blur_sigma)
        else:
            phase_smooth = phase[t]
        
        # Compute spatial gradients using central differences
        gy, gx = np.gradient(phase_smooth)
        grad_x[t] = gx
        grad_y[t] = gy
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Compute direction (negative gradient direction gives wave propagation direction)
        angle[t] = np.arctan2(-gy, -gx)
        
        # Compute temporal phase derivative for speed calculation
        if t < T - 1:
            # Phase difference to next frame (handle wrapping)
            phase_dt = np.angle(np.exp(1j * (phase[t+1] - phase[t])))
        else:
            # Use previous frame difference for last frame
            phase_dt = np.angle(np.exp(1j * (phase[t] - phase[t-1])))
        
        # Speed = temporal_phase_derivative / (2π * spatial_gradient_magnitude)
        # Avoid division by zero
        valid_mask = grad_mag > 1e-10
        speed[t][valid_mask] = np.abs(phase_dt[valid_mask]) / (2 * np.pi * grad_mag[valid_mask] * dt)
        
        # Compute frame statistics
        valid_speeds = speed[t][valid_mask]
        if len(valid_speeds) > 0:
            mean_speed[t] = np.mean(valid_speeds)
        
        valid_angles = angle[t][valid_mask]
        if len(valid_angles) > 0:
            mean_dir[t] = circmean(valid_angles)
    
    return {
        'grad_x': grad_x,
        'grad_y': grad_y,
        'speed': speed,
        'angle': angle,
        'mean_speed': mean_speed,
        'mean_dir': mean_dir
    }


def pipeline_analyse_band(movie, fps, band, blur_sigma=1.0):
    """
    One-stop analysis pipeline: band-pass → analytic cube → gradient cube.
    
    Args:
        movie (np.ndarray): Raw movie data (T, Y, X)
        fps (float): Frame rate in Hz
        band (tuple): Frequency band as (low_freq, high_freq)
        blur_sigma (float): Gaussian blur sigma for phase gradients (default: 1.0)
        
    Returns:
        dict: Complete analysis results containing all intermediate and final data
    """
    low_freq, high_freq = band
    
    # Step 1: Band-pass filter
    filtered_movie = bandpass_filter(movie, fps, low_freq, high_freq)
    
    # Step 2: Build analytic cube
    analytic_results = build_analytic_cube(filtered_movie, fps)
    
    # Step 3: Compute phase gradients
    gradient_results = phase_gradient_cube(analytic_results['phase'], fps, blur_sigma)
    
    # Combine all results into single dictionary
    results = {
        'filtered_movie': filtered_movie,
        'band': band,
        'fps': fps,
        **analytic_results,
        **gradient_results
    }
    
    return results


def show_quiver_on_frame(amplitude_frame, grad_x_frame, grad_y_frame, 
                        step=8, scale=20, alpha=0.7, title="Wave Direction"):
    """
    Plot a single frame with quiver overlay showing wave direction.
    
    Args:
        amplitude_frame (np.ndarray): Amplitude data for background (Y, X)
        grad_x_frame (np.ndarray): X-component of phase gradient (Y, X)
        grad_y_frame (np.ndarray): Y-component of phase gradient (Y, X)
        step (int): Sampling step for quiver arrows (default: 8)
        scale (float): Scaling factor for arrow length (default: 20)
        alpha (float): Transparency of quiver arrows (default: 0.7)
        title (str): Plot title
        
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Show amplitude as background
    im = ax.imshow(amplitude_frame, cmap='inferno', aspect='equal')
    
    # Create coordinate grids (subsample for clarity)
    Y, X = amplitude_frame.shape
    y_coords, x_coords = np.mgrid[0:Y:step, 0:X:step]
    
    # Subsample gradient data
    u = -grad_x_frame[::step, ::step]  # Negative for propagation direction
    v = -grad_y_frame[::step, ::step]
    
    # Add quiver plot
    ax.quiver(x_coords, y_coords, u, v, scale_units='xy', scale=scale, 
             alpha=alpha, color='white', width=0.003)
    
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Amplitude')
    
    return fig, ax


def plot_direction_histogram(directions, bins=36, title="Wave Propagation Directions"):
    """
    Plot polar histogram of wave propagation directions.
    
    Args:
        directions (np.ndarray): Array of direction angles in radians
        bins (int): Number of histogram bins (default: 36, i.e., 10-degree bins)
        title (str): Plot title
        
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Remove invalid values
    valid_dirs = directions[np.isfinite(directions)]
    
    if len(valid_dirs) == 0:
        print("No valid directions to plot")
        return None, None
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Create histogram
    counts, bin_edges = np.histogram(valid_dirs, bins=bins, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot as polar bar chart
    width = 2 * np.pi / bins
    bars = ax.bar(bin_centers, counts, width=width, alpha=0.7)
    
    ax.set_title(title, pad=20)
    ax.set_theta_zero_location('E')  # 0 degrees at east
    ax.set_theta_direction(1)  # Counterclockwise
    
    return fig, ax
