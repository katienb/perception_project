import h5py
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Path to your HDF5 file
h5_path = 'n:/GEVI_Wave/Analysis/Visual/cfm001mjr/20231208/meas00/cG_unmixed_dFF.h5'
tif_path = 'c:/Users/Katie/Documents/Katie/Data/cfm001_231208_00.tif'

# Open and extract the movie array
with h5py.File(h5_path, 'r') as h5f:
    movie = h5f["mov"][()][:]  # shape: (t, y, x)

print("Movie shape:", movie.shape)
print("Number of frames:", len(movie))

# Set up figure and axis
fig, ax = plt.subplots()
im = ax.imshow(movie[0], cmap='gray', vmin=np.min(movie), vmax=np.max(movie))
ax.set_title("Frame 0")

# Update function for animation
def update(frame):
    im.set_array(movie[frame])
    ax.set_title(f"Frame {frame}")
    return [im]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(movie), interval=30, blit=False)

plt.show()

# Save as multi-frame TIFF
tifffile.imwrite(tif_path, movie.astype(np.float32))  # or np.uint16 depending on dtype
print(f"Saved multi-frame TIFF to {tif_path}")

# Load TIFF movie
frames = tifffile.imread(tif_path)  # shape: (n_frames, height, width)

# Set up the figure and axes
fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='gray', vmin=frames.min(), vmax=frames.max())
ax.set_title("TIFF Movie Viewer")
ax.axis("off")

# Animation update function
def update(i):
    im.set_array(frames[i])
    return [im]

# Create and show animation
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=10, blit=True)
plt.show()