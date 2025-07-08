import tifffile
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load TIFF movie
#movie_path = "../LESS_ver_alpha/demoMovie_denoised_LESS..tif"
#movie_path = "../LESS_ver_alpha/demoMovie.tif"
#movie_path = 'c:/Users/Katie/Documents/Katie/Data/cfm001_231208_00.tif'
movie_path = 'c:/Users/Katie/Documents/Katie/Data/cfm001_231208_00_denoised_LESS..tif'
frames = tifffile.imread(movie_path)  # shape: (n_frames, height, width)

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