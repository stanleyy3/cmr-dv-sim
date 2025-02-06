'''
Original Author: Nanaki
Additional Authors: Stanley, Neil
'''

import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.interpolate import make_interp_spline

'''
    Visualize
'''
def visualise_3(true_points):
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 20])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-5, 5])
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], s=1)
    
    # Set initial view
    ax.view_init(elev=30, azim=-60)

    plt.show()

'''
    Generate elevated ground in concentric circles 
'''
def generate_inclined_ground(num_points=8000, x_range=(-50, 50), y_range=(-50, 50), 
                             incline=(0.2, 0.1), noise_scale=0.5, noise_octaves=4):
    """
    Generate a point cloud representing an uneven, inclined ground plane.
    
    Args:
        num_points: Number of points in the point cloud.
        x_range: Tuple defining the x-axis limits.
        y_range: Tuple defining the y-axis limits.
        incline: Tuple (a, b) for the incline in z = ax + by.
        noise_scale: Scale of the Perlin noise.
        noise_octaves: Number of octaves for the noise function.
        
    Returns:
        o3d.geometry.PointCloud: Generated point cloud.
    """
    a, b = incline  # Slope coefficients
    x_vals = np.random.uniform(x_range[0], x_range[1], num_points)
    y_vals = np.random.uniform(y_range[0], y_range[1], num_points)

    z_vals = a * x_vals + b * y_vals  # Base inclined plane

    # Add Perlin noise for terrain irregularities
    z_noise = np.array([pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves) 
                        for x, y in zip(x_vals, y_vals)])
    
    z_vals += z_noise  # Combine base incline with Perlin noise

    # Create point cloud
    points = np.vstack((x_vals, y_vals, z_vals)).T
    
    return points

def generate_inclined_ground_in_chunks(num_chunks=4, points_per_chunk=2000, 
                                       x_range=(-50, 50), y_range=(-50, 50),
                                       incline=(0.2, 0.1), noise_scale=0.5, noise_octaves=4):
    """
    Generate multiple smaller point clouds (chunks) that together form a larger domain.
    Each chunk has fewer points, helping keep performance manageable.
    """
    chunked_data = []
    chunk_width = (x_range[1] - x_range[0]) / num_chunks
    chunk_height = (y_range[1] - y_range[0]) / num_chunks

    for i in range(num_chunks):
        for j in range(num_chunks):
            # Determine each chunk’s sub-range
            x_min = x_range[0] + i * chunk_width
            x_max = x_min + chunk_width
            y_min = y_range[0] + j * chunk_height
            y_max = y_min + chunk_height

            # Generate points for this chunk
            x_vals = np.random.uniform(x_min, x_max, points_per_chunk)
            y_vals = np.random.uniform(y_min, y_max, points_per_chunk)

            z_vals = incline[0] * x_vals + incline[1] * y_vals
            z_noise = np.array([pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves)
                                for x, y in zip(x_vals, y_vals)])
            z_vals += z_noise

            chunk_points = np.vstack((x_vals, y_vals, z_vals)).T
            chunked_data.append(chunk_points)

    return chunked_data

def visualise_chunk(chunk_points):
    """
    Visualize a single chunk of points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(chunk_points[:, 0], chunk_points[:, 1], chunk_points[:, 2], s=1)
    ax.view_init(elev=30, azim=-60)
    plt.show()

def mask_concentric_rings(points, num_rings=15, ring_spacing=0.5, ring_width=0.3, max_radius=10):
    """
    Filters points to keep only concentric rings with a fixed spacing.

    Args:
        points: (N,3) array of (x, y, z) terrain points.
        num_rings: Number of rings to form.
        ring_spacing: Fixed separation between rings (e.g., 0.5m).
        ring_width: Thickness of each ring (controls how many points wide).
        max_radius: Maximum range for rings (e.g., 10m).

    Returns:
        Masked point cloud with only points that lie on the rings.
    """
    x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]
    
    # Compute radial distance from the origin
    r_vals = np.sqrt(x_vals**2 + y_vals**2)

    # Generate expected ring positions based on num_rings
    ring_positions = np.arange(ring_spacing, num_rings * ring_spacing, ring_spacing)

    # Define mask: Keep points close to any of the ring positions
    mask = np.any(np.abs(r_vals[:, None] - ring_positions) < (ring_width / 2), axis=1)

    return points[mask]

# Hyper params
num_rings = 10   # Number of rings
ring_spacing = 1.5  # Fixed distance between rings
ring_width = 0.45  # Width of each ring

# Generate points
terrain_points = generate_inclined_ground()
filtered_points = mask_concentric_rings(terrain_points, num_rings=num_rings, ring_spacing=ring_spacing, ring_width=ring_width)

def mask_straight_path(points, path_width=0.8, path_length=10, path_offset=2.0):
    """
    Filters points to keep only those within a straight path extending forward from (0,0).

    Args:
        points: (N,3) array of (x, y, z) terrain points.
        path_width: Thickness of the path (controls how many points wide).
        path_length: How far the path extends in the y-direction.

    Returns:
        Masked point cloud with only points that lie within the straight path.
    """
    x_vals, y_vals, z_vals = points[:, 1], points[:, 0], points[:, 2]
    
    # Define masks for the original path and parallel path (offset by `path_offset`)
    mask_original = (np.abs(x_vals) < (path_width / 2)) & (y_vals >= 0) & (y_vals <= path_length)
    mask_parallel = (np.abs(x_vals - path_offset) < (path_width / 2)) & (y_vals >= 0) & (y_vals <= path_length)

    # Combine masks to keep both paths
    mask = mask_original | mask_parallel

    return points[mask]

# masked_straight_path = mask_straight_path(filtered_points, path_width=0.3, path_length=10)

visualise_3(terrain_points)
visualise_3(filtered_points)

# Example usage
# chunks = generate_inclined_ground_in_chunks()
# visualise_chunk(chunks[0])  # Only visualize the first chunk