import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Original Author: Nanaki
Additional Authors: Stanley, Neil
'''

def visualize_terrain_with_cones(terrain_points, cone_points):
    """
    Visualize the full terrain and the cone points in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the terrain (blue)
    ax.scatter(
        terrain_points[:, 0],
        terrain_points[:, 1],
        terrain_points[:, 2],
        s=1,
        c='blue',
        label="Terrain"
    )

    # Plot the cones (lighter blue)
    ax.scatter(
        cone_points[:, 0],
        cone_points[:, 1],
        cone_points[:, 2],
        s=3,
        c='lightblue',
        label="Cones"
    )

    ax.set_xlim([np.min(terrain_points[:, 0]), np.max(terrain_points[:, 0])])
    ax.set_ylim([np.min(terrain_points[:, 1]), np.max(terrain_points[:, 1])])
    ax.set_zlim([np.min(terrain_points[:, 2]) - 1, np.max(terrain_points[:, 2]) + 2])

    ax.legend()
    plt.title("Full Environment: Terrain & Cones")
    plt.show()

def visualize_concentric_lidar_view(terrain_points, cone_points,
                                    ring_spacing=1.6667,  # 3x the density compared to 5.0
                                    ring_width=1.0,
                                    max_radius=20.0):
    """
    Visualize a concentric "ring-based" lidar view around (0,0).
    We keep points whose distance from (0,0) falls near a set of ring radii.

    Args:
        terrain_points: (N,3) array of all terrain points.
        cone_points: (M,3) array of cone points.
        ring_spacing: The distance between consecutive rings (reduced from 5.0 to ~1.6667).
        ring_width: Thickness of each ring (in radial distance).
        max_radius: Maximum radius we consider from (0,0).

    The rings are at distances ring_spacing, 2*ring_spacing, 3*ring_spacing, ... up to max_radius.
    """
    # 1) Define the ring positions
    ring_positions = np.arange(ring_spacing, max_radius + ring_spacing, ring_spacing)

    # 2) Compute radial distance for terrain
    r_terrain = np.sqrt(terrain_points[:, 0]**2 + terrain_points[:, 1]**2)
    # Keep points whose distance to any ring center is < ring_width/2
    mask_terrain = np.any(
        np.abs(r_terrain[:, None] - ring_positions) < (ring_width / 2),
        axis=1
    )
    local_terrain = terrain_points[mask_terrain]

    # 3) Compute radial distance for cones
    r_cones = np.sqrt(cone_points[:, 0]**2 + cone_points[:, 1]**2)
    mask_cones = np.any(
        np.abs(r_cones[:, None] - ring_positions) < (ring_width / 2),
        axis=1
    )
    local_cones = cone_points[mask_cones]

    # 4) Visualize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot local terrain points in blue
    ax.scatter(
        local_terrain[:, 0],
        local_terrain[:, 1],
        local_terrain[:, 2],
        s=3,
        c='blue',
        label="Terrain (Concentric)"
    )

    # Plot local cone points in lightblue
    ax.scatter(
        local_cones[:, 0],
        local_cones[:, 1],
        local_cones[:, 2],
        s=5,
        c='lightblue',
        label="Cones (Concentric)"
    )

    # Set axis limits to +/- max_radius in X and Y
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])

    # For Z-limits, derive from local points or just pick a fixed range
    if len(local_terrain) > 0 or len(local_cones) > 0:
        z_min = None
        z_max = None
        if len(local_terrain) > 0:
            z_min = np.min(local_terrain[:, 2])
            z_max = np.max(local_terrain[:, 2])
        if len(local_cones) > 0:
            z_min_cones = np.min(local_cones[:, 2])
            z_max_cones = np.max(local_cones[:, 2])
            if z_min is None or z_min_cones < z_min:
                z_min = z_min_cones
            if z_max is None or z_max_cones > z_max:
                z_max = z_max_cones

        # Add some padding
        pad = 1.0
        ax.set_zlim([z_min - pad, z_max + pad])
    else:
        # If no points fall into these rings
        ax.set_zlim([-2, 2])

    ax.legend()
    plt.title(
        f"Concentric Lidar View (High Density):\n"
        f"ring_spacing={ring_spacing}, ring_width={ring_width}, max_radius={max_radius}"
    )
    plt.show()

def generate_inclined_ground(num_points=8000,
                             x_range=(-20, 20),
                             y_range=(-60, 60),
                             incline=(0.2, 0.1),
                             noise_scale=0.5,
                             noise_octaves=4):
    """
    Generate a point cloud representing an uneven, inclined ground plane.
    """
    a, b = incline  # slope coefficients
    x_vals = np.random.uniform(x_range[0], x_range[1], num_points)
    y_vals = np.random.uniform(y_range[0], y_range[1], num_points)

    # Base plane
    z_vals = a * x_vals + b * y_vals

    # Add Perlin noise
    z_noise = np.array([
        pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves)
        for x, y in zip(x_vals, y_vals)
    ])
    z_vals += z_noise

    return np.vstack((x_vals, y_vals, z_vals)).T

def straight_path_function(y):
    """
    A simple function returning x=0 for all y.
    Replace with piecewise or non-linear functions as needed.
    """
    return 0.0

def get_ground_z(terrain_points, x, y):
    """
    Naive function to find the nearest (x, y) point in terrain_points and return its z.
    For large N, consider a spatial tree (KDTree) for efficiency.
    """
    dx = terrain_points[:, 0] - x
    dy = terrain_points[:, 1] - y
    dist_sq = dx*dx + dy*dy
    idx = np.argmin(dist_sq)
    return terrain_points[idx, 2]

def generate_one_cone(cx, cy, ground_z, cone_height=1.5,
                      base_radius=0.5, vertical_segments=5, radial_subdivisions=12):
    """
    Create a set of points forming a rough conical shape.
    """
    cone_pts = []
    for i in range(vertical_segments):
        frac = i / (vertical_segments - 1)  # goes 0 -> 1
        z = ground_z + frac * cone_height
        r = base_radius * (1 - frac)  # base radius shrinks to 0 at the tip

        for j in range(radial_subdivisions):
            theta = 2 * np.pi * j / radial_subdivisions
            px = cx + r * np.cos(theta)
            py = cy + r * np.sin(theta)
            cone_pts.append([px, py, z])

    return np.array(cone_pts)

def generate_cone_points_on_path(terrain_points,
                                 path_func,
                                 y_min=-60,
                                 y_max=60,
                                 step_size=5.0,
                                 left_offset=-2.0,
                                 right_offset=2.0,
                                 cone_height=1.5,
                                 base_radius=0.5):
    """
    For y in [y_min, y_max], place cones on the left and right sides of the path.
    """
    all_cones = []
    y_values = np.arange(y_min, y_max, step_size)

    for y in y_values:
        # Center of the path
        center_x = path_func(y)

        # Left cone
        x_left = center_x + left_offset
        z_left_ground = get_ground_z(terrain_points, x_left, y)
        left_cone = generate_one_cone(
            cx=x_left,
            cy=y,
            ground_z=z_left_ground,
            cone_height=cone_height,
            base_radius=base_radius
        )

        # Right cone
        x_right = center_x + right_offset
        z_right_ground = get_ground_z(terrain_points, x_right, y)
        right_cone = generate_one_cone(
            cx=x_right,
            cy=y,
            ground_z=z_right_ground,
            cone_height=cone_height,
            base_radius=base_radius
        )

        all_cones.append(left_cone)
        all_cones.append(right_cone)

    if len(all_cones) > 0:
        cone_points = np.vstack(all_cones)
    else:
        cone_points = np.empty((0, 3))

    return cone_points

def main():
    # 1) Generate the full environment (terrain)
    terrain_points = generate_inclined_ground(
        num_points=8000,
        x_range=(-20, 20),
        y_range=(-60, 60),
        incline=(0.2, 0.1),
        noise_scale=0.5,
        noise_octaves=4
    )

    # 2) Define a path function (straight line for now)
    path_func = straight_path_function

    # 3) Generate cones along that path
    cone_points = generate_cone_points_on_path(
        terrain_points=terrain_points,
        path_func=path_func,
        y_min=-60,
        y_max=60,
        step_size=5.0,
        left_offset=-2.0,
        right_offset=2.0,
        cone_height=1.5,
        base_radius=0.5
    )

    # 4) Visualize the full environment
    visualize_terrain_with_cones(terrain_points, cone_points)

    # 5) Visualize a concentric lidar view, but with 3x ring density
    visualize_concentric_lidar_view(
        terrain_points,
        cone_points,
        ring_spacing=1.6667,  # triple the density (5.0 / 3 ≈ 1.6667)
        ring_width=1.0,
        max_radius=20.0
    )

if __name__ == '__main__':
    main()
