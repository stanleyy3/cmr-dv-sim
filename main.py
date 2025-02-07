import numpy as np
from noise import pnoise2
import open3d as o3d


# ------------------------------------------------------------------------------
# Utilities to create Open3D point clouds and visualize them
# ------------------------------------------------------------------------------

def create_open3d_pcd(np_points, rgb=(0.0, 0.0, 1.0)):
    """
    Create an open3d.geometry.PointCloud from an Nx3 numpy array of points.
    Assign a uniform color 'rgb' to all points.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # Create Nx3 array of the same color
    colors = np.tile(np.array(rgb), (len(np_points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_with_open3d(terrain_points, cone_points,
                          terrain_color=(0.0, 0.0, 1.0),
                          cone_color=(0.5, 0.8, 1.0)):
    """
    Visualize the full environment (terrain + cones) using Open3D.
    """
    pcd_terrain = create_open3d_pcd(terrain_points, rgb=terrain_color)
    pcd_cones   = create_open3d_pcd(cone_points,     rgb=cone_color)

    # Open3D visualization in one window
    # You can rotate, zoom, etc. interactively
    o3d.visualization.draw_geometries([pcd_terrain, pcd_cones],
                                      window_name="Full Environment")


def visualize_concentric_with_open3d(terrain_points, cone_points,
                                     ring_spacing=5.0, ring_width=1.0,
                                     max_radius=20.0,
                                     terrain_color=(0.0, 0.0, 1.0),
                                     cone_color=(0.5, 0.8, 1.0)):
    """
    Visualize a concentric "ring-based" lidar view (multiple circular rings)
    around (0,0) up to max_radius. Only points whose radial distance from (0,0)
    is within ring_width/2 of a multiple of ring_spacing are included.
    """
    # Define ring distances: e.g. [5, 10, 15, 20] if spacing=5, max_radius=20
    ring_positions = np.arange(ring_spacing, max_radius + ring_spacing, ring_spacing)

    # Compute radial distances and filter terrain
    r_terrain = np.sqrt(terrain_points[:, 0]**2 + terrain_points[:, 1]**2)
    mask_terrain = np.any(np.abs(r_terrain[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_terrain = terrain_points[mask_terrain]

    # Filter cones
    r_cones = np.sqrt(cone_points[:, 0]**2 + cone_points[:, 1]**2)
    mask_cones = np.any(np.abs(r_cones[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_cones = cone_points[mask_cones]

    # Create Open3D point clouds
    pcd_terrain = create_open3d_pcd(local_terrain, rgb=terrain_color)
    pcd_cones   = create_open3d_pcd(local_cones,   rgb=cone_color)

    # Visualize
    o3d.visualization.draw_geometries([pcd_terrain, pcd_cones],
                                      window_name="Concentric Lidar View")


# ------------------------------------------------------------------------------
# Terrain & Cone Generation
# ------------------------------------------------------------------------------

def generate_inclined_ground(num_points=8000,
                             x_range=(-20, 20),
                             y_range=(-60, 60),
                             incline=(0.2, 0.1),
                             noise_scale=0.5,
                             noise_octaves=4):
    """
    Generate an inclined, noisy terrain as an Nx3 numpy array (x,y,z).
    """
    a, b = incline  # slope coefficients
    x_vals = np.random.uniform(x_range[0], x_range[1], num_points)
    y_vals = np.random.uniform(y_range[0], y_range[1], num_points)

    # Base incline
    z_vals = a * x_vals + b * y_vals

    # Perlin noise
    z_noise = np.array([
        pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves)
        for x, y in zip(x_vals, y_vals)
    ])
    z_vals += z_noise

    return np.vstack((x_vals, y_vals, z_vals)).T


def straight_path_function(y):
    """
    For a given y, return x=0 (straight line path).
    Replace with piecewise or non-linear logic as needed.
    """
    return 0.0


def get_ground_z(terrain_points, x, y):
    """
    Naive approach: find nearest terrain point in XY-plane and return its z.
    For large N, consider using a spatial data structure (e.g., KDTree).
    """
    dx = terrain_points[:, 0] - x
    dy = terrain_points[:, 1] - y
    dist_sq = dx*dx + dy*dy
    idx = np.argmin(dist_sq)
    return terrain_points[idx, 2]


def generate_one_cone(cx, cy, ground_z, cone_height=1.5,
                      base_radius=0.5, vertical_segments=5, radial_subdivisions=12):
    """
    Generate Nx3 array for a rough cone shape:
     - multiple horizontal rings from base (z=ground_z) up to the tip (z=ground_z+cone_height)
     - each ring shrinks in radius from base_radius down to 0 at the tip
    """
    cone_pts = []
    for i in range(vertical_segments):
        frac = i / (vertical_segments - 1)  # goes from 0 to 1
        z = ground_z + frac * cone_height
        # radius shrinks
        r = base_radius * (1 - frac)

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
    For y in [y_min, y_max], place cones at (path_center+left_offset, y)
    and (path_center+right_offset, y).
    """
    all_cones = []
    y_values = np.arange(y_min, y_max, step_size)

    for y in y_values:
        center_x = path_func(y)

        # Left cone
        x_left = center_x + left_offset
        z_left = get_ground_z(terrain_points, x_left, y)
        left_cone = generate_one_cone(
            cx=x_left,
            cy=y,
            ground_z=z_left,
            cone_height=cone_height,
            base_radius=base_radius
        )

        # Right cone
        x_right = center_x + right_offset
        z_right = get_ground_z(terrain_points, x_right, y)
        right_cone = generate_one_cone(
            cx=x_right,
            cy=y,
            ground_z=z_right,
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


# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------

def main():
    # Generate Terrain
    terrain_points = generate_inclined_ground(
        num_points=100000,
        x_range=(-50, 50),
        y_range=(-100, 100),
        incline=(0.2, 0.1),
        noise_scale=0.5,
        noise_octaves=4
    )

    # Define Path (straight line here, but can be replaced with piecewise logic)
    path_func = straight_path_function

    # Generate Cones Along the Path
    cone_points = generate_cone_points_on_path(
        terrain_points=terrain_points,
        path_func=path_func,
        y_min=-60,
        y_max=60,
        step_size=5.0,     # spacing between cones along y
        left_offset=-2.0,  # shift left
        right_offset=2.0,  # shift right
        cone_height=1.5,
        base_radius=0.5
    )

    # Visualize the Full Environment in Open3D
    visualize_with_open3d(terrain_points, cone_points,
                          terrain_color=(0, 0, 1),
                          cone_color=(0.5, 0.8, 1.0))

    # Visualize a Concentric Lidar View (rings around (0,0))
    visualize_concentric_with_open3d(
        terrain_points,
        cone_points,
        ring_spacing=5.0,
        ring_width=1.0,
        max_radius=20.0,
        terrain_color=(0, 0, 1),
        cone_color=(0.5, 0.8, 1.0)
    )

if __name__ == '__main__':
    main()
