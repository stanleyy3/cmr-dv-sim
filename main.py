import numpy as np
from noise import pnoise2
import open3d as o3d

# ==============================================================================
# Utility Functions for Open3D Point Cloud Creation and Visualization
# ==============================================================================

def create_open3d_pcd(np_points, rgb=(0.0, 0.0, 1.0)):
    """
    Create an Open3D PointCloud from an Nx3 numpy array of points.
    Assign a uniform RGB color to all points.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    colors = np.tile(np.array(rgb), (len(np_points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_with_open3d_split(terrain_points, left_cone_points, right_cone_points,
                                terrain_color=(0.0, 0.0, 1.0),
                                left_cone_color=(1.0, 1.0, 0.0),
                                right_cone_color=(0.5, 0.8, 1.0)):
    """
    Visualize the full environment (terrain and cones) in a single Open3D window.
    """
    pcd_terrain = create_open3d_pcd(terrain_points, rgb=terrain_color)
    pcd_left_cones = create_open3d_pcd(left_cone_points, rgb=left_cone_color)
    pcd_right_cones = create_open3d_pcd(right_cone_points, rgb=right_cone_color)
    
    geometries = [pcd_terrain, pcd_left_cones, pcd_right_cones]
    o3d.visualization.draw_geometries(geometries, window_name="Full Environment")


def visualize_concentric_with_open3d_split(terrain_points, left_cone_points, right_cone_points,
                                           ring_spacing=1.0, ring_width=0.5,
                                           max_radius=20.0,
                                           terrain_color=(0.0, 0.0, 1.0),
                                           left_cone_color=(1.0, 1.0, 0.0),
                                           right_cone_color=(0.5, 0.8, 1.0)):
    """
    Visualize a concentric "ring-based" view of the environment.
    Only points within ring_width/2 of each ring (spaced every ring_spacing units) are shown.
    """
    ring_positions = np.arange(ring_spacing, max_radius + ring_spacing, ring_spacing)
    
    # Filter terrain points by radial distance from origin.
    r_terrain = np.sqrt(terrain_points[:, 0]**2 + terrain_points[:, 1]**2)
    mask_terrain = np.any(np.abs(r_terrain[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_terrain = terrain_points[mask_terrain]
    
    # Filter left cones.
    r_left = np.sqrt(left_cone_points[:, 0]**2 + left_cone_points[:, 1]**2)
    mask_left = np.any(np.abs(r_left[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_left = left_cone_points[mask_left]
    
    # Filter right cones.
    r_right = np.sqrt(right_cone_points[:, 0]**2 + right_cone_points[:, 1]**2)
    mask_right = np.any(np.abs(r_right[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_right = right_cone_points[mask_right]
    
    pcd_terrain = create_open3d_pcd(local_terrain, rgb=terrain_color)
    pcd_left_cones = create_open3d_pcd(local_left, rgb=left_cone_color)
    pcd_right_cones = create_open3d_pcd(local_right, rgb=right_cone_color)
    
    geometries = [pcd_terrain, pcd_left_cones, pcd_right_cones]
    o3d.visualization.draw_geometries(geometries, window_name="Concentric Lidar View")

# ==============================================================================
# Terrain Generation Functions
# ==============================================================================

def generate_inclined_ground(num_points=8000,
                             x_range=(-20, 20),
                             y_range=(-60, 60),
                             incline=(0.2, 0.1),
                             noise_scale=0.5,
                             noise_octaves=4):
    """
    Generate a noisy, inclined terrain represented as an Nx3 numpy array.
    """
    a, b = incline
    x_vals = np.random.uniform(x_range[0], x_range[1], num_points)
    y_vals = np.random.uniform(y_range[0], y_range[1], num_points)
    z_vals = a * x_vals + b * y_vals
    
    # Add Perlin noise to simulate natural variations.
    z_noise = np.array([
        pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves)
        for x, y in zip(x_vals, y_vals)
    ])
    z_vals += z_noise
    
    return np.vstack((x_vals, y_vals, z_vals)).T

# ==============================================================================
# Ground Lookup Function
# ==============================================================================

def get_ground_z(terrain_points, x, y):
    """
    Naively find the nearest terrain point (in the XY plane) and return its z value.
    """
    dx = terrain_points[:, 0] - x
    dy = terrain_points[:, 1] - y
    dist_sq = dx * dx + dy * dy
    idx = np.argmin(dist_sq)
    return terrain_points[idx, 2]

# ==============================================================================
# Path Generation Functions (Piecewise Quadratic)
# ==============================================================================

def generate_random_piecewise_quadratic_function(y_min, y_max, num_segments=5,
                                                   x_range=(-110, 110),
                                                   A_range=(0.02, 0.5)):
    """
    Generate a continuous piecewise quadratic function on [y_min, y_max].
    
    For each segment [y0, y1]:
      x(y) = A * (y - y0)^2 + B * (y - y0) + x0,
    where:
      - A is randomly chosen from A_range (ensuring 0.02 ≤ A ≤ 0.5).
      - x0 and x1 (the endpoints) are chosen uniformly from x_range.
      - B is computed to ensure the quadratic passes through (y1, x1).
      
    The function clamps the output to x_range.
    """
    # Create y-breakpoints for the segments
    y_breaks = np.sort(np.random.uniform(y_min, y_max, num_segments - 1))
    y_breaks = np.concatenate(([y_min], y_breaks, [y_max]))
    
    # Generate corresponding x-breakpoints uniformly from x_range
    x_breaks = np.random.uniform(x_range[0], x_range[1], num_segments + 1)
    
    segments = []
    for i in range(num_segments):
        y0 = y_breaks[i]
        y1 = y_breaks[i+1]
        x0 = x_breaks[i]
        x1 = x_breaks[i+1]
        
        # Choose A from the given range.
        A = np.random.uniform(A_range[0], A_range[1])
        
        # Compute B so that the quadratic passes through (y1, x1)
        if y1 != y0:
            B = (x1 - x0 - A * (y1 - y0)**2) / (y1 - y0)
        else:
            B = 0.0
            
        segments.append((y0, y1, A, B, x0))
    
    def piecewise_quadratic(y):
        # Determine which segment y falls into.
        for (y0, y1, A, B, x0) in segments:
            if y0 <= y < y1:
                x_val = A * (y - y0)**2 + B * (y - y0) + x0
                return np.clip(x_val, x_range[0], x_range[1])
        # If y is exactly y_max, use the last segment.
        if y == y_max:
            y0, y1, A, B, x0 = segments[-1]
            x_val = A * (y - y0)**2 + B * (y - y0) + x0
            return np.clip(x_val, x_range[0], x_range[1])
        return 0.0

    return piecewise_quadratic

# ==============================================================================
# Cone Geometry Functions
# ==============================================================================

def generate_one_cone(cx, cy, ground_z, cone_height=1.5,
                      base_radius=0.5, vertical_segments=20, radial_subdivisions=30):
    """
    Generate a rough cone geometry as an Nx3 numpy array.
    
    The cone is built by stacking horizontal rings from the base (z = ground_z)
    to the tip (z = ground_z + cone_height). The radius of the rings decreases
    linearly from base_radius to 0.
    """
    cone_pts = []
    for i in range(vertical_segments):
        frac = i / (vertical_segments - 1)
        z = ground_z + frac * cone_height
        r = base_radius * (1 - frac)
        for j in range(radial_subdivisions):
            theta = 2 * np.pi * j / radial_subdivisions
            px = cx + r * np.cos(theta)
            py = cy + r * np.sin(theta)
            cone_pts.append([px, py, z])
    return np.array(cone_pts)

# ==============================================================================
# Cone Placement Along the Spline via Arc-Length Parameterization
# ==============================================================================

def generate_cone_points_on_path_by_arc_length(terrain_points, path_func, y_min, y_max,
                                               step_size, left_offset, right_offset,
                                               cone_height, base_radius):
    """
    Place cone pairs along the spline defined by path_func at fixed arc-length intervals.
    
    Process:
      1. Sample the spline finely over [y_min, y_max] to compute cumulative arc-length.
      2. Interpolate to determine target y values corresponding to each step_size along the curve.
      3. For each target y:
         a. Compute the base point on the spline.
         b. Estimate the derivative to obtain the tangent vector.
         c. Compute left_normal = (-T_y, T_x) and right_normal = (T_y, -T_x).
         d. Offset the base point by left_offset and right_offset along these normals.
         e. Use get_ground_z to determine the correct ground z value.
         f. Generate the cone geometry.
    """
    N_samples = 1000
    y_samples = np.linspace(y_min, y_max, N_samples)
    x_samples = np.array([path_func(y) for y in y_samples])
    dx_dy = np.gradient(x_samples, y_samples)
    dy = y_samples[1] - y_samples[0]
    ds_samples = np.sqrt(dx_dy**2 + 1) * dy
    s = np.zeros_like(y_samples)
    s[1:] = np.cumsum(ds_samples[1:])
    total_length = s[-1]
    
    target_s = np.arange(0, total_length, step_size)
    target_y = np.interp(target_s, s, y_samples)
    
    left_cones_list = []
    right_cones_list = []
    eps = 1e-5
    
    for y in target_y:
        x_val = path_func(y)
        base_point = np.array([x_val, y])
        derivative = (path_func(y + eps) - path_func(y - eps)) / (2 * eps)
        T = np.array([derivative, 1.0])
        T_norm = np.linalg.norm(T)
        if T_norm == 0:
            T_norm = 1.0
        T = T / T_norm
        left_normal = np.array([-T[1], T[0]])
        right_normal = -left_normal
        left_point = base_point + left_offset * left_normal
        right_point = base_point + right_offset * right_normal
        left_z = get_ground_z(terrain_points, left_point[0], left_point[1])
        right_z = get_ground_z(terrain_points, right_point[0], right_point[1])
        left_cone = generate_one_cone(cx=left_point[0],
                                      cy=left_point[1],
                                      ground_z=left_z,
                                      cone_height=cone_height,
                                      base_radius=base_radius)
        right_cone = generate_one_cone(cx=right_point[0],
                                       cy=right_point[1],
                                       ground_z=right_z,
                                       cone_height=cone_height,
                                       base_radius=base_radius)
        left_cones_list.append(left_cone)
        right_cones_list.append(right_cone)
    
    left_cone_points = np.vstack(left_cones_list) if left_cones_list else np.empty((0, 3))
    right_cone_points = np.vstack(right_cones_list) if right_cones_list else np.empty((0, 3))
    
    return left_cone_points, right_cone_points

# ==============================================================================
# Main Program
# ==============================================================================

def main():
    # Generate terrain.
    slope_range = (-0.2, 0.2)
    a, b = np.random.uniform(low=slope_range[0], high=slope_range[1], size=2)
    
    terrain_points = generate_inclined_ground(
        num_points=1000000,
        x_range=(-110, 110),
        y_range=(-110, 110),
        incline=(a, b),
        noise_scale=0.5,
        noise_octaves=4
    )
    
    # Generate the piecewise quadratic path function.
    # Now, every A coefficient will be in the range [0.02, 0.5].
    path_func = generate_random_piecewise_quadratic_function(
        y_min=-60, y_max=60,
        num_segments=5,
        x_range=(-110, 110),
        A_range=(0.02, 0.5)
    )
    
    # Place cone pairs along the spline using arc-length based placement.
    left_cone_points, right_cone_points = generate_cone_points_on_path_by_arc_length(
        terrain_points=terrain_points,
        path_func=path_func,
        y_min=-60,
        y_max=60,
        step_size=5.0,
        left_offset=1.5,
        right_offset=1.5,
        cone_height=1,
        base_radius=0.3
    )
    
    # Visualize the full environment.
    visualize_with_open3d_split(
        terrain_points,
        left_cone_points,
        right_cone_points,
        terrain_color=(0, 0, 1),
        left_cone_color=(1, 0.8, 0),
        right_cone_color=(0.5, 0.8, 1.0)
    )
    
    # Visualize a concentric lidar view.
    visualize_concentric_with_open3d_split(
        terrain_points,
        left_cone_points,
        right_cone_points,
        ring_spacing=0.3,
        ring_width=0.2,
        max_radius=15.0,
        terrain_color=(0, 0, 1),
        left_cone_color=(1, 0.8, 0),
        right_cone_color=(0.5, 0.8, 1.0)
    )

if __name__ == '__main__':
    main()
