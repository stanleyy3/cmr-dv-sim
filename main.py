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

# --- Modified Custom Visualizer with Key Callbacks for Zoom ---
def custom_draw_geometries(geometries, window_name="Open3D", lookat=None, front=None, up=None, zoom=None):
    """
    Create a custom Open3D visualizer with key callbacks for zooming.
    You can adjust the camera parameters (lookat, front, up, zoom) initially.
    Once the window is open, press "+" to zoom in and "-" to zoom out.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name)
    for geo in geometries:
        vis.add_geometry(geo)
    view_ctrl = vis.get_view_control()
    if lookat is not None:
        view_ctrl.set_lookat(lookat)
    if front is not None:
        view_ctrl.set_front(front)
    if up is not None:
        view_ctrl.set_up(up)
    # Set initial zoom value.
    current_zoom = zoom if zoom is not None else 1.0
    view_ctrl.set_zoom(current_zoom)
    # Store current zoom in a mutable dictionary.
    cam_params = {"zoom": current_zoom}

    def increase_zoom(vis_obj):
        vc = vis_obj.get_view_control()
        # Multiply zoom factor to zoom in
        cam_params["zoom"] *= 0.9
        vc.set_zoom(cam_params["zoom"])
        print("Increased zoom, new zoom factor:", cam_params["zoom"])
        return False

    def decrease_zoom(vis_obj):
        vc = vis_obj.get_view_control()
        # Multiply zoom factor to zoom out
        cam_params["zoom"] *= 1.1
        vc.set_zoom(cam_params["zoom"])
        print("Decreased zoom, new zoom factor:", cam_params["zoom"])
        return False

    vis.register_key_callback(ord("+"), increase_zoom)
    vis.register_key_callback(ord("-"), decrease_zoom)
    vis.run()
    vis.destroy_window()

def visualize_with_open3d_split(terrain_points, left_cone_points, right_cone_points,
                                terrain_color=(0.0, 0.0, 1.0),
                                left_cone_color=(1.0, 1.0, 0.0),
                                right_cone_color=(0.5, 0.8, 1.0),
                                lookat=None, front=None, up=None, zoom=None):
    """
    Visualize the full environment (terrain and cones) using a custom Open3D visualizer.
    The additional camera parameters (lookat, front, up, zoom) allow setting an initial view.
    After the window opens, use "+" and "-" to adjust the zoom.
    """
    pcd_terrain = create_open3d_pcd(terrain_points, rgb=terrain_color)
    pcd_left_cones = create_open3d_pcd(left_cone_points, rgb=left_cone_color)
    pcd_right_cones = create_open3d_pcd(right_cone_points, rgb=right_cone_color)
    
    geometries = [pcd_terrain, pcd_left_cones, pcd_right_cones]
    custom_draw_geometries(geometries, window_name="Full Environment",
                            lookat=lookat, front=front, up=up, zoom=zoom)

def visualize_concentric_with_open3d_split(terrain_points, left_cone_points, right_cone_points,
                                           ring_spacing=1.0, ring_width=0.5,
                                           max_radius=20.0,
                                           terrain_color=(0.0, 0.0, 1.0),
                                           left_cone_color=(1.0, 1.0, 0.0),
                                           right_cone_color=(0.5, 0.8, 1.0),
                                           lookat=None, front=None, up=None, zoom=None):
    """
    Visualize a concentric "ring-based" view of the environment using a custom visualizer.
    The additional camera parameters allow setting the initial view.
    Use "+" and "-" keys to adjust the zoom after the window opens.
    """
    ring_positions = np.arange(ring_spacing, max_radius + ring_spacing, ring_spacing)
    
    r_terrain = np.sqrt(terrain_points[:, 0]**2 + terrain_points[:, 1]**2)
    mask_terrain = np.any(np.abs(r_terrain[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_terrain = terrain_points[mask_terrain]
    
    r_left = np.sqrt(left_cone_points[:, 0]**2 + left_cone_points[:, 1]**2)
    mask_left = np.any(np.abs(r_left[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_left = left_cone_points[mask_left]
    
    r_right = np.sqrt(right_cone_points[:, 0]**2 + right_cone_points[:, 1]**2)
    mask_right = np.any(np.abs(r_right[:, None] - ring_positions) < (ring_width / 2), axis=1)
    local_right = right_cone_points[mask_right]
    
    pcd_terrain = create_open3d_pcd(local_terrain, rgb=terrain_color)
    pcd_left_cones = create_open3d_pcd(local_left, rgb=left_cone_color)
    pcd_right_cones = create_open3d_pcd(local_right, rgb=right_cone_color)
    
    geometries = [pcd_terrain, pcd_left_cones, pcd_right_cones]
    custom_draw_geometries(geometries, window_name="Concentric Lidar View",
                            lookat=lookat, front=front, up=up, zoom=zoom)

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
# Active Piecewise Quadratic Path Generation with Increased Variation 
# ============================================================================== 

def generate_active_piecewise_quadratic_function_with_variation(y_min, y_max, num_segments=5, x_range=(-110, 110)):
    """
    Actively generate a continuous piecewise quadratic function on [y_min, y_max] that lies entirely within x_range.
    
    We divide [y_min, y_max] into num_segments equal intervals. For each interval [y_{i-1}, y_i] (with Δy constant),
    we choose an endpoint x_i such that the difference d = x_i - x_{i-1} is randomly sampled from an allowed interval that ensures
    there remains enough room for the remaining segments. d may be positive or negative.
    
    The quadratic coefficient for each segment is then A = d/(Δy)² and the segment is defined by:
         x(y) = x_{i-1} + A · (y - y_{i-1})²   for y in [y_{i-1}, y_i].
    """
    n = num_segments
    y_breaks = np.linspace(y_min, y_max, n+1)
    delta_y = y_breaks[1] - y_breaks[0]
    min_step = 0.02 * (delta_y**2)
    max_step = 0.5 * (delta_y**2)
    
    x_min, x_max = x_range
    x_points = [np.random.uniform(x_min, x_max)]
    
    for i in range(1, n+1):
        current_x = x_points[-1]
        R = n - i  # remaining segments
        pos_lower = min_step
        pos_upper = min(max_step, x_max - current_x - R * min_step)
        neg_upper = -min_step
        neg_lower = max(-max_step, x_min - current_x + R * min_step)
        
        pos_feasible = pos_lower <= pos_upper
        neg_feasible = neg_lower <= neg_upper
        
        allowed_interval = None
        if pos_feasible and neg_feasible:
            if np.random.rand() < 0.5:
                allowed_interval = (pos_lower, pos_upper)
            else:
                allowed_interval = (neg_lower, neg_upper)
        elif pos_feasible:
            allowed_interval = (pos_lower, pos_upper)
        elif neg_feasible:
            allowed_interval = (neg_lower, neg_upper)
        else:
            raise ValueError(f"No feasible step for segment {i} from x={current_x:.2f}. Adjust parameters.")
        
        d = np.random.uniform(allowed_interval[0], allowed_interval[1])
        new_x = current_x + d
        x_points.append(new_x)
        print(f"Segment {i}: current_x = {current_x:.2f}, chosen d = {d:.2f}, new_x = {new_x:.2f}")
    
    A_coeffs = []
    for i in range(1, len(x_points)):
        d = x_points[i] - x_points[i-1]
        A = d / (delta_y**2)
        A_coeffs.append(A)
        print(f"Segment {i}: from x = {x_points[i-1]:.2f} to x = {x_points[i]:.2f}, A = {A:.3f}")
    
    def piecewise_quadratic(y):
        for i in range(1, len(y_breaks)):
            if y_breaks[i-1] <= y < y_breaks[i]:
                A = A_coeffs[i-1]
                x_start = x_points[i-1]
                return np.clip(x_start + A * (y - y_breaks[i-1])**2, x_min, x_max)
        if y == y_max:
            A = A_coeffs[-1]
            x_start = x_points[-2]
            return np.clip(x_start + A * (y - y_breaks[-2])**2, x_min, x_max)
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
    
    # Actively generate the piecewise quadratic path function with increased variation.
    path_func = generate_active_piecewise_quadratic_function_with_variation(
        y_min=-60, y_max=60,
        num_segments=5,
        x_range=(-110, 110)
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
    
    # Visualize the full environment with custom camera parameters.
    # Adjust the 'lookat', 'front', 'up', and 'zoom' parameters as needed.
    visualize_with_open3d_split(
        terrain_points,
        left_cone_points,
        right_cone_points,
        terrain_color=(0, 0, 1),
        left_cone_color=(1, 0.8, 0),
        right_cone_color=(0.5, 0.8, 1.0),
        lookat=[-30, -30, 0],
        front=[0, -1, 0],
        up=[0, 0, 1],
        zoom=0.45
    )
    
    # Visualize a concentric lidar view with custom camera parameters.
    visualize_concentric_with_open3d_split(
        terrain_points,
        left_cone_points,
        right_cone_points,
        ring_spacing=0.3,
        ring_width=0.2,
        max_radius=15.0,
        terrain_color=(0, 0, 1),
        left_cone_color=(1, 0.8, 0),
        right_cone_color=(0.5, 0.8, 1.0),
        lookat=[-30, -30, 0],
        front=[0, -1, 0],
        up=[0, 0, 1],
        zoom=0.45
    )

if __name__ == '__main__':
    main()
