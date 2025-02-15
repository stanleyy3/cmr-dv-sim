import numpy as np
from noise import pnoise2
import open3d as o3d
import cv2
from scipy.interpolate import make_interp_spline
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

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

# ==============================================================================
# New Function: run_gui_window using open3d.visualization.gui
# ==============================================================================

def run_gui_window(geometries, window_title="Open3D", width=1024, height=768,
                   lookat=None, front=None, up=None, zoom=None):
    """
    Create a new GUI window using the new Open3D GUI module,
    add the provided geometries to a SceneWidget, and run the application.
    """
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(window_title, width, height)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    
    # Create a default material.
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    
    # Add each geometry to the scene.
    for i, geo in enumerate(geometries):
        scene.scene.add_geometry(f"geom_{i}", geo, material)
    
    # Set up the camera. Use the scene's bounding box.
    bbox = scene.scene.bounding_box
    if lookat is None:
        lookat = bbox.get_center()
    scene.setup_camera(60, bbox, lookat)
    
    app.run()

# ==============================================================================
# Updated Visualization Functions Using the New GUI Module
# ==============================================================================

def visualize_with_open3d_split(terrain_points, left_cone_points, right_cone_points,
                                terrain_color=(0.0, 0.0, 1.0),
                                left_cone_color=(1.0, 1.0, 0.0),
                                right_cone_color=(0.5, 0.8, 1.0),
                                lookat=None, front=None, up=None, zoom=None,
                                spline_line=None, markers=None):
    """
    Visualize the full environment (terrain and cones) using the new Open3D GUI module.
    If spline_line and/or markers are provided, they are added to the scene.
    """
    pcd_terrain = create_open3d_pcd(terrain_points, rgb=terrain_color)
    pcd_left_cones = create_open3d_pcd(left_cone_points, rgb=left_cone_color)
    pcd_right_cones = create_open3d_pcd(right_cone_points, rgb=right_cone_color)
    geometries = [pcd_terrain, pcd_left_cones, pcd_right_cones]
    if spline_line is not None:
        geometries.append(spline_line)
    if markers is not None:
        geometries.extend(markers)
    run_gui_window(geometries, window_title="Full Environment", lookat=lookat, front=front, up=up, zoom=zoom)

def visualize_concentric_with_open3d_split(terrain_points, left_cone_points, right_cone_points,
                                           ring_spacing=1.0, ring_width=0.5,
                                           max_radius=20.0,
                                           terrain_color=(0.0, 0.0, 1.0),
                                           left_cone_color=(1.0, 1.0, 0.0),
                                           right_cone_color=(0.5, 0.8, 1.0),
                                           lookat=None, front=None, up=None, zoom=None):
    """
    Visualize a concentric "ring-based" view of the environment using the new Open3D GUI module.
    Note: No markers are shown in the concentric view.
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
    run_gui_window(geometries, window_title="Concentric Lidar View", lookat=lookat, front=front, up=up, zoom=zoom)

# ==============================================================================
# Terrain and Spline Generation Functions
# ==============================================================================

def generate_inclined_ground(num_points=8000,
                             x_range=(-20, 20),
                             y_range=(-60, 60),
                             incline=(0.2, 0.1),
                             noise_scale=0.5,
                             noise_octaves=4):
    a, b = incline
    x_vals = np.random.uniform(x_range[0], x_range[1], num_points)
    y_vals = np.random.uniform(y_range[0], y_range[1], num_points)
    z_vals = a * x_vals + b * y_vals
    z_noise = np.array([pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves)
                         for x, y in zip(x_vals, y_vals)])
    z_vals += z_noise
    return np.vstack((x_vals, y_vals, z_vals)).T

def get_ground_z(terrain_points, x, y):
    dx = terrain_points[:, 0] - x
    dy = terrain_points[:, 1] - y
    dist_sq = dx * dx + dy * dy
    idx = np.argmin(dist_sq)
    return terrain_points[idx, 2]

def generate_active_piecewise_quadratic_function_with_variation(y_min, y_max, num_segments=5, x_range=(-110, 110)):
    n = num_segments
    y_breaks = np.linspace(y_min, y_max, n+1)
    delta_y = y_breaks[1] - y_breaks[0]
    min_step = 0.005 * (delta_y**2)
    max_step = 0.5 * (delta_y**2)
    x_min, x_max = x_range
    x_points = [(x_min + x_max) / 2.0]
    for i in range(1, n+1):
        current_x = x_points[-1]
        R = n - i
        pos_lower = min_step
        pos_upper = min(max_step, x_max - current_x - R * min_step)
        neg_upper = -min_step
        neg_lower = max(-max_step, x_min - current_x + R * min_step)
        pos_feasible = pos_lower <= pos_upper
        neg_feasible = neg_lower <= neg_upper
        allowed_interval = (np.random.rand() < 0.5 and (pos_lower, pos_upper)) or (neg_lower, neg_upper)
        if not (pos_feasible or neg_feasible):
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

def generate_spline_from_image(filename, x_range, y_range):
    """
    Generate a spline function from an image using OpenCV and SciPy.
    """
    print("Loading image:", filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load image. Check the filename and directory.")
    print("Image loaded. Shape:", img.shape)
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    print("Image thresholded. Unique values:", np.unique(thresh))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image.")
    print("Found", len(contours), "contours.")
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2)
    pts = pts[np.argsort(pts[:, 1])]
    print("Extracted", pts.shape[0], "points from the largest contour.")
    height, width = img.shape
    x_scaled = (pts[:, 0] / width) * (x_range[1] - x_range[0]) + x_range[0]
    y_scaled = (pts[:, 1] / height) * (y_range[1] - y_range[0]) + y_range[0]
    print("Before rescaling: x_scaled min/max:", np.min(x_scaled), np.max(x_scaled))
    print("Before rescaling: y_scaled min/max:", np.min(y_scaled), np.max(y_scaled))
    x_min_img, x_max_img = np.min(x_scaled), np.max(x_scaled)
    y_min_img, y_max_img = np.min(y_scaled), np.max(y_scaled)
    x_scaled = (x_scaled - x_min_img) / (x_max_img - x_min_img) * (x_range[1] - x_range[0]) + x_range[0]
    y_scaled = (y_scaled - y_min_img) / (y_max_img - y_min_img) * (y_range[1] - y_range[0]) + y_range[0]
    print("After rescaling: x_scaled min/max:", np.min(x_scaled), np.max(x_scaled))
    print("After rescaling: y_scaled min/max:", np.min(y_scaled), np.max(y_scaled))
    unique_y, indices = np.unique(y_scaled, return_index=True)
    x_unique = x_scaled[indices]
    print("Number of unique y points:", unique_y.shape[0])
    spline = make_interp_spline(unique_y, x_unique, k=2)
    print("Spline function fitted from image.")
    return spline

# ==============================================================================
# Tube and Cone Generation Functions
# ==============================================================================

def generate_continuous_tube_mesh(curve_points, radius, resolution):
    n = len(curve_points)
    tangents = []
    for i in range(n):
        if i == 0:
            tangent = curve_points[1] - curve_points[0]
        elif i == n - 1:
            tangent = curve_points[-1] - curve_points[-2]
        else:
            tangent = curve_points[i+1] - curve_points[i-1]
        tangent = tangent / np.linalg.norm(tangent)
        tangents.append(tangent)
    if abs(np.dot(tangents[0], [0, 0, 1])) < 0.99:
        normal0 = np.array([0, 0, 1])
    else:
        normal0 = np.array([0, 1, 0])
    normal0 = normal0 - np.dot(normal0, tangents[0]) * tangents[0]
    normal0 = normal0 / np.linalg.norm(normal0)
    binormal0 = np.cross(tangents[0], normal0)
    binormal0 = binormal0 / np.linalg.norm(binormal0)
    normals = [normal0]
    binormals = [binormal0]
    for i in range(1, n):
        proj = normals[i-1] - np.dot(normals[i-1], tangents[i]) * tangents[i]
        if np.linalg.norm(proj) < 1e-6:
            proj = np.array([0, 0, 1]) if abs(np.dot(tangents[i], [0, 0, 1])) < 0.99 else np.array([0, 1, 0])
            proj = proj - np.dot(proj, tangents[i]) * tangents[i]
        new_normal = proj / np.linalg.norm(proj)
        new_binormal = np.cross(tangents[i], new_normal)
        new_binormal = new_binormal / np.linalg.norm(new_binormal)
        normals.append(new_normal)
        binormals.append(new_binormal)
    vertices = []
    for i in range(n):
        for j in range(resolution):
            theta = 2 * np.pi * j / resolution
            offset = radius * (np.cos(theta) * normals[i] + np.sin(theta) * binormals[i])
            vertices.append(curve_points[i] + offset)
    vertices = np.array(vertices)
    faces = []
    for i in range(n - 1):
        for j in range(resolution):
            next_j = (j + 1) % resolution
            idx0 = i * resolution + j
            idx1 = (i + 1) * resolution + j
            idx2 = (i + 1) * resolution + next_j
            idx3 = i * resolution + next_j
            faces.append([idx0, idx1, idx2])
            faces.append([idx0, idx2, idx3])
    faces = np.array(faces)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

def generate_thick_spline_line(terrain_points, path_func, y_min, y_max,
                               num_samples=1000, thickness=0.5, resolution=20, color=[1, 0, 0]):
    y_samples = np.linspace(y_min, y_max, num_samples)
    curve_points = []
    for y in y_samples:
        x = path_func(y)
        z = get_ground_z(terrain_points, x, y)
        curve_points.append([x, y, z])
    curve_points = np.array(curve_points)
    tube_mesh = generate_continuous_tube_mesh(curve_points, radius=thickness/10.0, resolution=resolution)
    tube_mesh.paint_uniform_color(color)
    return tube_mesh

def generate_one_cone(cx, cy, ground_z, cone_height=1.5,
                      base_radius=0.5, vertical_segments=20, radial_subdivisions=30):
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

def generate_cone_points_on_path_by_arc_length(terrain_points, path_func, y_min, y_max,
                                               step_size, left_offset, right_offset,
                                               cone_height, base_radius):
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

def generate_two_corner_markers(terrain_points, x_range, y_range, marker_radius=1.0):
    markers = []
    # Bottom-left corner
    x_bl, y_bl = x_range[0], y_range[0]
    z_bl = get_ground_z(terrain_points, x_bl, y_bl)
    sphere_bl = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
    sphere_bl.translate([x_bl, y_bl, z_bl])
    sphere_bl.paint_uniform_color([1, 0, 0])  # red
    markers.append(sphere_bl)
    print(f"Bottom-left marker at: ({x_bl}, {y_bl}, {z_bl})")
    
    # Top-right corner
    x_tr, y_tr = x_range[1], y_range[1]
    z_tr = get_ground_z(terrain_points, x_tr, y_tr)
    sphere_tr = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
    sphere_tr.translate([x_tr, y_tr, z_tr])
    sphere_tr.paint_uniform_color([0, 1, 0])  # green
    markers.append(sphere_tr)
    print(f"Top-right marker at: ({x_tr}, {y_tr}, {z_tr})")
    
    return markers

# ==============================================================================
# Main Program
# ==============================================================================

def main():
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
    
    method = input("Choose spline generation method (1: Random, 2: From image): ").strip()
    x_range = (-110, 110)
    y_range = (-110, 110)
    if method == "2":
        filename = input("Enter image file name (with .png or .jpg extension): ").strip()
        try:
            path_func = generate_spline_from_image(filename, x_range, y_range)
            print("Spline function generated from image.")
        except Exception as e:
            print("Error generating spline from image:", e)
            return
    else:
        path_func = generate_active_piecewise_quadratic_function_with_variation(
            y_min=y_range[0], y_max=y_range[1],
            num_segments=5,
            x_range=x_range
        )
        print("Randomly generated spline function.")
    
    thick_spline = generate_thick_spline_line(terrain_points, path_func, y_min=y_range[0], y_max=y_range[1],
                                              num_samples=1000, thickness=0.5, resolution=20, color=[1, 0, 0])
    
    left_cone_points, right_cone_points = generate_cone_points_on_path_by_arc_length(
        terrain_points=terrain_points,
        path_func=path_func,
        y_min=y_range[0],
        y_max=y_range[1],
        step_size=5.0,
        left_offset=1.5,
        right_offset=1.5,
        cone_height=1,
        base_radius=0.3
    )
    
    # Generate two corner markers for the full environment view.
    corner_markers = generate_two_corner_markers(terrain_points, x_range, y_range, marker_radius=1.0)
    
    # Visualize full environment.
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
        zoom=0.45,
        spline_line=thick_spline,
        markers=corner_markers
    )
    
    # Prompt user for concentric view center (x,y) and validate.
    while True:
        center_input = input("Enter center coordinates for concentric view (x,y): ").strip()
        try:
            center_xy = [float(val.strip()) for val in center_input.split(',')]
            if len(center_xy) != 2:
                raise ValueError("Please enter exactly two values separated by a comma.")
            if center_xy[0] < x_range[0] or center_xy[0] > x_range[1] or center_xy[1] < y_range[0] or center_xy[1] > y_range[1]:
                raise ValueError("Coordinates out of bounds.")
            break
        except Exception as e:
            print("Invalid input or out of bounds. Please try again.")
    z_val = get_ground_z(terrain_points, center_xy[0], center_xy[1])
    center_coords = [center_xy[0], center_xy[1], z_val]
    print("Using concentric view center:", center_coords)
    
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
        lookat=center_coords,
        front=[0, -1, 0],
        up=[0, 0, 1],
        zoom=0.45
    )

if __name__ == '__main__':
    main()
