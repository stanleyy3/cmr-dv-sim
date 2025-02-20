import numpy as np
from noise import pnoise2
import open3d as o3d
import cv2
from scipy.interpolate import make_interp_spline
from open3d.visualization import gui, rendering
import threading
import time

# ==============================================================================
# Utility Functions
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

def generate_inclined_ground(num_points=8000,
                             x_range=(-20, 20),
                             y_range=(-60, 60),
                             incline=(0.2, 0.1),
                             noise_scale=0.5,
                             noise_octaves=4):
    """
    Generate randomly inclined + noise terrain points.
    """
    a, b = incline
    x_vals = np.random.uniform(x_range[0], x_range[1], num_points)
    y_vals = np.random.uniform(y_range[0], y_range[1], num_points)
    z_vals = a * x_vals + b * y_vals
    z_noise = np.array([pnoise2(x * noise_scale, y * noise_scale, octaves=noise_octaves)
                        for x, y in zip(x_vals, y_vals)])
    z_vals += z_noise
    return np.vstack((x_vals, y_vals, z_vals)).T

def get_ground_z(terrain_points, x, y):
    """
    Find the Z value in terrain_points nearest (x, y).
    """
    dx = terrain_points[:, 0] - x
    dy = terrain_points[:, 1] - y
    dist_sq = dx * dx + dy * dy
    idx = np.argmin(dist_sq)
    return terrain_points[idx, 2]

def run_gui_window(geometries, window_title="Open3D", width=1024, height=768,
                   lookat=None, front=None, up=None, zoom=None):
    """
    Create a simple, non-interactive GUI window with the given geometries.
    """
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(window_title, width, height)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.point_size = 5.0  # Larger points

    for i, geo in enumerate(geometries):
        scene.scene.add_geometry(f"geom_{i}", geo, material)

    bbox = scene.scene.bounding_box
    if lookat is None:
        lookat = bbox.get_center()
    scene.setup_camera(60, bbox, lookat)

    app.run()

# ==============================================================================
# Interactive Tube Offset Widget
# ==============================================================================
class InteractiveSceneWidget(gui.SceneWidget):
    """
    SceneWidget that allows CTRL + Left Mouse drag to update offset along normal/binormal.
    """
    def __init__(self, tube_update_callback, tube_geom_key, material):
        super().__init__()
        self.tube_update_callback = tube_update_callback
        self.tube_geom_key = tube_geom_key
        self.material = material
        self.center_offset = [0.0, 0.0, 0.0]  # [offset_normal, offset_binormal, offset_tangent]
        self.roll_angle = 0.0
        self.dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

    def on_mouse_event(self, event):
        if event.type == gui.MouseEvent.Type.DOWN and event.button == gui.MouseButton.LEFT:
            if event.modifiers & gui.KeyModifier.CTRL:
                self.dragging = True
                self.last_mouse_x = event.x
                self.last_mouse_y = event.y
                return gui.Widget.EventResult.HANDLED

        elif event.type == gui.MouseEvent.Type.MOVE and self.dragging:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

            scale_factor = 0.001  # Adjust sensitivity
            self.center_offset[0] += dy * scale_factor
            self.center_offset[1] += dx * scale_factor

            new_tube = self.tube_update_callback(self.center_offset, self.roll_angle)
            self.scene.remove_geometry(self.tube_geom_key)
            self.scene.add_geometry(self.tube_geom_key, new_tube, self.material)
            self.force_redraw()
            return gui.Widget.EventResult.HANDLED

        elif event.type == gui.MouseEvent.Type.UP and self.dragging:
            self.dragging = False
            return gui.Widget.EventResult.HANDLED

        return gui.Widget.EventResult.IGNORED

def run_interactive_gui_window(geometries, tube_update_callback, tube_geom_key="tube",
                               window_title="Open3D", width=1024, height=768,
                               lookat=None, front=None, up=None, zoom=None):
    """
    GUI window that lets you drag (CTRL + Left Mouse) to offset the tube geometry.
    """
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(window_title, width, height)

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.point_size = 5.0

    scene = InteractiveSceneWidget(tube_update_callback, tube_geom_key, material)
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    for i, geo in enumerate(geometries):
        if i == 0:
            scene.scene.add_geometry(tube_geom_key, geo, material)
        else:
            scene.scene.add_geometry(f"geom_{i}", geo, material)

    bbox = scene.scene.bounding_box
    if lookat is None:
        lookat = bbox.get_center()
    scene.setup_camera(60, bbox, lookat)

    app.run()

# ==============================================================================
# Spline Generation
# ==============================================================================

def generate_active_piecewise_quadratic_function_with_variation(y_min, y_max, num_segments=5, x_range=(-110,110)):
    """
    Random piecewise quadratic function: y -> x. (OPEN track)
    """
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
        allowed_interval = (
            (np.random.rand() < 0.5 and (pos_lower, pos_upper)) or (neg_lower, neg_upper)
        )
        if not (pos_feasible or neg_feasible):
            raise ValueError("No feasible step. Adjust parameters.")
        d = np.random.uniform(allowed_interval[0], allowed_interval[1])
        new_x = current_x + d
        x_points.append(new_x)

    A_coeffs = []
    for i in range(1, len(x_points)):
        d = x_points[i] - x_points[i-1]
        A = d / (delta_y**2)
        A_coeffs.append(A)

    def piecewise_quadratic(y):
        for i in range(1, len(y_breaks)):
            if y_breaks[i-1] <= y < y_breaks[i]:
                A = A_coeffs[i-1]
                x_start = x_points[i-1]
                return np.clip(x_start + A*(y - y_breaks[i-1])**2, x_min, x_max)
        if y == y_max:
            A = A_coeffs[-1]
            x_start = x_points[-2]
            return np.clip(x_start + A*(y - y_breaks[-2])**2, x_min, x_max)
        return 0.0
    return piecewise_quadratic


def push_apart(points, min_dist=15.0, iterations=3):
    """
    Push points that are too close apart by 'min_dist'.
    Called multiple 'iterations' times.
    """
    dist2 = min_dist*min_dist

    def push_once(pts):
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dx = pts[j][0]-pts[i][0]
                dy = pts[j][1]-pts[i][1]
                sqr_dist = dx*dx+dy*dy
                if sqr_dist<dist2:
                    length = np.sqrt(sqr_dist)
                    if length<1e-6:
                        continue
                    ux = dx/length
                    uy = dy/length
                    diff = min_dist - length
                    half = diff*0.5
                    pts[j][0]+=ux*half
                    pts[j][1]+=uy*half
                    pts[i][0]-=ux*half
                    pts[i][1]-=uy*half

    for _ in range(iterations):
        push_once(points)

def compute_convex_hull(pts):
    """
    Compute the convex hull of a set of 2D points in CCW order.
    Using Andrew's monotone chain algorithm.
    """
    sorted_pts = sorted(pts, key=lambda p: (p[0], p[1]))

    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    lower=[]
    for p in sorted_pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p)<=0:
            lower.pop()
        lower.append(p)

    upper=[]
    for p in reversed(sorted_pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0:
            upper.pop()
        upper.append(p)

    lower.pop()
    upper.pop()
    return lower+upper

def generate_closed_spline_function(point_count=(10,20),
                                    push_iters=3,
                                    push_dist=15.0):
    """
    Generate a "closed" racetrack-like spline function
    but specifically bounding to 200x200 => from -100..+100 in X,Y.

    The code from the article used 250x250, but you want 200x200.
    We'll define our random points in [-100,100], for both X & Y,
    giving a 200x200 region centered at (0,0).
    """
    # The article's approach: 200x200 => -100..100 in x,y
    Xmin, Xmax = -100, 100
    Ymin, Ymax = -100, 100

    min_count, max_count = point_count
    npts = np.random.randint(min_count, max_count+1)

    rand_pts=[]
    for _ in range(npts):
        rx = np.random.uniform(Xmin, Xmax)
        ry = np.random.uniform(Ymin, Ymax)
        rand_pts.append([rx, ry])

    # push points apart
    push_apart(rand_pts, min_dist=push_dist, iterations=push_iters)
    hull_pts = compute_convex_hull(rand_pts)
    push_apart(hull_pts, min_dist=push_dist, iterations=push_iters)

    hull_pts = np.array(hull_pts)
    # close the loop
    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    seg_len=[]
    total_len=0.0
    for i in range(len(hull_pts)-1):
        dx = hull_pts[i+1,0]-hull_pts[i,0]
        dy = hull_pts[i+1,1]-hull_pts[i,1]
        ln = np.sqrt(dx*dx+dy*dy)
        seg_len.append(ln)
        total_len+=ln
    cum_len=0.0
    t_vals=[0.0]
    for i in range(len(seg_len)):
        cum_len+=seg_len[i]
        t_vals.append(cum_len/total_len)

    t_vals = np.array(t_vals)
    xs = hull_pts[:,0]
    ys = hull_pts[:,1]
    sx = make_interp_spline(t_vals, xs, k=3)
    sy = make_interp_spline(t_vals, ys, k=3)

    # We'll define the domain y in [-110..110], but your track is in -100..100, so let's find minY..maxY of the hull
    minY, maxY = ys.min(), ys.max()

    def closed_spline_func(given_y):
        # clamp
        if maxY==minY:
            return xs.mean()
        frac = (given_y-minY)/(maxY-minY)
        if frac<0: frac=0
        if frac>1: frac=1
        return sx(frac)

    return closed_spline_func


def generate_spline_from_image(filename, x_range, y_range):
    """
    (unchanged)
    """
    print("Loading image:", filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load image. Check the filename and directory.")
    print("Image loaded. Shape:", img.shape)
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image.")
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2)
    pts = pts[np.argsort(pts[:,1])]

    height, width = img.shape
    x_scaled = (pts[:,0]/width)*(x_range[1]-x_range[0])+x_range[0]
    y_scaled = (pts[:,1]/height)*(y_range[1]-y_range[0])+y_range[0]

    x_min_img, x_max_img = np.min(x_scaled), np.max(x_scaled)
    y_min_img, y_max_img = np.min(y_scaled), np.max(y_scaled)
    x_scaled = (x_scaled - x_min_img)/(x_max_img - x_min_img)*(x_range[1]-x_range[0])+x_range[0]
    y_scaled = (y_scaled - y_min_img)/(y_max_img - y_min_img)*(y_range[1]-y_range[0])+y_range[0]

    unique_y, indices = np.unique(y_scaled, return_index=True)
    x_unique = x_scaled[indices]
    spline = make_interp_spline(unique_y, x_unique, k=2)
    return spline

# ==============================================================================
# Tube / Spline Visualization
# ==============================================================================
def generate_continuous_tube_mesh(curve_points, radius, resolution,
                                  center_offset=(0.0,0.0,0.0), roll_angle=0.0):
    """
    (unchanged)
    """
    n = len(curve_points)
    tangents=[]
    for i in range(n):
        if i==0:
            tangent=curve_points[1]-curve_points[0]
        elif i==n-1:
            tangent=curve_points[-1]-curve_points[-2]
        else:
            tangent=curve_points[i+1]-curve_points[i-1]
        tangent = tangent/np.linalg.norm(tangent)
        tangents.append(tangent)

    if abs(np.dot(tangents[0],[0,0,1]))<0.99:
        normal0=np.array([0,0,1])
    else:
        normal0=np.array([0,1,0])
    normal0=normal0-np.dot(normal0,tangents[0])*tangents[0]
    normal0=normal0/np.linalg.norm(normal0)
    binormal0=np.cross(tangents[0],normal0)
    binormal0=binormal0/np.linalg.norm(binormal0)

    normals=[normal0]
    binormals=[binormal0]
    for i in range(1,n):
        proj=normals[i-1]-np.dot(normals[i-1],tangents[i])*tangents[i]
        if np.linalg.norm(proj)<1e-6:
            proj=np.array([0,0,1]) if abs(np.dot(tangents[i],[0,0,1]))<0.99 else np.array([0,1,0])
            proj=proj-np.dot(proj,tangents[i])*tangents[i]
        new_normal=proj/np.linalg.norm(proj)
        new_binormal=np.cross(tangents[i],new_normal)
        new_binormal=new_binormal/np.linalg.norm(new_binormal)
        normals.append(new_normal)
        binormals.append(new_binormal)

    vertices=[]
    for i in range(n):
        cos_r=np.cos(roll_angle)
        sin_r=np.sin(roll_angle)
        rotated_normal=normals[i]*cos_r - binormals[i]*sin_r
        rotated_binormal=normals[i]*sin_r + binormals[i]*cos_r
        center=(curve_points[i]
                + center_offset[0]*rotated_normal
                + center_offset[1]*rotated_binormal
                + center_offset[2]*tangents[i])
        for j in range(resolution):
            theta=2*np.pi*j/resolution
            offset=radius*(np.cos(theta)*rotated_normal + np.sin(theta)*rotated_binormal)
            vertices.append(center+offset)
    vertices=np.array(vertices)

    faces=[]
    for i in range(n-1):
        for j in range(resolution):
            next_j=(j+1)%resolution
            idx0=i*resolution+j
            idx1=(i+1)*resolution+j
            idx2=(i+1)*resolution+next_j
            idx3=i*resolution+next_j
            faces.append([idx0,idx1,idx2])
            faces.append([idx0,idx2,idx3])
    faces=np.array(faces)

    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

def generate_thick_spline_line(terrain_points, path_func, y_min, y_max,
                               num_samples=1000, thickness=0.5, resolution=20, color=[1,0,0],
                               center_offset=(0.0,0.0,0.0), roll_angle=0.0):
    """
    (unchanged) - sample y in [y_min,y_max], do x=path_func(y)
    """
    ys=np.linspace(y_min,y_max,num_samples)
    curve_points=[]
    for y in ys:
        x=path_func(y)
        z=get_ground_z(terrain_points,x,y)
        curve_points.append([x,y,z])
    curve_points=np.array(curve_points)
    tube_mesh=generate_continuous_tube_mesh(
        curve_points,
        radius=thickness/10.0,
        resolution=resolution,
        center_offset=center_offset,
        roll_angle=roll_angle
    )
    tube_mesh.paint_uniform_color(color)
    return tube_mesh

# ==============================================================================
# Cones and Markers
# ==============================================================================
def generate_one_cone(cx, cy, ground_z, cone_height=1.5,
                      base_radius=0.5, vertical_segments=20, radial_subdivisions=30):
    """
    (unchanged)
    """
    cone_pts=[]
    for i in range(vertical_segments):
        frac=i/(vertical_segments-1)
        z=ground_z+frac*cone_height
        r=base_radius*(1-frac)
        for j in range(radial_subdivisions):
            theta=2*np.pi*j/radial_subdivisions
            px=cx+r*np.cos(theta)
            py=cy+r*np.sin(theta)
            cone_pts.append([px,py,z])
    return np.array(cone_pts)

def generate_cone_points_on_path_by_arc_length(terrain_points, path_func, y_min, y_max,
                                               step_size,left_offset,right_offset,
                                               cone_height,base_radius):
    """
    (unchanged)
    """
    N_samples=1000
    y_samples=np.linspace(y_min,y_max,N_samples)
    x_samples=np.array([path_func(y) for y in y_samples])
    dx_dy=np.gradient(x_samples,y_samples)
    dy=y_samples[1]-y_samples[0]
    ds_samples=np.sqrt(dx_dy**2+1)*dy
    s=np.zeros_like(y_samples)
    s[1:]=np.cumsum(ds_samples[1:])
    total_length=s[-1]
    target_s=np.arange(0,total_length,step_size)
    target_y=np.interp(target_s,s,y_samples)

    left_cones_list=[]
    right_cones_list=[]
    eps=1e-5
    for y in target_y:
        x_val=path_func(y)
        base_point=np.array([x_val,y])
        derivative=(path_func(y+eps)-path_func(y-eps))/(2*eps)
        T=np.array([derivative,1.0])
        T_norm=np.linalg.norm(T)
        if T_norm==0:
            T_norm=1.0
        T=T/T_norm
        left_normal=np.array([-T[1],T[0]])
        right_normal=-left_normal

        left_point=base_point+left_offset*left_normal
        right_point=base_point+right_offset*right_normal
        left_z=get_ground_z(terrain_points,left_point[0],left_point[1])
        right_z=get_ground_z(terrain_points,right_point[0],right_point[1])

        left_cone=generate_one_cone(cx=left_point[0],cy=left_point[1],
                                    ground_z=left_z,cone_height=cone_height,
                                    base_radius=base_radius)
        right_cone=generate_one_cone(cx=right_point[0],cy=right_point[1],
                                     ground_z=right_z,cone_height=cone_height,
                                     base_radius=base_radius)
        left_cones_list.append(left_cone)
        right_cones_list.append(right_cone)

    left_cone_points=np.vstack(left_cones_list) if left_cones_list else np.empty((0,3))
    right_cone_points=np.vstack(right_cones_list) if right_cones_list else np.empty((0,3))
    return left_cone_points, right_cone_points

def generate_two_corner_markers(terrain_points,x_range,y_range,marker_radius=1.0):
    """
    (unchanged)
    """
    markers=[]
    x_bl,y_bl=x_range[0],y_range[0]
    z_bl=get_ground_z(terrain_points,x_bl,y_bl)
    sphere_bl=o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
    sphere_bl.translate([x_bl,y_bl,z_bl])
    sphere_bl.paint_uniform_color([1,0,0])
    markers.append(sphere_bl)
    print(f"Bottom-left marker at: ({x_bl}, {y_bl}, {z_bl})")

    x_tr,y_tr=x_range[1],y_range[1]
    z_tr=get_ground_z(terrain_points,x_tr,y_tr)
    sphere_tr=o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
    sphere_tr.translate([x_tr,y_tr,z_tr])
    sphere_tr.paint_uniform_color([0,1,0])
    markers.append(sphere_tr)
    print(f"Top-right marker at: ({x_tr}, {y_tr}, {z_tr})")

    return markers

# ==============================================================================
# Simple Full-Environment Visualization
# ==============================================================================
def visualize_with_open3d_split(terrain_points,left_cone_points,right_cone_points,
                                terrain_color=(0.0,0.0,1.0),
                                left_cone_color=(1.0,1.0,0.0),
                                right_cone_color=(0.5,0.8,1.0),
                                lookat=None,front=None,up=None,zoom=None,
                                spline_line=None,markers=None):
    """
    (unchanged)
    """
    pcd_terrain=create_open3d_pcd(terrain_points,rgb=terrain_color)
    pcd_left_cones=create_open3d_pcd(left_cone_points,rgb=left_cone_color)
    pcd_right_cones=create_open3d_pcd(right_cone_points,rgb=right_cone_color)
    geometries=[spline_line,pcd_terrain,pcd_left_cones,pcd_right_cones]
    if markers is not None:
        geometries.extend(markers)
    run_gui_window(geometries,window_title="Full Environment",lookat=lookat,front=front,up=up,zoom=zoom)

def visualize_concentric_with_open3d_split(terrain_points,left_cone_points,right_cone_points,
                                           ring_spacing=1.0,ring_width=0.5,
                                           max_radius=20.0,
                                           terrain_color=(0.0,0.0,1.0),
                                           left_cone_color=(1.0,1.0,0.0),
                                           right_cone_color=(0.5,0.8,1.0),
                                           lookat=None,front=None,up=None,zoom=None):
    """
    (unchanged)
    """
    ring_positions=np.arange(ring_spacing,max_radius+ring_spacing,ring_spacing)
    r_terrain=np.sqrt(terrain_points[:,0]**2+terrain_points[:,1]**2)
    mask_terrain=np.any(np.abs(r_terrain[:,None]-ring_positions)<(ring_width/2),axis=1)
    local_terrain=terrain_points[mask_terrain]

    r_left=np.sqrt(left_cone_points[:,0]**2+left_cone_points[:,1]**2)
    mask_left=np.any(np.abs(r_left[:,None]-ring_positions)<(ring_width/2),axis=1)
    local_left=left_cone_points[mask_left]

    r_right=np.sqrt(right_cone_points[:,0]**2+right_cone_points[:,1]**2)
    mask_right=np.any(np.abs(r_right[:,None]-ring_positions)<(ring_width/2),axis=1)
    local_right=right_cone_points[mask_right]

    pcd_terrain=create_open3d_pcd(local_terrain,rgb=terrain_color)
    pcd_left_cones=create_open3d_pcd(local_left,rgb=left_cone_color)
    pcd_right_cones=create_open3d_pcd(local_right,rgb=right_cone_color)
    geometries=[pcd_terrain,pcd_left_cones,pcd_right_cones]
    run_gui_window(geometries,window_title="Concentric Lidar View",lookat=lookat,front=front,up=up,zoom=zoom)

# ==============================================================================
# Dynamic Concentric View (Center Recomputed Each Step, but Window Remains Open)
# ==============================================================================
def run_dynamic_concentric_view(terrain_points, left_cone_points, right_cone_points,
                                centers, ring_spacing, ring_width, max_radius,
                                step_interval, window_title="Concentric Dynamic View",
                                width=1024,height=768):
    """
    (unchanged)
    """
    app=gui.Application.instance
    app.initialize()
    window=app.create_window(window_title,width,height)
    scene_widget=gui.SceneWidget()
    scene_widget.scene=rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    material=rendering.MaterialRecord()
    material.shader="defaultLit"
    material.point_size=5.0

    def add_concentric_geometry_for_center(center):
        dx_terrain=terrain_points[:,0]-center[0]
        dy_terrain=terrain_points[:,1]-center[1]
        r_terrain=np.sqrt(dx_terrain**2+dy_terrain**2)

        ring_positions=np.arange(ring_spacing,max_radius+ring_spacing,ring_spacing)
        mask_terrain=np.any(np.abs(r_terrain[:,None]-ring_positions)<(ring_width/2),axis=1)
        local_terrain=terrain_points[mask_terrain]

        dx_left=left_cone_points[:,0]-center[0]
        dy_left=left_cone_points[:,1]-center[1]
        r_left=np.sqrt(dx_left**2+dy_left**2)
        mask_left=np.any(np.abs(r_left[:,None]-ring_positions)<(ring_width/2),axis=1)
        local_left=left_cone_points[mask_left]

        dx_right=right_cone_points[:,0]-center[0]
        dy_right=right_cone_points[:,1]-center[1]
        r_right=np.sqrt(dx_right**2+dy_right**2)
        mask_right=np.any(np.abs(r_right[:,None]-ring_positions)<(ring_width/2),axis=1)
        local_right=right_cone_points[mask_right]

        pcd_terrain=o3d.geometry.PointCloud()
        pcd_terrain.points=o3d.utility.Vector3dVector(local_terrain)
        pcd_terrain.colors=o3d.utility.Vector3dVector(np.tile([0,0,1.0],(len(local_terrain),1)))

        pcd_left=o3d.geometry.PointCloud()
        pcd_left.points=o3d.utility.Vector3dVector(local_left)
        pcd_left.colors=o3d.utility.Vector3dVector(np.tile([1.0,0.8,0.0],(len(local_left),1)))

        pcd_right=o3d.geometry.PointCloud()
        pcd_right.points=o3d.utility.Vector3dVector(local_right)
        pcd_right.colors=o3d.utility.Vector3dVector(np.tile([0.5,0.8,1.0],(len(local_right),1)))

        scene_widget.scene.add_geometry("terrain",pcd_terrain,material)
        scene_widget.scene.add_geometry("left_cones",pcd_left,material)
        scene_widget.scene.add_geometry("right_cones",pcd_right,material)

        bbox=scene_widget.scene.bounding_box
        scene_widget.setup_camera(60,bbox,center)

    add_concentric_geometry_for_center(centers[0])

    def animate():
        for c in centers:
            def update():
                scene_widget.scene.clear_geometry()
                print(f"[DYNAMIC] Updating center to: {c}")
                add_concentric_geometry_for_center(c)
                scene_widget.force_redraw()
            gui.Application.instance.post_to_main_thread(window, update)
            time.sleep(step_interval)

    t=threading.Thread(target=animate)
    t.daemon=True
    t.start()

    app.run()

# ==============================================================================
# Main Program
# ==============================================================================
def main():
    slope_range=(-0.2,0.2)
    a,b=np.random.uniform(low=slope_range[0], high=slope_range[1], size=2)
    terrain_points=generate_inclined_ground(
        num_points=1000000,
        x_range=(-110,110),
        y_range=(-110,110),
        incline=(a,b),
        noise_scale=0.5,
        noise_octaves=4
    )

    # Spline generation method
    method=input("Choose spline generation method (1: Random, 2: From image): ").strip()
    x_range=(-110,110)
    y_range=(-110,110)

    if method=="2":
        filename=input("Enter image file name (with .png or .jpg extension): ").strip()
        try:
            path_func=generate_spline_from_image(filename, x_range, y_range)
            print("Spline function generated from image.")
        except Exception as e:
            print("Error generating spline from image:",e)
            return
    else:
        # Let user pick open or closed
        print("You chose 'Random' spline generation.")
        print("Choose 'open' or 'closed' random track approach.")
        while True:
            track_type=input("Open or Closed? (o/c): ").strip().lower()
            if track_type=="o":
                print("Generating an OPEN piecewise function (spanning -110..110).")
                path_func=generate_active_piecewise_quadratic_function_with_variation(
                    y_min=y_range[0], y_max=y_range[1],
                    num_segments=5,
                    x_range=x_range
                )
                break
            elif track_type=="c":
                print("Generating a CLOSED track (200×200 bounding, from -100..100).")
                # ignoring x_range=..., we'll just do the bounding in [-100..100]
                # so the track is 200 units wide & tall
                path_func=generate_closed_spline_function(
                    point_count=(10,20),
                    push_iters=3,
                    push_dist=15.0
                )
                break
            else:
                print("Please enter 'o' or 'c'.")

    # Generate the main "tube" along the chosen path_func
    initial_offset=(0.0,0.0,0.0)
    initial_roll=0.0
    thick_spline=generate_thick_spline_line(
        terrain_points, path_func,
        y_min=y_range[0], y_max=y_range[1],
        num_samples=1000, thickness=0.5,
        resolution=20, color=[1,0,0],
        center_offset=initial_offset,
        roll_angle=initial_roll
    )

    # Generate cones
    left_cone_points, right_cone_points=generate_cone_points_on_path_by_arc_length(
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

    # Corner markers
    corner_markers=generate_two_corner_markers(terrain_points, x_range, y_range, marker_radius=1.0)

    # Tube geometry update callback (for offset dragging)
    def tube_update_callback(new_offset,new_roll):
        return generate_thick_spline_line(
            terrain_points, path_func,
            y_min=y_range[0], y_max=y_range[1],
            num_samples=1000, thickness=0.5,
            resolution=20, color=[1,0,0],
            center_offset=tuple(new_offset),
            roll_angle=new_roll
        )

    # Full environment geometry list
    geometries_full=[
        thick_spline,
        create_open3d_pcd(terrain_points, rgb=(0,0,1)),
        create_open3d_pcd(left_cone_points, rgb=(1,0.8,0)),
        create_open3d_pcd(right_cone_points, rgb=(0.5,0.8,1.0))
    ]
    if corner_markers is not None:
        geometries_full.extend(corner_markers)

    print("Interactive Controls:")
    print("  Hold the Ctrl key and drag with the left mouse button to adjust offset along normal/binormal.")
    run_interactive_gui_window(
        geometries_full,
        tube_update_callback,
        tube_geom_key="tube",
        window_title="Full Environment (Interactive)"
    )

    print("\nChoose concentric view mode:")
    print("  1: Manual center point (enter coordinates)")
    print("  2: Dynamic center along spline (auto-animate from start to end)")
    mode=input("Enter mode (1 or 2): ").strip()

    ring_spacing=0.3
    ring_width=0.2
    max_radius=15.0

    if mode=="1":
        while True:
            center_input=input("Enter center coordinates for concentric view (x,y): ").strip()
            try:
                center_xy=[float(val.strip()) for val in center_input.split(',')]
                if len(center_xy)!=2:
                    raise ValueError("Please enter exactly two values separated by a comma.")
                if not (x_range[0]<=center_xy[0]<=x_range[1]) or not (y_range[0]<=center_xy[1]<=y_range[1]):
                    raise ValueError("Coordinates out of bounds.")
                break
            except Exception as e:
                print(f"Invalid input or out of bounds. Please try again.\n{e}")

        z_val=get_ground_z(terrain_points, center_xy[0], center_xy[1])
        manual_center=[center_xy[0], center_xy[1], z_val]
        visualize_concentric_with_open3d_split(
            terrain_points, left_cone_points, right_cone_points,
            ring_spacing=ring_spacing,
            ring_width=ring_width,
            max_radius=max_radius,
            lookat=manual_center,
            front=[0,-1,0],
            up=[0,0,1],
            zoom=0.45
        )

    elif mode=="2":
        dynamic_step_count=800
        step_interval=0.0125
        # => total ~10s

        centers=[]
        for y in np.linspace(y_range[0],y_range[1],dynamic_step_count):
            x=path_func(y)
            z=get_ground_z(terrain_points,x,y)
            centers.append(np.array([x,y,z]))

        run_dynamic_concentric_view(
            terrain_points,
            left_cone_points,
            right_cone_points,
            centers,
            ring_spacing,
            ring_width,
            max_radius,
            step_interval,
            window_title="Concentric View (Dynamic)"
        )
    else:
        print("Invalid mode selected. Exiting.")

if __name__=="__main__":
    main()
