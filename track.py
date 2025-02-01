import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass

@dataclass
class TrackPoint:
    x: float
    y: float
    width: float
    banking: float

class TrackGenerator:
    def __init__(self):
        """Initialize the track generator."""
        self.control_points = []
        self.track_mesh = None
        
    def add_point(self, x, y, width=10.0, banking=0.0):
        """Add a control point for the track."""
        point = TrackPoint(x, y, width, banking)
        self.control_points.append(point)
        
    def clear_points(self):
        """Clear all control points."""
        self.control_points = []
        
    def generate_spline(self, num_points=1000):
        """Generate a smooth spline through the control points."""
        if len(self.control_points) < 2:
            return None
            
        # Extract x and y coordinates
        x = np.array([p.x for p in self.control_points])
        y = np.array([p.y for p in self.control_points])
        
        # Create parameter array for spline
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, num_points)
        
        # Generate splines
        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')
        
        # Generate smooth curve
        x_smooth = cs_x(t_new)
        y_smooth = cs_y(t_new)
        
        return np.column_stack((x_smooth, y_smooth))
        
    def generate_track_mesh(self, spline_points, terrain_heights):
        """Generate the 3D mesh for the track."""
        track_vertices = []
        track_faces = []
        
        for i in range(len(spline_points)):
            point = spline_points[i]
            next_point = spline_points[(i + 1) % len(spline_points)]
            
            # Calculate track direction vector
            direction = next_point - point
            direction = direction / np.linalg.norm(direction)
            
            # Calculate normal vector (perpendicular to direction)
            normal = np.array([-direction[1], direction[0]])
            
            # Get track width at this point (interpolate from control points)
            width = self._interpolate_parameter('width', i / len(spline_points))
            banking = self._interpolate_parameter('banking', i / len(spline_points))
            
            # Generate left and right track edges
            left_edge = point + normal * width/2
            right_edge = point - normal * width/2
            
            # Add vertices with height information
            track_vertices.extend([
                [left_edge[0], left_edge[1], terrain_heights[i]],
                [right_edge[0], right_edge[1], terrain_heights[i]]
            ])
            
            # Add faces (triangles)
            if i < len(spline_points) - 1:
                v_idx = i * 2
                track_faces.extend([
                    [v_idx, v_idx + 1, v_idx + 2],
                    [v_idx + 1, v_idx + 3, v_idx + 2]
                ])
                
        return np.array(track_vertices), np.array(track_faces)
        
    def _interpolate_parameter(self, param_name, t):
        """Interpolate track parameters (width, banking) between control points."""
        values = [getattr(p, param_name) for p in self.control_points]
        t_points = np.linspace(0, 1, len(values))
        cs = CubicSpline(t_points, values, bc_type='periodic')
        return float(cs(t)) 